import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load environment variables
load_dotenv()

EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")

NEGATIVE_KEYWORDS = [
    # Technical Failures & Product Malfunction
    "fail", "break", "crash", "defect", "fault", "degrade", "damage",
    "trip", "malfunction", "blank", "shutdown", "discharge",
    # Customer Dissatisfaction & Escalations
    "dissatisfy", "frustrate", "complain", "reject", "delay", "ignore",
    "escalate", "displease", "noncompliance", "neglect",
    # Support Gaps & Operational Delays
    "wait", "pending", "slow", "incomplete", "miss", "omit",
    "unresolved", "shortage", "no response",
    # Hazardous Conditions & Safety Risks
    "fire", "burn", "flashover", "arc", "explode", "unsafe",
    "leak", "corrode", "alarm", "incident",
    # Business Risk & Impact
    "impact", "loss", "risk", "downtime", "interrupt", "cancel",
    "terminate", "penalty"
]

# Database setup: SQLite for escalations
conn = sqlite3.connect("escalations.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS escalations (
    escalation_id TEXT PRIMARY KEY,
    customer TEXT,
    issue TEXT,
    date TEXT,
    status TEXT,
    sentiment TEXT,
    priority TEXT,
    escalation_flag INTEGER,
    action_taken TEXT,
    action_owner TEXT,
    status_update_date TEXT,
    user_feedback TEXT
)
""")
conn.commit()

analyzer = SentimentIntensityAnalyzer()

# Load or initialize escalation prediction model artifacts
MODEL_PATH = "escalation_model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

# Initialize global model and vectorizer variables
model = None
vectorizer = None

def load_model_and_vectorizer():
    """Load or initialize the escalation prediction model and vectorizer."""
    global model, vectorizer
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)
    except:
        # If not found, initialize dummy model and vectorizer (for demo)
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        model = LogisticRegression()
        # Model training/loading should be done separately with historical data

load_model_and_vectorizer()

def fetch_gmail_emails():
    """Fetch latest unseen emails from Gmail inbox."""
    if not EMAIL or not APP_PASSWORD:
        st.error("Gmail credentials not set in environment variables.")
        return []
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL, APP_PASSWORD)
        mail.select("inbox")
        result, data = mail.search(None, "UNSEEN")
        if result != "OK":
            st.info("No new emails found.")
            mail.logout()
            return []

        email_ids = data[0].split()
        emails = []

        # Limit to last 10 unseen emails for performance
        for eid in email_ids[-10:]:
            res, msg_data = mail.fetch(eid, "(RFC822)")
            if res != "OK":
                continue
            msg = email.message_from_bytes(msg_data[0][1])

            # Decode subject
            subject, encoding = decode_header(msg.get("Subject"))[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or "utf-8", errors="ignore")

            from_ = msg.get("From")
            date = msg.get("Date")

            # Extract plain text body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    ctype = part.get_content_type()
                    cdisp = str(part.get("Content-Disposition"))
                    if ctype == "text/plain" and "attachment" not in cdisp:
                        try:
                            body = part.get_payload(decode=True).decode()
                        except:
                            pass
                        break
            else:
                try:
                    body = msg.get_payload(decode=True).decode()
                except:
                    pass

            emails.append({
                "customer": from_,
                "issue": body.strip(),
                "subject": subject,
                "date": date
            })

            # Mark email as read
            mail.store(eid, '+FLAGS', '\\Seen')

        mail.logout()
        return emails
    except Exception as e:
        st.error(f"Error fetching emails: {e}")
        return []

def analyze_issue(issue_text):
    """
    Analyze sentiment, priority, escalation flag for given issue text.
    Uses VADER sentiment and keyword matching.
    """
    text_lower = issue_text.lower()
    vs = analyzer.polarity_scores(issue_text)
    sentiment = "Positive" if vs["compound"] >= 0 else "Negative"
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    priority = "High" if sentiment == "Negative" and neg_count >= 2 else "Low"
    escalation_flag = 1 if priority == "High" else 0
    return sentiment, priority, escalation_flag

def predict_escalation(issue_text):
    """
    Use ML model to predict escalation likelihood.
    Returns True if predicted escalation.
    """
    global model, vectorizer
    if not model or not vectorizer:
        return False  # Model not loaded, default no escalation

    try:
        vect = vectorizer.transform([issue_text])
        pred = model.predict(vect)
        return bool(pred[0])
    except Exception as e:
        st.warning(f"Prediction error: {e}")
        return False

def save_emails_to_db(emails):
    """
    Save fetched emails to the SQLite database,
    avoiding duplicates by checking customer and issue text.
    Sends MS Teams alerts for high priority escalations.
    """
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]
    new_entries = 0
    for e in emails:
        # Check for duplicate (customer + issue)
        cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (e['customer'], e['issue'][:500]))
        if cursor.fetchone():
            continue
        count += 1
        esc_id = f"SESICE-{count+250000}"
        sentiment, priority, escalation_flag = analyze_issue(e['issue'])

        # Additionally apply ML model prediction
        ml_pred = predict_escalation(e['issue'])
        if ml_pred:
            escalation_flag = 1
            priority = "High"

        now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
        cursor.execute("""
            INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, action_taken, action_owner, status_update_date, user_feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (esc_id, e['customer'], e['issue'][:500], e['date'], "Open", sentiment, priority, escalation_flag, "", "", now, ""))
        new_entries += 1

        # Send MS Teams alert if escalation flagged
        if escalation_flag == 1:
            send_ms_teams_alert(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {e['customer']}\nIssue: {e['issue'][:200]}...")

    conn.commit()
    return new_entries

def send_ms_teams_alert(message):
    """
    Send alert message to MS Teams channel via webhook.
    """
    if not MS_TEAMS_WEBHOOK_URL:
        st.warning("MS Teams webhook URL not set; cannot send alerts.")
        return
    headers = {"Content-Type": "application/json"}
    payload = {"text": message}
    try:
        response = requests.post(MS_TEAMS_WEBHOOK_URL, json=payload, headers=headers)
        if response.status_code != 200:
            st.error(f"MS Teams alert failed: {response.status_code} {response.text}")
    except Exception as e:
        st.error(f"Error sending MS Teams alert: {e}")

def load_escalations_df():
    """
    Load escalation records from DB as Pandas DataFrame.
    """
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    return df

def upload_excel_and_analyze(file):
    """
    Process uploaded Excel file containing complaints,
    extract relevant columns, analyze and save to DB.
    """
    try:
        df = pd.read_excel(file)
        df.columns = [c.lower().strip() for c in df.columns]
        customer_col = next((c for c in df.columns if "customer" in c or "email" in c), None)
        issue_col = next((c for c in df.columns if "issue" in c or "text" in c or "complaint" in c), None)
        date_col = next((c for c in df.columns if "date" in c), None)

        if not customer_col or not issue_col:
            st.error("Excel must contain customer/email and issue/text columns.")
            return 0

        count = 0
        cursor.execute("SELECT COUNT(*) FROM escalations")
        existing_count = cursor.fetchone()[0]

        for idx, row in df.iterrows():
            customer = str(row[customer_col])
            issue = str(row[issue_col])
            date = str(row[date_col]) if date_col and pd.notna(row[date_col]) else datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (customer, issue[:500]))
            if cursor.fetchone():
                continue
            existing_count += 1
            esc_id = f"SESICE-{existing_count+250000}"
            sentiment, priority, escalation_flag = analyze_issue(issue)

            # ML prediction override
            if predict_escalation(issue):
                escalation_flag = 1
                priority = "High"

            now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("""
                INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, action_taken, action_owner, status_update_date, user_feedback)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (esc_id, customer, issue[:500], date, "Open", sentiment, priority, escalation_flag, "", "", now, ""))
            count += 1
            if escalation_flag == 1:
                send_ms_teams_alert(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {customer}\nIssue: {issue[:200]}...")
        conn.commit()
        return count
    except Exception as e:
        st.error(f"Error processing uploaded Excel: {e}")
        return 0

def manual_entry_process(customer, issue):
    """
    Allow manual entry of escalation, analyze and save.
    """
    if not customer or not issue:
        st.sidebar.error("Please fill customer and issue.")
        return False
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]
    esc_id = f"SESICE-{count+250001}"
    sentiment, priority, escalation_flag = analyze_issue(issue)

    if predict_escalation(issue):
        escalation_flag = 1
        priority = "High"

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
    cursor.execute("""
        INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, action_taken, action_owner, status_update_date, user_feedback)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (esc_id, customer, issue[:500], now, "Open", sentiment, priority, escalation_flag, "", "", now, ""))
    conn.commit()
    st.sidebar.success(f"Added escalation {esc_id}")
    if escalation_flag == 1:
        send_ms_teams_alert(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {customer}\nIssue: {issue[:200]}...")
    return True

def display_kanban_card(row):
    """
    Display a single escalation card with color coding and expandable details.
    Allows status update, action taken, and action owner editing with save button.
    """
    esc_id = row['escalation_id']
    sentiment = row['sentiment']
    priority = row['priority']
    status = row['status']

    # Colors for sentiment, priority, and status
    sentiment_colors = {"Positive": "#27ae60", "Negative": "#e74c3c"}
    priority_colors = {"High": "#c0392b", "Low": "#27ae60"}
    status_colors = {"Open": "#f1c40f", "In Progress": "#2980b9", "Resolved": "#2ecc71"}

    border_color = priority_colors.get(priority, "#000000")
    status_color = status_colors.get(status, "#bdc3c7")
    sentiment_color = sentiment_colors.get(sentiment, "#7f8c8d")

    # Card header with colored border and labels
    header_html = f"""
    <div style="
        border-left: 6px solid {border_color};
        padding-left: 10px;
        margin-bottom: 10px;
        font-weight:bold;
        font-size: 14px;">
        {esc_id} &nbsp; 
        <span style='color:{sentiment_color}; font-weight:bold;'>● {sentiment}</span> / 
        <span style='color:{priority_colors.get(priority, '#000')}; font-weight:bold;'>■ {priority}</span> / 
        <span style='color:{status_color}; font-weight:bold;'>{status}</span>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    with st.expander("Details", expanded=False):
        st.markdown(f"**Customer:** {row['customer']}")
        st.markdown(f"**Issue:** {row['issue']}")
        st.markdown(f"**Date:** {row['date']}")

        # Inputs for updating status, action taken, owner
        new_status = st.selectbox(
            "Update Status",
            ["Open", "In Progress", "Resolved"],
            index=["Open", "In Progress", "Resolved"].index(status),
            key=f"{esc_id}_status"
        )
        new_action_taken = st.text_area(
            "Action Taken",
            value=row['action_taken'] or "",
            key=f"{esc_id}_action"
        )
        new_action_owner = st.text_input(
            "Action Owner",
            value=row['action_owner'] or "",
            key=f"{esc_id}_owner"
        )

        if st.button("Save Updates", key=f"save_{esc_id}"):
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("""
                UPDATE escalations SET status=?, action_taken=?, action_owner=?, status_update_date=?
                WHERE escalation_id=?
            """, (new_status, new_action_taken, new_action_owner, now, esc_id))
            conn.commit()
            st.success("Updated successfully!")
            st.experimental_rerun()

def save_complaints_excel():
    """
    Save the entire escalation data as Excel for download.
    """
    df = load_escalations_df()
    filename = "complaints_data.xlsx"
    df.to_excel(filename, index=False)
    return filename

def check_sla_and_alert():
    """
    Check for SLA breaches: open high priority escalations older than 10 minutes.
    Sends MS Teams alerts.
    """
    df = load_escalations_df()
    now = datetime.datetime.now(datetime.timezone.utc)
    breached = df[(df['priority'] == "High") & (df['status'] == "Open")]

    alerts_sent = 0
    for _, row in breached.iterrows():
        try:
            last_update = datetime.datetime.strptime(row['status_update_date'], "%a, %d %b %Y %H:%M:%S %z")
        except Exception:
            continue
        elapsed = now - last_update
        if elapsed.total_seconds() > 10 * 60:  # 10 minutes for testing
            send_ms_teams_alert(
                f"⚠️ SLA breach detected:\nID: {row['escalation_id']}\nCustomer: {row['customer']}\nOpen for: {elapsed.seconds // 60} minutes\nIssue: {row['issue'][:200]}..."
            )
            alerts_sent += 1
    return alerts_sent

def render_kanban():
    """
    Render the main Kanban board UI with status buckets as vertical columns,
    each with color-coded header and counts, filter options, and buttons for email fetching and SLA check.
    """
    st.markdown(
        """
        <style>
        /* Sticky header styling */
        .sticky-header {
            position: sticky;
            top: 0;
            background-color: white;
            padding: 10px 20px 10px 0;
            z-index: 100;
            border-bottom: 1px solid #ddd;
        }
        /* Container for buttons in header */
        .button-container > div {
            display: inline-block;
            margin-right: 15px;
        }
        /* Kanban columns container */
        .kanban-columns {
            display: flex;
            flex-direction: row;
            justify-content: flex-start;
            gap: 15px;
            margin-top: 20px;
        }
        /* Each Kanban column */
        .kanban-column {
            background-color: #f4f6f7;
            border-radius: 5px;
            padding: 10px;
            width: 300px; /* Fixed width for compactness */
            max-height: 600px;
            overflow-y: auto;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
        }
        /* Header of each Kanban column */
        .kanban-header {
            font-weight: bold;
            font-size: 1.2rem;
            padding: 8px 12px;
            border-radius: 5px;
            margin-bottom: 10px;
            color: white;
            user-select: none;
        }
        /* Status-specific colors */
        .status-open {
            background-color: #f1c40f; /* yellow */
        }
        .status-inprogress {
            background-color: #2980b9; /* blue */
        }
        .status-resolved {
            background-color: #2ecc71; /* green */
        }
        /* Card container inside column scroll */
        .kanban-cards {
            flex-grow: 1;
            overflow-y: auto;
        }
        /* Individual card spacing */
        .kanban-card {
            margin-bottom: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sticky header with title and action buttons
    st.markdown('<div class="sticky-header">', unsafe_allow_html=True)
    st.title("EscalateAI - AI-Powered Escalation Management")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        filter_option = st.radio("Filter Escalations:", ["All", "Escalated Only"], horizontal=True)
    with col2:
        if st.button("Fetch Latest Emails"):
            new_emails = fetch_gmail_emails()
            saved = save_emails_to_db(new_emails)
            st.success(f"Fetched and saved {saved} new escalations.")
    with col3:
        if st.button("Check SLA Breaches"):
            alerts = check_sla_and_alert()
            st.info(f"Sent {alerts} SLA breach alerts if any.")
    st.markdown("</div>", unsafe_allow_html=True)  # close sticky header div

    df = load_escalations_df()
    if filter_option == "Escalated Only":
        df = df[df['escalation_flag'] == 1]

    if df.empty:
        st.info("No escalations to display.")
        return

    # Prepare counts per status
    status_list = ["Open", "In Progress", "Resolved"]
    status_counts = {status: df[df['status'] == status].shape[0] for status in status_list}

    # Filter by status bucket selection (optional)
    selected_status = st.radio(
        label=f"Select Status Bucket (counts shown)",
        options=["All"] + status_list,
        index=0,
        format_func=lambda x: f"{x} ({'All' if x == 'All' else status_counts.get(x, 0)})"
    )

    # Filter dataframe based on selected status bucket
    if selected_status != "All":
        df = df[df['status'] == selected_status]

    # Map status to CSS class for header color
    status_class_map = {
        "Open": "status-open",
        "In Progress": "status-inprogress",
        "Resolved": "status-resolved",
    }

    # Kanban columns container div
    st.markdown('<div class="kanban-columns">', unsafe_allow_html=True)

    # For each status, render a column with header and cards
    for status in status_list:
        st.markdown(f'<div class="kanban-column">', unsafe_allow_html=True)

        # Status header with color and count
        st.markdown(
            f'<div class="kanban-header {status_class_map[status]}">{status} ({status_counts.get(status,0)})</div>',
            unsafe_allow_html=True,
        )

        # Container for cards (scrollable)
        st.markdown('<div class="kanban-cards">', unsafe_allow_html=True)

        filtered = df[df['status'] == status]

        if filtered.empty:
            st.markdown("<i>No escalations</i>", unsafe_allow_html=True)
        else:
            # Display each escalation card with margin spacing
            for _, row in filtered.iterrows():
                with st.container():
                    display_kanban_card(row)
                    st.markdown('<div style="margin-bottom:15px;"></div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # close cards container
        st.markdown("</div>", unsafe_allow_html=True)  # close kanban column

    st.markdown("</div>", unsafe_allow_html=True)  # close kanban-columns div

def main():
    """
    Main Streamlit app entry point.
    Includes sidebar for manual entry and bulk upload.
    Calls Kanban rendering.
    """
    st.sidebar.title("EscalateAI Controls")

    # Manual Entry
    st.sidebar.header("Manual Escalation Entry")
    cust_input = st.sidebar.text_input("Customer Email or Name")
    issue_input = st.sidebar.text_area("Issue Description")
    if st.sidebar.button("Add Escalation"):
        if cust_input.strip() and issue_input.strip():
            manual_entry_process(cust_input.strip(), issue_input.strip())
        else:
            st.sidebar.error("Please fill both customer and issue fields.")

    # Bulk Upload via Excel
    st.sidebar.header("Upload Excel for Bulk Escalations")
    uploaded_file = st.sidebar.file_uploader("Choose Excel file", type=["xlsx"])
    if uploaded_file:
        count = upload_excel_and_analyze(uploaded_file)
        st.sidebar.success(f"Uploaded and saved {count} escalations from Excel.")

    # Download current escalations as Excel
    if st.sidebar.button("Download Escalations Excel"):
        filename = save_complaints_excel()
        with open(filename, "rb") as f:
            st.sidebar.download_button(
                label="Download complaints_data.xlsx",
                data=f,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    render_kanban()

if __name__ == "__main__":
    main()
