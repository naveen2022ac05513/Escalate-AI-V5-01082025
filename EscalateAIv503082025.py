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
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load environment variables from .env file
load_dotenv()

EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")

# List of negative keywords used for priority determination
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

# Setup SQLite database connection and table creation for escalations
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

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Globals for predictive model and vectorizer
model = None
vectorizer = None

# Load or initialize predictive model & vectorizer
def load_predictive_model():
    global model, vectorizer
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
    except Exception:
        # Initialize dummy model if none exists yet
        vectorizer = TfidfVectorizer(max_features=1000)
        model = LogisticRegression()
        # Model is untrained initially

# Save predictive model & vectorizer to disk
def save_predictive_model():
    global model, vectorizer
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

# Predict escalation risk from issue text using ML model
def predict_escalation_risk(issue_text):
    global model, vectorizer
    if model is None or vectorizer is None:
        load_predictive_model()
    X = vectorizer.transform([issue_text])
    pred_prob = model.predict_proba(X)[0][1]
    return pred_prob

# Function to fetch unseen emails from Gmail inbox
def fetch_gmail_emails():
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

        for eid in email_ids[-10:]:  # last 10 unseen emails
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

# Analyze issue text to get sentiment, priority and escalation flag
def analyze_issue(issue_text):
    text_lower = issue_text.lower()
    vs = analyzer.polarity_scores(issue_text)
    sentiment = "Positive" if vs["compound"] >= 0 else "Negative"
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    priority = "High" if sentiment == "Negative" and neg_count >= 2 else "Low"
    escalation_flag = 1 if priority == "High" else 0
    return sentiment, priority, escalation_flag

# Save fetched emails to database with analysis
def save_emails_to_db(emails):
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]
    new_entries = 0
    for e in emails:
        # Avoid duplicates by customer + issue snippet check
        cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (e['customer'], e['issue'][:500]))
        if cursor.fetchone():
            continue
        count += 1
        esc_id = f"SESICE-{count+250000}"
        sentiment, priority, escalation_flag = analyze_issue(e['issue'])
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
        cursor.execute("""
            INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, action_taken, action_owner, status_update_date, user_feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (esc_id, e['customer'], e['issue'][:500], e['date'], "Open", sentiment, priority, escalation_flag, "", "", now, ""))
        new_entries += 1
        if escalation_flag == 1:
            send_ms_teams_alert(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {e['customer']}\nIssue: {e['issue'][:200]}...")
    conn.commit()
    return new_entries

# Send alert message to MS Teams channel via webhook
def send_ms_teams_alert(message):
    if not MS_TEAMS_WEBHOOK_URL:
        st.warning("MS Teams webhook URL not set; cannot send alerts.")
        return
    headers = {"Content-Type": "application/json"}
    payload = {
        "text": message
    }
    try:
        response = requests.post(MS_TEAMS_WEBHOOK_URL, json=payload, headers=headers)
        if response.status_code != 200:
            st.error(f"MS Teams alert failed: {response.status_code} {response.text}")
    except Exception as e:
        st.error(f"Error sending MS Teams alert: {e}")

# Load all escalations from DB as a DataFrame
def load_escalations_df():
    df = pd.read_sql_query("SELECT * FROM escalations ORDER BY date DESC", conn)
    return df

# Upload and analyze Excel file of escalations
def upload_excel_and_analyze(file):
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

# Manual escalation entry from sidebar form
def manual_entry_process(customer, issue):
    if not customer or not issue:
        st.sidebar.error("Please fill customer and issue.")
        return False
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]
    esc_id = f"SESICE-{count+250001}"
    sentiment, priority, escalation_flag = analyze_issue(issue)
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

# Display an individual escalation card inside the Kanban board with editable fields
def display_kanban_card(row):
    esc_id = row['escalation_id']
    sentiment = row['sentiment']
    priority = row['priority']
    status = row['status']

    sentiment_colors = {"Positive": "#27ae60", "Negative": "#e74c3c"}
    priority_colors = {"High": "#c0392b", "Low": "#27ae60"}
    status_colors = {"Open": "#f1c40f", "In Progress": "#2980b9", "Resolved": "#2ecc71"}

    border_color = priority_colors.get(priority, "#000000")
    status_color = status_colors.get(status, "#bdc3c7")
    sentiment_color = sentiment_colors.get(sentiment, "#7f8c8d")

    header_html = f"""
    <div style="
        border-left: 6px solid {border_color};
        padding-left: 10px;
        margin-bottom: 10px;
        font-weight:bold;">
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

# Save the escalations data to Excel file for download
def save_complaints_excel():
    df = load_escalations_df()
    filename = "complaints_data.xlsx"
    df.to_excel(filename, index=False)
    return filename

# Check SLA for open high priority escalations and send alerts
def check_sla_and_alert():
    """
    Checks for SLA breaches: escalations with High priority and Open status
    exceeding 10 minutes (for testing purposes).
    Sends MS Teams alerts accordingly.
    """
    df = load_escalations_df()
    now = datetime.datetime.now(datetime.timezone.utc)  # timezone aware now
    breached = df[
        (df['priority'] == "High") &
        (df['status'] == "Open")
    ]

    alerts_sent = 0
    for _, row in breached.iterrows():
        try:
            last_update = datetime.datetime.strptime(row['status_update_date'], "%a, %d %b %Y %H:%M:%S %z")
        except Exception:
            continue
        elapsed = now - last_update
        if elapsed.total_seconds() > 10 * 60:  # 10 minutes SLA for testing
            send_ms_teams_alert(
                f"⚠️ SLA breach detected:\nID: {row['escalation_id']}\nCustomer: {row['customer']}\nOpen for: {elapsed.seconds // 60} minutes\nIssue: {row['issue'][:200]}..."
            )
            alerts_sent += 1
    return alerts_sent

# Render the Kanban board UI with status buckets and counts
def render_kanban():
    """
    Render the main Kanban board UI with status buckets and counts,
    filter options, and buttons for email fetching and SLA check.
    """
    st.markdown(
        """
        <style>
        .sticky-header {
            position: sticky;
            top: 0;
            background-color: white;
            padding: 10px 20px 10px 0;
            z-index: 100;
            border-bottom: 1px solid #ddd;
        }
        .button-container > div {
            display: inline-block;
            margin-right: 15px;
        }
        .kanban-column {
            background-color: #f4f6f7;
            border-radius: 5px;
            padding: 10px;
            margin-right: 15px;
            min-width: 280px;
            max-width: 320px;
            height: 600px;
            overflow-y: auto;
        }
        .kanban-columns {
            display: flex;
            flex-direction: row;
            justify-content: flex-start;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sticky-header">', unsafe_allow_html=True)
    st.title("EscalateAI - AI-Powered Escalation Management")

    col1, col2, col3 = st.columns([2,1,1])
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

    # Show status buckets with counts and allow filtering by clicking
    selected_status = st.radio(
        label=f"Select Status Bucket (counts shown)",
        options=["All"] + status_list,
        index=0,
        format_func=lambda x: f"{x} ({'All' if x == 'All' else status_counts.get(x, 0)})"
    )

    # Filter dataframe based on selected status bucket
    if selected_status != "All":
        df = df[df['status'] == selected_status]

    # Display Kanban columns for each status with the cards inside
    st.markdown('<div class="kanban-columns">', unsafe_allow_html=True)

    for status in status_list:
        st.markdown(f'<div class="kanban-column"><h3>{status} ({status_counts.get(status,0)})</h3>', unsafe_allow_html=True)

        filtered = df[df['status'] == status]

        if filtered.empty:
            st.markdown("<i>No escalations</i>", unsafe_allow_html=True)
        else:
            for _, row in filtered.iterrows():
                display_kanban_card(row)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Main app function that controls sidebar and main page
def main():
    st.sidebar.header("EscalateAI - Escalation Entry & Upload")

    # Sidebar: Manual Entry
    st.sidebar.subheader("Manual Escalation Entry")
    customer = st.sidebar.text_input("Customer Email or Name")
    issue = st.sidebar.text_area("Issue Description")
    if st.sidebar.button("Add Escalation"):
        manual_entry_process(customer, issue)

    # Sidebar: Bulk Upload Excel
    st.sidebar.subheader("Bulk Upload Escalations (Excel)")
    uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    if uploaded_file:
        count = upload_excel_and_analyze(uploaded_file)
        if count > 0:
            st.sidebar.success(f"Imported {count} escalations from Excel.")

    # Sidebar: Download Excel of all escalations
    if st.sidebar.button("Download Escalations Excel"):
        filename = save_complaints_excel()
        with open(filename, "rb") as f:
            data = f.read()
        st.sidebar.download_button("Download Excel", data, file_name=filename)

    # Render Kanban board main view
    render_kanban()


if __name__ == "__main__":
    load_predictive_model()
    main()
