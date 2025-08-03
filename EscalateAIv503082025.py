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
import pickle

# Load environment variables from .env file for sensitive info like email and webhook URLs
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

# Initialize SQLite connection and create table for escalations if it doesn't exist
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

# Initialize Sentiment Analyzer (VADER)
analyzer = SentimentIntensityAnalyzer()

# Globals for ML model and vectorizer (initialized later)
model = None
vectorizer = None

def fetch_gmail_emails():
    """
    Connect to Gmail IMAP server, fetch last 10 unseen emails, decode,
    extract sender, subject, and body, then mark them as read.
    """
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

        for eid in email_ids[-10:]:  # limit to last 10 unseen emails
            res, msg_data = mail.fetch(eid, "(RFC822)")
            if res != "OK":
                continue
            msg = email.message_from_bytes(msg_data[0][1])

            # Decode email subject
            subject, encoding = decode_header(msg.get("Subject"))[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or "utf-8", errors="ignore")

            from_ = msg.get("From")
            date = msg.get("Date")

            # Extract plain text body from email
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    ctype = part.get_content_type()
                    cdisp = str(part.get("Content-Disposition"))
                    if ctype == "text/plain" and "attachment" not in cdisp:
                        try:
                            body = part.get_payload(decode=True).decode()
                        except Exception:
                            pass
                        break
            else:
                try:
                    body = msg.get_payload(decode=True).decode()
                except Exception:
                    pass

            emails.append({
                "customer": from_,
                "issue": body.strip(),
                "subject": subject,
                "date": date
            })

            # Mark email as read to avoid refetching
            mail.store(eid, '+FLAGS', '\\Seen')

        mail.logout()
        return emails
    except Exception as e:
        st.error(f"Error fetching emails: {e}")
        return []

def analyze_issue(issue_text):
    """
    Analyze sentiment with VADER and check negative keywords count.
    Assign priority and escalation_flag based on sentiment and negative keyword frequency.
    """
    text_lower = issue_text.lower()
    vs = analyzer.polarity_scores(issue_text)
    sentiment = "Positive" if vs["compound"] >= 0 else "Negative"
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    priority = "High" if sentiment == "Negative" and neg_count >= 2 else "Low"
    escalation_flag = 1 if priority == "High" else 0
    return sentiment, priority, escalation_flag

def save_emails_to_db(emails):
    """
    Save fetched emails to the SQLite DB, avoiding duplicates.
    Generate unique escalation IDs.
    Trigger MS Teams alert for high priority escalations.
    """
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]
    new_entries = 0
    for e in emails:
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

def send_ms_teams_alert(message):
    """
    Send alert message to MS Teams via webhook URL.
    """
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

def load_escalations_df():
    """
    Load all escalation records from the database into a pandas DataFrame.
    """
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    return df

def upload_excel_and_analyze(file):
    """
    Parse uploaded Excel file, extract necessary columns, analyze issues,
    and store new escalations in the database with unique IDs.
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
    Allow manual entry of escalation via sidebar form.
    Analyze and save to database.
    """
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

def display_kanban_card(row):
    """
    Display each escalation as a card in the Kanban board with
    color-coded priority, status, and sentiment.
    Allows inline editing of status, action taken, and owner.
    """
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

    # Header with colored bars and status indicators
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

        # Select box to update status
        new_status = st.selectbox(
            "Update Status",
            ["Open", "In Progress", "Resolved"],
            index=["Open", "In Progress", "Resolved"].index(status),
            key=f"{esc_id}_status"
        )
        # Text area for action taken update
        new_action_taken = st.text_area(
            "Action Taken",
            value=row['action_taken'] or "",
            key=f"{esc_id}_action"
        )
        # Text input for action owner update
        new_action_owner = st.text_input(
            "Action Owner",
            value=row['action_owner'] or "",
            key=f"{esc_id}_owner"
        )

        # Save updates button to write changes back to DB
        if st.button("Save Updates", key=f"save_{esc_id}"):
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("""
                UPDATE escalations SET status=?, action_taken=?, action_owner=?, status_update_date=?
                WHERE escalation_id=?
            """, (new_status, new_action_taken, new_action_owner, now, esc_id))
            conn.commit()
            st.success("Updated successfully!")
            st.experimental_rerun()  # Rerun app to reflect changes

def save_complaints_excel():
    """
    Export the current escalation data to an Excel file for download.
    """
    df = load_escalations_df()
    filename = "complaints_data.xlsx"
    df.to_excel(filename, index=False)
    return filename

def check_sla_and_alert():
    """
    Check for SLA breaches: high priority escalations still open beyond 10 minutes.
    Send alerts to MS Teams.
    """
    df = load_escalations_df()
    now = datetime.datetime.now(datetime.timezone.utc)
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
        if elapsed.total_seconds() > 10 * 60:  # 10 minutes for testing
            send_ms_teams_alert(
                f"⚠️ SLA breach detected:\nID: {row['escalation_id']}\nCustomer: {row['customer']}\nOpen for: {elapsed.seconds // 60} minutes\nIssue: {row['issue'][:200]}..."
            )
            alerts_sent += 1
    return alerts_sent

def train_escalation_model():
    """
    Train a simple logistic regression model to predict escalation priority
    based on the issue text using TF-IDF features.
    Stores trained model and vectorizer globally for prediction.
    """
    global model, vectorizer  # Must declare globals before usage

    df = load_escalations_df()
    if df.empty:
        st.warning("No data to train model.")
        return

    # Filter rows with issues and known priorities
    df = df[df['issue'].notna() & df['priority'].notna()]

    # Prepare labels: High = 1, Low = 0
    y = df['priority'].apply(lambda x: 1 if x == 'High' else 0).values

    # Initialize TF-IDF vectorizer and transform issue text
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['issue'])

    # Initialize and train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    st.success("Model trained successfully on current data.")

def predict_escalation_priority(issue_text):
    """
    Use trained model to predict if an issue is High or Low priority.
    Falls back to NLP + keyword heuristic if model unavailable.
    """
    global model, vectorizer
    if model is None or vectorizer is None:
        # Model not trained yet, fallback to heuristic
        return analyze_issue(issue_text)

    X = vectorizer.transform([issue_text])
    pred = model.predict(X)[0]
    sentiment = "Negative" if pred == 1 else "Positive"
    priority = "High" if pred == 1 else "Low"
    escalation_flag = 1 if priority == "High" else 0
    return sentiment, priority, escalation_flag

def render_kanban():
    """
    Render the main Kanban board UI with filter options, color coding,
    and buttons to fetch emails and check SLA.
    """
    # Sticky header style for title and buttons
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
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sticky container start
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

    # Sort by priority and date
    df = df.sort_values(by=['priority', 'date'], ascending=[False, True])

    # Display Kanban cards for each escalation
    for _, row in df.iterrows():
        display_kanban_card(row)

def main():
    """
    Main function to orchestrate Streamlit UI and logic.
    """
    st.sidebar.title("EscalateAI Sidebar")

    # Manual entry form
    with st.sidebar.form("manual_entry_form"):
        st.write("Add Manual Escalation")
        customer = st.text_input("Customer Email or Name")
        issue = st.text_area("Issue Description")
        submitted = st.form_submit_button("Add Escalation")
        if submitted:
            manual_entry_process(customer, issue)

    # Upload Excel file to bulk add escalations
    uploaded_file = st.sidebar.file_uploader("Upload Excel file with escalations", type=["xls", "xlsx"])
    if uploaded_file:
        added = upload_excel_and_analyze(uploaded_file)
        st.sidebar.success(f"Added {added} escalations from Excel.")

    # Train escalation prediction ML model button
    if st.sidebar.button("Train Escalation Prediction Model"):
        train_escalation_model()

    # Export escalation data as Excel
    if st.sidebar.button("Export escalations to Excel"):
        filename = save_complaints_excel()
        st.sidebar.markdown(f"[Download escalations Excel]({filename})")

    # Load and render Kanban board main UI
    render_kanban()

if __name__ == "__main__":
    main()
