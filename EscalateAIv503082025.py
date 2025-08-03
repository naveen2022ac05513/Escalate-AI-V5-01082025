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
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Gmail & MS Teams webhook credentials from environment variables
EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")

# Negative keywords list for escalation detection
NEGATIVE_KEYWORDS = [
    "fail", "break", "crash", "defect", "fault", "degrade", "damage",
    "trip", "malfunction", "blank", "shutdown", "discharge",
    "dissatisfy", "frustrate", "complain", "reject", "delay", "ignore",
    "escalate", "displease", "noncompliance", "neglect",
    "wait", "pending", "slow", "incomplete", "miss", "omit",
    "unresolved", "shortage", "no response",
    "fire", "burn", "flashover", "arc", "explode", "unsafe",
    "leak", "corrode", "alarm", "incident",
    "impact", "loss", "risk", "downtime", "interrupt", "cancel",
    "terminate", "penalty"
]

# Connect to SQLite DB to store escalations
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

# Declare global ML model and vectorizer variables
model = None
vectorizer = None

def load_ml_model():
    """
    Initialize and train a dummy Logistic Regression model with minimal data
    to allow predictions without errors.
    """
    global model, vectorizer
    vectorizer = TfidfVectorizer()
    model = LogisticRegression()

    # Dummy training data with two classes to avoid ValueError
    X_train = vectorizer.fit_transform([
        "dummy negative example",
        "dummy positive example"
    ])
    y_train = [0, 1]
    model.fit(X_train, y_train)

def fetch_gmail_emails():
    """
    Connect to Gmail IMAP and fetch last 10 unseen emails,
    extract sender, subject, date, and plain text body.
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

        for eid in email_ids[-10:]:  # Process last 10 unseen emails only
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
    Analyze sentiment using VADER and check negative keywords
    to determine priority and escalation flag.
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
    Save new email escalations to database and send MS Teams alerts for
    high priority cases.
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
    Send alert message to MS Teams channel via webhook.
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
    Load all escalations from the database as a pandas DataFrame.
    """
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    return df

def upload_excel_and_analyze(file):
    """
    Process uploaded Excel file with complaints,
    analyze and save new entries to database.
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
    Save manual escalation entry from user input and send alert if high priority.
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

def predict_escalation(issue_text):
    """
    Predict escalation flag (0 or 1) using the ML model.
    """
    global model, vectorizer
    if model is None or vectorizer is None:
        load_ml_model()
    X = vectorizer.transform([issue_text])
    pred = model.predict(X)
    return int(pred[0])

def display_kanban_card(row):
    """
    Display a single escalation card with status, priority, sentiment, and editable fields.
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

        # Use unique keys to avoid Streamlit duplicate key errors
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
        new_feedback = st.text_input(
            "User Feedback",
            value=row.get('user_feedback', "") or "",
            key=f"{esc_id}_feedback"
        )

        if st.button("Save Updates", key=f"save_{esc_id}"):
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("""
                UPDATE escalations SET status=?, action_taken=?, action_owner=?, status_update_date=?, user_feedback=?
                WHERE escalation_id=?
            """, (new_status, new_action_taken, new_action_owner, now, new_feedback, esc_id))
            conn.commit()
            st.success("Updated successfully!")
            st.experimental_rerun()

def save_complaints_excel():
    """
    Export the current escalations database as an Excel file for download.
    """
    df = load_escalations_df()
    filename = "complaints_data.xlsx"
    df.to_excel(filename, index=False)
    return filename

def check_sla_and_alert():
    """
    Check for SLA breaches where high priority escalations
    have been open for more than 10 minutes (testing threshold).
    Sends MS Teams alert if breached.
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

def render_kanban():
    """
    Render the Kanban board with three status columns (Open, In Progress, Resolved).
    Shows counts and colored headers for visual appeal.
    """
    st.markdown(
        """
        <style>
        .sticky-header {
            position: sticky;
            top: 0;
            background-color: white;
            padding: 10px 0 10px 0;
            z-index: 100;
            border-bottom: 1px solid #ddd;
        }
        .kanban-column {
            background-color: #f9f9f9;
            border-radius: 5px;
            padding: 10px;
            min-height: 500px;
            max-height: 800px;
            overflow-y: auto;
        }
        .kanban-header {
            font-weight: bold;
            font-size: 20px;
            margin-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Sticky header with title and buttons
    st.markdown('<div class="sticky-header">', unsafe_allow_html=True)
    st.markdown("<h1>🚀 EscalateAI - Escalations & Complaints Kanban Board</h1>", unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns([1,1])
    with col_btn1:
        if st.button("📧 Fetch Emails Manually"):
            emails = fetch_gmail_emails()
            if emails:
                new_count = save_emails_to_db(emails)
                st.success(f"Fetched and saved {new_count} new emails.")
            else:
                st.info("No new emails or error.")
    with col_btn2:
        if st.button("⏰ Trigger SLA Alert Check"):
            alerts_sent = check_sla_and_alert()
            if alerts_sent > 0:
                st.success(f"Sent {alerts_sent} SLA breach alert(s) to MS Teams.")
            else:
                st.info("No SLA breaches detected at this time.")
    st.markdown('</div>', unsafe_allow_html=True)

    df = load_escalations_df()
    filter_choice = st.radio("Filter Escalations by Status", options=["All", "Open", "In Progress", "Resolved"])

    if filter_choice != "All":
        df = df[df['status'] == filter_choice]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🟡 Open")
        open_df = df[df['status'] == "Open"]
        for _, row in open_df.iterrows():
            display_kanban_card(row)

    with col2:
        st.markdown("### 🔵 In Progress")
        inprogress_df = df[df['status'] == "In Progress"]
        for _, row in inprogress_df.iterrows():
            display_kanban_card(row)

    with col3:
        st.markdown("### ✅ Resolved")
        resolved_df = df[df['status'] == "Resolved"]
        for _, row in resolved_df.iterrows():
            display_kanban_card(row)

def main():
    st.set_page_config(page_title="EscalateAI", layout="wide")
    st.title("EscalateAI - AI-based Customer Escalation Management Tool")

    load_ml_model()

    menu = st.sidebar.selectbox("Menu", ["Dashboard / Kanban", "Add Escalation Manually", "Upload Complaints Excel", "Download Complaints Excel"])

    if menu == "Dashboard / Kanban":
        render_kanban()

    elif menu == "Add Escalation Manually":
        st.sidebar.header("Manual Escalation Entry")
        customer = st.sidebar.text_input("Customer Email or Name")
        issue = st.sidebar.text_area("Issue / Complaint Description")
        if st.sidebar.button("Add Escalation"):
            if manual_entry_process(customer, issue):
                st.sidebar.success("Escalation added successfully!")

    elif menu == "Upload Complaints Excel":
        uploaded_file = st.sidebar.file_uploader("Upload Excel file with complaints", type=["xlsx"])
        if uploaded_file:
            count = upload_excel_and_analyze(uploaded_file)
            st.sidebar.success(f"Processed and added {count} new escalations from Excel.")

    elif menu == "Download Complaints Excel":
        filename = save_complaints_excel()
        st.sidebar.markdown(f"[Download Complaints Excel](./{filename})")

if __name__ == "__main__":
    main()
