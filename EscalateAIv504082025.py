import streamlit as st
import pandas as pd
import sqlite3
import os
import re
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import imaplib
import email
import smtplib
from email.mime.text import MIMEText
import requests
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load environment variables
load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", 587))
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", EMAIL_USER)
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# Initialize SQLite DB
conn = sqlite3.connect('escalate_ai.db', check_same_thread=False)
c = conn.cursor()

# Create table if not exists
c.execute('''CREATE TABLE IF NOT EXISTS escalations (
    id TEXT PRIMARY KEY,
    timestamp TEXT,
    account TEXT,
    issue_description TEXT,
    severity TEXT,
    criticality TEXT,
    category TEXT,
    sentiment REAL,
    urgency INTEGER,
    escalation_triggered INTEGER,
    status TEXT,
    action_taken TEXT,
    action_owner TEXT
)''')
conn.commit()

# VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Negative keyword list (expanded)
NEGATIVE_WORDS = set("""
fail break crash defect fault degrade damage trip malfunction blank shutdown discharge
dissatisfy frustrate complain reject delay ignore escalate displease noncompliance neglect
wait pending slow incomplete miss omit unresolved shortage no_response
fire burn flashover arc explode unsafe leak corrode alarm incident
impact loss risk downtime interrupt cancel terminate penalty
""".split())

# Status options
STATUS_OPTIONS = ['Open', 'In Progress', 'Resolved', 'Escalated']

# Generate next escalation ID
def get_next_id():
    c.execute("SELECT id FROM escalations ORDER BY id DESC LIMIT 1")
    last = c.fetchone()
    if last:
        num = int(last[0].split('-')[1])
        return f"SESICE-{num+1:07d}"
    else:
        return "SESICE-2500001"

# NLP and Tagging functions
def analyze_sentiment(text):
    vs = analyzer.polarity_scores(text)
    return vs['compound']

def detect_urgency(text):
    text_lower = text.lower()
    urgency = 0
    for word in NEGATIVE_WORDS:
        if word in text_lower:
            urgency += 1
    # VADER negative sentiment adds to urgency
    sentiment = analyze_sentiment(text)
    if sentiment < -0.5:
        urgency += 2
    elif sentiment < 0:
        urgency += 1
    return urgency

def classify_severity(urgency):
    if urgency >= 4:
        return "High"
    elif urgency >= 2:
        return "Medium"
    else:
        return "Low"

def classify_criticality(text):
    # Simple heuristic based on keywords - can be enhanced
    critical_keywords = ['fire', 'explode', 'arc', 'shutdown', 'incident']
    if any(k in text.lower() for k in critical_keywords):
        return "Critical"
    return "Normal"

def classify_category(text):
    categories = {
        "Technical Failure": ["fail", "crash", "defect", "malfunction", "shutdown", "discharge"],
        "Customer Dissatisfaction": ["complain", "reject", "delay", "ignore", "escalate", "frustrate"],
        "Support Delay": ["wait", "pending", "slow", "unresolved", "miss"],
        "Safety Risk": ["fire", "burn", "explode", "unsafe", "leak", "corrode", "alarm", "incident"],
        "Business Impact": ["impact", "loss", "risk", "downtime", "cancel", "terminate", "penalty"],
    }
    text_lower = text.lower()
    for cat, keys in categories.items():
        if any(k in text_lower for k in keys):
            return cat
    return "General"

# Database operations
def insert_escalation(entry):
    c.execute('''
        INSERT INTO escalations (id, timestamp, account, issue_description, severity, criticality, category,
            sentiment, urgency, escalation_triggered, status, action_taken, action_owner)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (entry['id'], entry['timestamp'], entry['account'], entry['issue_description'], entry['severity'],
         entry['criticality'], entry['category'], entry['sentiment'], entry['urgency'],
         entry['escalation_triggered'], entry['status'], entry['action_taken'], entry['action_owner'])
    )
    conn.commit()

def update_escalation_status(escalation_id, status):
    c.execute("UPDATE escalations SET status = ? WHERE id = ?", (status, escalation_id))
    conn.commit()

def update_action(escalation_id, action_taken, action_owner):
    c.execute("UPDATE escalations SET action_taken = ?, action_owner = ? WHERE id = ?",
              (action_taken, action_owner, escalation_id))
    conn.commit()

def get_all_escalations():
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    return df

def filter_escalations_by_status(status):
    df = pd.read_sql_query("SELECT * FROM escalations WHERE status=?", conn, params=(status,))
    return df

# Fetch unread emails from Gmail
def fetch_unread_emails():
    mail = imaplib.IMAP4_SSL(EMAIL_SERVER)
    mail.login(EMAIL_USER, EMAIL_PASS)
    mail.select("inbox")
    status, messages = mail.search(None, '(UNSEEN)')
    email_ids = messages[0].split()
    fetched_emails = []
    for e_id in email_ids:
        status, msg_data = mail.fetch(e_id, '(RFC822)')
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                subject = msg["subject"] or ""
                from_ = msg["from"] or ""
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode(errors="ignore")
                            break
                else:
                    body = msg.get_payload(decode=True).decode(errors="ignore")
                fetched_emails.append({"subject": subject, "from": from_, "body": body})
    mail.logout()
    return fetched_emails

# Send email alert via SMTP
def send_email_alert(subject, message, to_email=EMAIL_RECEIVER):
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = EMAIL_USER
    msg['To'] = to_email
    try:
        with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, to_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Email alert failed: {e}")
        return False

# Send MS Teams alert
def send_teams_alert(message):
    headers = {"Content-Type": "application/json"}
    payload = {"text": message}
    try:
        response = requests.post(MS_TEAMS_WEBHOOK_URL, headers=headers, data=json.dumps(payload))
        return response.status_code == 200
    except Exception as e:
        st.error(f"Teams alert failed: {e}")
        return False

# Process a single issue text to generate escalation record
def process_issue(account, issue_description):
    sentiment = analyze_sentiment(issue_description)
    urgency = detect_urgency(issue_description)
    severity = classify_severity(urgency)
    criticality = classify_criticality(issue_description)
    category = classify_category(issue_description)
    escalation_triggered = 1 if severity == "High" else 0
    esc_id = get_next_id()
    timestamp = datetime.now().isoformat(timespec='seconds')
    status = "Open"
    action_taken = ""
    action_owner = ""
    entry = {
        "id": esc_id,
        "timestamp": timestamp,
        "account": account,
        "issue_description": issue_description,
        "severity": severity,
        "criticality": criticality,
        "category": category,
        "sentiment": sentiment,
        "urgency": urgency,
        "escalation_triggered": escalation_triggered,
        "status": status,
        "action_taken": action_taken,
        "action_owner": action_owner,
    }
    insert_escalation(entry)
    return entry

# SLA check: find high severity issues open > 10 minutes
def check_sla_breaches():
    df = pd.read_sql_query("SELECT * FROM escalations WHERE severity='High' AND status!='Resolved'", conn)
    breaches = []
    for _, row in df.iterrows():
        timestamp = datetime.fromisoformat(row['timestamp'])
        if datetime.now() - timestamp > timedelta(minutes=10):
            breaches.append(row)
    return pd.DataFrame(breaches)

# Streamlit UI
def main():
    st.set_page_config(page_title="EscalateAI", layout="wide")
    st.title("EscalateAI - Customer Escalation Management")

    # Sidebar - Authentication
    password = st.sidebar.text_input("Admin Password", type="password")
    if password != ADMIN_PASSWORD:
        st.sidebar.error("Incorrect password.")
        st.stop()

    st.sidebar.header("Actions")

    # Fetch Emails button
    if st.sidebar.button("Fetch Emails from Gmail"):
        with st.spinner("Fetching unread emails..."):
            emails = fetch_unread_emails()
        if not emails:
            st.sidebar.info("No new unread emails found.")
        else:
            st.sidebar.success(f"Fetched {len(emails)} unread emails.")
            # Process fetched emails into escalation records
            count_processed = 0
            for em in emails:
                # Basic parsing: use email 'from' as account, email body as issue description
                account = em['from']
                desc = em['body']
                if desc.strip():
                    process_issue(account, desc)
                    count_processed += 1
            st.sidebar.success(f"Processed {count_processed} emails into escalations.")

    # Send Alerts buttons
    if st.sidebar.button("Send Email Alerts for High Priority"):
        high_priority = pd.read_sql_query("SELECT * FROM escalations WHERE severity='High' AND status!='Resolved'", conn)
        if high_priority.empty:
            st.sidebar.info("No high priority issues to alert.")
        else:
            for _, row in high_priority.iterrows():
                subj = f"Escalation Alert: {row['id']} - {row['account']}"
                msg = f"Issue: {row['issue_description']}\nSeverity: {row['severity']}\nStatus: {row['status']}\nTimestamp: {row['timestamp']}"
                send_email_alert(subj, msg)
            st.sidebar.success(f"Sent email alerts for {len(high_priority)} high priority issues.")

    if st.sidebar.button("Send MS Teams Alerts for High Priority"):
        high_priority = pd.read_sql_query("SELECT * FROM escalations WHERE severity='High' AND status!='Resolved'", conn)
        if high_priority.empty:
            st.sidebar.info("No high priority issues to alert.")
        else:
            for _, row in high_priority.iterrows():
                msg = f"**Escalation Alert:**\nID: {row['id']}\nAccount: {row['account']}\nIssue: {row['issue_description']}\nSeverity: {row['severity']}\nStatus: {row['status']}\nTimestamp: {row['timestamp']}"
                send_teams_alert(msg)
            st.sidebar.success(f"Sent MS Teams alerts for {len(high_priority)} high priority issues.")

    # Excel Upload for bulk issues
    st.sidebar.header("Bulk Upload Issues (Excel)")
    uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=['xlsx'])
    if uploaded_file:
        try:
            df_upload = pd.read_excel(uploaded_file)
            st.sidebar.write("Preview of uploaded file:")
            st.sidebar.dataframe(df_upload.head())
            if st.sidebar.button("Process Uploaded Issues"):
                count_processed = 0
                for idx, row in df_upload.iterrows():
                    account = str(row.get('account') or row.get('Account') or "Unknown")
                    desc = str(row.get('issue_description') or row.get('Issue_Description') or row.get('description') or "")
                    if desc.strip():
                        process_issue(account, desc)
                        count_processed += 1
                st.sidebar.success(f"Processed {count_processed} issues from uploaded file.")
        except Exception as e:
            st.sidebar.error(f"Failed to read uploaded file: {e}")

    # Filters and Kanban Board
    st.header("Escalations Kanban Board")
    filter_status = st.multiselect("Filter by Status", STATUS_OPTIONS, default=STATUS_OPTIONS)
    filter_severity = st.multiselect("Filter by Severity", ['High', 'Medium', 'Low'], default=['High', 'Medium', 'Low'])
    filter_category = st.multiselect("Filter by Category",
                                    ["Technical Failure", "Customer Dissatisfaction", "Support Delay", "Safety Risk", "Business Impact", "General"],
                                    default=["Technical Failure", "Customer Dissatisfaction", "Support Delay", "Safety Risk", "Business Impact", "General"])

    df_all = get_all_escalations()
    df_filtered = df_all[
        (df_all['status'].isin(filter_status)) &
        (df_all['severity'].isin(filter_severity)) &
        (df_all['category'].isin(filter_category))
    ].copy()

    # Show counts
    counts = df_all['status'].value_counts().to_dict()
    st.sidebar.markdown("### Escalation Counts by Status")
    for status in STATUS_OPTIONS:
        st.sidebar.write(f"- {status}: {counts.get(status,0)}")

    # Show Kanban by Status columns
    cols = st.columns(len(filter_status))
    for idx, status in enumerate(filter_status):
        with cols[idx]:
            st.subheader(status)
            df_status = df_filtered[df_filtered['status'] == status]
            for _, row in df_status.iterrows():
                with st.expander(f"{row['id']} - {row['account']}"):
                    st.write(f"**Issue:** {row['issue_description']}")
                    st.write(f"**Severity:** {row['severity']}")
                    st.write(f"**Criticality:** {row['criticality']}")
                    st.write(f"**Category:** {row['category']}")
                    st.write(f"**Sentiment Score:** {row['sentiment']:.2f}")
                    st.write(f"**Urgency Score:** {row['urgency']}")
                    st.write(f"**Status:** {row['status']}")
                    st.write(f"**Action Taken:** {row['action_taken']}")
                    st.write(f"**Action Owner:** {row['action_owner']}")

                    # Status update selector
                    new_status = st.selectbox(f"Update Status for {row['id']}", STATUS_OPTIONS, index=STATUS_OPTIONS.index(row['status']), key=f"status_{row['id']}")
                    if new_status != row['status']:
                        update_escalation_status(row['id'], new_status)
                        st.experimental_rerun()

                    # Action taken input
                    new_action = st.text_input(f"Action Taken for {row['id']}", value=row['action_taken'], key=f"action_{row['id']}")
                    new_owner = st.text_input(f"Action Owner for {row['id']}", value=row['action_owner'], key=f"owner_{row['id']}")
                    if st.button(f"Save Action for {row['id']}", key=f"save_{row['id']}"):
                        update_action(row['id'], new_action, new_owner)
                        st.success("Action updated")
                        st.experimental_rerun()

    # SLA breaches display
    sla_breaches = check_sla_breaches()
    if not sla_breaches.empty:
        st.warning(f"⚠️ SLA Breaches: {len(sla_breaches)} high severity issues open > 10 minutes.")
        st.dataframe(sla_breaches[['id', 'account', 'issue_description', 'timestamp', 'status']])
    else:
        st.info("No SLA breaches detected.")

    # Export all escalations
    st.header("Export Escalations Data")
    if st.button("Download Escalations Excel"):
        df_all = get_all_escalations()
        towrite = pd.ExcelWriter("escalations_export.xlsx", engine='xlsxwriter')
        df_all.to_excel(towrite, index=False, sheet_name='Escalations')
        towrite.close()
        with open("escalations_export.xlsx", "rb") as f:
            st.download_button(label="Download Excel", data=f, file_name="escalations_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
