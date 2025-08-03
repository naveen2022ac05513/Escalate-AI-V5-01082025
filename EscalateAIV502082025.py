# EscalateAI – Full Deployment Ready Version

import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh
import base64

# Load environment variables
load_dotenv()
EMAIL = os.getenv("EMAIL_USER")
PASSWORD = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER")
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")

# Constants
DB_FILE = "escalations.db"
ESCALATION_PREFIX = "SESICE"
ESCALATION_START = 250001
NEGATIVE_WORDS = [
    "delay", "poor", "worst", "angry", "bad", "escalate", "frustrated",
    "not happy", "disappointed", "unacceptable", "issue", "problem", "urgent", "critical"
]

# Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS escalations (
        id TEXT PRIMARY KEY,
        timestamp TEXT,
        customer TEXT,
        issue TEXT,
        sentiment TEXT,
        urgency TEXT,
        severity TEXT,
        criticality TEXT,
        category TEXT,
        status TEXT DEFAULT 'Open',
        action_taken TEXT,
        action_owner TEXT
    )''')
    conn.commit()
    conn.close()

def generate_escalation_id():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM escalations")
    count = c.fetchone()[0]
    conn.close()
    return f"{ESCALATION_PREFIX}-{ESCALATION_START + count}"

def analyze_issue(issue):
    sentiment_score = analyzer.polarity_scores(issue)['compound']
    sentiment = "Negative" if sentiment_score < -0.05 else ("Positive" if sentiment_score > 0.05 else "Neutral")
    urgency = "High" if any(word in issue.lower() for word in NEGATIVE_WORDS) else "Normal"
    severity = "Critical" if urgency == "High" else "Medium"
    criticality = "Urgent" if urgency == "High" else "Routine"
    category = "Complaint" if sentiment == "Negative" else "Feedback"
    return sentiment, urgency, severity, criticality, category

def insert_escalation(customer, issue):
    sentiment, urgency, severity, criticality, category = analyze_issue(issue)
    esc_id = generate_escalation_id()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''INSERT INTO escalations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (esc_id, timestamp, customer, issue, sentiment, urgency, severity, criticality, category, 'Open', '', ''))
    conn.commit()
    conn.close()
    return esc_id, urgency

def fetch_emails():
    emails = []
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL, PASSWORD)
        mail.select("inbox")
        _, msgnums = mail.search(None, "UNSEEN")
        for num in msgnums[0].split():
            _, data = mail.fetch(num, "(RFC822)")
            msg = email.message_from_bytes(data[0][1])
            subject = decode_header(msg["Subject"])[0][0]
            if isinstance(subject, bytes):
                subject = subject.decode()
            from_ = msg.get("From")
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body += part.get_payload(decode=True).decode(errors='ignore')
            else:
                body = msg.get_payload(decode=True).decode(errors='ignore')
            emails.append((from_, body))
    except Exception as e:
        st.error(f"Email parsing error: {e}")
    return emails

def process_emails():
    emails = fetch_emails()
    for sender, body in emails:
        esc_id, urgency = insert_escalation(sender, body)
        if urgency == "High":
            send_alert(esc_id, sender, body)

def process_excel(file):
    df = pd.read_excel(file)
    for _, row in df.iterrows():
        customer = str(row.get("Customer", "Unknown"))
        issue = str(row.get("Issue", ""))
        if issue:
            esc_id, urgency = insert_escalation(customer, issue)
            if urgency == "High":
                send_alert(esc_id, customer, issue)

def fetch_escalations():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    conn.close()
    return df

def update_escalation(id, field, value):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(f"UPDATE escalations SET {field} = ? WHERE id = ?", (value, id))
    conn.commit()
    conn.close()

def send_alert(esc_id, customer, issue):
    message = {
        "title": "🚨 Escalation Alert",
        "text": f"Escalation ID: {esc_id}\nCustomer: {customer}\nIssue: {issue}"
    }
    try:
        requests.post(MS_TEAMS_WEBHOOK_URL, json=message)
    except:
        st.warning("Failed to send Teams alert.")

def display_kanban(df, status_filter):
    status_buckets = ["Open", "In Progress", "Resolved"]
    cols = st.columns(3)
    for i, status in enumerate(status_buckets):
        filtered = df[df['status'] == status]
        if status_filter != "All":
            filtered = filtered[filtered['status'] == status_filter]
        with cols[i]:
            st.subheader(f"{status} ({len(filtered)})")
            for idx, row in filtered.iterrows():
                with st.expander(f"{row.get('id', '')} - {row.get('customer', '')}"):
                    st.write(f"**Issue:** {row.get('issue', '')}")
                    st.write(f"**Sentiment:** {row.get('sentiment', '')} | **Urgency:** {row.get('urgency', '')} | **Severity:** {row.get('severity', '')} | **Criticality:** {row.get('criticality', '')} | **Category:** {row.get('category', '')}")
                    action = st.text_input(f"Action Taken - {row['id']}", row.get('action_taken', ''), key=f"action_{row['id']}")
                    owner = st.text_input(f"Owner - {row['id']}", row.get('action_owner', ''), key=f"owner_{row['id']}")
                    new_status = st.selectbox(f"Status - {row['id']}", status_buckets, index=status_buckets.index(row['status']), key=f"status_{row['id']}")
                    if st.button(f"Update - {row['id']}"):
                        update_escalation(row['id'], 'action_taken', action)
                        update_escalation(row['id'], 'action_owner', owner)
                        update_escalation(row['id'], 'status', new_status)
                        st.success("Updated successfully.")

def sidebar_controls():
    st.sidebar.title("EscalateAI Options")
    if st.sidebar.button("🔄 Refresh Emails"):
        process_emails()
        st.sidebar.success("Emails processed.")
    excel_file = st.sidebar.file_uploader("📤 Upload Complaints Excel", type=[".xls", ".xlsx"])
    if excel_file:
        process_excel(excel_file)
        st.sidebar.success("Excel data processed.")
    df = fetch_escalations()
    if not df.empty:
        towrite = pd.ExcelWriter("Escalation_Export.xlsx", engine='openpyxl')
        df.to_excel(towrite, index=False, sheet_name='Escalations')
        towrite.save()
        with open("Escalation_Export.xlsx", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="Escalations.xlsx">📥 Download All Escalations</a>'
            st.sidebar.markdown(href, unsafe_allow_html=True)

def main():
    st.set_page_config(layout="wide")
    st.title("EscalateAI – Escalation Management")
    sidebar_controls()
    df = fetch_escalations()
    status_filter = st.selectbox("Filter by Status", ["All", "Open", "In Progress", "Resolved"])
    display_kanban(df, status_filter)

if __name__ == "__main__":
    init_db()
    main()
