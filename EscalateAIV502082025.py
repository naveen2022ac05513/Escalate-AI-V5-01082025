# EscalateAI – Escalation Management Tool with Email Parsing, Excel Upload, NLP, Kanban, Alerts, and Filters

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
import smtplib
from email.mime.text import MIMEText
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import requests
import uuid

# Load environment variables
load_dotenv()
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_SERVER = os.getenv("EMAIL_SERVER")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT"))
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

DB_FILE = "escalations.db"

# === Initialize DB ===
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS escalations (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            customer TEXT,
            issue TEXT,
            sentiment TEXT,
            urgency TEXT,
            escalation_flag INTEGER,
            status TEXT,
            action_taken TEXT,
            action_owner TEXT
        )
    ''')
    conn.commit()
    conn.close()

# === Generate Unique ID ===
def generate_id():
    return f"SESICE-{str(uuid.uuid4().int)[-6:]}"

# === Sentiment + Urgency Analyzer ===
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    if score['compound'] <= -0.5:
        return 'Negative', 'High', 1
    elif score['compound'] < 0:
        return 'Slightly Negative', 'Medium', 1
    else:
        return 'Neutral/Positive', 'Low', 0

# === Insert Escalation ===
def insert_escalation(customer, issue, sentiment, urgency, escalation_flag):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    eid = generate_id()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO escalations (id, timestamp, customer, issue, sentiment, urgency, escalation_flag, status, action_taken, action_owner)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (eid, timestamp, customer, issue, sentiment, urgency, escalation_flag, "Open", "", ""))
    conn.commit()
    conn.close()
    return eid, escalation_flag

# === Parse Emails ===
def fetch_emails():
    mail = imaplib.IMAP4_SSL(EMAIL_SERVER)
    mail.login(EMAIL_USER, EMAIL_PASS)
    mail.select("inbox")
    status, messages = mail.search(None, 'UNSEEN')
    for num in messages[0].split():
        _, data = mail.fetch(num, '(RFC822)')
        msg = email.message_from_bytes(data[0][1])
        subject, encoding = decode_header(msg["Subject"])[0]
        if isinstance(subject, bytes):
            subject = subject.decode(encoding or "utf-8")
        from_ = msg.get("From")
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            body = msg.get_payload(decode=True).decode()
        sentiment, urgency, flag = analyze_sentiment(body)
        insert_escalation(from_, subject + " " + body[:200], sentiment, urgency, flag)
    mail.logout()

# === Upload Excel File ===
def upload_excel(file):
    df = pd.read_excel(file)
    for _, row in df.iterrows():
        issue_text = str(row.get("Issue", ""))
        customer = str(row.get("Customer", "Unknown"))
        sentiment, urgency, flag = analyze_sentiment(issue_text)
        insert_escalation(customer, issue_text, sentiment, urgency, flag)

# === Load from DB ===
def load_escalations_df():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    conn.close()
    return df

# === Update Action ===
def update_status_action(eid, new_status, new_action, new_owner):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        UPDATE escalations SET status = ?, action_taken = ?, action_owner = ? WHERE id = ?
    ''', (new_status, new_action, new_owner, eid))
    conn.commit()
    conn.close()

# === Send MS Teams Alert ===
def send_teams_alert(msg):
    payload = {"text": msg}
    requests.post(MS_TEAMS_WEBHOOK_URL, json=payload)

# === Send Email Alert ===
def send_email_alert(subject, body):
    msg = MIMEText(body)
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = subject
    server = smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT)
    server.starttls()
    server.login(EMAIL_USER, EMAIL_PASS)
    server.sendmail(EMAIL_USER, EMAIL_RECEIVER, msg.as_string())
    server.quit()

# === Sidebar ===
def sidebar_content():
    st.sidebar.title("Filters & Actions")
    status_filter = st.sidebar.multiselect("Status", ["Open", "In Progress", "Resolved"], default=["Open", "In Progress"])
    escalation_filter = st.sidebar.radio("Show Escalations", ["All", "Escalated Only"], index=0)
    st.sidebar.markdown("---")

    if st.sidebar.button("Trigger MS Teams Alert"):
        send_teams_alert("⚠️ SLA breach triggered manually from EscalateAI.")

    if st.sidebar.button("Trigger Email Alert"):
        send_email_alert("⚠️ SLA Breach Alert", "SLA breach manually triggered from EscalateAI.")

    st.sidebar.markdown("---")
    uploaded = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])
    if uploaded:
        upload_excel(uploaded)
        st.sidebar.success("Uploaded and processed.")

    if st.sidebar.button("Download All Complaints"):
        df = load_escalations_df()
        df.to_excel("all_escalations.xlsx", index=False)
        with open("all_escalations.xlsx", "rb") as f:
            st.sidebar.download_button("Download Excel", f, file_name="all_escalations.xlsx")

    return status_filter, escalation_filter

# === Display Kanban ===
def display_kanban(df, status_filter):
    buckets = ["Open", "In Progress", "Resolved"]
    cols = st.columns(len(buckets))
    for i, bucket in enumerate(buckets):
        with cols[i]:
            st.markdown(f"### {bucket}")
            filtered = df[df["status"] == bucket]
            st.markdown(f"**{len(filtered)} issues**")
            for _, row in filtered.iterrows():
                if row['status'] in status_filter:
                    with st.expander(f"{row['id']} - {row['customer']}"):
                        st.write(f"Issue: {row['issue']}")
                        st.write(f"Sentiment: {row['sentiment']}")
                        st.write(f"Urgency: {row['urgency']}")
                        st.write(f"Escalated: {'Yes' if row['escalation_flag'] else 'No'}")
                        new_status = st.selectbox("Update Status", buckets, index=buckets.index(row['status']), key=f"{row['id']}_status")
                        new_action = st.text_input("Action Taken", row['action_taken'], key=f"{row['id']}_action")
                        new_owner = st.text_input("Action Owner", row['action_owner'], key=f"{row['id']}_owner")
                        if st.button("Update", key=row['id']):
                            update_status_action(row['id'], new_status, new_action, new_owner)
                            st.success("Updated")

# === Main ===
def main():
    st.set_page_config(layout="wide")
    st.title("📍 EscalateAI – Customer Escalation Management")
    st_autorefresh(interval=60000, limit=100, key="refresh")

    fetch_emails()
    status_filter, escalation_filter = sidebar_content()
    df = load_escalations_df()

    if escalation_filter == "Escalated Only":
        df = df[df['escalation_flag'] == 1]

    display_kanban(df, status_filter)

# === Entry Point ===
if __name__ == "__main__":
    init_db()
    main()
