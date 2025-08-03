# EscalateAI – Full App with Email Parsing, Classification, Logging, Alerting, and MS Teams Integration

import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import uuid
import requests

# Constants
DB_FILE = "escalations.db"
ESCALATION_PREFIX = "SESICE-"
START_ESCALATION_ID = 250001
GMAIL_IMAP = "imap.gmail.com"
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")

# Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize DB
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS escalations (
        escalation_id TEXT PRIMARY KEY,
        timestamp TEXT,
        sender TEXT,
        subject TEXT,
        issue TEXT,
        sentiment TEXT,
        urgency TEXT,
        status TEXT,
        action_taken TEXT,
        status_update_date TEXT,
        priority TEXT,
        customer TEXT,
        owner TEXT
    )''')
    conn.commit()
    conn.close()

# Unique Escalation ID Generator
def generate_escalation_id():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM escalations")
    count = c.fetchone()[0]
    conn.close()
    return f"{ESCALATION_PREFIX}{START_ESCALATION_ID + count}"

# Load Escalations DataFrame
def load_escalations_df():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM escalations", conn)
    conn.close()
    return df

# Save Escalation to DB
def save_escalation(data):
    conn = sqlite3.connect(DB_FILE)
    df = pd.DataFrame([data])
    df.to_sql("escalations", conn, if_exists="append", index=False)
    conn.close()

# Analyze Email Content
def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score <= -0.5:
        return "Negative"
    elif score >= 0.5:
        return "Positive"
    else:
        return "Neutral"

def determine_urgency(text):
    keywords = ["urgent", "immediately", "asap", "critical", "severe", "now"]
    text = text.lower()
    return "High" if any(k in text for k in keywords) else "Normal"

def extract_customer_and_owner(sender):
    domain = sender.split("@")[1]
    customer = domain.split(".")[0].capitalize()
    owner = "Unassigned"
    return customer, owner

# Send MS Teams Notification
def send_ms_teams_alert(message):
    if not MS_TEAMS_WEBHOOK_URL:
        return
    payload = {"text": message}
    try:
        requests.post(MS_TEAMS_WEBHOOK_URL, json=payload)
    except Exception as e:
        print("MS Teams alert error:", e)

# Parse Emails from Gmail
def parse_emails(email_user, email_pass):
    mail = imaplib.IMAP4_SSL(GMAIL_IMAP)
    mail.login(email_user, email_pass)
    mail.select("inbox")

    status, messages = mail.search(None, "UNSEEN")
    messages = messages[0].split()
    for num in messages:
        _, msg_data = mail.fetch(num, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                sender = msg.get("From")
                subject = msg.get("Subject")
                date = msg.get("Date")

                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode(errors="ignore")
                            break
                else:
                    body = msg.get_payload(decode=True).decode(errors="ignore")

                sentiment = analyze_sentiment(body)
                urgency = determine_urgency(body)
                customer, owner = extract_customer_and_owner(sender)

                escalation = {
                    "escalation_id": generate_escalation_id(),
                    "timestamp": date,
                    "sender": sender,
                    "subject": subject,
                    "issue": body,
                    "sentiment": sentiment,
                    "urgency": urgency,
                    "status": "Open",
                    "action_taken": "",
                    "status_update_date": date,
                    "priority": urgency,
                    "customer": customer,
                    "owner": owner
                }
                save_escalation(escalation)
                send_ms_teams_alert(f"New escalation logged: {escalation['escalation_id']} from {sender}\nSubject: {subject}\nUrgency: {urgency}\nSentiment: {sentiment}")
    mail.logout()

# SLA Alert Checker
def check_sla_and_alert():
    df = load_escalations_df()
    now = datetime.datetime.now(datetime.timezone.utc)
    breached = df[(df['priority'] == "High") & (df['status'] == "Open")]
    for _, row in breached.iterrows():
        try:
            last_update = datetime.datetime.strptime(row['status_update_date'], "%a, %d %b %Y %H:%M:%S %z")
        except Exception:
            continue
        elapsed = now - last_update
        if elapsed.total_seconds() > 48 * 3600:
            send_ms_teams_alert(f"⚠️ SLA breach detected:\nID: {row['escalation_id']}\nCustomer: {row['customer']}\nOpen for: {elapsed.days} days\nIssue: {row['issue'][:200]}...")

# Streamlit UI
def main():
    st.set_page_config(page_title="EscalateAI", layout="wide")
    st.title("📩 EscalateAI – Escalation Management Tool")
    init_db()

    menu = ["Escalation Dashboard", "Manual Entry", "Parse Emails"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Escalation Dashboard":
        df = load_escalations_df()
        st.dataframe(df)
        for idx, row in df.iterrows():
            st.markdown(f"### {row['escalation_id']} – {row['subject']}")
            st.markdown(f"**From:** {row['sender']}")
            st.markdown(f"**Customer:** {row['customer']}, **Priority:** {row['priority']}, **Status:** {row['status']}")
            st.markdown(f"**Sentiment:** {row['sentiment']}, **Urgency:** {row['urgency']}")
            st.text_area("Issue", row['issue'], height=150, key=f"issue_{idx}")
            new_status = st.selectbox("Update Status", ["Open", "In Progress", "Resolved"], index=["Open", "In Progress", "Resolved"].index(row['status']), key=f"status_{idx}")
            new_action = st.text_input("Action Taken", value=row['action_taken'], key=f"action_{idx}")
            if st.button("Update", key=f"update_{idx}"):
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute("""
                    UPDATE escalations SET status=?, action_taken=?, status_update_date=? WHERE escalation_id=?
                """, (new_status, new_action, datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z"), row['escalation_id']))
                conn.commit()
                conn.close()
                st.success("Updated successfully!")

    elif choice == "Manual Entry":
        with st.form("manual_form"):
            sender = st.text_input("Sender Email")
            subject = st.text_input("Subject")
            issue = st.text_area("Issue")
            submitted = st.form_submit_button("Submit")
            if submitted:
                sentiment = analyze_sentiment(issue)
                urgency = determine_urgency(issue)
                customer, owner = extract_customer_and_owner(sender)
                escalation = {
                    "escalation_id": generate_escalation_id(),
                    "timestamp": datetime.datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z"),
                    "sender": sender,
                    "subject": subject,
                    "issue": issue,
                    "sentiment": sentiment,
                    "urgency": urgency,
                    "status": "Open",
                    "action_taken": "",
                    "status_update_date": datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z"),
                    "priority": urgency,
                    "customer": customer,
                    "owner": owner
                }
                save_escalation(escalation)
                send_ms_teams_alert(f"Manual escalation logged: {escalation['escalation_id']} from {sender}\nSubject: {subject}\nUrgency: {urgency}\nSentiment: {sentiment}")
                st.success("Escalation logged successfully!")

    elif choice == "Parse Emails":
        email_user = st.text_input("Gmail ID")
        email_pass = st.text_input("Gmail App Password", type="password")
        if st.button("Parse Emails"):
            parse_emails(email_user, email_pass)
            st.success("Emails parsed and escalations logged!")

    check_sla_and_alert()

if __name__ == '__main__':
    main()
