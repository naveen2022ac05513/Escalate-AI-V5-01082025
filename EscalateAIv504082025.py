# EscalateAI: Streamlit App for Escalation Management

import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
import base64
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh
import smtplib
from email.mime.text import MIMEText
import requests
import random

# Load environment variables
load_dotenv()
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_SERVER = os.getenv("EMAIL_SERVER")
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", 587))
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

DB_FILE = "escalate_ai.db"
NEGATIVE_KEYWORDS = ['delay', 'escalate', 'urgent', 'problem', 'failure', 'frustrated', 'not working', 'issue', 'angry', 'disappointed', 'slow', 'late']

# ========== Initialize DB ==========
def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute('''CREATE TABLE IF NOT EXISTS escalations (
        id TEXT PRIMARY KEY,
        customer TEXT,
        issue TEXT,
        sentiment REAL,
        urgency TEXT,
        severity TEXT,
        criticality TEXT,
        category TEXT,
        status TEXT,
        action_taken TEXT,
        owner TEXT,
        timestamp TEXT,
        escalated INTEGER DEFAULT 0
    )''')
    conn.commit()
    conn.close()

# ========== Unique ID Generator ==========
def generate_escalation_id():
    base = 250000
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT id FROM escalations", conn)
    conn.close()
    existing = [int(x.split('-')[-1]) for x in df['id']] if not df.empty else []
    next_id = max(existing) + 1 if existing else base + 1
    return f'SESICE-{next_id}'

# ========== NLP Analysis ==========
analyzer = SentimentIntensityAnalyzer()

def analyze_issue(issue):
    sentiment_score = analyzer.polarity_scores(issue)['compound']
    urgency = 'High' if any(word in issue.lower() for word in NEGATIVE_KEYWORDS) else 'Low'
    severity = 'Critical' if 'down' in issue.lower() or 'outage' in issue.lower() else 'Moderate'
    criticality = 'High' if urgency == 'High' else 'Normal'
    category = 'Technical' if any(k in issue.lower() for k in ['error', 'bug', 'fail']) else 'General'
    return sentiment_score, urgency, severity, criticality, category

# ========== Email Parsing ==========
def fetch_emails():
    try:
        mail = imaplib.IMAP4_SSL(EMAIL_SERVER)
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select("inbox")
        _, messages = mail.search(None, "UNSEEN")
        email_ids = messages[0].split()

        issues = []
        for e_id in email_ids:
            _, msg_data = mail.fetch(e_id, "(RFC822)")
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)

            subject, _ = decode_header(msg["Subject"])[0]
            subject = subject.decode() if isinstance(subject, bytes) else subject
            from_ = msg.get("From")

            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode()
                        break
            else:
                body = msg.get_payload(decode=True).decode()

            customer = from_.split('<')[0].strip()
            issues.append((customer, subject + " " + body))
        return issues
    except Exception as e:
        print(f"Email fetch error: {e}")
        return []

# ========== Save to DB ==========
def save_to_db(customer, issue, sentiment, urgency, severity, criticality, category):
    eid = generate_escalation_id()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM escalations WHERE customer=? AND issue=?", (customer, issue))
    if not cursor.fetchone():
        cursor.execute('''INSERT INTO escalations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                       (eid, customer, issue, sentiment, urgency, severity, criticality, category,
                        'Open', '', '', now, 0))
        conn.commit()
    conn.close()

# ========== SLA Check ==========
def check_sla():
    df = pd.read_sql_query("SELECT * FROM escalations WHERE status = 'Open'", sqlite3.connect(DB_FILE))
    alerts = []
    now = datetime.datetime.now()
    for _, row in df.iterrows():
        ts = datetime.datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S")
        diff = (now - ts).total_seconds()
        if row['urgency'] == 'High' and diff > 600 and row['escalated'] == 0:
            alerts.append(row)
            conn = sqlite3.connect(DB_FILE)
            conn.execute("UPDATE escalations SET escalated = 1 WHERE id = ?", (row['id'],))
            conn.commit()
            conn.close()
    return alerts

def send_alerts(alerts):
    for row in alerts:
        msg = f"⚠️ Escalation Alert\nID: {row['id']}\nCustomer: {row['customer']}\nIssue: {row['issue']}\nUrgency: {row['urgency']}"
        # Email
        try:
            server = smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT)
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            message = MIMEText(msg)
            message["From"] = EMAIL_USER
            message["To"] = EMAIL_RECEIVER
            message["Subject"] = f"SLA Breach Alert - {row['id']}"
            server.sendmail(EMAIL_USER, EMAIL_RECEIVER, message.as_string())
            server.quit()
        except Exception as e:
            print("Email alert error:", e)

        # MS Teams
        try:
            requests.post(MS_TEAMS_WEBHOOK_URL, json={"text": msg})
        except Exception as e:
            print("MS Teams alert error:", e)

# ========== Streamlit App ==========
def main():
    st.set_page_config(layout="wide")
    st.title("🚨 EscalateAI - Escalation Management")

    st_autorefresh(interval=60000, key="refresh")

    init_db()
    new_emails = fetch_emails()
    for customer, issue in new_emails:
        s, u, sev, crit, cat = analyze_issue(issue)
        save_to_db(customer, issue, s, u, sev, crit, cat)

    alerts = check_sla()
    if alerts:
        send_alerts(alerts)

    df = pd.read_sql_query("SELECT * FROM escalations", sqlite3.connect(DB_FILE))

    with st.sidebar:
        st.header("📥 Upload Excel")
        uploaded = st.file_uploader("Upload complaints Excel", type=["xlsx"])
        if uploaded:
            df_xl = pd.read_excel(uploaded)
            for _, row in df_xl.iterrows():
                s, u, sev, crit, cat = analyze_issue(row['Issue'])
                save_to_db(row['Customer'], row['Issue'], s, u, sev, crit, cat)
            st.success("Uploaded and analyzed")

        status_filter = st.radio("Filter by:", ["All", "Escalated"])
        if st.button("🔁 Trigger SLA Alerts"):
            send_alerts(check_sla())

        @st.cache_data
        def convert_df(df):
            return df.to_excel(index=False)

        st.download_button("📥 Download All", data=convert_df(df), file_name="escalations_all.xlsx")

    if status_filter == "Escalated":
        df = df[df['escalated'] == 1]

    st.subheader("📊 Escalation Kanban")
    counts = df['status'].value_counts().to_dict()
    cols = st.columns(3)
    for i, status in enumerate(["Open", "In Progress", "Resolved"]):
        with cols[i]:
            st.markdown(f"### {status} ({counts.get(status, 0)})")
            for _, row in df[df['status'] == status].iterrows():
                with st.expander(f"{row['id']} - {row['customer']}"):
                    st.write(row['issue'])
                    st.write(f"Sentiment: {row['sentiment']}, Urgency: {row['urgency']}, Severity: {row['severity']}")
                    new_status = st.selectbox("Update status", ["Open", "In Progress", "Resolved"], index=["Open", "In Progress", "Resolved"].index(row['status']), key=f"status_{row['id']}")
                    action = st.text_input("Action Taken", value=row['action_taken'], key=f"action_{row['id']}")
                    owner = st.text_input("Owner", value=row['owner'], key=f"owner_{row['id']}")
                    if st.button("💾 Save", key=f"save_{row['id']}"):
                        conn = sqlite3.connect(DB_FILE)
                        conn.execute("UPDATE escalations SET status=?, action_taken=?, owner=? WHERE id=?", (new_status, action, owner, row['id']))
                        conn.commit()
                        conn.close()
                        st.success("Updated")

if __name__ == "__main__":
    main()
