# ===============================
# EscalateAI – Part 1: Setup, NLP, Email Parsing, DB
# ===============================

import streamlit as st
import pandas as pd
import sqlite3
import imaplib
import email
from email.header import decode_header
import datetime
import re
import os
import smtplib
from email.mime.text import MIMEText
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import uuid
import requests
import json

DB_PATH = "escalations.db"
NEGATIVE_WORDS = ['fail', 'break', 'crash', 'defect', 'fault', 'degrade', 'damage', 'malfunction', 'shutdown',
                  'dissatisfy', 'frustrate', 'complain', 'reject', 'delay', 'ignore', 'escalate', 'displease',
                  'wait', 'pending', 'slow', 'incomplete', 'unresolved', 'shortage', 'no response',
                  'fire', 'burn', 'flashover', 'explode', 'unsafe', 'leak', 'alarm', 'incident',
                  'impact', 'loss', 'risk', 'downtime', 'interrupt', 'cancel']

# Initialize DB schema
def ensure_schema():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS escalations (
        id TEXT PRIMARY KEY,
        customer TEXT,
        issue TEXT,
        sentiment TEXT,
        urgency TEXT,
        severity TEXT,
        criticality TEXT,
        category TEXT,
        status TEXT,
        action_taken TEXT,
        owner TEXT,
        created_at TEXT,
        source TEXT
    )''')
    conn.commit()
    conn.close()

# Fetch from DB
def fetch_escalations():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM escalations", conn)
    conn.close()
    return df

# NLP analysis
analyzer = SentimentIntensityAnalyzer()
def analyze_issue(text):
    sentiment_score = analyzer.polarity_scores(text)["compound"]
    sentiment = "Negative" if sentiment_score < -0.05 else "Positive" if sentiment_score > 0.05 else "Neutral"
    urgency = "High" if any(word in text.lower() for word in NEGATIVE_WORDS) else "Low"
    severity = "Critical" if "crash" in text.lower() or "explode" in text.lower() else "Moderate"
    criticality = "High" if urgency == "High" and sentiment == "Negative" else "Normal"
    category = "Technical" if "fail" in text.lower() or "defect" in text.lower() else "Operational"
    return sentiment, urgency, severity, criticality, category

# Insert new escalation
def insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, source):
    esc_id = f"SESICE-{str(uuid.uuid4().int)[-6:]}"
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO escalations (id, customer, issue, sentiment, urgency, severity, criticality, category, status, action_taken, owner, created_at, source)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (esc_id, customer, issue, sentiment, urgency, severity, criticality, category, 'Open', '', '', datetime.datetime.now().isoformat(), source))
    conn.commit()
    conn.close()

# Email parsing
def parse_gmail_emails():
    email_user = os.getenv("EMAIL_USER")
    email_pass = os.getenv("EMAIL_PASS")
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(email_user, email_pass)
    mail.select("inbox")
    _, search_data = mail.search(None, 'UNSEEN')
    for num in search_data[0].split():
        _, data = mail.fetch(num, "(RFC822)")
        _, bytes_data = data[0]
        msg = email.message_from_bytes(bytes_data)
        subject = decode_header(msg["subject"])[0][0]
        if isinstance(subject, bytes):
            subject = subject.decode()
        from_ = msg.get("From")
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode(errors="ignore")
        else:
            body = msg.get_payload(decode=True).decode(errors="ignore")
        sentiment, urgency, severity, criticality, category = analyze_issue(body)
        insert_escalation(from_, body.strip(), sentiment, urgency, severity, criticality, category, source="email")
    mail.logout()

# Send MS Teams alert
def send_teams_alert(message):
    webhook_url = os.getenv("TEAMS_WEBHOOK_URL")
    if webhook_url:
        headers = {'Content-Type': 'application/json'}
        payload = {"text": message}
        requests.post(webhook_url, headers=headers, data=json.dumps(payload))

# Send Email alert
def send_email_alert(subject, body):
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    email_user = os.getenv("EMAIL_USER")
    email_pass = os.getenv("EMAIL_PASS")
    to_email = os.getenv("ALERT_EMAIL_TO")
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = email_user
    msg["To"] = to_email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(email_user, email_pass)
        server.sendmail(email_user, to_email, msg.as_string())
# ===============================
# EscalateAI – Part 2: ML, Feedback, SLA, Kanban
# ===============================

# Train ML model to predict escalation likelihood
def train_model():
    df = fetch_escalations()
    if df.empty or 'urgency' not in df or 'sentiment' not in df:
        return None
    df["label"] = ((df["urgency"] == "High") & (df["sentiment"] == "Negative")).astype(int)
    X = pd.get_dummies(df[["urgency", "sentiment", "severity", "criticality"]])
    y = df["label"]
    if len(X) < 10:
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    joblib.dump(clf, "escalation_model.pkl")
    return clf

# Predict escalation likelihood
def predict_escalation(issue_text):
    sentiment, urgency, severity, criticality, _ = analyze_issue(issue_text)
    try:
        model = joblib.load("escalation_model.pkl")
        df = pd.DataFrame([{
            "urgency": urgency,
            "sentiment": sentiment,
            "severity": severity,
            "criticality": criticality
        }])
        X = pd.get_dummies(df)
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)
        prediction = model.predict(X)[0]
        return bool(prediction)
    except:
        return False

# SLA check – detect escalations open >10 mins
def check_sla_breaches():
    df = fetch_escalations()
    breaches = []
    now = datetime.datetime.now()
    for _, row in df.iterrows():
        created_time = datetime.datetime.fromisoformat(row['created_at'])
        minutes_open = (now - created_time).total_seconds() / 60
        if row['status'] == 'Open' and row['urgency'] == 'High' and minutes_open > 10:
            breaches.append(row)
    return breaches

# Retrain model (triggered manually or on feedback loop)
def retrain_model():
    model = train_model()
    if model:
        st.success("Model retrained with current data.")

# Feedback loop UI
def feedback_loop():
    df = fetch_escalations()
    editable_df = st.data_editor(df[["id", "customer", "issue", "status", "action_taken"]], num_rows="dynamic")
    for index, row in editable_df.iterrows():
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE escalations SET status=?, action_taken=? WHERE id=?", (row["status"], row["action_taken"], row["id"]))
        conn.commit()
        conn.close()

# Kanban board display
def show_kanban():
    df = fetch_escalations()
    status_tabs = ["Open", "In Progress", "Resolved", "Escalated"]
    cols = st.columns(len(status_tabs))
    for i, status in enumerate(status_tabs):
        with cols[i]:
            st.markdown(f"### {status} ({(df['status'] == status).sum()})")
            for _, row in df[df["status"] == status].iterrows():
                st.markdown(f"""
                    - **ID**: {row['id']}
                    - **Customer**: {row['customer']}
                    - **Issue**: {row['issue'][:80]}...
                    - **Sentiment**: {row['sentiment']}
                    - **Urgency**: {row['urgency']}
                    - **Severity**: {row['severity']}
                    - **Criticality**: {row['criticality']}
                    - **Created**: {row['created_at'][:19]}
                """)
# ===============================
# EscalateAI – Part 3: Streamlit Layout & UI
# ===============================

st.set_page_config(page_title="EscalateAI", layout="wide")
st.title("📡 EscalateAI – Customer Escalation Management")

ensure_schema()

# ========== Sidebar ==========
st.sidebar.header("📁 Options")
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])
if uploaded_file:
    df_uploaded = pd.read_excel(uploaded_file)
    for _, row in df_uploaded.iterrows():
        customer = row.get("Customer", "")
        issue = row.get("Issue", "")
        if isinstance(issue, str) and issue.strip():
            add_escalation(customer, issue)

if st.sidebar.button("📤 Export All Issues"):
    df = fetch_escalations()
    df.to_excel("Escalations_Export.xlsx", index=False)
    with open("Escalations_Export.xlsx", "rb") as f:
        st.sidebar.download_button("⬇️ Download Exported Excel", f, file_name="Escalations_Export.xlsx")

if st.sidebar.button("📬 Fetch Emails"):
    fetch_emails()
    st.sidebar.success("Fetched and analyzed emails.")

if st.sidebar.button("🚨 Check SLA Breaches"):
    breaches = check_sla_breaches()
    if breaches:
        st.sidebar.warning(f"SLA Breaches Detected: {len(breaches)}")
        for b in breaches:
            st.sidebar.write(f"{b['id']} | {b['customer']} | {b['issue'][:30]}")
    else:
        st.sidebar.success("No SLA Breaches.")

if st.sidebar.button("🧠 Retrain Escalation Model"):
    retrain_model()

# ========== Main Panel ==========
with st.expander("➕ Add Escalation Manually"):
    customer = st.text_input("Customer Name")
    issue = st.text_area("Issue Description")
    if st.button("Log Escalation"):
        if customer and issue:
            escalation_detected = add_escalation(customer, issue)
            if escalation_detected:
                st.success("Escalation detected and logged.")
            else:
                st.info("Issue logged without escalation.")
        else:
            st.warning("Please enter both customer and issue.")

with st.expander("📊 Feedback Editor"):
    feedback_loop()

with st.expander("📌 Kanban Board View"):
    show_kanban()

# ========== Notification Buttons ==========
st.markdown("---")
st.subheader("🔔 Manual Alerts")
if st.button("Send MS Teams & Email Notification"):
    df = fetch_escalations()
    for _, row in df.iterrows():
        if row["status"] == "Open" and row["urgency"] == "High":
            send_email_notification(row)
            send_teams_notification(row)
    st.success("Notifications sent for active escalations.")

# ========== Auto-Send SLA Alert on Breach ==========
sla_breaches = check_sla_breaches()
for breach in sla_breaches:
    send_email_notification(breach)
    send_teams_notification(breach)
