# escalate_ai.py – EscalateAI unified app with env vars integration and sidebar controls

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import re
import uuid
import time
import datetime
import base64
import imaplib
import email
from email.header import decode_header
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import smtplib
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import threading
from dotenv import load_dotenv

# ---------------------------- LOAD ENV VARS ----------------------------
load_dotenv()

EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
SMTP_EMAIL = EMAIL_USER
SMTP_PASS = EMAIL_PASS
ALERT_RECIPIENT = os.getenv("EMAIL_RECEIVER")
TEAMS_WEBHOOK = os.getenv("MS_TEAMS_WEBHOOK_URL")

# ---------------------------- GLOBALS ----------------------------
DB_PATH = "escalations.db"
ESCALATION_PREFIX = "SESICE-"
analyzer = SentimentIntensityAnalyzer()

NEGATIVE_KEYWORDS = {
    "technical": ["fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip", "malfunction", "blank", "shutdown", "discharge"],
    "dissatisfaction": ["dissatisfy", "frustrate", "complain", "reject", "delay", "ignore", "escalate", "displease", "noncompliance", "neglect"],
    "support": ["wait", "pending", "slow", "incomplete", "miss", "omit", "unresolved", "shortage", "no response"],
    "safety": ["fire", "burn", "flashover", "arc", "explode", "unsafe", "leak", "corrode", "alarm", "incident"],
    "business": ["impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate", "penalty"]
}

# Track processed email UIDs to avoid duplicates
processed_email_uids = set()
processed_email_uids_lock = threading.Lock()

# ---------------------------- DATABASE FUNCTIONS ----------------------------

def ensure_schema():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS escalations (
            id TEXT PRIMARY KEY,
            customer TEXT,
            issue TEXT,
            sentiment TEXT,
            urgency TEXT,
            severity TEXT,
            criticality TEXT,
            category TEXT,
            status TEXT,
            timestamp TEXT,
            action_taken TEXT,
            owner TEXT,
            escalated TEXT,
            priority TEXT,
            escalation_flag TEXT,
            action_owner TEXT,
            status_update_date TEXT,
            user_feedback TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    new_id = ESCALATION_PREFIX + str(uuid.uuid4())[:8].upper()
    now = datetime.datetime.now().isoformat()
    cursor.execute('''
        INSERT INTO escalations (
            id, customer, issue, sentiment, urgency, severity, criticality, category,
            status, timestamp, escalated, priority, escalation_flag,
            action_taken, owner, action_owner, status_update_date, user_feedback
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        new_id, customer, issue, sentiment, urgency, severity, criticality, category,
        "Open", now, escalation_flag, "normal", escalation_flag,
        "", "", "", "", ""
    ))
    conn.commit()
    conn.close()

def fetch_escalations():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM escalations", conn)
    except Exception as e:
        st.error(f"Error reading escalations table: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def update_escalation_status(esc_id, status, action_taken, action_owner, feedback=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE escalations
        SET status = ?, action_taken = ?, action_owner = ?, status_update_date = ?, user_feedback = ?
        WHERE id = ?
    ''', (status, action_taken, action_owner, datetime.datetime.now().isoformat(), feedback, esc_id))
    conn.commit()
    conn.close()

# ---------------------------- EMAIL PARSING ----------------------------

def parse_emails():
    new_emails = []
    try:
        mail = imaplib.IMAP4_SSL(EMAIL_SERVER)
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select("inbox")

        _, data = mail.search(None, "UNSEEN")
        email_uids = data[0].split()

        for uid in email_uids:
            with processed_email_uids_lock:
                if uid in processed_email_uids:
                    continue  # skip already processed

            _, msg_data = mail.fetch(uid, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject = decode_header(msg["Subject"])[0][0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(errors="ignore")
                    from_ = msg.get("From")
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                body = part.get_payload(decode=True).decode(errors="ignore")
                                break
                    else:
                        body = msg.get_payload(decode=True).decode(errors="ignore")
                    issue_text = f"{subject} - {body[:500]}"
                    new_emails.append({"customer": from_, "issue": issue_text})

                    with processed_email_uids_lock:
                        processed_email_uids.add(uid)

        mail.logout()
    except Exception as e:
        st.error(f"Error fetching emails: {e}")
    return new_emails

# ---------------------------- NLP + ESCALATION TAGGING ----------------------------

def analyze_issue(issue_text):
    sentiment_score = analyzer.polarity_scores(issue_text)
    compound = sentiment_score["compound"]
    sentiment = "negative" if compound < -0.05 else "positive" if compound > 0.05 else "neutral"

    # Check urgency by keywords presence
    issue_lower = issue_text.lower()
    urgency = "high" if any(word in issue_lower for category in NEGATIVE_KEYWORDS.values() for word in category) else "normal"

    category, matched_keywords = None, []
    for cat, keywords in NEGATIVE_KEYWORDS.items():
        if any(k in issue_lower for k in keywords):
            category = cat
            matched_keywords = [k for k in keywords if k in issue_lower]
            break

    severity = "critical" if category in ["safety", "technical"] else "major" if category in ["support", "business"] else "minor"
    criticality = "high" if sentiment == "negative" and urgency == "high" else "medium"

    escalation_flag = "Yes" if urgency == "high" or sentiment == "negative" else "No"

    return sentiment, urgency, severity, criticality, category, escalation_flag

# ---------------------------- ML MODEL ----------------------------

def train_model():
    df = fetch_escalations()
    if df.shape[0] < 20:
        return None
    df = df.dropna(subset=['sentiment', 'urgency', 'severity', 'criticality', 'escalated'])
    X = pd.get_dummies(df[['sentiment', 'urgency', 'severity', 'criticality']])
    y = df['escalated'].apply(lambda x: 1 if x == 'Yes' else 0)
    if y.nunique() < 2:
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_escalation(model, sentiment, urgency, severity, criticality):
    X_pred = pd.DataFrame([{
        f"sentiment_{sentiment}": 1,
        f"urgency_{urgency}": 1,
        f"severity_{severity}": 1,
        f"criticality_{criticality}": 1
    }])
    X_pred = X_pred.reindex(columns=model.feature_names_in_, fill_value=0)
    pred = model.predict(X_pred)
    return "Yes" if pred[0] == 1 else "No"

# ---------------------------- SLA ALERT ----------------------------

def send_alert(message, via="email"):
    if via == "email":
        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_EMAIL, SMTP_PASS)
                server.sendmail(SMTP_EMAIL, ALERT_RECIPIENT, message)
        except Exception as e:
            st.error(f"Email alert failed: {e}")
    elif via == "teams":
        try:
            headers = {"Content-Type": "application/json"}
            payload = {"text": message}
            response = requests.post(TEAMS_WEBHOOK, json=payload, headers=headers)
            if response.status_code != 200:
                st.error(f"Teams alert failed with status {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Teams alert failed: {e}")

# ---------------------------- BACKGROUND EMAIL POLLING ----------------------------

def email_polling_job():
    while True:
        try:
            new_emails = parse_emails()
            for mail in new_emails:
                sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(mail["issue"])
                insert_escalation(mail["customer"], mail["issue"], sentiment, urgency, severity, criticality, category, escalation_flag)
        except Exception as e:
            # Avoid crashing thread; log in Streamlit UI if possible
            print(f"Email polling error: {e}")
        time.sleep(60)  # Poll every 60 seconds

# ---------------------------- STREAMLIT UI ----------------------------

ensure_schema()

st.set_page_config(layout="wide")
st.title("🚨 EscalateAI – Customer Escalation Management")

st.sidebar.header("⚙️ Controls")

# Manual fetch emails button
if st.sidebar.button("📥 Fetch Emails Now"):
    new_emails = parse_emails()
    count_new = 0
    for mail in new_emails:
        sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(mail["issue"])
        insert_escalation(mail["customer"], mail["issue"], sentiment, urgency, severity, criticality, category, escalation_flag)
        count_new += 1
    st.sidebar.success(f"Fetched and processed {count_new} new emails.")

# Upload Excel for bulk upload
uploaded_file = st.sidebar.file_uploader("📥 Upload Excel (Customer complaints)", type=["xlsx"])
if uploaded_file:
    df_excel = pd.read_excel(uploaded_file)
    count_new = 0
    for _, row in df_excel.iterrows():
        issue = str(row.get("issue", ""))
        customer = str(row.get("customer", "Unknown"))
        sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)
        insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag)
        count_new += 1
    st.sidebar.success(f"Uploaded and processed {count_new} Excel entries successfully.")

# Download CSV
if st.sidebar.button("📤 Download All Complaints"):
    df = fetch_escalations()
    csv = df.to_csv(index=False)
    st.sidebar.download_button("Download CSV", csv, file_name="escalations.csv", mime="text/csv")

# Trigger SLA Alert button
if st.sidebar.button("📣 Trigger SLA Alert"):
    df = fetch_escalations()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    breaches = df[(df['status'] != 'Resolved') & (df['priority'] == 'high') & 
                  ((datetime.datetime.now() - df['timestamp']) > datetime.timedelta(minutes=10))]
    if not breaches.empty:
        alert_msg = f"🚨 SLA breach detected for {len(breaches)} cases!"
        send_alert(alert_msg, via="teams")
        send_alert(alert_msg, via="email")
        st.sidebar.success("SLA breach alert sent.")
    else:
        st.sidebar.info("No SLA breaches detected.")

# Model training and prediction
model = train_model()

# Main Tabs
tabs = st.tabs(["🗃️ All", "🚩 Escalated", "🔁 Feedback & Retraining"])

with tabs[0]:
    st.subheader("📊 Escalation Kanban Board")

    df = fetch_escalations()
    # Show counts
    counts = df['status'].value_counts()
    open_count = counts.get('Open', 0)
    inprogress_count = counts.get('In Progress', 0)
    resolved_count = counts.get('Resolved', 0)
    st.markdown(f"**Open:** {open_count} | **In Progress:** {inprogress_count} | **Resolved:** {resolved_count}")

    col1, col2, col3 = st.columns(3)
    for status, col in zip(["Open", "In Progress", "Resolved"], [col1, col2, col3]):
        with col:
            st.markdown(f"### {status}")
            bucket = df[df["status"] == status]
            for i, row in bucket.iterrows():
                esc_pred = None
                if model:
                    esc_pred = predict_escalation(model, row['sentiment'], row['urgency'], row['severity'], row['criticality'])
                pred_text = f" | ML Prediction: {esc_pred}" if esc_pred else ""
                expander_label = f"{row['id']} - {row['customer']} {'🚩' if row['escalated']=='Yes' else ''}{pred_text}"
                with st.expander(expander_label, expanded=False):
                    st.write(f"**Issue:** {row['issue']}")
                    st.write(f"**Severity:** {row['severity']}")
                    st.write(f"**Criticality:** {row['criticality']}")
                    st.write(f"**Category:** {row['category']}")
                    st.write(f"**Sentiment:** {row['sentiment']}")
                    st.write(f"**Urgency:** {row['urgency']}")
                    st.write(f"**Escalated:** {row['escalated']}")
                    new_status = st.selectbox("Update Status", ["Open", "In Progress", "Resolved"], index=["Open", "In Progress", "Resolved"].index(row["status"]), key=f"status_{row['id']}")
                    new_action = st.text_input("Action Taken", row.get("action_taken", ""), key=f"action_{row['id']}")
                    new_owner = st.text_input("Owner", row.get("owner", ""), key=f"owner_{row['id']}")
                    if st.button("💾 Save Changes", key=f"save_{row['id']}"):
                        update_escalation_status(row['id'], new_status, new_action, new_owner)
                        st.success("Escalation updated.")

with tabs[1]:
    st.subheader("🚩 Escalated Issues")
    df = fetch_escalations()
    df_esc = df[df["escalated"] == "Yes"]
    st.dataframe(df_esc)

with tabs[2]:
    st.subheader("🔁 Feedback & Retraining")
    df = fetch_escalations()
    df_feedback = df[df["escalated"].notnull()]
    feedback_map = {"Correct": 1, "Incorrect": 0}
    for i, row in df_feedback.iterrows():
        feedback = st.selectbox(f"Is escalation for {row['id']} correct?", ["Correct", "Incorrect"], key=f"fb_{row['id']}")
        if st.button(f"Submit Feedback for {row['id']}", key=f"fb_btn_{row['id']}"):
            update_escalation_status(row['id'], row['status'], row.get('action_taken',''), row.get('owner',''), feedback_map[feedback])
            st.success("Feedback saved.")

    if st.button("🔁 Retrain Model"):
        st.info("Retraining model with feedback...")
        model = train_model()
        if model:
            st.success("Model retrained successfully.")
        else:
            st.warning("Not enough data to retrain model.")

# Background email polling thread start
def start_email_polling():
    thread = threading.Thread(target=email_polling_job, daemon=True)
    thread.start()

if st.session_state.get("email_polling_started", False) is False:
    start_email_polling()
    st.session_state["email_polling_started"] = True

# ---------------------------- DEV OPTIONS ----------------------------

if st.sidebar.checkbox("🧪 View Raw Database"):
    df = fetch_escalations()
    st.sidebar.dataframe(df)

if st.sidebar.button("🗑️ Reset Database (Dev Only)"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS escalations")
    conn.commit()
    conn.close()
    st.sidebar.warning("Database reset. Please restart the app.")

# ---------------------------- END OF FILE ----------------------------
