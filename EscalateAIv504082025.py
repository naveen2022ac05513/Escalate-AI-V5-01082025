# escalate_ai.py – EscalateAI unified app

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

# ---------------------------- GLOBALS ----------------------------
DB_PATH = "escalations.db"
ESCALATION_PREFIX = "SESICE-"
analyzer = SentimentIntensityAnalyzer()

# Expanded keyword list (predefined negative indicators)
NEGATIVE_KEYWORDS = {
    "technical": ["fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip", "malfunction", "blank", "shutdown", "discharge"],
    "dissatisfaction": ["dissatisfy", "frustrate", "complain", "reject", "delay", "ignore", "escalate", "displease", "noncompliance", "neglect"],
    "support": ["wait", "pending", "slow", "incomplete", "miss", "omit", "unresolved", "shortage", "no response"],
    "safety": ["fire", "burn", "flashover", "arc", "explode", "unsafe", "leak", "corrode", "alarm", "incident"],
    "business": ["impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate", "penalty"]
}
def fetch_escalations():
    conn = connect_db()
    try:
        df = pd.read_sql("SELECT * FROM escalations", conn)
    except Exception as e:
        st.error(f"Error reading escalations table: {e}")
        df = pd.DataFrame()  # fallback empty
    finally:
        conn.close()
    return df

# Run this once on app start
ensure_schema()
# ---------------------------- DB INIT ----------------------------
def init_db():
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

def insert_into_db(data):
    conn = sqlite3.connect(DB_PATH)
    df = pd.DataFrame([data])
    df.to_sql("escalations", conn, if_exists="append", index=False)
    conn.close()

def fetch_escalations():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM escalations", conn)
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
def parse_emails(imap_server, email_user, email_pass):
    conn = imaplib.IMAP4_SSL(imap_server)
    conn.login(email_user, email_pass)
    conn.select("inbox")

    _, messages = conn.search(None, "UNSEEN")
    emails = []
    for num in messages[0].split():
        _, msg_data = conn.fetch(num, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                subject = decode_header(msg["Subject"])[0][0]
                if isinstance(subject, bytes):
                    subject = subject.decode()
                from_ = msg.get("From")
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode()
                            break
                else:
                    body = msg.get_payload(decode=True).decode()
                emails.append({
                    "customer": from_,
                    "issue": f"{subject} - {body[:200]}"
                })
    conn.logout()
    return emails

# ---------------------------- NLP + ESCALATION TAGGING ----------------------------
def analyze_issue(issue_text):
    sentiment_score = analyzer.polarity_scores(issue_text)
    sentiment = "negative" if sentiment_score["compound"] < -0.05 else "positive" if sentiment_score["compound"] > 0.05 else "neutral"
    
    urgency = "high" if any(word in issue_text.lower() for category in NEGATIVE_KEYWORDS.values() for word in category) else "normal"
    
    category, matched_keywords = None, []
    for cat, keywords in NEGATIVE_KEYWORDS.items():
        if any(k in issue_text.lower() for k in keywords):
            category = cat
            matched_keywords = [k for k in keywords if k in issue_text.lower()]
            break

    severity = "critical" if category in ["safety", "technical"] else "major" if category in ["support", "business"] else "minor"
    criticality = "high" if sentiment == "negative" and urgency == "high" else "medium"

    escalation_flag = "Yes" if urgency == "high" or sentiment == "negative" else "No"

    return sentiment, urgency, severity, criticality, category, escalation_flag

# ---------------------------- ML PREDICTION STUB ----------------------------
def train_model():
    df = fetch_escalations()
    if df.shape[0] < 20:
        return None  # not enough data
    df = df.dropna()
    X = pd.get_dummies(df[['sentiment', 'urgency', 'severity', 'criticality']])
    y = df['escalated'].apply(lambda x: 1 if x == 'Yes' else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier()
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
        # Fill with valid SMTP credentials
        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login("your_email@gmail.com", "your_app_password")
                server.sendmail("your_email@gmail.com", "alert_recipient@example.com", message)
        except Exception as e:
            st.error(f"Email alert failed: {e}")
    elif via == "teams":
        try:
            webhook_url = "https://outlook.office.com/webhook/..."  # fill with valid MS Teams webhook
            requests.post(webhook_url, json={"text": message})
        except Exception as e:
            st.error(f"Teams alert failed: {e}")
# ---------------------------- STREAMLIT UI ----------------------------
st.set_page_config(layout="wide")
st.title("🚨 EscalateAI – Customer Escalation Management")

# Sidebar
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("📥 Upload Excel", type=["xlsx"])
if uploaded_file:
    df_excel = pd.read_excel(uploaded_file)
    for _, row in df_excel.iterrows():
        issue = row.get("issue", "")
        customer = row.get("customer", "Unknown")
        sentiment, urgency, severity, criticality, category, escalation = analyze_issue(issue)
        insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation)

if st.sidebar.button("📤 Download All Complaints"):
    df = fetch_escalations()
    st.sidebar.download_button("Download Complaints", df.to_csv(index=False), file_name="escalations.csv")

if st.sidebar.button("📣 Trigger SLA Alert"):
    overdue = fetch_escalations()
    overdue['created_at'] = pd.to_datetime(overdue['created_at'])
    breaches = overdue[(overdue['status'] != 'Resolved') & (overdue['priority'] == 'high') & 
                       ((datetime.datetime.now() - overdue['created_at']) > datetime.timedelta(minutes=10))]
    if not breaches.empty:
        send_alert("🚨 SLA breach detected!", via="teams")
        send_alert("🚨 SLA breach detected!", via="email")
    else:
        st.sidebar.success("No SLA breaches detected")

# ---------------------------- MAIN DISPLAY ----------------------------
tabs = st.tabs(["🗃️ All", "🚩 Escalated", "🛠️ Feedback"])

with tabs[0]:
    st.subheader("📊 Escalation Kanban Board")
    df = fetch_escalations()

    col1, col2, col3 = st.columns(3)
    for status, col in zip(["Open", "In Progress", "Resolved"], [col1, col2, col3]):
        bucket = df[df["status"] == status]
        col.metric(f"{status} Cases", len(bucket))
        for i, row in bucket.iterrows():
            with col.expander(f"{row['id']} - {row['customer']}", expanded=False):
                st.markdown(f"**Issue:** {row['issue']}")
                st.markdown(f"**Severity:** {row['severity']}")
                st.markdown(f"**Criticality:** {row['criticality']}")
                st.markdown(f"**Escalated?** {'🚨' if row['escalated'] == 'Yes' else '✅'}")
                new_status = st.selectbox("Update Status", ["Open", "In Progress", "Resolved"], index=["Open", "In Progress", "Resolved"].index(row["status"]), key=f"status_{row['id']}")
                new_action = st.text_input("Action Taken", row.get("action_taken", ""), key=f"action_{row['id']}")
                new_owner = st.text_input("Owner", row.get("owner", ""), key=f"owner_{row['id']}")
                if st.button("💾 Save", key=f"save_{row['id']}"):
                    def update_escalation(row_id, status, action_taken, owner):
                        conn = sqlite3.connect(DB_PATH)
                        cursor = conn.cursor()
                        cursor.execute("""
                            UPDATE escalations
                            SET status = ?, action_taken = ?, owner = ?
                            WHERE id = ?
                        """, (status, action_taken, owner, row_id))
                        conn.commit()
                        conn.close()
                    update_escalation(row['id'], new_status, new_action, new_owner)
                    st.success("Saved")

with tabs[1]:
    st.subheader("🚩 Escalated Issues")
    df = fetch_escalations()
    df_esc = df[df["escalated"] == "Yes"]
    st.write(df_esc)

with tabs[2]:
    st.subheader("🔁 Feedback & Retraining")
    df = fetch_escalations()
    df_feedback = df[df["escalated"].notnull()]
    feedback_map = {"Correct": 1, "Incorrect": 0}
    feedback_list = []
    for i, row in df_feedback.iterrows():
        feedback = st.selectbox(f"Is Escalation for {row['id']} correct?", ["Correct", "Incorrect"], key=f"fb_{row['id']}")
        feedback_list.append((row['id'], feedback_map[feedback]))
    if st.button("🔁 Retrain Model"):
        st.info("Retraining model with feedback (stubbed logic)...")
        model = train_model()
        st.success("Model retrained (simulated).")
# ---------------------------- FINAL SETUP & BACKGROUND TASKS ----------------------------

import threading
import time

# Background email parsing every 60 seconds (can be daemonized)
def email_polling_job():
    while True:
        parse_emails()
        time.sleep(60)

# Start polling thread only if running as main script
if __name__ == "__main__":
    if os.environ.get("RUN_EMAIL_PARSER", "1") == "1":  # Toggle via env var
        threading.Thread(target=email_polling_job, daemon=True).start()
    st.sidebar.markdown("✅ Email polling active")

# ---------------------------- UTILITIES FOR TESTING / DEBUG ----------------------------

if st.sidebar.checkbox("🧪 View Raw Database"):
    df = fetch_escalations()
    st.sidebar.write(df)

if st.sidebar.button("🗑️ Reset Database (Dev Only)"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS escalations")
    conn.commit()
    conn.close()
    st.sidebar.warning("Database reset. Restart the app.")

# ---------------------------- NOTES ----------------------------
# 🔁 Model training logic is stubbed. Replace with actual classifier logic.
# 🧠 To make it smarter: store model.pkl, add sklearn pipeline, retrain on feedback
# 📨 For real Gmail use: setup OAuth2 with Gmail API, replace imaplib logic

# ---------------------------- END OF CODE ----------------------------



