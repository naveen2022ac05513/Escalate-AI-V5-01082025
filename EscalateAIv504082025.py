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

# ---------------------------- DATABASE FUNCTIONS ----------------------------

def ensure_schema():
    """Ensure the database and escalations table exist with the required schema."""
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
    """Insert a new escalation record into the database."""
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
    """Fetch all escalation records as a DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM escalations", conn)
    except Exception as e:
        st.error(f"Error reading escalations table: {e}")
        df = pd.DataFrame()  # fallback empty dataframe
    finally:
        conn.close()
    return df

def update_escalation_status(esc_id, status, action_taken, action_owner, feedback=None):
    """Update escalation details in the DB."""
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
    """Parse unseen emails from IMAP and return list of dict with customer and issue."""
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

# ---------------------------- ML MODEL (Stub) ----------------------------

def train_model():
    df = fetch_escalations()
    if df.shape[0] < 20:
        return None  # Not enough data to train
    df = df.dropna(subset=['sentiment', 'urgency', 'severity', 'criticality', 'escalated'])
    X = pd.get_dummies(df[['sentiment', 'urgency', 'severity', 'criticality']])
    y = df['escalated'].apply(lambda x: 1 if x == 'Yes' else 0)
    if y.nunique() < 2:
        return None  # Not enough class diversity
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
        # Configure your SMTP credentials
        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login("your_email@gmail.com", "your_app_password")
                server.sendmail("your_email@gmail.com", "alert_recipient@example.com", message)
        except Exception as e:
            st.error(f"Email alert failed: {e}")
    elif via == "teams":
        try:
            webhook_url = "https://outlook.office.com/webhook/..."  # Replace with your MS Teams webhook URL
            requests.post(webhook_url, json={"text": message})
        except Exception as e:
            st.error(f"Teams alert failed: {e}")

# ---------------------------- STREAMLIT UI ----------------------------

ensure_schema()  # Initialize DB on app start

st.set_page_config(layout="wide")
st.title("🚨 EscalateAI – Customer Escalation Management")

# Sidebar: Upload Excel, Download data, SLA Alerts
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("📥 Upload Excel (Customer complaints)", type=["xlsx"])
if uploaded_file:
    df_excel = pd.read_excel(uploaded_file)
    for _, row in df_excel.iterrows():
        issue = str(row.get("issue", ""))
        customer = str(row.get("customer", "Unknown"))
        sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)
        insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag)
    st.sidebar.success("Uploaded and processed Excel file successfully.")

if st.sidebar.button("📤 Download All Complaints"):
    df = fetch_escalations()
    csv = df.to_csv(index=False)
    st.sidebar.download_button("Download CSV", csv, file_name="escalations.csv", mime="text/csv")

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

# Main tabs: All, Escalated, Feedback

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
                expander_label = f"{row['id']} - {row['customer']} {'🚩' if row['escalated']=='Yes' else ''}"
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
        st.info("Retraining model with feedback (stubbed)...")
        model = train_model()
        if model:
            st.success("Model retrained successfully.")
        else:
            st.warning("Not enough data to retrain model.")

# ---------------------------- BACKGROUND EMAIL POLLING ----------------------------

import threading

def email_polling_job():
    # Placeholder for polling logic - add your IMAP credentials and call parse_emails()
    while True:
        # TODO: call parse_emails() and insert new escalations
        time.sleep(60)

if __name__ == "__main__":
    if os.environ.get("RUN_EMAIL_PARSER", "1") == "1":
        threading.Thread(target=email_polling_job, daemon=True).start()

# ---------------------------- DEBUG / DEV OPTIONS ----------------------------

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

# ---------------------------- NOTES ----------------------------
# - Update SMTP and MS Teams webhook URLs before use.
# - Add real email credentials and OAuth2 for Gmail integration.
# - The ML model is a stub; expand training and save/load model for production.
# - Email polling job is a stub - implement fetching & insertion of new escalations.
# - This app assumes Streamlit >= 1.10.

# ---------------------------- END OF FILE ----------------------------
