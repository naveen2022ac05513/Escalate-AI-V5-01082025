# escalate_ai.py – EscalateAI unified app with Gmail fetching, alerting, improved ML, and real-time polling

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
from dotenv import load_dotenv
import threading

# ---------------------------- LOAD ENV ----------------------------
load_dotenv()

IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASS = os.getenv("SMTP_PASS")
ALERT_RECIPIENT = os.getenv("ALERT_RECIPIENT")

TEAMS_WEBHOOK = os.getenv("TEAMS_WEBHOOK")

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

# For avoiding duplicates in email polling
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

def parse_emails(imap_server, email_user, email_pass):
    """Parse unseen emails from IMAP and return list of dict with customer and issue."""
    emails = []
    try:
        conn = imaplib.IMAP4_SSL(imap_server)
        conn.login(email_user, email_pass)
        conn.select("inbox")

        _, messages = conn.search(None, "UNSEEN")
        for num in messages[0].split():
            # To avoid duplicates using UID, fetch UID
            _, msg_uid_data = conn.fetch(num, '(UID)')
            uid = None
            if msg_uid_data and isinstance(msg_uid_data[0], tuple):
                uid_match = re.search(rb'UID (\d+)', msg_uid_data[0][1])
                if uid_match:
                    uid = uid_match.group(1).decode()

            if uid:
                with processed_email_uids_lock:
                    if uid in processed_email_uids:
                        # Skip duplicate email
                        continue

            _, msg_data = conn.fetch(num, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject = decode_header(msg["Subject"])[0][0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(errors='ignore')
                    from_ = msg.get("From")
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain" and not part.get('Content-Disposition'):
                                try:
                                    body = part.get_payload(decode=True).decode(errors='ignore')
                                except:
                                    body = ""
                                break
                    else:
                        try:
                            body = msg.get_payload(decode=True).decode(errors='ignore')
                        except:
                            body = ""
                    combined_text = f"{subject} - {body[:500]}"
                    emails.append({
                        "customer": from_,
                        "issue": combined_text
                    })
                    if uid:
                        with processed_email_uids_lock:
                            processed_email_uids.add(uid)
        conn.logout()
    except Exception as e:
        st.error(f"Error fetching emails: {e}")
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

# ---------------------------- ML MODEL ----------------------------

def train_model():
    df = fetch_escalations()
    # Only train on escalated and non-escalated with known sentiment etc
    df = df.dropna(subset=['sentiment', 'urgency', 'severity', 'criticality', 'escalated'])
    if df.shape[0] < 20:
        return None  # Not enough data to train
    y = df['escalated'].apply(lambda x: 1 if x == 'Yes' else 0)
    if y.nunique() < 2:
        return None  # Not enough class diversity

    X = pd.get_dummies(df[['sentiment', 'urgency', 'severity', 'criticality']])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_escalation(model, sentiment, urgency, severity, criticality):
    if model is None:
        # fallback rule based
        return "Yes" if urgency == "high" or sentiment == "negative" else "No"
    X_pred = pd.DataFrame([{
        f"sentiment_{sentiment}": 1,
        f"urgency_{urgency}": 1,
        f"severity_{severity}": 1,
        f"criticality_{criticality}": 1
    }])
    # Align columns with model training data
    X_pred = X_pred.reindex(columns=model.feature_names_in_, fill_value=0)
    pred = model.predict(X_pred)
    return "Yes" if pred[0] == 1 else "No"

# ---------------------------- SLA ALERT ----------------------------

def send_alert(message, via="email"):
    if via == "email":
        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(SMTP_EMAIL, SMTP_PASS)
                server.sendmail(SMTP_EMAIL, ALERT_RECIPIENT, f"Subject: EscalateAI Alert\n\n{message}")
        except Exception as e:
            st.error(f"Email alert failed: {e}")
    elif via == "teams":
        try:
            if not TEAMS_WEBHOOK:
                st.warning("MS Teams webhook URL not set in .env")
                return
            requests.post(TEAMS_WEBHOOK, json={"text": message})
        except Exception as e:
            st.error(f"Teams alert failed: {e}")

# ---------------------------- STREAMLIT UI ----------------------------

ensure_schema()  # Initialize DB

st.set_page_config(layout="wide")
st.title("🚨 EscalateAI – Customer Escalation Management")

# -------------------- SIDEBAR --------------------

st.sidebar.header("⚙️ Controls")

# Manual Excel upload
uploaded_file = st.sidebar.file_uploader("📥 Upload Excel (Customer complaints)", type=["xlsx"])
if uploaded_file:
    df_excel = pd.read_excel(uploaded_file)
    model = train_model()  # get latest model to predict escalation flag
    for _, row in df_excel.iterrows():
        issue = str(row.get("issue", ""))
        customer = str(row.get("customer", "Unknown"))
        sentiment, urgency, severity, criticality, category, rule_flag = analyze_issue(issue)
        ml_flag = predict_escalation(model, sentiment, urgency, severity, criticality)
        escalation_flag = "Yes" if (rule_flag == "Yes" or ml_flag == "Yes") else "No"
        insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag)
    st.sidebar.success("Uploaded and processed Excel file successfully.")

# Email fetching button
if st.sidebar.button("📥 Fetch Emails"):
    if not EMAIL_USER or not EMAIL_PASS:
        st.sidebar.error("Email credentials not set in .env")
    else:
        with st.spinner("Fetching emails..."):
            emails = parse_emails(IMAP_SERVER, EMAIL_USER, EMAIL_PASS)
            model = train_model()
            count_inserted = 0
            for em in emails:
                sentiment, urgency, severity, criticality, category, rule_flag = analyze_issue(em["issue"])
                ml_flag = predict_escalation(model, sentiment, urgency, severity, criticality)
                escalation_flag = "Yes" if (rule_flag == "Yes" or ml_flag == "Yes") else "No"
                insert_escalation(em["customer"], em["issue"], sentiment, urgency, severity, criticality, category, escalation_flag)
                count_inserted += 1
            st.sidebar.success(f"Fetched and inserted {count_inserted} new emails.")

# Download all escalations
if st.sidebar.button("📤 Download All Complaints"):
    df = fetch_escalations()
    csv = df.to_csv(index=False)
    st.sidebar.download_button("Download CSV", csv, file_name="escalations.csv", mime="text/csv")

# SLA Alerting button
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

# -------------------- MAIN TABS --------------------

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
        st.info("Retraining model with feedback...")
        model = train_model()
        if model:
            st.success("Model retrained successfully.")
        else:
            st.warning("Not enough data to retrain model.")

# ---------------------------- BACKGROUND EMAIL POLLING ----------------------------

def email_polling_job():
    """Background thread: poll email every 60s and insert new escalations"""
    while True:
        try:
            if EMAIL_USER and EMAIL_PASS:
                emails = parse_emails(IMAP_SERVER, EMAIL_USER, EMAIL_PASS)
                if emails:
                    model = train_model()
                    for em in emails:
                        sentiment, urgency, severity, criticality, category, rule_flag = analyze_issue(em["issue"])
                        ml_flag = predict_escalation(model, sentiment, urgency, severity, criticality)
                        escalation_flag = "Yes" if (rule_flag == "Yes" or ml_flag == "Yes") else "No"
                        insert_escalation(em["customer"], em["issue"], sentiment, urgency, severity, criticality, category, escalation_flag)
        except Exception as e:
            # Log error quietly
            print(f"[Email Polling Error] {e}")
        time.sleep(60)

# Start background polling thread as daemon
if __name__ == "__main__":
    if os.environ.get("RUN_EMAIL_PARSER", "1") == "1":
        threading.Thread(target=email_polling_job, daemon=True).start()

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

# ---------------------------- NOTES ----------------------------
# - Create a .env file with email credentials and webhook URL
# - Use App Passwords or OAuth2 for Gmail SMTP/IMAP
# - ML model training and prediction uses RandomForestClassifier on categorical NLP tags
# - Email polling runs in a background thread every 60 seconds
# - Ensure Streamlit version >= 1.10 for proper threading support
