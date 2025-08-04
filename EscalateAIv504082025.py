# escalate_ai.py – Full EscalateAI with UI colors & requested features

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import re
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

DB_PATH = "escalations.db"
ESCALATION_PREFIX = "SESICE-25"

analyzer = SentimentIntensityAnalyzer()

NEGATIVE_KEYWORDS = {
    "technical": ["fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip", "malfunction", "blank", "shutdown", "discharge"],
    "dissatisfaction": ["dissatisfy", "frustrate", "complain", "reject", "delay", "ignore", "escalate", "displease", "noncompliance", "neglect"],
    "support": ["wait", "pending", "slow", "incomplete", "miss", "omit", "unresolved", "shortage", "no response"],
    "safety": ["fire", "burn", "flashover", "arc", "explode", "unsafe", "leak", "corrode", "alarm", "incident"],
    "business": ["impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate", "penalty"]
}

processed_email_uids = set()
processed_email_uids_lock = threading.Lock()

# --- ID generation with SESICE-25XXXXX format ---
def get_next_escalation_id():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f'''
        SELECT id FROM escalations WHERE id LIKE '{ESCALATION_PREFIX}%'
        ORDER BY id DESC LIMIT 1
    ''')
    last = cursor.fetchone()
    if last:
        last_num = int(last[0].replace(ESCALATION_PREFIX, ""))
        next_num = last_num + 1
    else:
        next_num = 1
    conn.close()
    return f"{ESCALATION_PREFIX}{str(next_num).zfill(5)}"

# --- DATABASE FUNCTIONS ---

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
    new_id = get_next_escalation_id()
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

# --- EMAIL PARSING ---

def parse_emails(imap_server, email_user, email_pass):
    try:
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
                        subject = subject.decode(errors='ignore')
                    from_ = msg.get("From")
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                body = part.get_payload(decode=True).decode(errors='ignore')
                                break
                    else:
                        body = msg.get_payload(decode=True).decode(errors='ignore')
                    emails.append({
                        "customer": from_,
                        "issue": f"{subject} - {body[:200]}"
                    })
        conn.logout()
        return emails
    except Exception as e:
        st.error(f"Failed to parse emails: {e}")
        return []

# --- NLP + Escalation Tagging ---

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

# --- ML MODEL (stub) ---

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

# --- ALERTING ---

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
            requests.post(TEAMS_WEBHOOK, json={"text": message})
        except Exception as e:
            st.error(f"Teams alert failed: {e}")

# --- BACKGROUND EMAIL POLLING ---

def email_polling_job():
    while True:
        emails = parse_emails(EMAIL_SERVER, EMAIL_USER, EMAIL_PASS)
        with processed_email_uids_lock:
            for e in emails:
                issue = e["issue"]
                customer = e["customer"]
                sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)
                insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag)
        time.sleep(60)

# --- COLOR SETS for UI ---

STATUS_COLORS = {
    "Open": "#FFA500",
    "In Progress": "#1E90FF",
    "Resolved": "#32CD32"
}

SEVERITY_COLORS = {
    "critical": "#FF4500",
    "major": "#FF8C00",
    "minor": "#228B22"
}

URGENCY_COLORS = {
    "high": "#DC143C",
    "normal": "#008000"
}

def colored_text(text, color):
    return f'<span style="color:{color};font-weight:bold;">{text}</span>'

# --- STREAMLIT UI ---

ensure_schema()

st.set_page_config(layout="wide")
st.title("🚨 EscalateAI – Customer Escalation Management")

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

if st.sidebar.button("📤 Download All Complaints (CSV)"):
    df = fetch_escalations()
    csv = df.to_csv(index=False)
    st.sidebar.download_button("Download CSV", csv, file_name="escalations.csv", mime="text/csv")

# New Download escalated cases button
if st.sidebar.button("📥 Download Escalated Cases (Excel)"):
    df = fetch_escalations()
    df_esc = df[df["escalated"] == "Yes"]
    if df_esc.empty:
        st.sidebar.info("No escalated cases to download.")
    else:
        towrite = pd.ExcelWriter("escalated_cases.xlsx", engine='xlsxwriter')
        df_esc.to_excel(towrite, index=False, sheet_name='EscalatedCases')
        towrite.save()
        with open("escalated_cases.xlsx", "rb") as file:
            st.sidebar.download_button(
                label="Download Escalated Cases Excel",
                data=file,
                file_name="escalated_cases.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if st.sidebar.button("📩 Fetch Emails (IMAP)"):
    emails = parse_emails(EMAIL_SERVER, EMAIL_USER, EMAIL_PASS)
    count = len(emails)
    for e in emails:
        issue = e["issue"]
        customer = e["customer"]
        sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)
        insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag)
    st.sidebar.success(f"Fetched and processed {count} new emails.")

if st.sidebar.button("📣 Trigger SLA Alert"):
    df = fetch_escalations()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    breaches = df[(df['status'] != 'Resolved') & (df['priority'] == 'high') & 
                  ((datetime.datetime.now() - df['timestamp']) > datetime.timedelta(minutes=10))]
    if not breaches.empty:
        alert_msg = f"🚨 SLA breach detected for {len(breaches)} case(s)!"
        send_alert(alert_msg, via="teams")
        send_alert(alert_msg, via="email")
        st.sidebar.success("SLA breach alert sent.")
    else:
        st.sidebar.info("No SLA breaches detected.")

# Display SLA breach banner in sidebar if any
df_all = fetch_escalations()
df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], errors='coerce')
breaches = df_all[(df_all['status'] != 'Resolved') & (df_all['priority'] == 'high') & 
                  ((datetime.datetime.now() - df_all['timestamp']) > datetime.timedelta(minutes=10))]
if not breaches.empty:
    st.sidebar.markdown(
        f"<div style='background-color:#FF6347;color:white;padding:10px;border-radius:5px;margin-bottom:10px;text-align:center;'>"
        f"🚨 SLA breach detected for {len(breaches)} case(s)!"
        f"</div>", unsafe_allow_html=True
    )

# --- Tabs for main UI ---

tabs = st.tabs(["🗃️ All", "🚩 Escalated", "🔁 Feedback & Retraining"])

with tabs[0]:
    st.subheader("📊 Escalation Kanban Board")

    df = fetch_escalations()
    counts = df['status'].value_counts()
    open_count = counts.get('Open', 0)
    inprogress_count = counts.get('In Progress', 0)
    resolved_count = counts.get('Resolved', 0)
    st.markdown(f"**Open:** {open_count} | **In Progress:** {inprogress_count} | **Resolved:** {resolved_count}")

    col1, col2, col3 = st.columns(3)
    for status, col in zip(["Open", "In Progress", "Resolved"], [col1, col2, col3]):
        with col:
            col.markdown(f"<h3 style='background-color:{STATUS_COLORS[status]};color:white;padding:8px;border-radius:5px;text-align:center;'>{status}</h3>", unsafe_allow_html=True)
            bucket = df[df["status"] == status]
            for i, row in bucket.iterrows():
                flag = "🚩" if row['escalated'] == 'Yes' else ""
                severity_color = SEVERITY_COLORS.get(row['severity'], "black")
                urgency_color = URGENCY_COLORS.get(row['urgency'], "black")
                
                exp_label = f"{row['id']} - {row['customer']} {flag}"
                with st.expander(exp_label, expanded=False):
                    st.markdown(f"**Issue:** {row['issue']}")
                    st.markdown(f"**Severity:** {colored_text(row['severity'].capitalize(), severity_color)}", unsafe_allow_html=True)
                    st.markdown(f"**Criticality:** {row['criticality'].capitalize()}")
                    st.markdown(f"**Category:** {row['category'].capitalize() if row['category'] else 'N/A'}")
                    st.markdown(f"**Sentiment:** {row['sentiment'].capitalize()}")
                    st.markdown(f"**Urgency:** {colored_text(row['urgency'].capitalize(), urgency_color)}", unsafe_allow_html=True)
                    st.markdown(f"**Escalated:** {flag if flag else 'No'}")

                    new_status = st.selectbox(
                        "Update Status",
                        ["Open", "In Progress", "Resolved"],
                        index=["Open", "In Progress", "Resolved"].index(row["status"]),
                        key=f"status_{row['id']}"
                    )
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

# --- BACKGROUND THREAD START ---
if 'email_thread' not in st.session_state:
    email_thread = threading.Thread(target=email_polling_job, daemon=True)
    email_thread.start()
    st.session_state['email_thread'] = email_thread

