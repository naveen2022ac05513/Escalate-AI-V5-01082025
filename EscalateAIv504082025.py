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

# Configuration (set these in your .env or environment)
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

def get_next_escalation_id():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id FROM escalations WHERE id LIKE '{ESCALATION_PREFIX}%' ORDER BY id DESC LIMIT 1")
    last = cursor.fetchone()
    conn.close()
    if last:
        last_num = int(last[0].replace(ESCALATION_PREFIX, ""))
        next_num = last_num + 1
    else:
        next_num = 1
    return f"{ESCALATION_PREFIX}{str(next_num).zfill(5)}"

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

def analyze_issue(issue_text):
    sentiment_score = analyzer.polarity_scores(issue_text)
    compound = sentiment_score["compound"]
    if compound < -0.05:
        sentiment = "negative"
    elif compound > 0.05:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    urgency = "high" if any(word in issue_text.lower() for category in NEGATIVE_KEYWORDS.values() for word in category) else "normal"

    category = None
    for cat, keywords in NEGATIVE_KEYWORDS.items():
        if any(k in issue_text.lower() for k in keywords):
            category = cat
            break

    if category in ["safety", "technical"]:
        severity = "critical"
    elif category in ["support", "business"]:
        severity = "major"
    else:
        severity = "minor"

    criticality = "high" if sentiment == "negative" and urgency == "high" else "medium"
    escalation_flag = "Yes" if urgency == "high" or sentiment == "negative" else "No"

    return sentiment, urgency, severity, criticality, category, escalation_flag

def train_model():
    df = fetch_escalations()
    if df.shape[0] < 20:
        return None
    df = df.dropna(subset=['sentiment', 'urgency', 'severity', 'criticality', 'escalated'])
    if df.empty:
        return None
    X = pd.get_dummies(df[['sentiment', 'urgency', 'severity', 'criticality']])
    y = df['escalated'].apply(lambda x: 1 if x == 'Yes' else 0)
    if y.nunique() < 2:
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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

STATUS_COLORS = {
    "Open": "#FFA500",
    "In Progress": "#1E90FF",
    "Resolved": "#32CD32",
    "Escalated": "#DC143C"
}

def colored_text(text, color):
    return f'<span style="color:{color};font-weight:bold;">{text}</span>'

ensure_schema()

# Inject CSS for fixed sticky title below Streamlit header
st.markdown(
    """
    <style>
    #app-title {
        position: fixed;
        top: 50px;
        left: 0;
        width: 100%;
        background-color: white;
        padding: 10px 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        z-index: 9998;
    }
    #main-content {
        padding-top: 90px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit default header stays as is

st.markdown(
    """
    <div id="app-title">
        <h1 style="margin:0;">🚨 EscalateAI – Customer Escalation Management System</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div id="main-content">', unsafe_allow_html=True)

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

if st.sidebar.button("📥 Download Escalated Cases (Excel)"):
    df = fetch_escalations()
    df_esc = df[df["escalated"] == "Yes"]
    if df_esc.empty:
        st.sidebar.info("No escalated cases to download.")
    else:
        with pd.ExcelWriter("escalated_cases.xlsx", engine='xlsxwriter') as writer:
            df_esc.to_excel(writer, index=False, sheet_name='EscalatedCases')
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

tabs = st.tabs(["🗃️ All", "🚩 Escalated", "🔁 Feedback & Retraining"])

with tabs[0]:
    st.subheader("All Customer Escalations")
    df = fetch_escalations()
    if df.empty:
        st.info("No escalations found.")
    else:
        status_counts = df['status'].value_counts()
        cols = st.columns(len(status_counts))
        for idx, (status, count) in enumerate(status_counts.items()):
            color = STATUS_COLORS.get(status, "black")
            cols[idx].markdown(f"### {colored_text(status, color)}")
            cols[idx].markdown(f"### {count}")
        selected_status = st.selectbox("Filter by status", options=["All", "Open", "In Progress", "Resolved", "Escalated"])
        if selected_status != "All":
            df = df[df["status"] == selected_status]
        for _, row in df.iterrows():
            with st.expander(f"{row['id']} - {row['customer']} - {row['category'] or 'Uncategorized'}"):
                st.markdown(f"**Issue:** {row['issue']}")
                st.markdown(f"**Sentiment:** {row['sentiment']} | **Urgency:** {row['urgency']} | **Severity:** {row['severity']} | **Criticality:** {row['criticality']} | **Escalated:** {row['escalated']}")
                new_status = st.selectbox(f"Status for {row['id']}", options=["Open", "In Progress", "Resolved", "Escalated"], index=["Open", "In Progress", "Resolved", "Escalated"].index(row["status"]))
                action_taken = st.text_area(f"Action Taken for {row['id']}", value=row.get("action_taken", ""))
                action_owner = st.text_input(f"Action Owner for {row['id']}", value=row.get("action_owner", ""))
                if st.button(f"Update {row['id']}"):
                    update_escalation_status(row['id'], new_status, action_taken, action_owner)
                    st.success(f"Updated escalation {row['id']} successfully.")
                    st.experimental_rerun()

with tabs[1]:
    st.subheader("Escalated Cases")
    df_esc = df_all[df_all["escalated"] == "Yes"]
    if df_esc.empty:
        st.info("No escalated cases at the moment.")
    else:
        st.dataframe(df_esc[["id", "customer", "issue", "severity", "criticality", "status", "timestamp"]])

with tabs[2]:
    st.subheader("Feedback & Retrain Model")
    st.info("Please provide feedback on escalation predictions to improve the model.")
    df_feedback = df_all[df_all["user_feedback"].isnull() | (df_all["user_feedback"] == "")]
    if df_feedback.empty:
        st.success("No pending feedback entries.")
    else:
        for _, row in df_feedback.iterrows():
            st.markdown(f"### {row['id']} - {row['customer']}")
            st.markdown(f"**Issue:** {row['issue']}")
            feedback = st.radio(f"Was the escalation prediction correct for {row['id']}?", options=["Yes", "No"], key=row['id'])
            if st.button(f"Submit Feedback {row['id']}", key="fb_" + row['id']):
                update_escalation_status(row['id'], row['status'], row['action_taken'], row['action_owner'], feedback)
                st.success("Feedback submitted. Thank you!")
                st.experimental_rerun()
    if st.button("Retrain Model Now"):
        model = train_model()
        if model:
            st.success("Model retrained successfully.")
            # Save model if needed - omitted for brevity
        else:
            st.warning("Not enough data to train model. Need at least 20 varied samples.")

st.markdown('</div>', unsafe_allow_html=True)

# Start email polling thread on app start
if "email_thread" not in st.session_state:
    thread = threading.Thread(target=email_polling_job, daemon=True)
    thread.start()
    st.session_state.email_thread = thread
