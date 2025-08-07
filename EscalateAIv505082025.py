# 🚨 EscalateAI – Enhanced Version
# Author: Chanakya Gandham
# Description: AI-powered escalation management with ML, UI, alerts, and deduplication

# -------------------
# --- Imports -------
# -------------------
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
import hashlib
from email.header import decode_header
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import smtplib
import requests
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.message import EmailMessage

# -------------------
# --- Config -------
# -------------------
load_dotenv()
EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
ALERT_RECIPIENT = os.getenv("EMAIL_RECEIVER")
TEAMS_WEBHOOK = os.getenv("MS_TEAMS_WEBHOOK_URL")
EMAIL_SUBJECT = os.getenv("EMAIL_SUBJECT", "🚨 EscalateAI Alert")

# -------------------
# --- Constants -----
# -------------------
DB_PATH = "escalations.db"
ESCALATION_PREFIX = "SESICE-25"
analyzer = SentimentIntensityAnalyzer()
global_seen_hashes = set()

NEGATIVE_KEYWORDS = {
    "technical": ["fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip", "malfunction", "blank", "shutdown", "discharge", "leak"],
    "dissatisfaction": ["dissatisfy", "frustrate", "complain", "reject", "delay", "ignore", "escalate", "displease", "noncompliance", "neglect"],
    "support": ["wait", "pending", "slow", "incomplete", "miss", "omit", "unresolved", "shortage", "no response"],
    "safety": ["fire", "burn", "flashover", "arc", "explode", "unsafe", "leak", "corrode", "alarm", "incident"],
    "business": ["impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate", "penalty"]
}
# -------------------
# --- DB Setup -------
# -------------------
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
        user_feedback TEXT,
        location TEXT,
        region TEXT,
        activity TEXT
    )
    ''')
    conn.commit()
    conn.close()

# -------------------
# --- Helper Functions -------
# -------------------
def summarize_issue_text(issue_text):
    clean_text = re.sub(r'\s+', ' ', issue_text).strip()
    return clean_text[:120] + "..." if len(clean_text) > 120 else clean_text

def generate_issue_hash(issue_text):
    clean_text = re.sub(r'\s+', ' ', issue_text.lower().strip())
    return hashlib.md5(clean_text.encode()).hexdigest()

def compute_ageing(ts):
    if not ts or pd.isnull(ts):
        return "00:00"
    try:
        ts_dt = pd.to_datetime(ts, errors='coerce')
        now = datetime.datetime.now()
        elapsed = now - ts_dt
        total_minutes = int(elapsed.total_seconds() // 60)
        hours, minutes = divmod(total_minutes, 60)
        return f"{hours:02d}:{minutes:02d}"
    except Exception:
        return "00:00"

def get_next_escalation_id():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f'''
    SELECT id FROM escalations WHERE id LIKE '{ESCALATION_PREFIX}%'
    ORDER BY id DESC LIMIT 1
    ''')
    last = cursor.fetchone()
    conn.close()
    next_num = int(last[0].replace(ESCALATION_PREFIX, "")) + 1 if last else 1
    return f"{ESCALATION_PREFIX}{str(next_num).zfill(5)}"

def insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag, location="", region="", activity=""):
    issue_hash = generate_issue_hash(issue)
    if issue_hash in global_seen_hashes:
        return
    global_seen_hashes.add(issue_hash)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    new_id = get_next_escalation_id()
    now = datetime.datetime.now().isoformat()
    cursor.execute('''
    INSERT INTO escalations (
        id, customer, issue, sentiment, urgency, severity, criticality, category,
        status, timestamp, escalated, priority, escalation_flag,
        action_taken, owner, action_owner, status_update_date, user_feedback,
        location, region, activity
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        new_id, customer, issue, sentiment, urgency, severity, criticality, category,
        "Open", now, escalation_flag, "normal", escalation_flag,
        "", "", "", "", "", location, region, activity
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
    # -------------------
# --- DB Setup -------
# -------------------
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
        user_feedback TEXT,
        location TEXT,
        region TEXT,
        activity TEXT
    )
    ''')
    conn.commit()
    conn.close()

# -------------------
# --- Helper Functions -------
# -------------------
def summarize_issue_text(issue_text):
    clean_text = re.sub(r'\s+', ' ', issue_text).strip()
    return clean_text[:120] + "..." if len(clean_text) > 120 else clean_text

def generate_issue_hash(issue_text):
    clean_text = re.sub(r'\s+', ' ', issue_text.lower().strip())
    return hashlib.md5(clean_text.encode()).hexdigest()

def compute_ageing(ts):
    if not ts or pd.isnull(ts):
        return "00:00"
    try:
        ts_dt = pd.to_datetime(ts, errors='coerce')
        now = datetime.datetime.now()
        elapsed = now - ts_dt
        total_minutes = int(elapsed.total_seconds() // 60)
        hours, minutes = divmod(total_minutes, 60)
        return f"{hours:02d}:{minutes:02d}"
    except Exception:
        return "00:00"

def get_next_escalation_id():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f'''
    SELECT id FROM escalations WHERE id LIKE '{ESCALATION_PREFIX}%'
    ORDER BY id DESC LIMIT 1
    ''')
    last = cursor.fetchone()
    conn.close()
    next_num = int(last[0].replace(ESCALATION_PREFIX, "")) + 1 if last else 1
    return f"{ESCALATION_PREFIX}{str(next_num).zfill(5)}"

def insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag, location="", region="", activity=""):
    issue_hash = generate_issue_hash(issue)
    if issue_hash in global_seen_hashes:
        return
    global_seen_hashes.add(issue_hash)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    new_id = get_next_escalation_id()
    now = datetime.datetime.now().isoformat()
    cursor.execute('''
    INSERT INTO escalations (
        id, customer, issue, sentiment, urgency, severity, criticality, category,
        status, timestamp, escalated, priority, escalation_flag,
        action_taken, owner, action_owner, status_update_date, user_feedback,
        location, region, activity
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        new_id, customer, issue, sentiment, urgency, severity, criticality, category,
        "Open", now, escalation_flag, "normal", escalation_flag,
        "", "", "", "", "", location, region, activity
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

# -------------------
# --- Email Parsing -------
# -------------------
def parse_emails():
    emails = []
    try:
        conn = imaplib.IMAP4_SSL(EMAIL_SERVER)
        conn.login(EMAIL_USER, EMAIL_PASS)
        conn.select("inbox")
        _, messages = conn.search(None, "UNSEEN")
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
                    full_text = f"{subject} - {body}"
                    issue_hash = generate_issue_hash(full_text)
                    if issue_hash not in global_seen_hashes:
                        global_seen_hashes.add(issue_hash)
                        summary = summarize_issue_text(full_text)
                        emails.append({
                            "customer": from_,
                            "issue": summary
                        })
        conn.logout()
    except Exception as e:
        st.error(f"Failed to parse emails: {e}")
    return emails

# -------------------
# --- Excel Upload -------
# -------------------
def process_uploaded_excel(uploaded_file):
    df_excel = pd.read_excel(uploaded_file)
    for _, row in df_excel.iterrows():
        issue = str(row.get("issue", ""))
        customer = str(row.get("customer", "Unknown"))
        location = str(row.get("location", ""))
        region = str(row.get("region", ""))
        activity = str(row.get("activity", ""))
        issue_summary = summarize_issue_text(issue)
        issue_hash = generate_issue_hash(issue)
        if issue_hash in global_seen_hashes:
            continue
        global_seen_hashes.add(issue_hash)
        sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)
        insert_escalation(customer, issue_summary, sentiment, urgency, severity, criticality, category, escalation_flag, location, region, activity)

# -------------------
# --- UI Setup -------
# -------------------
ensure_schema()
st.set_page_config(layout="wide", page_title="EscalateAI", page_icon="🚨")
st.markdown("<h1>🚨 EscalateAI – Enhanced Escalation Management</h1>", unsafe_allow_html=True)

# Theme toggle
theme = st.sidebar.radio("Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("<style>body{background-color:#1e1e1e;color:white;}</style>", unsafe_allow_html=True)

# 📥 Excel Upload
st.sidebar.markdown("### 📥 Upload Excel")
uploaded_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])
if uploaded_file:
    process_uploaded_excel(uploaded_file)
    st.sidebar.success("✅ File processed successfully")

# 📩 Email Fetch
st.sidebar.markdown("### 📩 Fetch Emails")
if st.sidebar.button("Fetch Emails"):
    emails = parse_emails()
    for e in emails:
        issue, customer = e["issue"], e["customer"]
        sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)
        insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag)
    st.sidebar.success(f"✅ {len(emails)} emails processed")

# 🔍 Filters
df = fetch_escalations()
status = st.sidebar.selectbox("Status", ["All"] + sorted(df["status"].dropna().unique()))
severity = st.sidebar.selectbox("Severity", ["All"] + sorted(df["severity"].dropna().unique()))
sentiment = st.sidebar.selectbox("Sentiment", ["All"] + sorted(df["sentiment"].dropna().unique()))
category = st.sidebar.selectbox("Category", ["All"] + sorted(df["category"].dropna().unique()))
view = st.sidebar.radio("Escalation View", ["All", "Escalated", "Non-Escalated"])

filtered_df = df.copy()
if status != "All":
    filtered_df = filtered_df[filtered_df["status"] == status]
if severity != "All":
    filtered_df = filtered_df[filtered_df["severity"] == severity]
if sentiment != "All":
    filtered_df = filtered_df[filtered_df["sentiment"] == sentiment]
if category != "All":
    filtered_df = filtered_df[filtered_df["category"] == category]
if view == "Escalated":
    filtered_df = filtered_df[filtered_df["escalated"] == "Yes"]
elif view == "Non-Escalated":
    filtered_df = filtered_df[filtered_df["escalated"] != "Yes"]

# 📊 Kanban Board
st.subheader("📊 Escalation Kanban Board")
col1, col2, col3 = st.columns(3)
for status, col in zip(["Open", "In Progress", "Resolved"], [col1, col2, col3]):
    col.markdown(f"### {status}")
    bucket = filtered_df[filtered_df["status"] == status]
    for _, row in bucket.iterrows():
        ageing_value = compute_ageing(row["timestamp"])
        expander_label = f"{row['id']} - {row['customer']} ⏳ {ageing_value}"
        with col.expander(expander_label, expanded=False):
            st.markdown(f"**Issue:** {row['issue']}")
            st.markdown(f"**Severity:** {row['severity']}")
            st.markdown(f"**Urgency:** {row['urgency']}")
            st.markdown(f"**Criticality:** {row['criticality']}")
            st.markdown(f"**Category:** {row['category']}")
            st.markdown(f"**Sentiment:** {row['sentiment']}")
            st.markdown(f"**Escalated:** {row['escalated']}")
            st.markdown(f"**Location:** {row.get('location', '')}")
            st.markdown(f"**Region:** {row.get('region', '')}")
            st.markdown(f"**Activity:** {row.get('activity', '')}")

            # Editable fields
            new_status = st.selectbox("Update Status", ["Open", "In Progress", "Resolved"], index=["Open", "In Progress", "Resolved"].index(row["status"]))
            new_action = st.text_input("Action Taken", row.get("action_taken", ""), key=f"action_{row['id']}")
            new_owner = st.text_input("Owner", row.get("owner", ""), key=f"owner_{row['id']}")
            new_owner_email = st.text_input("Owner Email", row.get("action_owner", ""), key=f"email_{row['id']}")
            location = st.text_input("Location", row.get("location", ""), key=f"loc_{row['id']}")
            region = st.text_input("Region", row.get("region", ""), key=f"reg_{row['id']}")
            activity = st.text_input("Activity", row.get("activity", ""), key=f"act_{row['id']}")

            if st.button("💾 Save Changes", key=f"save_{row['id']}"):
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute('''
                UPDATE escalations SET status=?, action_taken=?, owner=?, action_owner=?, location=?, region=?, activity=?, status_update_date=?
                WHERE id=?
                ''', (new_status, new_action, new_owner, new_owner_email, location, region, activity, datetime.datetime.now().isoformat(), row['id']))
                conn.commit()
                conn.close()
                st.success("Escalation updated.")
                send_alert(f"🔔 Escalation {row['id']} updated.", via="email", recipient=new_owner_email)

            # Escalation actions
            if st.button("🚀 Escalate to N+1", key=f"n1_{row['id']}"):
                send_alert(f"Escalation #{row['id']} moved to Tier 2", via="teams", recipient="tier2@company.com")
            if st.button("📣 Escalate to N+2", key=f"n2_{row['id']}"):
                send_alert(f"Escalation #{row['id']} escalated to Management", via="email", recipient="management@company.com")

# -------------------
# --- SLA Monitoring -------
# -------------------
st.sidebar.markdown("### ⏰ SLA Monitor")
df_all = fetch_escalations()
df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], errors='coerce')
breaches = df_all[(df_all['status'] != 'Resolved') & (df_all['priority'] == 'high') &
                  ((datetime.datetime.now() - df_all['timestamp']) > datetime.timedelta(minutes=10))]

if not breaches.empty:
    st.sidebar.markdown(
        f"<div style='background:#dc3545;padding:8px;border-radius:5px;color:white;text-align:center;'>"
        f"<strong>🚨 {len(breaches)} SLA Breach(s) Detected</strong></div>",
        unsafe_allow_html=True
    )
    if st.sidebar.button("Trigger SLA Alerts"):
        alert_msg = f"🚨 SLA breach for {len(breaches)} case(s)!"
        send_alert(alert_msg, via="teams")
        send_alert(alert_msg, via="email")
        st.sidebar.success("✅ Alerts sent")
else:
    st.sidebar.info("All SLAs healthy")

# -------------------
# --- Manual Alerts -------
# -------------------
st.sidebar.markdown("### 🔔 Manual Notifications")
manual_msg = st.sidebar.text_area("Compose Alert", "🚨 Test alert from EscalateAI")
if st.sidebar.button("Send MS Teams"):
    send_alert(manual_msg, via="teams")
    st.sidebar.success("✅ MS Teams alert sent")
if st.sidebar.button("Send Email"):
    send_alert(manual_msg, via="email")
    st.sidebar.success("✅ Email alert sent")

# -------------------
# --- Optional: Drag-and-Drop Kanban (Plugin) -------
# -------------------
# Uncomment below if using streamlit-dnd
# pip install streamlit-dnd

# from streamlit_dnd import dnd_list
# st.subheader("🧲 Drag-and-Drop Kanban (Experimental)")
# for status in ["Open", "In Progress", "Resolved"]:
#     items = df[df["status"] == status]["id"].tolist()
#     new_order = dnd_list(items, direction="vertical", title=status)
#     for item in new_order:
#         update_escalation_status(item, status, "", "", "")

# -------------------
# --- Feedback & Retraining -------
# -------------------
st.subheader("🔁 Feedback & Retraining")
df_feedback = df[df["escalated"].notnull()]
fb_map = {"Correct": 1, "Incorrect": 0}

for _, row in df_feedback.iterrows():
    with st.expander(f"🆔 {row['id']}"):
        fb = st.selectbox("Escalation Accuracy", ["Correct", "Incorrect"], key=f"fb_{row['id']}")
        sent = st.selectbox("Sentiment", ["positive", "neutral", "negative"], key=f"sent_{row['id']}")
        crit = st.selectbox("Criticality", ["low", "medium", "high"], key=f"crit_{row['id']}")
        notes = st.text_area("Notes", key=f"note_{row['id']}")
        if st.button("Submit Feedback", key=f"btn_{row['id']}"):
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
            UPDATE escalations SET user_feedback=?, sentiment=?, criticality=?, status_update_date=?
            WHERE id=?
            ''', (fb_map[fb], sent, crit, datetime.datetime.now().isoformat(), row['id']))
            conn.commit()
            conn.close()
            st.success("Feedback saved.")

# Retrain model
if st.button("🔁 Retrain Model"):
    st.info("Retraining model with feedback...")
    model = train_model()
    if model:
        st.success("Model retrained successfully.")
    else:
        st.warning("Not enough data to retrain model.")

# -------------------
# --- Developer Options -------
# -------------------
st.sidebar.markdown("### 🧪 Developer Tools")
if st.sidebar.checkbox("View Raw Database"):
    st.sidebar.dataframe(fetch_escalations())

if st.sidebar.button("🗑️ Reset Database"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS escalations")
    conn.commit()
    conn
