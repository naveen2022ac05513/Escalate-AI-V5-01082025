# --- Imports ---
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
from dotenv import load_dotenv
import threading

# --- Load config ---
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
processed_email_uids = set()
processed_email_uids_lock = threading.Lock()

NEGATIVE_KEYWORDS = {
    "technical": ["fail", "break", "crash", "defect", "fault"],
    "dissatisfaction": ["complain", "reject", "delay", "ignore"],
    "support": ["wait", "pending", "slow", "miss"],
    "safety": ["fire", "burn", "unsafe", "alarm"],
    "business": ["impact", "loss", "risk", "downtime"]
}

# --- DB Setup ---
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
        customer_phone TEXT
    )
    ''')
    conn.commit()
    conn.close()

def get_next_escalation_id():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id FROM escalations WHERE id LIKE '{ESCALATION_PREFIX}%' ORDER BY id DESC LIMIT 1")
    last = cursor.fetchone()
    conn.close()
    next_num = int(last[0].replace(ESCALATION_PREFIX, "")) + 1 if last else 1
    return f"{ESCALATION_PREFIX}{str(next_num).zfill(5)}"

def insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag, customer_phone=""):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    new_id = get_next_escalation_id()
    now = datetime.datetime.now().isoformat()
    cursor.execute('''
    INSERT INTO escalations (
        id, customer, issue, sentiment, urgency, severity, criticality, category,
        status, timestamp, escalated, priority, escalation_flag,
        action_taken, owner, action_owner, status_update_date, user_feedback, customer_phone
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        new_id, customer, issue, sentiment, urgency, severity, criticality, category,
        "Open", now, escalation_flag, "normal", escalation_flag,
        "", "", "", "", "", customer_phone
    ))
    conn.commit()
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

# --- Alerts ---
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

# --- Sentiment & Analysis ---
def analyze_issue(issue_text):
    sentiment_score = analyzer.polarity_scores(issue_text)
    compound = sentiment_score["compound"]
    sentiment = "negative" if compound < -0.05 else "positive" if compound > 0.05 else "neutral"
    urgency = "high" if any(word in issue_text.lower() for category in NEGATIVE_KEYWORDS.values() for word in category) else "normal"
    category = next((cat for cat, kws in NEGATIVE_KEYWORDS.items() if any(k in issue_text.lower() for k in kws)), None)
    severity = "critical" if category in ["safety", "technical"] else "major" if category in ["support", "business"] else "minor"
    criticality = "high" if sentiment == "negative" and urgency == "high" else "medium"
    escalation_flag = "Yes" if sentiment == "negative" or urgency == "high" else "No"
    return sentiment, urgency, severity, criticality, category, escalation_flag

# --- UI Setup ---
ensure_schema()
st.set_page_config(layout="wide")
st.markdown("<h1>🚨 EscalateAI – Escalation System with WhatsApp</h1>", unsafe_allow_html=True)

st.sidebar.header("📥 Upload")
uploaded_file = st.sidebar.file_uploader("Excel File", type=["xlsx"])
if uploaded_file:
    df_excel = pd.read_excel(uploaded_file)
    for _, row in df_excel.iterrows():
        customer = str(row.get("customer", "Unknown"))
        issue = str(row.get("issue", ""))
        customer_phone = str(row.get("customer_phone", ""))
        sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)
        insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag, customer_phone)
    st.sidebar.success("Uploaded and processed.")

# --- Kanban Tab ---
st.subheader("📊 Escalation Kanban Board")
df = fetch_escalations()
statuses = ["Open", "In Progress", "Resolved"]
cols = st.columns(3)

for status, col in zip(statuses, cols):
    with col:
        st.markdown(f"### {status}")
        bucket = df[df["status"] == status]
        for i, row in bucket.iterrows():
            with st.expander(f"{row['id']} - {row['customer']}", expanded=False):
                st.markdown(f"**Issue:** {row['issue']}")
                st.markdown(f"**Sentiment:** {row['sentiment']} | **Urgency:** {row['urgency']} | **Severity:** {row['severity']}")
                st.markdown(f"**Category:** {row['category']} | **Escalated:** {row['escalated']}")

                new_status = st.selectbox("Update Status", statuses, index=statuses.index(row["status"]), key=f"status_{row['id']}")
                new_action = st.text_input("Action Taken", row.get("action_taken", ""), key=f"action_{row['id']}")
                new_owner = st.text_input("Owner", row.get("owner", ""), key=f"owner_{row['id']}")

                if st.button("💾 Save", key=f"save_{row['id']}"):
                    update_escalation_status(row['id'], new_status, new_action, new_owner)
                    st.success("Saved.")

                # 🟢 WhatsApp Notification for Resolved Cases
                if row["status"] == "Resolved" and row["customer_phone"]:
                    phone_number = f"whatsapp:{row['customer_phone']}"
                    if st.button("📱 Notify via WhatsApp", key=f"whatsapp_{row['id']}"):
                        message = (
                            f"Hello {row['customer']},\n"
                            f"Your escalation [{row['id']}] has been resolved.\n"
                            "We appreciate your patience and feedback."
                       
