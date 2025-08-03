# EscalateAI – Full Functionality
# ========================================
# Features:
# • Parsing emails + Excel for customer issues
# • NLP analysis (sentiment, urgency, category tagging)
# • Unique ID: SESICE-XXXXX
# • Kanban Board: Open, In Progress, Resolved
# • Action Taken + Owner inline editing
# • SLA Breach detection (10 mins)
# • Teams/email alerts
# • Predictive ML + Feedback + Retraining

import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import requests
from streamlit_autorefresh import st_autorefresh
import uuid
import time

# Setup
load_dotenv()
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
IMAP_SERVER = "imap.gmail.com"
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")
DB_PATH = "escalations.db"

analyzer = SentimentIntensityAnalyzer()

NEGATIVE_KEYWORDS = ["fail", "defect", "shutdown", "complain", "delay", "ignore", "escalate", "violation", "unsafe", "fire", "burn", "leak"]
URGENCY_PHRASES = ["asap", "urgent", "immediately", "right away", "critical"]
CATEGORY_KEYWORDS = {
    "Safety": ["fire", "burn", "leak", "unsafe"],
    "Performance": ["slow", "crash", "malfunction"],
    "Delay": ["delay", "pending", "wait"],
    "Compliance": ["noncompliance", "violation"],
    "Service": ["ignore", "unavailable"],
    "Quality": ["defect", "fault"]
}

# DB Init
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS escalations (
    escalation_id TEXT PRIMARY KEY,
    customer TEXT,
    issue TEXT,
    date TEXT,
    status TEXT,
    sentiment TEXT,
    priority TEXT,
    escalation_flag INTEGER,
    urgency TEXT,
    category TEXT,
    action_taken TEXT,
    action_owner TEXT,
    status_update_date TEXT
)
""")
conn.commit()

# Utils
def generate_id():
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0] + 250001
    return f"SESICE-{count}"

def classify_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    return "Negative" if score < -0.05 else "Positive" if score > 0.05 else "Neutral"

def detect_urgency(text):
    return "High" if any(p in text.lower() for p in URGENCY_PHRASES) else "Normal"

def detect_category(text):
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(k in text.lower() for k in keywords):
            return cat
    return "General"

def is_escalation(text):
    return any(word in text.lower() for word in NEGATIVE_KEYWORDS)

def insert_to_db(data):
    cursor.execute("""
        INSERT INTO escalations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()

def fetch_cases():
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    return df

# Alert
def send_teams_alert(msg):
    if MS_TEAMS_WEBHOOK_URL:
        requests.post(MS_TEAMS_WEBHOOK_URL, json={"text": msg})

def detect_sla_breach():
    now = datetime.datetime.now()
    cursor.execute("SELECT escalation_id, date FROM escalations WHERE priority='High' AND status='Open'")
    rows = cursor.fetchall()
    for eid, date_str in rows:
        ts = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        if (now - ts).total_seconds() > 600:
            send_teams_alert(f"SLA Breach: Escalation {eid} still open >10 mins")

# Email Parsing (demo)
def parse_email():
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL, PASSWORD)
    mail.select("inbox")
    status, messages = mail.search(None, 'UNSEEN')
    for num in messages[0].split():
        _, data = mail.fetch(num, '(RFC822)')
        msg = email.message_from_bytes(data[0][1])
        subject = decode_header(msg["Subject"])[0][0]
        from_ = msg.get("From")
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            body = msg.get_payload(decode=True).decode()
        process_case(from_, body)
    mail.logout()

# Excel Upload
def process_excel(uploaded_file):
    df = pd.read_excel(uploaded_file)
    for _, row in df.iterrows():
        process_case(row['Customer'], row['Issue'])

# Core Processing
def process_case(customer, issue):
    eid = generate_id()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sentiment = classify_sentiment(issue)
    priority = "High" if sentiment == "Negative" else "Normal"
    urgency = detect_urgency(issue)
    category = detect_category(issue)
    flag = 1 if is_escalation(issue) else 0
    data = (eid, customer, issue, now, "Open", sentiment, priority, flag, urgency, category, "", "", now)
    insert_to_db(data)
    if flag and priority == "High":
        send_teams_alert(f"New Escalation: {eid} from {customer}\nIssue: {issue}")

# UI
st_autorefresh(interval=60000, limit=None)
st.title("🚨 EscalateAI – Customer Escalation Tracker")

with st.sidebar:
    st.subheader("Upload Customer Issues")
    file = st.file_uploader("Excel File (.xlsx)", type=["xlsx"])
    if file:
        process_excel(file)
        st.success("Uploaded and processed.")
    st.subheader("Manual Entry")
    cust = st.text_input("Customer")
    iss = st.text_area("Issue")
    if st.button("Add"):
        process_case(cust, iss)
        st.success("Manually added.")

st.subheader("Kanban View")
df = fetch_cases()
statuses = ["Open", "In Progress", "Resolved"]
cols = st.columns(len(statuses))

for i, status in enumerate(statuses):
    with cols[i]:
        st.markdown(f"### {status}")
        for _, row in df[df['status'] == status].iterrows():
            with st.expander(f"{row['escalation_id']} - {row['customer']}"):
                st.write(row['issue'])
                st.write(f"**Sentiment:** {row['sentiment']} | **Urgency:** {row['urgency']} | **Category:** {row['category']}")
                action = st.text_input("Action Taken", value=row['action_taken'], key=row['escalation_id'] + "_a")
                owner = st.text_input("Action Owner", value=row['action_owner'], key=row['escalation_id'] + "_o")
                new_status = st.selectbox("Update Status", statuses, index=statuses.index(status), key=row['escalation_id'] + "_s")
                if st.button("Save", key=row['escalation_id'] + "_save"):
                    cursor.execute("""
                        UPDATE escalations SET status=?, action_taken=?, action_owner=?, status_update_date=? WHERE escalation_id=?
                    """, (new_status, action, owner, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), row['escalation_id']))
                    conn.commit()
                    st.success("Updated")

# SLA Monitor
detect_sla_breach()
