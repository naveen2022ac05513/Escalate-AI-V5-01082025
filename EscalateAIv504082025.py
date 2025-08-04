# EscalateAIv504082025_Final.py

import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
import re
import smtplib
from email.mime.text import MIMEText
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import threading
import time
import requests
import xlsxwriter
import base64

# Load environment variables
load_dotenv()
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_SERVER = os.getenv("EMAIL_SERVER")
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT"))
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Negative keyword categories
NEGATIVE_KEYWORDS = {
    "Technical": ["fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip", "malfunction", "blank", "shutdown", "discharge"],
    "Customer": ["dissatisfy", "frustrate", "complain", "reject", "delay", "ignore", "escalate", "displease", "noncompliance", "neglect"],
    "Support": ["wait", "pending", "slow", "incomplete", "miss", "omit", "unresolved", "shortage", "no response"],
    "Hazard": ["fire", "burn", "flashover", "arc", "explode", "unsafe", "leak", "corrode", "alarm", "incident"],
    "Business": ["impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate", "penalty"]
}

# SQLite DB setup
DB_FILE = "escalate_ai.db"
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS escalations (
    id TEXT PRIMARY KEY,
    timestamp TEXT,
    sender TEXT,
    subject TEXT,
    issue TEXT,
    sentiment TEXT,
    urgency TEXT,
    severity TEXT,
    criticality TEXT,
    category TEXT,
    status TEXT,
    action_taken TEXT,
    action_owner TEXT
)
''')
conn.commit()

# Helper: Generate next unique ID starting from SESICE-2500001
def get_next_id():
    cursor.execute("SELECT id FROM escalations ORDER BY id DESC LIMIT 1")
    result = cursor.fetchone()
    if result:
        last_num = int(result[0].split("-")[-1])
        return f"SESICE-{last_num + 1}"
    else:
        return "SESICE-2500001"

# Helper: NLP logic
def analyze_issue(text):
    sentiment_score = analyzer.polarity_scores(text)['compound']
    sentiment = "Negative" if sentiment_score < -0.05 else "Positive" if sentiment_score > 0.05 else "Neutral"
    
    urgency = any(word in text.lower() for cat in NEGATIVE_KEYWORDS.values() for word in cat)
    category = None
    for cat, words in NEGATIVE_KEYWORDS.items():
        if any(word in text.lower() for word in words):
            category = cat
            break
    severity = "High" if urgency else "Low"
    criticality = "Critical" if sentiment == "Negative" and urgency else "Normal"
    
    return sentiment, "Urgent" if urgency else "Normal", severity, criticality, category or "General"

# Helper: Fetch emails from Gmail
def fetch_emails():
    mail = imaplib.IMAP4_SSL(EMAIL_SERVER)
    mail.login(EMAIL_USER, EMAIL_PASS)
    mail.select("inbox")
    status, messages = mail.search(None, 'UNSEEN')
    for num in messages[0].split():
        status, msg_data = mail.fetch(num, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                subject = decode_header(msg["Subject"])[0][0]
                subject = subject.decode() if isinstance(subject, bytes) else subject
                from_ = msg.get("From")
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body += part.get_payload(decode=True).decode()
                else:
                    body = msg.get_payload(decode=True).decode()
                sentiment, urgency, severity, criticality, category = analyze_issue(body)
                escalation_id = get_next_id()
                cursor.execute('''INSERT INTO escalations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                               (escalation_id, str(datetime.datetime.now()), from_, subject, body, sentiment, urgency, severity, criticality, category, "Open", "", ""))
                conn.commit()

# Helper: Send alert
def send_alert(message, via="teams"):
    if via == "teams":
        requests.post(MS_TEAMS_WEBHOOK_URL, json={"text": message})
    else:
        msg = MIMEText(message)
        msg['Subject'] = "Escalation Alert"
        msg['From'] = EMAIL_USER
        msg['To'] = EMAIL_RECEIVER
        with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)

# Helper: ML Prediction & Training
MODEL_FILE = "escalation_model.pkl"
def train_model():
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    if len(df) >= 10:
        df = df.fillna("")
        le = LabelEncoder()
        df['label'] = df['status'].apply(lambda x: 1 if x == "Escalated" else 0)
        features = df[['sentiment', 'urgency', 'severity', 'criticality', 'category']]
        for col in features.columns:
            features[col] = le.fit_transform(features[col])
        X = features
        y = df['label']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, MODEL_FILE)

def predict_escalation(row):
    if not os.path.exists(MODEL_FILE): return "Unknown"
    model = joblib.load(MODEL_FILE)
    le = LabelEncoder()
    features = pd.DataFrame([row], columns=["sentiment", "urgency", "severity", "criticality", "category"])
    for col in features.columns:
        features[col] = le.fit_transform(features[col])
    pred = model.predict(features)[0]
    return "Likely" if pred == 1 else "Unlikely"

# Streamlit UI
st.set_page_config(layout="wide", page_title="EscalateAI", page_icon="🚨")
st.title("🚨 EscalateAI - Intelligent Escalation Detection")

with st.sidebar:
    if st.button("📥 Fetch Emails"):
        fetch_emails()
        st.success("Emails fetched successfully!")
    if st.button("🚨 Trigger Alerts"):
        send_alert("New high-priority escalation detected!", via="teams")
        st.success("Alert sent to MS Teams!")
    if st.button("📊 Retrain Model"):
        train_model()
        st.success("ML model retrained!")
    if st.button("⬇️ Download Escalations"):
        df_export = pd.read_sql_query("SELECT * FROM escalations", conn)
        df_export.to_excel("escalated_cases.xlsx", index=False, engine='xlsxwriter')
        with open("escalated_cases.xlsx", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="escalated_cases.xlsx">Download Excel</a>'
            st.markdown(href, unsafe_allow_html=True)

# Load data
df = pd.read_sql_query("SELECT * FROM escalations", conn)

# Kanban-style display
statuses = ["Open", "In Progress", "Resolved", "Escalated"]
cols = st.columns(len(statuses))
for i, status in enumerate(statuses):
    with cols[i]:
        st.markdown(f"### {status} ({len(df[df['status']==status])})")
        for _, row in df[df['status']==status].iterrows():
            color = "#f8d7da" if row['severity'] == "High" else "#d4edda"
            st.markdown(
                f"<div style='background-color:{color};padding:10px;margin-bottom:5px;border-radius:5px'>"
                f"<b>{row['id']}</b><br>{row['issue'][:100]}...<br><small>{row['timestamp']}</small></div>",
                unsafe_allow_html=True
            )
