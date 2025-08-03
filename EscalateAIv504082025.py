# EscalateAI - End-to-End Escalation Management Tool

import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import requests
import base64
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import smtplib
from email.mime.text import MIMEText

# Load environment variables
load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_SERVER = os.getenv("EMAIL_SERVER")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT"))
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# Setup
DB_FILE = "escalations.db"
NEGATIVE_WORDS = [
    "fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip", "malfunction", "blank",
    "shutdown", "discharge", "dissatisfy", "frustrate", "complain", "reject", "delay", "ignore", "escalate",
    "displease", "noncompliance", "neglect", "wait", "pending", "slow", "incomplete", "miss", "omit",
    "unresolved", "shortage", "no response", "fire", "burn", "flashover", "arc", "explode", "unsafe", "leak",
    "corrode", "alarm", "incident", "impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate", "penalty"
]

analyzer = SentimentIntensityAnalyzer()

# Ensure database
def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute('''CREATE TABLE IF NOT EXISTS escalations (
        id TEXT PRIMARY KEY,
        customer TEXT,
        issue TEXT,
        sentiment REAL,
        urgency TEXT,
        severity TEXT,
        criticality TEXT,
        category TEXT,
        status TEXT,
        timestamp TEXT,
        action_taken TEXT,
        owner TEXT,
        escalated INTEGER
    )''')
    conn.close()

def generate_id():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM escalations")
    count = cur.fetchone()[0]
    conn.close()
    return f"SESICE-{250000 + count + 1}"

def fetch_emails():
    mail = imaplib.IMAP4_SSL(EMAIL_SERVER)
    mail.login(EMAIL_USER, EMAIL_PASS)
    mail.select("inbox")
    _, search_data = mail.search(None, "UNSEEN")
    messages = []
    for num in search_data[0].split():
        _, data = mail.fetch(num, "(RFC822)")
        msg = email.message_from_bytes(data[0][1])
        subject = decode_header(msg["Subject"])[0][0]
        if isinstance(subject, bytes):
            subject = subject.decode()
        from_ = msg.get("From")
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == "text/plain":
                    body = part.get_payload(decode=True).decode(errors='ignore')
                    break
        else:
            body = msg.get_payload(decode=True).decode(errors='ignore')
        messages.append((from_, subject, body))
    mail.logout()
    return messages

def analyze_text(issue):
    sentiment_score = analyzer.polarity_scores(issue)["compound"]
    urgency = "High" if any(word in issue.lower() for word in NEGATIVE_WORDS) else "Normal"
    severity = "High" if sentiment_score < -0.5 else "Medium" if sentiment_score < 0 else "Low"
    criticality = "Critical" if urgency == "High" and severity == "High" else "Moderate"
    category = "Technical" if any(w in issue.lower() for w in ["fail", "crash", "malfunction"]) else "General"
    return sentiment_score, urgency, severity, criticality, category

def save_escalation(customer, issue):
    sentiment, urgency, severity, criticality, category = analyze_text(issue)
    entry_id = generate_id()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    escalated = 1 if urgency == "High" and severity == "High" else 0
    conn = sqlite3.connect(DB_FILE)
    conn.execute("INSERT OR IGNORE INTO escalations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (entry_id, customer, issue, sentiment, urgency, severity, criticality, category,
         "Open", timestamp, "", "", escalated))
    conn.commit()
    conn.close()
    return entry_id, urgency, severity

def send_alert(issue, id, channel="teams"):
    message = f"🚨 Escalation Alert 🚨\nID: {id}\nIssue: {issue}"
    if channel == "teams":
        requests.post(MS_TEAMS_WEBHOOK_URL, json={"text": message})
    elif channel == "email":
        msg = MIMEText(message)
        msg["Subject"] = "Escalation Alert"
        msg["From"] = EMAIL_USER
        msg["To"] = EMAIL_RECEIVER
        with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, EMAIL_RECEIVER, msg.as_string())

def load_data():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    conn.close()
    return df

def update_status(entry_id, status, action_taken, owner):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE escalations SET status=?, action_taken=?, owner=? WHERE id=?",
                 (status, action_taken, owner, entry_id))
    conn.commit()
    conn.close()
