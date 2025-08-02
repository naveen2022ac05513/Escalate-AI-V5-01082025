Certainly! Here's the full updated code again, with **detailed explanations** for each major block to help you understand exactly what it does and how it fits into the overall EscalateAI system.

---

```python
# ==============================================================
# EscalateAI – Escalation Management Tool with Email Parsing
# --------------------------------------------------------------
# • Parses emails from seservices.schneider@se.com (IMAP)
# • Logs escalations directly into database every minute
# • Predicts sentiment, urgency, and risk in real-time using transformers and rule-based
# • Streamlit dashboard for escalation tracking
# • Supports manual entry, Excel upload, and CSV download
# • Logs scheduler activity and allows pause/resume controls
# • Notifies when new escalation is added
# • Filters by urgency, sentiment (rule & transformer), date, escalation status
# --------------------------------------------------------------
# Author: Naveen Gandham • v1.5.0 • August 2025
# ==============================================================

import os
import re
import sqlite3
import uuid
import email
from email.mime.text import MIMEText
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from imapclient import IMAPClient
from bs4 import BeautifulSoup
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import smtplib

# Hugging Face transformers imports
from transformers import pipeline


# ----------------------- Paths & Environment Variables -----------------------
# Define directory paths for the app, data storage, and the database file.
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)  # Make sure data folder exists

DB_PATH = DATA_DIR / "escalateai.db"  # SQLite database path

# Load environment variables from a .env file for sensitive info like email credentials
load_dotenv()
IMAP_USER = os.getenv("EMAIL_USER")  # Email address used for login to IMAP server
IMAP_PASS = os.getenv("EMAIL_PASS")  # Password for the email
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")  # Default to Gmail IMAP server

# Only process emails from authorized senders (your escalation email)
AUTHORIZED_EMAILS = ["seservices.schneider@se.com"]


# ----------------------- Logging Setup -----------------------
# Create logs directory if it doesn't exist and configure logging
logfile = Path("logs")
logfile.mkdir(exist_ok=True)

logging.basicConfig(
    filename=logfile / "escalateai.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# This logging will help trace the program’s behavior and capture errors.


# ----------------------- Rule-based Sentiment Analysis -----------------------
# List of negative keywords/phrases typically indicating customer dissatisfaction
NEG_WORDS = [
    r"\b(delay|issue|failure|dissatisfaction|unacceptable|complaint|escalation|critical|risk|faulty|broken|error|problem|defect|bad|slow|down|crash|disconnect|poor|frustrated|angry|disappointed|not working|lost|missing|unresponsive|confused|trouble|inefficient|late|wrong|expired|refund|cancel|broken|unable|incomplete|fail|bug|glitch|overcharge|mismatch|conflict|inaccurate|delay|damaged|refund|missing|unavailable|denied|wrongful|halt|lag|freeze|stuck|unfair|violation|complain|angry|unhappy|bad experience|hate|sucks|terrible)\b"
]

def rule_sent(text: str) -> str:
    """
    Simple rule-based sentiment analysis.
    Scans the text for any negative keywords.
    Returns 'Negative' if any found, otherwise 'Positive'.
    """
    return "Negative" if any(re.search(p, text, re.I) for p in NEG_WORDS) else "Positive"


# ----------------------- Transformer-based Sentiment Analysis -----------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    """
    Loads a pre-trained Hugging Face transformer model for sentiment analysis.
    Caches the model so it only loads once per Streamlit session.
    """
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Instantiate model on app load
sentiment_analyzer = load_sentiment_model()

def transformer_sent(text: str) -> str:
    """
    Uses transformer model to classify sentiment.
    Truncates input to 512 tokens max (model limit).
    Returns 'Negative' or 'Positive' based on model output.
    """
    result = sentiment_analyzer(text[:512])[0]
    return "Negative" if result['label'] == 'NEGATIVE' else "Positive"


# ----------------------- Combined Issue Analysis -----------------------
def analyze_issue(text: str) -> Tuple[str, str, str, bool]:
    """
    Performs full analysis on the issue text:
    - rule_sentiment: based on keywords
    - transformer_sentiment: ML-based classification
    - urgency: 'High' if urgent keywords present, else 'Low'
    - escalate: boolean, true if transformer sentiment negative & urgency high

    Returns a tuple: (rule_sentiment, transformer_sentiment, urgency, escalate)
    """
    rb_sentiment = rule_sent(text)
    tf_sentiment = transformer_sent(text)
    urgency = "High" if any(k in text.lower() for k in ["urgent", "immediate", "critical", "asap"]) else "Low"
    escalate = tf_sentiment == "Negative" and urgency == "High"
    return rb_sentiment, tf_sentiment, urgency, escalate


# ----------------------- Email Alert Notification -----------------------
def send_alert_email(issue_summary):
    """
    Sends an alert email for high-risk escalations.
    Uses SMTP with SSL (configured for Gmail SMTP).
    """
    try:
        sender = os.getenv("EMAIL_USER")
        password = os.getenv("EMAIL_PASS")
        receiver = os.getenv("ALERT_RECEIVER") or sender
        msg = MIMEText(issue_summary)
        msg["Subject"] = "🚨 High-Risk Escalation Detected"
        msg["From"] = sender
        msg["To"] = receiver

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())

        logging.info(f"📧 Alert email sent to {receiver}")
    except Exception as e:
        logging.error(f"❌ Failed to send alert email: {e}")


# ----------------------- Insert Escalation into Database -----------------------
def insert_escalation(data: dict):
    """
    Inserts a new escalation record into SQLite DB.
    Creates table if not exists.
    Sends alert if high-risk escalation.
    """
    data["id"] = f"SESICE-{str(uuid.uuid4())[:8].upper()}"
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS escalations (
                id TEXT PRIMARY KEY,
                customer TEXT,
                issue TEXT,
                date_reported TEXT,
                rule_sentiment TEXT,
                transformer_sentiment TEXT,
                urgency TEXT,
                escalated INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cols = ",".join(data.keys())
        vals = tuple(data.values())
        placeholders = ",".join(["?"] * len(data))
        conn.execute(f"INSERT INTO escalations ({cols}) VALUES ({placeholders})", vals)
        conn.commit()

    # Send alert for high-risk
    if data["transformer_sentiment"] == "Negative" and data["urgency"] == "High":
        send_alert_email(f"Customer: {data['customer']}\nIssue: {data['issue'][:200]}")


# ----------------------- Parse Emails from IMAP -----------------------
def parse_emails():
    """
    Connects to the IMAP server and fetches unseen emails from authorized senders.
    Extracts email body, cleans HTML, runs analysis, and logs escalations.
    """
    parsed_count = 0
    try:
        with IMAPClient(IMAP_SERVER) as client:
            client.login(IMAP_USER, IMAP_PASS)
            client.select_folder("INBOX", readonly=True)
            messages = client.search(["UNSEEN"])
            for uid, msg_data in client.fetch(messages, ["RFC822"]).items():
                msg = email.message_from_bytes(msg_data[b"RFC822"])
                from_email = email.utils.parseaddr(msg.get("From"))[1].lower()

                # Only process authorized emails
                if from_email not in AUTHORIZED_EMAILS:
                    continue

                date = msg.get("Date") or datetime.utcnow().isoformat()

                # Extract plain text from email (handle multipart)
                if msg.is_multipart():
                    body = next((part.get_payload(decode=True).decode(errors='ignore')
                                 for part in msg.walk() if part.get_content_type() == "text/plain"), "")
                else:
                    body = msg.get_payload(decode=True).decode(errors='ignore')

                # Remove any HTML tags if present
                soup = BeautifulSoup(body, "html.parser")
                clean_body = soup.get_text()

                # Analyze sentiment, urgency, escalation
                rb_sent, tf_sent, urgency, escalate = analyze_issue(clean_body)

                # Insert to DB
                insert_escalation({
                    "customer": from_email,
                    "issue": clean_body[:500],  # limit issue length
                    "date_reported": date,
                    "rule_sentiment": rb_sent,
                    "transformer_sentiment": tf_sent,
                    "urgency": urgency,
                    "escalated": int(escalate)
                })

                parsed_count += 1
                logging.info(f"🔔 Logged escalation from {from_email} with urgency={urgency}, rule_sent={rb_sent}, transformer_sent={tf_sent}.")

        if parsed_count:
            st.success(f"✅ Parsed and logged {parsed_count} new emails.")
        else:
            st.info("No new authorized emails found.")

    except Exception as e:
        logging.error(f"❌ Error parsing emails: {e}")
        st.error(f"Error parsing emails: {e}")


# ----------------------- Scheduler to fetch emails every minute -----------------------
def schedule_email_fetch():
    """
    Starts a background scheduler that runs parse_emails() every 1 minute.
    """
    scheduler = BackgroundScheduler()
    scheduler.add_job(parse_emails, 'interval', minutes=1, id='email_job', replace_existing=True)
    scheduler.start()
    logging.info("✅ Email scheduler started (every 1 minute)")
    return scheduler

# Initialize scheduler and track its status in Streamlit session state
if 'email_scheduler' not in st.session_state:
    st.session_state['email_scheduler'] = schedule_email_fetch()
    st.session_state['scheduler_status'] = True


# ----------------------- Streamlit Sidebar Controls -----------------------
# Immediate manual email parsing trigger button
st.sidebar.button("📩 Parse Inbox Emails (Now)", on_click=parse_emails)

st.sidebar.markdown("---")
st.sidebar.subheader("⏱ Scheduler Control")

# Pause the scheduler job if running
if st.sidebar.button("⏸ Pause Scheduler"):
    if st.session_state.get("scheduler_status"):
        st.session_state['email_scheduler'].shutdown(wait=False)
        st.session_state['scheduler_status'] = False
        logging.info("⏸ Scheduler paused by user.")
        st.sidebar.info("Scheduler paused.")

# Resume the scheduler job if paused
if st.sidebar.button("▶️ Resume Scheduler"):
    if not st.session_state.get("scheduler_status"):
        st.session_state['email_scheduler'] = schedule_email_fetch()
        st.session_state['scheduler_status'] = True
        logging.info("▶️ Scheduler resumed by user.")
        st.sidebar.success("Scheduler resumed.")

st.sidebar.markdown("---")

# Manual escalation entry form
st.sidebar.header("📝 Manual Escalation Entry")
with st.sidebar.form("manual_entry"):
    cust = st.text_input("Customer")
    issue = st.text_area("Issue")
    date = st.date_input("Date Reported", datetime.utcnow().date())
    submitted = st.form_submit_button("Submit")
    if submitted and issue:
        rb_sent, tf_sent, urgency, escalate = analyze_issue(issue)
        insert_escalation({
            "customer": cust or "Unknown",
            "issue": issue,
            "date_reported": str(date),
            "rule_sentiment": rb_sent,
            "transformer_sentiment": tf_sent,
            "urgency": urgency,
            "escalated": int(escalate)
        })
        logging.info(f"📝 Manual entry logged: customer={cust}, urgency={urgency}, rule_sentiment={rb_sent}, transformer_sentiment={tf_sent}.")
        st.sidebar.success("✅ Escalation logged.")

# Excel upload for bulk escalations
st.sidebar.header("📤 Upload Escalation File")
uploaded_file = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
if uploaded_file:
    df_upload = pd.read_excel(uploaded_file)
    for _, row in df_upload.iterrows():
        issue = str(row.get("Issue", ""))
        rb_sent, tf_sent, urgency, escalate = analyze_issue(issue)
        insert_escalation({
            "customer": row.get("Customer", "Unknown"),
            "issue": issue[:500],
            "date_reported": str(row.get("Date Reported", datetime.utcnow().date())),
            "rule_sentiment": rb_sent,
            "transformer_sentiment": tf_sent,
            "urgency": urgency,
            "escalated": int(escalate)
        })
        logging.info(f"📥 Uploaded escalation from Excel logged for {row.get('Customer', 'Unknown')}.")
    st.sidebar.success("✅ File processed and entries logged.")


# ----------------------- Main Dashboard -----------------------
st.title("📌 Escalation Dashboard")

# Load data from SQLite DB
with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql("SELECT * FROM escalations ORDER BY datetime(created_at) DESC", conn)

if df.empty:
    st.info("No escalations logged yet.")
else:
    # Filters UI - 5 columns for filters
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        filter_choice = st.radio("🔎 Escalation Status:", ["All", "Escalated Only", "Non-Escalated"], horizontal=False)
    with col2:
        urgency_filter = st.selectbox("⚠️ Urgency:", ["All"] + sorted(df["urgency"].unique()))
    with col3:
        rule_sent_filter = st.selectbox("📝 Rule Sentiment:", ["All"] + sorted(df["rule_sentiment"].unique()))
    with col4:
        tf_sent_filter = st.selectbox("🧠 Transformer Sentiment:", ["All"] + sorted(df["transformer_sentiment"].unique()))
    with col5:
        date_range = st.date_input("📅 Date Range:", [])

    # Apply filters
    if filter_choice == "Escalated Only":
        df = df[df["escalated"] == 1]
    elif filter_choice == "Non-Escalated":
        df = df[df["escalated"] == 0]

    if urgency_filter != "All":
        df = df[df["urgency"] == urgency_filter]

    if rule_sent_filter != "All":
        df = df[df["rule_sentiment"] == rule_sent_filter]

    if tf_sent_filter != "All":
        df = df[df["transformer_sentiment"] == tf_sent_filter]

    if date_range and len(date_range) == 2:
        start, end = map(str, date_range)
        df = df[(df["date_reported"] >= start) & (df["date_reported"] <= end)]

    # Display escalation entries
    for idx, row in df.iterrows():
        with st.expander(f"{row['id']} - {row['customer']} (Rule: {row['rule_sentiment']} / Transformer: {row['transformer_sentiment']} / {row['urgency']})"):
            st.markdown(f"**Issue:** {row['issue']}")
            st.markdown(f"**Escalated:** {'Yes' if row['escalated'] else 'No'}")
            st.markdown(f"**Date:** {row['date_reported']}")

    # CSV download of filtered data
    st.download_button(
        "📥 Download Filtered Escalations (CSV)",
        df.to_csv(index
```
