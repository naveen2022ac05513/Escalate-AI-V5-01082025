import os
import sqlite3
import pandas as pd
import streamlit as st
from imapclient import IMAPClient
import email
from bs4 import BeautifulSoup
from datetime import datetime
from transformers import pipeline
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
from pathlib import Path

# Load env variables
load_dotenv()
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
ALERT_RECEIVER = os.getenv("ALERT_RECEIVER", EMAIL_USER)
SMTP_USER = os.getenv("SMTP_USER", EMAIL_USER)
SMTP_PASS = os.getenv("SMTP_PASS", EMAIL_PASS)

# Database file
DB_PATH = "escalations.db"

# Logging setup
logfile = Path("logs")
logfile.mkdir(exist_ok=True)
logging.basicConfig(
    filename=logfile / "escalateai.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Create DB table if not exists
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS escalations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer TEXT,
    issue TEXT,
    sentiment TEXT,
    urgency TEXT,
    date_reported TEXT,
    escalated INTEGER,
    created_at TEXT,
    rule_sentiment TEXT,
    transformer_sentiment TEXT
);
"""

# Initialize DB
with sqlite3.connect(DB_PATH) as conn:
    conn.execute(CREATE_TABLE_SQL)

# Load sentiment analysis pipeline (HuggingFace transformers)
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    import torch  # Make sure torch installed
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_analyzer = load_sentiment_model()

# Simple rule-based negative words list
NEGATIVE_WORDS = set([
    "not working", "issue", "error", "fail", "problem", "delay", "unable", "complain",
    "bad", "poor", "broken", "late", "slow", "crash", "downtime", "urgent", "critical"
])

# Insert escalation into DB
def insert_escalation(data: dict):
    with sqlite3.connect(DB_PATH) as conn:
        try:
            cols = ", ".join(data.keys())
            placeholders = ", ".join("?" for _ in data)
            vals = tuple(data.values())
            conn.execute(f"INSERT INTO escalations ({cols}) VALUES ({placeholders})", vals)
            conn.commit()
            logging.info(f"Inserted escalation: {data['customer']}, issue: {data['issue'][:30]}")
            # Send alert if high risk
            if data.get("sentiment") == "NEGATIVE" and data.get("urgency") == "High":
                send_alert_email(data)
        except Exception as e:
            logging.error(f"Insert escalation error: {e}")

# Send email alert
def send_alert_email(data):
    try:
        issue_summary = (
            f"Customer: {data['customer']}\n"
            f"Issue: {data['issue']}\n"
            f"Urgency: {data['urgency']}\n"
            f"Sentiment: {data['sentiment']}"
        )
        msg = MIMEText(issue_summary)
        msg["Subject"] = "🚨 High-Risk Escalation Detected"
        msg["From"] = SMTP_USER
        msg["To"] = ALERT_RECEIVER

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, ALERT_RECEIVER, msg.as_string())
        logging.info(f"Alert email sent to {ALERT_RECEIVER}")
    except Exception as e:
        logging.error(f"Failed to send alert email: {e}")

# Analyze text for sentiment & urgency
def analyze_issue(issue_text):
    # Rule-based negative detection
    rule_neg = any(kw in issue_text.lower() for kw in NEGATIVE_WORDS)
    rule_sentiment = "NEGATIVE" if rule_neg else "POSITIVE"

    # Transformer sentiment
    try:
        result = sentiment_analyzer(issue_text[:512])[0]  # truncate long text
        transformer_sentiment = "NEGATIVE" if result['label'].upper() == "NEGATIVE" else "POSITIVE"
    except Exception as e:
        logging.error(f"Sentiment analysis error: {e}")
        transformer_sentiment = "POSITIVE"

    # Urgency detection by keywords
    urgency = "High" if any(w in issue_text.lower() for w in ["urgent", "immediately", "asap", "critical", "fail"]) else "Low"

    # Decide escalated if either sentiment negative or urgency high
    escalated = 1 if rule_sentiment == "NEGATIVE" or transformer_sentiment == "NEGATIVE" or urgency == "High" else 0

    return rule_sentiment, transformer_sentiment, urgency, escalated

# Fetch unread emails from Gmail via IMAP
def fetch_unread_emails():
    new_cases = 0
    try:
        with IMAPClient(EMAIL_SERVER) as client:
            client.login(EMAIL_USER, EMAIL_PASS)
            client.select_folder('INBOX')
            messages = client.search(['UNSEEN'])
            if not messages:
                return 0

            response = client.fetch(messages, ['RFC822'])
            for uid, data in response.items():
                msg = email.message_from_bytes(data[b'RFC822'])
                subject = msg.get('Subject', '')
                from_ = msg.get('From', '')
                date_ = msg.get('Date', '')
                body = ""

                if msg.is_multipart():
                    for part in msg.walk():
                        ctype = part.get_content_type()
                        cdispo = str(part.get('Content-Disposition'))

                        if ctype == 'text/plain' and 'attachment' not in cdispo:
                            body = part.get_payload(decode=True).decode(errors='ignore')
                            break
                else:
                    body = msg.get_payload(decode=True).decode(errors='ignore')

                # Extract customer (from email)
                customer = from_
                issue = body.strip()[:1000]

                # Analyze issue text
                rule_sentiment, transformer_sentiment, urgency, escalated = analyze_issue(issue)

                # Insert to DB
                insert_escalation({
                    "customer": customer,
                    "issue": issue,
                    "sentiment": transformer_sentiment,
                    "urgency": urgency,
                    "date_reported": date_ if date_ else datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "escalated": escalated,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "rule_sentiment": rule_sentiment,
                    "transformer_sentiment": transformer_sentiment
                })

                new_cases += 1
            logging.info(f"Fetched {new_cases} new emails.")
    except Exception as e:
        logging.error(f"Email fetch error: {e}")
    return new_cases

# Scheduler setup for periodic email fetch
scheduler = BackgroundScheduler()
scheduler.start()

# Control flags for pause/resume
if "fetch_paused" not in st.session_state:
    st.session_state.fetch_paused = False

def scheduled_fetch():
    if not st.session_state.fetch_paused:
        count = fetch_unread_emails()
        if count > 0:
            st.experimental_rerun()

scheduler.add_job(scheduled_fetch, "interval", seconds=60)

# Streamlit UI
st.title("🚀 EscalateAI - Escalation Management")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    if st.button("Pause Email Fetching"):
        st.session_state.fetch_paused = True
        st.success("Email fetching paused.")
    if st.button("Resume Email Fetching"):
        st.session_state.fetch_paused = False
        st.success("Email fetching resumed.")

    st.markdown("---")

    st.header("Manual Entry")
    manual_customer = st.text_input("Customer Email")
    manual_issue = st.text_area("Issue Description")
    if st.button("Add Escalation Manually"):
        if manual_customer.strip() and manual_issue.strip():
            rs, ts, urgency, escalated = analyze_issue(manual_issue)
            insert_escalation({
                "customer": manual_customer,
                "issue": manual_issue,
                "sentiment": ts,
                "urgency": urgency,
                "date_reported": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "escalated": escalated,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rule_sentiment": rs,
                "transformer_sentiment": ts
            })
            st.success("Escalation added!")
        else:
            st.error("Please enter both customer email and issue.")

    st.markdown("---")

    st.header("Upload Excel")
    uploaded_file = st.file_uploader("Upload Excel with columns: customer, issue, date_reported (optional)", type=["xlsx"])
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            for idx, row in df.iterrows():
                cust = row.get("customer", "Unknown")
                issue = str(row.get("issue", ""))
                date_reported = row.get("date_reported", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                rs, ts, urgency, escalated = analyze_issue(issue)
                insert_escalation({
                    "customer": cust,
                    "issue": issue,
                    "sentiment": ts,
                    "urgency": urgency,
                    "date_reported": date_reported,
                    "escalated": escalated,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "rule_sentiment": rs,
                    "transformer_sentiment": ts
                })
            st.success("Escalations imported successfully!")
        except Exception as e:
            st.error(f"Failed to process file: {e}")

# Load escalations for dashboard
with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql("SELECT * FROM escalations ORDER BY datetime(created_at) DESC", conn)

# Filters
st.header("Escalations Dashboard")

filter_choice = st.radio("Escalation Status:", ["All", "Escalated Only", "Non-Escalated"])
urgency_filter = st.selectbox("Urgency:", ["All"] + sorted(df["urgency"].unique()) if not df.empty else [])
sentiment_filter = st.selectbox("Sentiment:", ["All"] + sorted(df["sentiment"].unique()) if not df.empty else [])
date_range = st.date_input("Date Range:", [])

if filter_choice == "Escalated Only":
    df = df[df["escalated"] == 1]
elif filter_choice == "Non-Escalated":
    df = df[df["escalated"] == 0]

if urgency_filter != "All":
    df = df[df["urgency"] == urgency_filter]

if sentiment_filter != "All":
    df = df[df["sentiment"] == sentiment_filter]

if date_range and len(date_range) == 2:
    start_date = date_range[0].strftime("%Y-%m-%d")
    end_date = date_range[1].strftime("%Y-%m-%d")
    df = df[(df["date_reported"] >= start_date) & (df["date_reported"] <= end_date)]

if df.empty:
    st.info("No escalations found.")
else:
    for _, row in df.iterrows():
        with st.expander(f"{row['id']} - {row['customer']} ({row['sentiment']}/{row['urgency']})"):
            st.markdown(f"**Issue:** {row['issue']}")
            st.markdown(f"**Escalated:** {'Yes' if row['escalated'] else 'No'}")
            st.markdown(f"**Date:** {row['date_reported']}")

    st.download_button(
        "📥 Download Filtered Escalations (CSV)",
        df.to_csv(index=False).encode(),
        file_name="escalations_filtered.csv",
        mime="text/csv"
    )
