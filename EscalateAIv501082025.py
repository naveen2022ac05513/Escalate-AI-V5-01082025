# ==============================================================
# EscalateAI – Escalation Management Tool with Email Parsing
# --------------------------------------------------------------
# • Parses emails from inbox configured in .env
# • Logs escalations directly into database every minute
# • Predicts sentiment (rule-based + transformer), urgency, and risk in real-time
# • Streamlit dashboard for escalation tracking
# • Supports manual entry, Excel upload, and CSV download
# • Logs scheduler activity and allows pause/resume controls
# • Notifies when new escalation is added
# • Filters by urgency, sentiment, date, escalation status
# --------------------------------------------------------------
# Author: Naveen Gandham • v1.5.0 • August 2025
# ==============================================================

import os, re, sqlite3, uuid, email
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

from transformers import pipeline

# ----------------------- Paths & ENV -----------------------
APP_DIR   = Path(__file__).resolve().parent
DATA_DIR  = APP_DIR / "data"; DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR = APP_DIR / "models"; MODEL_DIR.mkdir(exist_ok=True)
DB_PATH   = DATA_DIR / "escalateai.db"

load_dotenv()
IMAP_USER   = os.getenv("EMAIL_USER")
IMAP_PASS   = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
ALERT_RECEIVER = os.getenv("ALERT_RECEIVER", IMAP_USER)

# ----------------------- Logging -----------------------
logfile = Path("logs"); logfile.mkdir(exist_ok=True)
logging.basicConfig(
    filename=logfile / "escalateai.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------- Sentiment Models -----------------------
NEG_WORDS = [r"\b(delay|issue|failure|dissatisfaction|unacceptable|complaint|escalation|critical|risk|faulty|bad|poor|slow|crash|urgent|asap|immediately)\b"]

@st.cache_resource(show_spinner=False)
def load_transformer_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

transformer_model = load_transformer_model()

def rule_sent(text: str) -> str:
    return "Negative" if any(re.search(p, text, re.I) for p in NEG_WORDS) else "Positive"

def transformer_sent(text: str) -> str:
    try:
        result = transformer_model(text[:512])[0]
        return "Negative" if result['label'].upper() == "NEGATIVE" else "Positive"
    except:
        return "Positive"

def analyze_issue(text: str) -> Tuple[str, str, str, bool]:
    rule = rule_sent(text)
    transformer = transformer_sent(text)
    urgency = "High" if any(k in text.lower() for k in ["urgent", "immediate", "critical", "asap"]) else "Low"
    escalate = (rule == "Negative" or transformer == "Negative") and urgency == "High"
    return rule, transformer, urgency, escalate

# ----------------------- Notification -----------------------
def send_alert_email(issue_summary):
    try:
        msg = MIMEText(issue_summary)
        msg["Subject"] = "🚨 High-Risk Escalation Detected"
        msg["From"] = IMAP_USER
        msg["To"] = ALERT_RECEIVER
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(IMAP_USER, IMAP_PASS)
            server.sendmail(IMAP_USER, ALERT_RECEIVER, msg.as_string())
        logging.info(f"📧 Alert email sent to {ALERT_RECEIVER}")
    except Exception as e:
        logging.error(f"❌ Failed to send alert: {e}")

# ----------------------- Insert -----------------------
def insert_escalation(data: dict):
    data["escalation_id"] = f"SESICE-{str(uuid.uuid4())[:8].upper()}"
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS escalations (
                escalation_id TEXT PRIMARY KEY,
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
    if data["rule_sentiment"] == "Negative" or data["transformer_sentiment"] == "Negative":
        if data["urgency"] == "High":
            send_alert_email(f"Escalation ID: {data['escalation_id']}\nCustomer: {data['customer']}\nIssue: {data['issue'][:200]}")

def parse_emails():
    parsed_count = 0
    with IMAPClient(IMAP_SERVER) as client:
        client.login(IMAP_USER, IMAP_PASS)
        logging.info(f"🔐 Logged in as {IMAP_USER}")
        
        folders = client.list_folders()
        logging.info(f"📂 Available folders: {[f[2] for f in folders]}")
        
        client.select_folder("INBOX", readonly=True)
        messages = client.search(["UNSEEN"])
        logging.info(f"📨 Unseen messages: {messages}")
        
        for uid, msg_data in client.fetch(messages, ["RFC822"]).items():
            msg = email.message_from_bytes(msg_data[b"RFC822"])
            from_email = email.utils.parseaddr(msg.get("From"))[1].lower()
            date = msg.get("Date") or datetime.utcnow().isoformat()
            
            # Extract body
            if msg.is_multipart():
                body = next((part.get_payload(decode=True).decode(errors='ignore') 
                             for part in msg.walk() 
                             if part.get_content_type() == "text/plain"), "")
            else:
                body = msg.get_payload(decode=True).decode(errors='ignore')
            
            soup = BeautifulSoup(body, "html.parser")
            clean_body = soup.get_text()

            # Sentiment & Escalation
            rule, transformer, urgency, escalate = analyze_issue(clean_body)
            insert_escalation({
                "customer": from_email,
                "issue": clean_body[:500],
                "date_reported": date,
                "rule_sentiment": rule,
                "transformer_sentiment": transformer,
                "urgency": urgency,
                "escalated": int(escalate)
            })

            logging.info(f"✅ Processed: From={from_email}, Rule={rule}, Transformer={transformer}, Urgency={urgency}, Escalated={escalate}")
            parsed_count += 1

    if parsed_count:
        st.success(f"✅ Parsed and logged {parsed_count} new emails.")
    else:
        st.info("No new emails found.")

# ----------------------- Scheduler -----------------------
def schedule_email_fetch():
    scheduler = BackgroundScheduler()
    scheduler.add_job(parse_emails, 'interval', minutes=1, id='email_job', replace_existing=True)
    scheduler.start()
    logging.info("✅ Email scheduler started and running every 1 minute")

if 'email_scheduler' not in st.session_state:
    schedule_email_fetch()
    st.session_state['scheduler_status'] = True
    st.session_state['email_scheduler'] = True


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
