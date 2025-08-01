# ==============================================================
# EscalateAI – Escalation Management Tool with Email Parsing
# --------------------------------------------------------------
# • Parses emails from seservices.schneider@se.com (IMAP)
# • Logs escalations directly into database
# • Predicts sentiment, urgency, and risk in real-time
# • Streamlit dashboard for escalation tracking
# --------------------------------------------------------------
# Author: Naveen Gandham • v1.3.1 • August 2025
# ==============================================================

"""Setup
pip install streamlit pandas openpyxl python-dotenv scikit-learn joblib xlsxwriter imapclient beautifulsoup4
# Optional for better NLP
pip install transformers torch --index-url https://download.pytorch.org/whl/cpu
"""

# ========== Standard Lib ==========
import os, re, sqlite3, smtplib, uuid, io, email
from email.mime.text import MIMEText
from datetime import datetime
from pathlib import Path
from typing import Tuple

# ========== Third‑party ==========
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imapclient import IMAPClient
from bs4 import BeautifulSoup

# Optional sentiment NLP ------------------------------------------------------
try:
    from transformers import pipeline as hf_pipeline
    import torch
    HAS_NLP = True
except:
    HAS_NLP = False

# ========== Paths & ENV ==========
APP_DIR   = Path(__file__).resolve().parent
DATA_DIR  = APP_DIR / "data";  DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR = APP_DIR / "models"; MODEL_DIR.mkdir(exist_ok=True)
DB_PATH   = DATA_DIR / "escalateai.db"

load_dotenv()
IMAP_USER     = os.getenv("EMAIL_USER")  # seservices.schneider@se.com
IMAP_PASS     = os.getenv("EMAIL_PASS")
IMAP_SERVER   = os.getenv("EMAIL_SERVER", "imap.gmail.com")

AUTHORIZED_EMAILS = ["seservices.schneider@se.com"]

# ========== Sentiment Model ==========
@st.cache_resource(show_spinner=False)
def load_sentiment():
    if not HAS_NLP:
        return None
    try:
        return hf_pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    except:
        return None

sent_model = load_sentiment()
NEG_WORDS = [r"\b(delay|issue|failure|dissatisfaction|unacceptable|complaint|escalation|critical|risk|faulty)\b"]

def rule_sent(text: str) -> str:
    return "Negative" if any(re.search(p, text, re.I) for p in NEG_WORDS) else "Positive"

def analyze_issue(text: str) -> Tuple[str, str, bool]:
    if sent_model:
        lbl = sent_model(text[:512])[0]["label"].lower()
        sentiment = "Negative" if lbl == "negative" else "Positive"
    else:
        sentiment = rule_sent(text)
    urgency = "High" if any(k in text.lower() for k in ["urgent", "immediate", "critical", "asap"]) else "Low"
    return sentiment, urgency, sentiment == "Negative" and urgency == "High"

# ========== DB Setup ==========
def insert_escalation(data: dict):
    data["id"] = f"SESICE-{str(uuid.uuid4())[:8].upper()}"
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS escalations (
                id TEXT PRIMARY KEY,
                customer TEXT,
                issue TEXT,
                date_reported TEXT,
                sentiment TEXT,
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

# ========== Email Parsing ==========
def parse_emails():
    parsed_count = 0
    with IMAPClient(IMAP_SERVER) as client:
        client.login(IMAP_USER, IMAP_PASS)
        client.select_folder("INBOX", readonly=True)
        messages = client.search(["UNSEEN"])
        for uid, msg_data in client.fetch(messages, ["RFC822"]).items():
            msg = email.message_from_bytes(msg_data[b"RFC822"])
            from_email = email.utils.parseaddr(msg.get("From"))[1].lower()
            if from_email not in AUTHORIZED_EMAILS:
                continue
            subject = msg.get("Subject", "(No Subject)")
            date = msg.get("Date") or datetime.utcnow().isoformat()
            if msg.is_multipart():
                body = next((part.get_payload(decode=True).decode(errors='ignore') for part in msg.walk() if part.get_content_type() == "text/plain"), "")
            else:
                body = msg.get_payload(decode=True).decode(errors='ignore')
            soup = BeautifulSoup(body, "html.parser")
            clean_body = soup.get_text()
            sentiment, urgency, escalate = analyze_issue(clean_body)
            insert_escalation({
                "customer": from_email,
                "issue": clean_body[:500],
                "date_reported": date,
                "sentiment": sentiment,
                "urgency": urgency,
                "escalated": int(escalate)
            })
            parsed_count += 1
    if parsed_count:
        st.success(f"✅ Parsed and logged {parsed_count} new emails.")
    else:
        st.info("No new authorized emails found.")

# ========== Sidebar Trigger ==========
st.sidebar.button("📩 Parse Inbox Emails", on_click=parse_emails)

# ========== Dashboard View ==========
st.title("📌 Escalation Dashboard")
with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql("SELECT * FROM escalations ORDER BY datetime(created_at) DESC", conn)
if df.empty:
    st.info("No escalations logged yet.")
else:
    for idx, row in df.iterrows():
        with st.expander(f"{row['id']} - {row['customer']} ({row['sentiment']}/{row['urgency']})"):
            st.markdown(f"**Issue:** {row['issue']}")
            st.markdown(f"**Escalated:** {'Yes' if row['escalated'] else 'No'}")
            st.markdown(f"**Date:** {row['date_reported']}")
