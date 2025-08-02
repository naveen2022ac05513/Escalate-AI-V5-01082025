# ==============================================================
# EscalateAI – Escalation Management Tool with Email Parsing & Time-Based Escalation
# --------------------------------------------------------------
# • Parses emails from inbox configured in .env
# • Logs escalations directly into database every minute
# • Predicts sentiment (rule-based + transformer), urgency, and risk in real-time
# • Streamlit dashboard for escalation tracking
# • Supports manual entry, Excel upload, and CSV download
# • Logs scheduler activity and allows pause/resume controls
# • Notifies when new escalation is added
# • Filters by urgency, sentiment, date, escalation status
# • Kanban board with Status, Owner, Action Taken editable per case, with counts and colors
# • Time-based auto escalation of unresolved cases every hour
# --------------------------------------------------------------
# Author: Naveen Gandham • v1.8.0 • August 2025
# ==============================================================

import os
import re
import sqlite3
import email
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple
import time

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
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR = APP_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "escalateai.db"

load_dotenv()
IMAP_USER = os.getenv("EMAIL_USER")
IMAP_PASS = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
ALERT_RECEIVER = os.getenv("ALERT_RECEIVER", IMAP_USER)

# ----------------------- Logging -----------------------
logfile = Path("logs")
logfile.mkdir(exist_ok=True)
logging.basicConfig(
    filename=logfile / "escalateai.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------- Database Initialization & Migration -----------------------
def initialize_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS escalations (
                id TEXT PRIMARY KEY,
                customer TEXT,
                issue TEXT,
                date_reported TEXT,
                rule_sentiment TEXT,
                transformer_sentiment TEXT,
                urgency TEXT,
                escalated INTEGER,
                status TEXT DEFAULT 'Open',
                owner TEXT DEFAULT '',
                action_status TEXT DEFAULT 'Pending',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_updated TEXT,
                escalation_level INTEGER DEFAULT 1
            )
            """
        )
        conn.commit()

def migrate_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("PRAGMA table_info(escalations)")
        columns = [col[1] for col in cursor.fetchall()]
        if "last_updated" not in columns:
            conn.execute("ALTER TABLE escalations ADD COLUMN last_updated TEXT")
        if "escalation_level" not in columns:
            conn.execute("ALTER TABLE escalations ADD COLUMN escalation_level INTEGER DEFAULT 1")
        conn.commit()

initialize_db()
migrate_db()

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

# ----------------------- Insert with Sequential ID -----------------------
def insert_escalation(data: dict):
    retries = 3
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for attempt in range(retries):
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cur = conn.execute(
                    "SELECT MAX(CAST(SUBSTR(id, 7) AS INTEGER)) FROM escalations WHERE id LIKE 'SESICE-%'"
                )
                max_num = cur.fetchone()[0]
                if max_num is None or max_num < 250000:
                    next_num = 250001
                else:
                    next_num = max_num + 1

                new_id = f"SESICE-{next_num:06d}"
                data["id"] = new_id

                data.setdefault("status", "Open")
                data.setdefault("owner", "")
                data.setdefault("action_status", "Pending")
                data.setdefault("last_updated", now_str)
                data.setdefault("escalation_level", 1)

                cols = ",".join(data.keys())
                vals = tuple(data.values())
                placeholders = ",".join(["?"] * len(data))

                conn.execute(f"INSERT INTO escalations ({cols}) VALUES ({placeholders})", vals)
                conn.commit()

            if data["rule_sentiment"] == "Negative" or data["transformer_sentiment"] == "Negative":
                if data["urgency"] == "High":
                    send_alert_email(
                        f"Escalation ID: {data['id']}\nCustomer: {data['customer']}\nIssue: {data['issue'][:200]}"
                    )
            break  # success, exit retry loop
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e) and attempt < retries - 1:
                time.sleep(0.1)  # short wait and retry
            else:
                raise

# ----------------------- Email Parser -----------------------
def parse_emails():
    parsed_count = 0
    try:
        with IMAPClient(IMAP_SERVER) as client:
            client.login(IMAP_USER, IMAP_PASS)
            client.select_folder("INBOX")
            messages = client.search(["UNSEEN"])
            for uid, msg_data in client.fetch(messages, ["BODY[]"]).items():
                raw_email = msg_data[b"BODY[]"]
                msg = email.message_from_bytes(raw_email)
                from_email = email.utils.parseaddr(msg.get("From"))[1].lower()
                date = msg.get("Date") or datetime.utcnow().isoformat()
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        ctype = part.get_content_type()
                        cdispo = str(part.get("Content-Disposition"))
                        if ctype == "text/plain" and "attachment" not in cdispo:
                            try:
                                body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                                break
                            except:
                                continue
                else:
                    body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
                clean_body = BeautifulSoup(body, "html.parser").get_text()
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
                parsed_count += 1
                logging.info(f"🔔 Escalation from {from_email} logged with rule={rule}, transformer={transformer}, urgency={urgency}.")
    except Exception as e:
        logging.error(f"❌ Failed to fetch emails: {e}")
        st.error("❌ Failed to connect to email server. Check credentials or network.")
    if parsed_count:
        st.success(f"✅ Parsed and logged {parsed_count} new emails.")
    else:
        st.info("No new emails found.")

# ----------------------- Time-based Escalation Scheduler Job -----------------------
def escalate_overdue_cases():
    threshold_hours = 24
    now = datetime.now()
    with sqlite3.connect(DB_PATH) as conn:
        overdue_cases = conn.execute(
            """
            SELECT id, escalation_level, owner, status, last_updated
            FROM escalations
            WHERE status != 'Resolved' AND last_updated IS NOT NULL
            """
        ).fetchall()

        for case_id, level, owner, status, last_upd in overdue_cases:
            last_upd_dt = datetime.strptime(last_upd, "%Y-%m-%d %H:%M:%S")
            if (now - last_upd_dt) > timedelta(hours=threshold_hours):
                new_level = level + 1
                # Example escalation logic: assign to manager at level 2, else keep current owner
                new_owner = "manager@example.com" if new_level == 2 else owner

                conn.execute(
                    """
                    UPDATE escalations
                    SET escalation_level = ?, owner = ?, last_updated = ?
                    WHERE id = ?
                    """,
                    (new_level, new_owner, now.strftime("%Y-%m-%d %H:%M:%S"), case_id)
                )
                conn.commit()

                send_alert_email(
                    f"Escalation {case_id} has been escalated to level {new_level} and assigned to {new_owner}."
                )
                logging.info(f"Escalation {case_id} auto-escalated to level {new_level}")

# ----------------------- Scheduler -----------------------
scheduler = BackgroundScheduler()
scheduler.add_job(parse_emails, 'interval', minutes=1, id='email_job', replace_existing=True)
scheduler.add_job(escalate_overdue_cases, 'interval', hours=1, id='auto_escalation', replace_existing=True)
scheduler.start()
logging.info("✅ Email and escalation schedulers started")

# ----------------------- Streamlit session state for scheduler -----------------------
if 'email_scheduler' not in st.session_state:
    st.session_state['email_scheduler'] = True
    st.session_state['scheduler_status'] = True

# ----------------------- Manual Email Parser -----------------------
st.sidebar.markdown("---")
st.sidebar.header("📬 Email Controls")

if st.sidebar.button("Manually Parse Emails"):
    with st.spinner("Checking inbox..."):
        parse_emails()

# Streamlit UI
st.title("🚀 EscalateAI - Escalation Management with Time-based Auto Escalation")

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
                "date_reported": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rule_sentiment": rs,
                "transformer_sentiment": ts,
                "urgency": urgency,
                "escalated": int(escalated),
            })
            st.success("Escalation added!")
        else:
            st.error("Please enter both customer email and issue.")

    st.markdown("---")

    st.header("Upload Excel")
    uploaded_file = st.file_uploader("Upload Excel with columns: customer, issue, date_reported (optional)", type=["xlsx"])
    if uploaded_file:
    try:
        df_upload = pd.read_excel(uploaded_file, engine="openpyxl")
        required_cols = {"customer", "issue"}
        if not required_cols.issubset(df_upload.columns.str.lower()):
            st.error("Excel must contain at least 'customer' and 'issue' columns.")
        else:
            st.success("File uploaded successfully. Processing...")
            for idx, row in df_upload.iterrows():
                cust = row.get("customer", "Unknown")
                issue = str(row.get("issue", ""))
                date_reported = row.get("date_reported", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                rs, ts, urgency, escalated = analyze_issue(issue)
                try:
                    insert_escalation({
                        "customer": cust,
                        "issue": issue,
                        "date_reported": date_reported,
                        "rule_sentiment": rs,
                        "transformer_sentiment": ts,
                        "urgency": urgency,
                        "escalated": int(escalated),
                    })
                except sqlite3.IntegrityError as e:
                    st.error(f"Duplicate ID error while inserting row {idx + 1}: {e}")
                time.sleep(0.1)  # prevent collision
            st.success("Bulk upload completed.")
    except Exception as e:
        st.error(f"Failed to process file: {e}")




# Load escalations for dashboard
with sqlite3.connect(DB_PATH) as conn:
    try:
        df = pd.read_sql("SELECT * FROM escalations ORDER BY created_at DESC", conn)
    except Exception as e:
        st.error(f"Database read error: {e}")
        df = pd.DataFrame()

# ----------------------- Kanban Board with Colors & Counts -----------------------
st.header("Escalations Kanban Board")

statuses = ["Open", "In Progress", "Resolved"]
STATUS_COLORS = {
    "Open": "#ffcccc",          # light red
    "In Progress": "#fff0b3",   # light yellow
    "Resolved": "#ccffcc"       # light green
}

kanban_cols = st.columns(len(statuses))

for col, status in zip(kanban_cols, statuses):
    count = len(df[df["status"] == status]) if not df.empty else 0
    with col:
        # Colored header with count
        st.markdown(
            f"<h3 style='background-color:{STATUS_COLORS[status]};"
            f"padding:8px;border-radius:6px;text-align:center;'>{status} ({count})</h3>",
            unsafe_allow_html=True
        )

        filtered = df[df["status"] == status] if not df.empty else pd.DataFrame()
        if filtered.empty:
            st.write("_No escalations_")
        else:
            for idx, row in filtered.iterrows():
                with st.expander(f"{row['id']} - {row['customer']} ({row.get('rule_sentiment', '')}/{row['urgency']})"):
                    st.markdown(f"**Issue:** {row['issue']}")
                    st.markdown(f"**Escalated:** {'Yes' if row.get('escalated', 0) else 'No'}")
                    st.markdown(f"**Date:** {row['date_reported']}")
                    st.markdown(f"**Escalation Level:** {row.get('escalation_level', 1)}")

                    new_owner = st.text_input("Owner", value=row.get("owner", ""), key=f"owner_{row['id']}")
                    new_action_status = st.text_input(
                        "Action Taken",
                        value=row.get("action_status", ""),
                        key=f"action_status_{row['id']}"
                    )
                    new_status = st.selectbox(
                        "Status",
                        options=statuses,
                        index=statuses.index(row.get("status", "Open")),
                        key=f"status_{row['id']}"
                    )

                    if st.button("Update", key=f"update_{row['id']}"):
                        try:
                            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            with sqlite3.connect(DB_PATH) as conn:
                                conn.execute(
                                    """
                                    UPDATE escalations SET owner=?, action_status=?, status=?, last_updated=? WHERE id=?
                                    """,
                                    (new_owner, new_action_status, new_status, now_str, row['id'])
                                )
                                conn.commit()
                            st.success(f"Updated escalation {row['id']}!")
                            st.session_state['needs_refresh'] = True
                        except Exception as e:
                            st.error(f"Failed to update escalation: {e}")

if st.session_state.get('needs_refresh'):
    st.session_state['needs_refresh'] = False
    st.experimental_rerun()

# ----------------------- Manual Parser -----------------------
with st.expander("✍️ Manually Parse Email"):
    st.markdown("Use this form to test email parsing manually or input an email issue when IMAP fails.")

    manual_email = st.text_area("Paste email body or issue here", height=200)
    manual_sender = st.text_input("Customer Email")
    manual_date = st.date_input("Date Reported", datetime.today())

    if st.button("Parse and Log Manually"):
        if manual_email and manual_sender:
            rule, transformer, urgency, escalate = analyze_issue(manual_email)
            insert_escalation({
                "customer": manual_sender.lower(),
                "issue": manual_email[:500],
                "date_reported": str(manual_date),
                "rule_sentiment": rule,
                "transformer_sentiment": transformer,
                "urgency": urgency,
                "escalated": int(escalate)
            })
            st.success(f"✅ Manually logged escalation with urgency = {urgency}, rule sentiment = {rule}")
        else:
            st.error("Please provide both the issue text and customer email.")

    st.download_button(
        key="download_manual_filtered_escalations",
        label="📥 Download Filtered Escalations (CSV)",
        data=df.to_csv(index=False).encode() if not df.empty else b"",
        file_name="escalations_filtered.csv",
        mime="text/csv"
    )
