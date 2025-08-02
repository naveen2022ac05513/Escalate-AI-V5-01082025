# ==============================================================
# EscalateAI – Escalation Management Tool with Email Parsing & Time-Based Escalation & Alerts
# --------------------------------------------------------------
# Author: Naveen Gandham • v1.9.0 • August 2025
# ==============================================================

import os
import re
import sqlite3
import email
from email.mime.text import MIMEText
from email.message import EmailMessage
from datetime import datetime, timedelta
from pathlib import Path
import time
import smtplib

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from imapclient import IMAPClient
from bs4 import BeautifulSoup
from apscheduler.schedulers.background import BackgroundScheduler
import logging

from transformers import pipeline
import imaplib
import email
from email.header import decode_header
import os
from dotenv import load_dotenv

import streamlit as st
import imaplib
import email
from email.header import decode_header
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Correct variable names as per your .env file
EMAIL = os.getenv("EMAIL_USER") or st.text_input("Enter your Gmail address")
APP_PASSWORD = os.getenv("EMAIL_PASS") or st.text_input("Enter your Gmail App Password", type="password")
EMAIL_SERVER = os.getenv("EMAIL_SERVER") or "imap.gmail.com"

def connect_and_fetch_emails():
    if not EMAIL or not APP_PASSWORD:
        st.warning("🔐 Please provide your Gmail and App Password.")
        return []

    try:
        mail = imaplib.IMAP4_SSL(EMAIL_SERVER)
        mail.login(EMAIL, APP_PASSWORD)
        st.success("📡 Connected to Gmail.")
    except imaplib.IMAP4.error as e:
        st.error(f"❌ Gmail login failed: {str(e)}")
        return []

    status, _ = mail.select("inbox")
    if status != 'OK':
        st.error("❌ Could not select the inbox.")
        return []

    # DEBUG: Try different search filters
    result, data = mail.search(None, 'ALL')
    st.write("🔍 Search result:", result)
    st.write("📬 Raw data from search:", data)

    if result != 'OK' or not data or not data[0]:
        st.info("📭 No new unread emails.")
        return []

    email_ids = data[0].split()
    st.write(f"📧 Found {len(email_ids)} unread emails.")

    fetched_emails = []

    for num in email_ids[-10:]:  # Read only last 10 unseen
        result, msg_data = mail.fetch(num, '(RFC822)')
        if result != 'OK':
            st.warning(f"⚠️ Failed to fetch email ID {num}")
            continue

        msg = email.message_from_bytes(msg_data[0][1])

        subject, encoding = decode_header(msg["Subject"])[0]
        if isinstance(subject, bytes):
            subject = subject.decode(encoding or 'utf-8', errors='ignore')

        from_ = msg.get("From")
        date = msg.get("Date")

        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition")):
                    try:
                        body = part.get_payload(decode=True).decode()
                    except:
                        pass
                    break
        else:
            try:
                body = msg.get_payload(decode=True).decode()
            except:
                pass

        fetched_emails.append({
            "from": from_,
            "subject": subject,
            "body": body,
            "date": date
        })

        mail.store(num, '+FLAGS', '\\Seen')  # Mark as read

    mail.logout()
    return fetched_emails


# Streamlit UI
st.header("📬 Gmail Escalation Fetcher")
if st.button("🔄 Fetch Emails"):
    emails = connect_and_fetch_emails()
    if emails:
        st.success(f"✅ {len(emails)} new emails fetched.")
        st.dataframe(pd.DataFrame(emails))
    else:
        st.warning("📪 No new unread emails or error in fetching.")


# ----------------------- Paths & ENV -----------------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "escalateai.db"

load_dotenv()
IMAP_USER = os.getenv("EMAIL_USER")
IMAP_PASS = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
ALERT_RECEIVER = os.getenv("ALERT_RECEIVER", IMAP_USER)

# SMTP email alert config (adjust as needed)
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", IMAP_USER)
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", IMAP_PASS)
FROM_EMAIL = SMTP_USERNAME

# ----------------------- Logging -----------------------
LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "escalateai.log",
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
        # Migration: add columns if missing
        cursor = conn.execute("PRAGMA table_info(escalations)")
        columns = [col[1] for col in cursor.fetchall()]
        if "last_updated" not in columns:
            conn.execute("ALTER TABLE escalations ADD COLUMN last_updated TEXT")
        if "escalation_level" not in columns:
            conn.execute("ALTER TABLE escalations ADD COLUMN escalation_level INTEGER DEFAULT 1")
        if "owner" not in columns:
            conn.execute("ALTER TABLE escalations ADD COLUMN owner TEXT DEFAULT ''")
        if "action_status" not in columns:
            conn.execute("ALTER TABLE escalations ADD COLUMN action_status TEXT DEFAULT 'Pending'")
        conn.commit()

initialize_db()

# ----------------------- Sentiment & Keyword Lists -----------------------
NEG_WORDS = sorted(set([
    # Technical Failures & Product Malfunction
    "fail", "break", "crash", "defect", "fault", "degrade",
    "damage", "trip", "malfunction", "blank", "shutdown", "discharge",
    # Customer Dissatisfaction & Escalations
    "dissatisfy", "frustrate", "complain", "reject", "delay",
    "ignore", "escalate", "displease", "noncompliance", "neglect",
    # Support Gaps & Operational Delays
    "wait", "pending", "slow", "incomplete", "miss", "omit",
    "unresolved", "shortage", "no response",
    # Hazardous Conditions & Safety Risks
    "fire", "burn", "flashover", "arc", "explode", "unsafe",
    "leak", "corrode", "alarm", "incident",
    # Business Risk & Impact
    "impact", "loss", "risk", "downtime", "interrupt", "cancel",
    "terminate", "penalty"
]))

@st.cache_resource(show_spinner=False)
def load_transformer_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

transformer_model = load_transformer_model()

def rule_sent(text: str) -> str:
    return "Negative" if any(re.search(rf"\b{re.escape(word)}\b", text, re.I) for word in NEG_WORDS) else "Positive"

def transformer_sent(text: str) -> str:
    try:
        result = transformer_model(text[:512])[0]
        return "Negative" if result['label'].upper() == "NEGATIVE" else "Positive"
    except Exception as e:
        logging.error(f"Transformer sentiment error: {e}")
        return "Positive"

def analyze_issue(text: str):
    rule = rule_sent(text)
    transformer = transformer_sent(text)
    urgency = "High" if any(k in text.lower() for k in ["urgent", "immediate", "critical", "asap"]) else "Low"
    escalate = (rule == "Negative" or transformer == "Negative") and urgency == "High"
    return rule, transformer, urgency, escalate

# ----------------------- Notification -----------------------
def send_alert_email(to_email, subject, body):
    try:
        if not to_email:
            to_email = ALERT_RECEIVER
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = FROM_EMAIL
        msg["To"] = to_email
        msg.set_content(body)
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        logging.info(f"📧 Alert email sent to {to_email} with subject '{subject}'")
    except Exception as e:
        logging.error(f"❌ Failed to send alert email: {e}")

# ----------------------- Insert with Unique ID -----------------------
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
                next_num = 250001 if max_num is None or max_num < 250000 else max_num + 1
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
            # Alert if escalated
            if (data["rule_sentiment"] == "Negative" or data["transformer_sentiment"] == "Negative") and data["urgency"] == "High":
                send_alert_email(
                    ALERT_RECEIVER,
                    "🚨 High-Risk Escalation Detected",
                    f"Escalation ID: {data['id']}\nCustomer: {data['customer']}\nIssue: {data['issue'][:200]}"
                )
            break
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e) and attempt < retries - 1:
                time.sleep(0.1)
            else:
                raise

# ----------------------- Email Parser (Gmail IMAP) -----------------------
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
                            except Exception:
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

# ----------------------- Time-Based Auto Escalation -----------------------
AUTO_ESCALATION_HOURS = 72  # 3 days

def auto_escalate_cases():
    now = datetime.now()
    with sqlite3.connect(DB_PATH) as conn:
        overdue_cases = conn.execute(
            """
            SELECT id, escalation_level, owner, status, last_updated, issue, customer
            FROM escalations
            WHERE status != 'Resolved' AND last_updated IS NOT NULL AND escalation_level < 3
            """
        ).fetchall()

        for case_id, level, owner, status, last_upd, issue, customer in overdue_cases:
            try:
                last_upd_dt = datetime.strptime(last_upd, "%Y-%m-%d %H:%M:%S")
                hours_passed = (now - last_upd_dt).total_seconds() / 3600
                if hours_passed >= AUTO_ESCALATION_HOURS:
                    new_level = level + 1
                    new_owner = owner if owner else ALERT_RECEIVER
                    conn.execute(
                        """
                        UPDATE escalations
                        SET escalation_level = ?, escalated=1, last_updated = ?, owner = ?
                        WHERE id = ?
                        """,
                        (new_level, now.strftime("%Y-%m-%d %H:%M:%S"), new_owner, case_id)
                    )
                    conn.commit()
                    send_alert_email(
                        new_owner,
                        f"⚠️ Escalation Level Increased for {case_id}",
                        f"Escalation {case_id} for customer {customer} has been escalated to level {new_level} due to inactivity.\n\nIssue: {issue}"
                    )
                    logging.info(f"Auto-escalated {case_id} to level {new_level}")
            except Exception as e:
                logging.error(f"Error in auto escalation for {case_id}: {e}")

# ----------------------- Scheduler -----------------------
scheduler = BackgroundScheduler()
scheduler.add_job(parse_emails, 'interval', minutes=1, id='email_job', replace_existing=True)
scheduler.add_job(auto_escalate_cases, 'interval', hours=1, id='auto_escalation', replace_existing=True)
scheduler.start()
logging.info("Schedulers started")

# ----------------------- Streamlit App -----------------------
st.title("🚀 EscalateAI - Escalation Management with Auto Escalation & Email Alerts")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    if st.button("Manually Parse Emails"):
        with st.spinner("Checking inbox..."):
            parse_emails()

    st.markdown("---")
    st.header("Manual Escalation Entry")
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

    st.subheader("📥 Bulk Upload from Excel")
    uploaded_file = st.file_uploader("Upload Escalation Excel", type=["xlsx"])
    if uploaded_file:
        try:
            df_upload = pd.read_excel(uploaded_file, engine="openpyxl")
            df_upload.columns = [col.lower().strip() for col in df_upload.columns]
            required_cols = {"customer", "issue"}
            if not required_cols.issubset(df_upload.columns):
                st.error("Excel must contain at least 'customer' and 'issue' columns.")
            else:
                st.success("File uploaded successfully. Processing entries...")
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
                            "status": "Open",
                            "action_status": "Pending",
                            "owner": "",
                            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "escalation_level": 1 if escalated else 0
                        })
                    except Exception as e:
                        st.warning(f"Row {idx+1}: {e}")
                    time.sleep(0.05)
                st.success("✅ Bulk upload completed.")
        except Exception as e:
            st.error(f"Failed to process file: {e}")

# Load escalations for dashboard
with sqlite3.connect(DB_PATH) as conn:
    try:
        df = pd.read_sql("SELECT * FROM escalations ORDER BY created_at DESC", conn)
    except Exception as e:
        st.error(f"Database read error: {e}")
        df = pd.DataFrame()

# Filters
st.header("Escalations Kanban Board")
filter_escalated = st.checkbox("Show Escalated Cases Only", value=False)

if filter_escalated:
    df = df[df["escalated"] == 1]

statuses = ["Open", "In Progress", "Resolved"]
STATUS_COLORS = {
    "Open": "#ffcccc",
    "In Progress": "#fff0b3",
    "Resolved": "#ccffcc"
}

cols = st.columns(len(statuses))

for col, status in zip(cols, statuses):
    count = len(df[df["status"] == status]) if not df.empty else 0
    with col:
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
                        key=f"action_{row['id']}"
                    )
                    new_status = st.selectbox("Status", options=statuses, index=statuses.index(row["status"]), key=f"status_{row['id']}")

                    if st.button("Update", key=f"update_{row['id']}"):
                        try:
                            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            with sqlite3.connect(DB_PATH) as conn:
                                conn.execute(
                                    """
                                    UPDATE escalations
                                    SET status=?, action_status=?, last_updated=?, owner=?
                                    WHERE id=?
                                    """,
                                    (new_status, new_action_status, now_str, new_owner, row["id"])
                                )
                                conn.commit()

                            # Notify owner on assignment
                            if new_owner:
                                send_alert_email(
                                    new_owner,
                                    f"You have been assigned Escalation {row['id']}",
                                    f"Hello,\n\nYou have been assigned the escalation case {row['id']}.\n\nIssue:\n{row['issue']}\n\nPlease attend to this promptly."
                                )
                            st.success(f"Escalation {row['id']} updated.")
                            if st.session_state.get('needs_refresh'):
                               st.session_state['needs_refresh'] = False
                               st.info("🔄 Please refresh the page to see the updated escalation data.")
                        except Exception as e:
                            st.error(f"Update failed: {e}")

# Footer note
st.markdown("---")
st.markdown("⚙️ Powered by EscalateAI | Developed by Naveen Gandham")

