import os
import re
import sqlite3
import email
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple
import time
import smtplib
import logging

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from imapclient import IMAPClient
from bs4 import BeautifulSoup
from apscheduler.schedulers.background import BackgroundScheduler
from transformers import pipeline

# Load environment variables
load_dotenv()

EMAIL = os.getenv("GMAIL_USER") or os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD") or os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
ALERT_RECEIVER = os.getenv("ALERT_RECEIVER") or EMAIL

if not EMAIL or not APP_PASSWORD:
    st.error("⚠️ Gmail credentials not found in .env file. Please add GMAIL_USER and GMAIL_APP_PASSWORD.")
    st.stop()

# Paths & DB
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "escalateai.db"

# Logging
log_dir = APP_DIR / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "escalateai.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Database Init
def initialize_db():
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
            status TEXT DEFAULT 'Open',
            owner TEXT DEFAULT '',
            action_status TEXT DEFAULT 'Pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_updated TEXT,
            escalation_level INTEGER DEFAULT 1
        )
        """)
        conn.commit()

initialize_db()

# Negative Words for rule-based sentiment
NEG_WORDS = set([
    # Technical Failures & Product Malfunction
    "fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip",
    "malfunction", "blank", "shutdown", "discharge",
    # Customer Dissatisfaction & Escalations
    "dissatisfy", "frustrate", "complain", "reject", "delay", "ignore", "escalate",
    "displease", "noncompliance", "neglect",
    # Support Gaps & Operational Delays
    "wait", "pending", "slow", "incomplete", "miss", "omit", "unresolved",
    "shortage", "no response",
    # Hazardous Conditions & Safety Risks
    "fire", "burn", "flashover", "arc", "explode", "unsafe", "leak", "corrode",
    "alarm", "incident",
    # Business Risk & Impact
    "impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate",
    "penalty"
])

@st.cache_resource(show_spinner=False)
def load_transformer_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

transformer_model = load_transformer_model()

def rule_sent(text: str) -> str:
    return "Negative" if any(word in text.lower() for word in NEG_WORDS) else "Positive"

def transformer_sent(text: str) -> str:
    try:
        result = transformer_model(text[:512])[0]
        return "Negative" if result['label'].upper() == "NEGATIVE" else "Positive"
    except Exception:
        return "Positive"

def analyze_issue(text: str) -> Tuple[str, str, str, bool]:
    rule = rule_sent(text)
    transformer = transformer_sent(text)
    urgency = "High" if any(k in text.lower() for k in ["urgent", "immediate", "critical", "asap"]) else "Low"
    escalate = (rule == "Negative" or transformer == "Negative") and urgency == "High"
    return rule, transformer, urgency, escalate

def send_alert_email(issue_summary: str):
    try:
        msg = MIMEText(issue_summary)
        msg["Subject"] = "🚨 High-Risk Escalation Detected"
        msg["From"] = EMAIL
        msg["To"] = ALERT_RECEIVER
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL, APP_PASSWORD)
            server.sendmail(EMAIL, ALERT_RECEIVER, msg.as_string())
        logging.info(f"Alert email sent to {ALERT_RECEIVER}")
    except Exception as e:
        logging.error(f"Failed to send alert email: {e}")

def generate_new_id() -> str:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT MAX(CAST(SUBSTR(id, 7) AS INTEGER)) FROM escalations WHERE id LIKE 'SESICE-%'")
        max_num = cur.fetchone()[0]
        if max_num is None or max_num < 250000:
            next_num = 250001
        else:
            next_num = max_num + 1
        return f"SESICE-{next_num:06d}"

def insert_escalation(data: dict):
    retries = 3
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for attempt in range(retries):
        try:
            with sqlite3.connect(DB_PATH) as conn:
                data["id"] = generate_new_id()
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
            if data["urgency"] == "High" and (data["rule_sentiment"] == "Negative" or data["transformer_sentiment"] == "Negative"):
                send_alert_email(
                    f"Escalation ID: {data['id']}\nCustomer: {data['customer']}\nIssue: {data['issue'][:200]}"
                )
            break
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e) and attempt < retries - 1:
                time.sleep(0.1)
            else:
                raise

def fetch_emails():
    parsed_count = 0
    try:
        with IMAPClient(IMAP_SERVER) as client:
            client.login(EMAIL, APP_PASSWORD)
            client.select_folder("INBOX")
            messages = client.search(["UNSEEN"])
            if not messages:
                st.info("📪 No new emails.")
                return
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
                logging.info(f"Escalation from {from_email} logged.")
            st.success(f"✅ Parsed and logged {parsed_count} new emails.")
    except Exception as e:
        logging.error(f"Failed to fetch emails: {e}")
        st.error("Failed to fetch emails. Check credentials and network.")

def escalate_overdue_cases():
    threshold_hours = 24
    now = datetime.now()
    with sqlite3.connect(DB_PATH) as conn:
        overdue_cases = conn.execute(
            "SELECT id, escalation_level, owner, status, last_updated FROM escalations WHERE status != 'Resolved' AND last_updated IS NOT NULL"
        ).fetchall()
        for case_id, level, owner, status, last_upd in overdue_cases:
            last_upd_dt = datetime.strptime(last_upd, "%Y-%m-%d %H:%M:%S")
            if (now - last_upd_dt) > timedelta(hours=threshold_hours):
                new_level = level + 1
                new_owner = "manager@example.com" if new_level == 2 else owner
                conn.execute(
                    "UPDATE escalations SET escalation_level = ?, owner = ?, last_updated = ? WHERE id = ?",
                    (new_level, new_owner, now.strftime("%Y-%m-%d %H:%M:%S"), case_id)
                )
                conn.commit()
                send_alert_email(f"Escalation {case_id} escalated to level {new_level}, assigned to {new_owner}.")
                logging.info(f"Escalation {case_id} auto-escalated.")

# Scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(fetch_emails, 'interval', minutes=1, id='email_fetcher', replace_existing=True)
scheduler.add_job(escalate_overdue_cases, 'interval', hours=1, id='auto_escalation', replace_existing=True)
scheduler.start()
logging.info("Schedulers started")

# Streamlit UI
st.title("🚀 EscalateAI - Escalation Management with Auto Escalation & Email Alerts")

with st.sidebar:
    st.header("Controls")
    if st.button("Fetch Emails Now"):
        fetch_emails()
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
            st.error("Please enter customer email and issue.")

    st.markdown("---")
    st.header("Bulk Upload Escalations (Excel)")
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
    if uploaded_file:
        try:
            df_upload = pd.read_excel(uploaded_file)
            df_upload.columns = [c.lower().strip() for c in df_upload.columns]
            if not {"customer", "issue"}.issubset(df_upload.columns):
                st.error("Excel must contain 'customer' and 'issue' columns.")
            else:
                for _, row in df_upload.iterrows():
                    cust = row["customer"]
                    issue = str(row["issue"])
                    date_reported = row.get("date_reported", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    rs, ts, urgency, escalated = analyze_issue(issue)
                    insert_escalation({
                        "customer": cust,
                        "issue": issue,
                        "date_reported": date_reported,
                        "rule_sentiment": rs,
                        "transformer_sentiment": ts,
                        "urgency": urgency,
                        "escalated": int(escalated),
                    })
                st.success("Bulk upload complete!")
        except Exception as e:
            st.error(f"Failed to process Excel file: {e}")

# Load and display escalations with filters
with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql("SELECT * FROM escalations ORDER BY created_at DESC", conn)

st.header("Escalations Kanban Board")

statuses = ["Open", "In Progress", "Resolved"]
status_colors = {
    "Open": "#ffcccc",
    "In Progress": "#fff0b3",
    "Resolved": "#ccffcc"
}

# Filter by status and escalated flag
status_filter = st.multiselect("Filter by Status", options=statuses, default=statuses)
escalated_filter = st.selectbox("Show Escalations", options=["All", "Only Escalated", "Only Non-Escalated"], index=0)

filtered_df = df[df["status"].isin(status_filter)]
if escalated_filter == "Only Escalated":
    filtered_df = filtered_df[filtered_df["escalated"] == 1]
elif escalated_filter == "Only Non-Escalated":
    filtered_df = filtered_df[filtered_df["escalated"] == 0]

cols = st.columns(len(statuses))

for col, status in zip(cols, statuses):
    count = len(filtered_df[filtered_df["status"] == status])
    with col:
        st.markdown(f"<h3 style='background-color:{status_colors[status]};padding:8px;border-radius:6px;text-align:center;'>{status} ({count})</h3>", unsafe_allow_html=True)
        sub_df = filtered_df[filtered_df["status"] == status]
        if sub_df.empty:
            st.write("_No escalations_")
        else:
            for _, row in sub_df.iterrows():
                with st.expander(f"{row['id']} - {row['customer']} ({row['rule_sentiment']}/{row['urgency']})"):
                    st.write(f"**Issue:** {row['issue']}")
                    st.write(f"**Escalated:** {'Yes' if row['escalated'] else 'No'}")
                    st.write(f"**Date Reported:** {row['date_reported']}")
                    st.write(f"**Escalation Level:** {row.get('escalation_level', 1)}")

                    owner = st.text_input(f"Owner ({row['id']})", value=row.get("owner", ""), key=f"owner_{row['id']}")
                    action_status = st.text_input(f"Action Taken ({row['id']})", value=row.get("action_status", ""), key=f"action_{row['id']}")
                    status_val = st.selectbox(f"Status ({row['id']})", options=statuses, index=statuses.index(row["status"]), key=f"status_{row['id']}")

                    if st.button("Update", key=f"update_{row['id']}"):
                        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with sqlite3.connect(DB_PATH) as conn:
                            conn.execute("UPDATE escalations SET owner=?, action_status=?, status=?, last_updated=? WHERE id=?",
                                         (owner, action_status, status_val, now_str, row['id']))
                            conn.commit()
                        st.success(f"Escalation {row['id']} updated.")
                        st.experimental_rerun()
