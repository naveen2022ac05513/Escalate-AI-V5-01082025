import os
import re
import sqlite3
import email
from email.header import decode_header
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import time
import smtplib

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import imaplib
from bs4 import BeautifulSoup
from transformers import pipeline
from apscheduler.schedulers.background import BackgroundScheduler
import logging

# Load environment variables from .env file
load_dotenv()

# Configurations
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
ALERT_RECEIVER = os.getenv("ALERT_RECEIVER", EMAIL_USER)
DB_PATH = "escalateai.db"
IMAP_FOLDER = "INBOX"

# Setup logging
logging.basicConfig(
    filename="escalateai.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize DB and create table if needed
def initialize_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS escalations (
                escalation_id TEXT PRIMARY KEY,
                customer TEXT,
                issue TEXT,
                date_reported TEXT,
                sentiment TEXT,
                priority TEXT,
                status TEXT DEFAULT 'Open',
                owner TEXT DEFAULT '',
                action_taken TEXT DEFAULT '',
                escalation_level INTEGER DEFAULT 1,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

initialize_db()

# Load transformer sentiment analysis model
@st.cache_resource(show_spinner=False)
def load_transformer_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

transformer_model = load_transformer_model()

# Simple negative keywords for rule-based sentiment
NEG_WORDS = [
    "fail", "break", "crash", "defect", "fault", "delay", "complaint",
    "escalate", "urgent", "immediate", "critical", "problem", "issue"
]

def rule_sentiment(text: str) -> str:
    pattern = re.compile(r"\b(" + "|".join(NEG_WORDS) + r")\b", re.I)
    return "Negative" if pattern.search(text) else "Positive"

def transformer_sentiment(text: str) -> str:
    try:
        result = transformer_model(text[:512])[0]
        return "Negative" if result['label'].upper() == "NEGATIVE" else "Positive"
    except Exception as e:
        logging.error(f"Transformer sentiment error: {e}")
        return "Positive"

def determine_priority(text: str) -> str:
    high_urgency_words = ["urgent", "immediately", "asap", "critical", "fail"]
    return "High" if any(word in text.lower() for word in high_urgency_words) else "Low"

def generate_escalation_id() -> str:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT MAX(CAST(SUBSTR(escalation_id, 8) AS INTEGER)) FROM escalations")
        max_num = cur.fetchone()[0]
        next_num = max_num + 1 if max_num else 250001
        return f"SESICE-{next_num:06d}"

def send_alert_email(subject: str, body: str):
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = EMAIL_USER
        msg["To"] = ALERT_RECEIVER
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, ALERT_RECEIVER, msg.as_string())
        logging.info(f"Alert email sent: {subject}")
    except Exception as e:
        logging.error(f"Failed to send alert email: {e}")

def insert_escalation(data: Dict):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Check duplicate (same customer + issue)
            cursor = conn.execute(
                "SELECT 1 FROM escalations WHERE customer=? AND issue=?",
                (data["customer"], data["issue"])
            )
            if cursor.fetchone():
                logging.info("Duplicate escalation skipped.")
                return False

            escalation_id = generate_escalation_id()
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute("""
                INSERT INTO escalations (
                    escalation_id, customer, issue, date_reported, sentiment, priority,
                    status, escalation_level, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                escalation_id, data["customer"], data["issue"], data["date_reported"],
                data["sentiment"], data["priority"], "Open", 1, now_str
            ))
            conn.commit()

            if data["sentiment"] == "Negative" and data["priority"] == "High":
                send_alert_email(
                    "🚨 High-Risk Escalation Detected",
                    f"Escalation ID: {escalation_id}\nCustomer: {data['customer']}\nIssue: {data['issue'][:200]}"
                )
            return True
    except Exception as e:
        logging.error(f"Insert escalation error: {e}")
        return False

def fetch_unseen_emails():
    try:
        mail = imaplib.IMAP4_SSL(EMAIL_SERVER)
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select(IMAP_FOLDER)
        status, response = mail.search(None, "UNSEEN")
        if status != 'OK':
            logging.warning("No unseen emails found or error in search.")
            return []

        email_ids = response[0].split()
        fetched_emails = []
        for eid in email_ids[-10:]:  # Limit to last 10 unseen emails
            status, msg_data = mail.fetch(eid, "(RFC822)")
            if status != "OK":
                continue
            msg = email.message_from_bytes(msg_data[0][1])
            from_email = email.utils.parseaddr(msg.get("From"))[1].lower()
            date = msg.get("Date") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain" and 'attachment' not in str(part.get("Content-Disposition")):
                        body = part.get_payload(decode=True).decode(errors="ignore")
                        break
            else:
                body = msg.get_payload(decode=True).decode(errors="ignore")

            clean_body = BeautifulSoup(body, "html.parser").get_text()
            fetched_emails.append({
                "customer": from_email,
                "issue": clean_body.strip()[:1000],
                "date_reported": date
            })
            # Mark as seen so not fetched again
            mail.store(eid, '+FLAGS', '\\Seen')
        mail.logout()
        return fetched_emails
    except Exception as e:
        logging.error(f"Failed fetching emails: {e}")
        return []

def analyze_and_log_emails(emails: List[Dict]):
    count = 0
    for email_data in emails:
        sentiment_rule = rule_sentiment(email_data["issue"])
        sentiment_trans = transformer_sentiment(email_data["issue"])
        # If either says Negative, consider Negative
        sentiment = "Negative" if "Negative" in [sentiment_rule, sentiment_trans] else "Positive"
        priority = determine_priority(email_data["issue"])
        inserted = insert_escalation({
            "customer": email_data["customer"],
            "issue": email_data["issue"],
            "date_reported": email_data["date_reported"],
            "sentiment": sentiment,
            "priority": priority
        })
        if inserted:
            count += 1
    return count

def escalate_overdue_cases():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            now = datetime.now()
            threshold = timedelta(hours=24)
            rows = conn.execute("""
                SELECT escalation_id, escalation_level, last_updated, status, owner 
                FROM escalations WHERE status != 'Resolved'
            """).fetchall()
            for row in rows:
                escalation_id, level, last_updated, status, owner = row
                last_upd_dt = datetime.strptime(last_updated, "%Y-%m-%d %H:%M:%S")
                if now - last_upd_dt > threshold:
                    new_level = level + 1
                    new_owner = owner if new_level == 1 else "manager@example.com"  # example logic
                    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
                    conn.execute("""
                        UPDATE escalations SET escalation_level=?, owner=?, last_updated=? WHERE escalation_id=?
                    """, (new_level, new_owner, now_str, escalation_id))
                    conn.commit()
                    send_alert_email(
                        "⚠️ Escalation Auto-Escalated",
                        f"Escalation {escalation_id} has been escalated to level {new_level} assigned to {new_owner}."
                    )
                    logging.info(f"Escalation {escalation_id} escalated to level {new_level}")
    except Exception as e:
        logging.error(f"Error in auto escalation: {e}")

# Scheduler setup
scheduler = BackgroundScheduler()
scheduler.add_job(lambda: analyze_and_log_emails(fetch_unseen_emails()), 'interval', minutes=1, id='email_fetch_job', replace_existing=True)
scheduler.add_job(escalate_overdue_cases, 'interval', hours=1, id='auto_escalation_job', replace_existing=True)
scheduler.start()

# Streamlit UI starts here
st.title("🚀 EscalateAI - AI-driven Escalation Management")

# Button to manually fetch and log new emails
if st.button("🔄 Fetch New Emails"):
    st.info("Fetching new emails...")
    emails = fetch_unseen_emails()
    count = analyze_and_log_emails(emails)
    st.success(f"Fetched and logged {count} new escalation(s).")

# Load data from DB for display
def load_escalations_df():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql("SELECT * FROM escalations ORDER BY last_updated DESC", conn)

df = load_escalations_df()

# Filters
with st.sidebar.expander("Filters", expanded=True):
    filter_status = st.multiselect("Filter by Status", options=["Open", "In Progress", "Resolved"], default=["Open", "In Progress", "Resolved"])
    filter_priority = st.multiselect("Filter by Priority", options=["High", "Low"], default=["High", "Low"])
    filter_escalation_level = st.slider("Escalation Level", min_value=1, max_value=int(df["escalation_level"].max() if not df.empty else 5), value=(1, int(df["escalation_level"].max() if not df.empty else 5)))
    filter_escalated_only = st.checkbox("Show only escalated cases", value=False)

filtered_df = df[
    (df["status"].isin(filter_status)) &
    (df["priority"].isin(filter_priority)) &
    (df["escalation_level"] >= filter_escalation_level[0]) &
    (df["escalation_level"] <= filter_escalation_level[1])
]

if filter_escalated_only:
    filtered_df = filtered_df[filtered_df["priority"] == "High"]

# Kanban Board Display
st.header("📋 Escalations Kanban Board")

status_columns = ["Open", "In Progress", "Resolved"]
status_colors = {
    "Open": "#ffcccc",
    "In Progress": "#fff0b3",
    "Resolved": "#ccffcc"
}

cols = st.columns(len(status_columns))

def render_kanban():
    for col, status in zip(cols, status_columns):
        with col:
            subset = filtered_df[filtered_df["status"] == status]
            st.markdown(f"<h3 style='background-color:{status_colors[status]};padding:6px;border-radius:6px;text-align:center;'>{status} ({len(subset)})</h3>", unsafe_allow_html=True)
            if subset.empty:
                st.write("_No escalations_")
            else:
                for _, row in subset.iterrows():
                    exp_label = f"{row['escalation_id']} - {row['customer']} ({row['sentiment']}/{row['priority']})"
                    with st.expander(exp_label):
                        st.write(f"**Issue:** {row['issue']}")
                        st.write(f"**Date Reported:** {row['date_reported']}")
                        st.write(f"**Escalation Level:** {row['escalation_level']}")
                        st.write(f"**Owner:** {row.get('owner','')}")
                        st.write(f"**Action Taken:** {row.get('action_taken','')}")

                        # Editable fields
                        new_owner = st.text_input("Owner", value=row.get('owner', ''), key=f"owner_{row['escalation_id']}")
                        new_action = st.text_input("Action Taken", value=row.get('action_taken', ''), key=f"action_{row['escalation_id']}")

                        status_value = row.get('status') or "Open"
                        try:
                            selected_idx = status_columns.index(status_value)
                        except ValueError:
                            selected_idx = 0
                        new_status = st.selectbox("Update Status", status_columns, index=selected_idx, key=f"status_{row['escalation_id']}")

                        if st.button("Update Escalation", key=f"update_{row['escalation_id']}"):
                            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            try:
                                with sqlite3.connect(DB_PATH) as conn:
                                    conn.execute("""
                                        UPDATE escalations SET owner=?, action_taken=?, status=?, last_updated=?
                                        WHERE escalation_id=?
                                    """, (new_owner, new_action, new_status, now_str, row['escalation_id']))
                                    conn.commit()
                                st.success(f"Updated escalation {row['escalation_id']}")
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Failed to update: {e}")

render_kanban()

# Manual escalation entry form
st.header("✍️ Manually Add Escalation")
with st.form("manual_form"):
    cust_email = st.text_input("Customer Email")
    issue_text = st.text_area("Issue Description")
    submit_manual = st.form_submit_button("Add Escalation")

    if submit_manual:
        if cust_email.strip() == "" or issue_text.strip() == "":
            st.error("Please enter both customer email and issue description.")
        else:
            sentiment_rule = rule_sentiment(issue_text)
            sentiment_trans = transformer_sentiment(issue_text)
            sentiment = "Negative" if "Negative" in [sentiment_rule, sentiment_trans] else "Positive"
            priority = determine_priority(issue_text)

            inserted = insert_escalation({
                "customer": cust_email.lower(),
                "issue": issue_text.strip()[:1000],
                "date_reported": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sentiment": sentiment,
                "priority": priority
            })
            if inserted:
                st.success("Escalation logged successfully.")
                st.experimental_rerun()
            else:
                st.warning("Duplicate escalation found, not added.")

# Export escalations to Excel button
if not df.empty:
    excel_data = df.to_excel(index=False)
    st.download_button(
        label="📥 Download Escalations as Excel",
        data=excel_data,
        file_name=f"escalations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("No escalations available to download.")

# Shutdown scheduler gracefully on exit (only works if app restarted manually)
import atexit
atexit.register(lambda: scheduler.shutdown())

