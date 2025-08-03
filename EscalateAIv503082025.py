# EscalateAI – Complete Functional Code with Debugging and Defensive Coding
# ========================================================================
# Features:
# • Parsing emails + Excel for customer issues
# • NLP analysis (sentiment, urgency, category tagging)
# • Unique ID: SESICE-XXXXX
# • Kanban Board with inline editing of status, action taken, owner
# • SLA Breach detection (10 minutes)
# • Teams/email alerts on high priority or SLA breach
# • Predictive ML model with continuous feedback loop
# • Defensive coding and debug info for missing fields

import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import requests
from streamlit_autorefresh import st_autorefresh
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import smtplib
from email.mime.text import MIMEText

# Load environment variables from .env file
load_dotenv()

# ------------------- Configuration -------------------
EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")

MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")

EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", 587))
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")  # Change default in .env!

DB_PATH = "escalations.db"

# List of negative keywords grouped by category
NEGATIVE_KEYWORDS = [
    # Technical Failures & Product Malfunction
    "fail", "break", "crash", "defect", "fault", "degrade", "damage",
    "trip", "malfunction", "blank", "shutdown", "discharge",
    # Customer Dissatisfaction & Escalations
    "dissatisfy", "frustrate", "complain", "reject", "delay", "ignore",
    "escalate", "displease", "noncompliance", "neglect",
    # Support Gaps & Operational Delays
    "wait", "pending", "slow", "incomplete", "miss", "omit",
    "unresolved", "shortage", "no response",
    # Hazardous Conditions & Safety Risks
    "fire", "burn", "flashover", "arc", "explode", "unsafe",
    "leak", "corrode", "alarm", "incident",
    # Business Risk & Impact
    "impact", "loss", "risk", "downtime", "interrupt", "cancel",
    "terminate", "penalty"
]

URGENCY_PHRASES = ["asap", "urgent", "immediately", "right away", "critical"]

CATEGORY_KEYWORDS = {
    "Safety": ["fire", "burn", "leak", "unsafe", "alarm", "incident", "explode", "flashover", "arc", "corrode"],
    "Performance": ["slow", "crash", "malfunction", "fail", "break", "shutdown", "trip"],
    "Delay": ["delay", "pending", "wait", "miss", "omit", "incomplete", "shortage", "no response"],
    "Compliance": ["noncompliance", "violation", "penalty", "reject", "cancel", "terminate"],
    "Service": ["ignore", "unresolved", "unavailable", "dissatisfy", "frustrate", "complain", "displease", "neglect"],
    "Quality": ["defect", "fault", "damage", "degrade", "blank", "discharge"],
    "Business Risk": ["impact", "loss", "risk", "downtime", "interrupt"],
}

# ------------------- Database Setup -------------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS escalations (
    escalation_id TEXT PRIMARY KEY,
    customer TEXT,
    issue TEXT,
    date TEXT,
    status TEXT,
    sentiment TEXT,
    priority TEXT,
    escalation_flag INTEGER,
    urgency TEXT,
    category TEXT,
    action_taken TEXT,
    action_owner TEXT,
    status_update_date TEXT,
    user_feedback INTEGER DEFAULT 0
)
""")
conn.commit()

# ------------------- Sentiment Analyzer -------------------
analyzer = SentimentIntensityAnalyzer()

# ------------------- Utility Functions -------------------

def generate_escalation_id():
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]
    new_id = f"SESICE-{count + 250001}"
    return new_id

def classify_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def detect_urgency(text):
    lowered = text.lower()
    for phrase in URGENCY_PHRASES:
        if phrase in lowered:
            return "High"
    return "Normal"

def detect_category(text):
    lowered = text.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in lowered:
                return cat
    return "General"

def count_negative_keywords(text):
    lowered = text.lower()
    count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in lowered)
    return count

def analyze_issue(issue_text, use_ml=True):
    """
    Analyze issue text to determine sentiment, priority, escalation flag, urgency, category
    If ML model is available and use_ml=True, use it for priority prediction.
    Else, use heuristic based on sentiment and negative keyword count.
    """
    sentiment = classify_sentiment(issue_text)
    urgency = detect_urgency(issue_text)
    category = detect_category(issue_text)
    neg_count = count_negative_keywords(issue_text)

    priority = "Low"
    escalation_flag = 0

    if use_ml:
        model, vectorizer = load_predictive_model()
        if model and vectorizer:
            predicted_priority = predict_priority(issue_text, sentiment, model, vectorizer)
            priority = predicted_priority
            escalation_flag = 1 if priority == "High" else 0
        else:
            # fallback heuristic
            if sentiment == "Negative" and neg_count >= 2:
                priority = "High"
                escalation_flag = 1
    else:
        if sentiment == "Negative" and neg_count >= 2:
            priority = "High"
            escalation_flag = 1

    return sentiment, priority, escalation_flag, urgency, category

# ------------------- Database Operations -------------------

def insert_escalation(data):
    """
    Data tuple:
    (escalation_id, customer, issue, date, status, sentiment,
     priority, escalation_flag, urgency, category, action_taken,
     action_owner, status_update_date)
    """
    cursor.execute("""
        INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment,
            priority, escalation_flag, urgency, category, action_taken, action_owner, status_update_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()

def fetch_all_escalations():
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    return df

# ------------------- Email Fetching -------------------

def fetch_gmail_emails():
    """
    Fetch latest 10 unseen emails from Gmail inbox,
    parse customer (from), issue (body), subject, date
    """
    if not EMAIL or not APP_PASSWORD:
        st.error("Gmail credentials not set in environment variables.")
        return []

    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL, APP_PASSWORD)
        mail.select("inbox")
        result, data = mail.search(None, "UNSEEN")
        if result != "OK":
            mail.logout()
            return []

        email_ids = data[0].split()
        emails = []

        for eid in email_ids[-10:]:  # last 10 unseen emails
            res, msg_data = mail.fetch(eid, "(RFC822)")
            if res != "OK":
                continue
            msg = email.message_from_bytes(msg_data[0][1])

            subject, encoding = decode_header(msg.get("Subject"))[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or "utf-8", errors="ignore")

            from_ = msg.get("From")
            date = msg.get("Date")

            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    ctype = part.get_content_type()
                    cdisp = str(part.get("Content-Disposition"))
                    if ctype == "text/plain" and "attachment" not in cdisp:
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

            emails.append({
                "customer": from_,
                "issue": body.strip(),
                "subject": subject,
                "date": date
            })

            mail.store(eid, '+FLAGS', '\\Seen')

        mail.logout()
        return emails

    except Exception as e:
        st.error(f"Error fetching emails: {e}")
        return []

# ------------------- Alerting -------------------

def send_ms_teams_alert(message):
    if not MS_TEAMS_WEBHOOK_URL:
        st.warning("MS Teams webhook URL not set; cannot send alerts.")
        return
    headers = {"Content-Type": "application/json"}
    payload = {"text": message}
    try:
        response = requests.post(MS_TEAMS_WEBHOOK_URL, json=payload, headers=headers)
        if response.status_code != 200:
            st.error(f"MS Teams alert failed: {response.status_code} {response.text}")
    except Exception as e:
        st.error(f"Error sending MS Teams alert: {e}")

def send_email_alert(subject, message):
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        st.warning("Email sender/receiver credentials not set.")
        return

    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER

    try:
        server = smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, [EMAIL_RECEIVER], msg.as_string())
        server.quit()
        st.info("Email alert sent.")
    except Exception as e:
        st.error(f"Failed to send email alert: {e}")

def send_alerts(message):
    send_ms_teams_alert(message)
    send_email_alert("EscalateAI Alert", message)

# ------------------- Processing Incoming Data -------------------

def save_emails_to_db(emails):
    """
    Save parsed emails to DB if not duplicate,
    analyze sentiment, priority, escalation, urgency, category,
    send alerts if high priority.
    """
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]
    new_entries = 0

    for e in emails:
        # Check duplicate (customer + issue)
        cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (e['customer'], e['issue'][:500]))
        if cursor.fetchone():
            continue
        count += 1
        esc_id = f"SESICE-{count + 250000}"
        sentiment, priority, escalation_flag, urgency, category = analyze_issue(e['issue'])
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")

        insert_escalation((
            esc_id, e['customer'], e['issue'][:500], e['date'], "Open",
            sentiment, priority, escalation_flag, urgency, category, "", "", now
        ))

        new_entries += 1
        if escalation_flag == 1:
            send_alerts(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {e['customer']}\nIssue: {e['issue'][:200]}...")
    return new_entries

def upload_excel_and_analyze(file):
    """
    Upload Excel file with escalations, extract customer, issue, date columns,
    analyze and insert new records.
    """
    try:
        df = pd.read_excel(file)
        df.columns = [c.lower().strip() for c in df.columns]
        customer_col = next((c for c in df.columns if "customer" in c or "email" in c), None)
        issue_col = next((c for c in df.columns if "issue" in c or "text" in c or "complaint" in c), None)
        date_col = next((c for c in df.columns if "date" in c), None)

        if not customer_col or not issue_col:
            st.error("Excel must contain customer/email and issue/text columns.")
            return 0

        count = 0
        cursor.execute("SELECT COUNT(*) FROM escalations")
        existing_count = cursor.fetchone()[0]

        for _, row in df.iterrows():
            customer = str(row[customer_col])
            issue = str(row[issue_col])
            date = str(row[date_col]) if date_col and pd.notna(row[date_col]) else datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (customer, issue[:500]))
            if cursor.fetchone():
                continue
            existing_count += 1
            esc_id = f"SESICE-{existing_count + 250000}"
            sentiment, priority, escalation_flag, urgency, category = analyze_issue(issue)
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")

            insert_escalation((
                esc_id, customer, issue[:500], date, "Open",
                sentiment, priority, escalation_flag, urgency, category, "", "", now
            ))

            count += 1
            if escalation_flag == 1:
                send_alerts(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {customer}\nIssue: {issue[:200]}...")
        return count
    except Exception as e:
        st.error(f"Error processing uploaded Excel: {e}")
        return 0

def manual_entry_process(customer, issue):
    """
    Process manual entry from sidebar.
    """
    if not customer or not issue:
        st.sidebar.error("Please fill customer and issue.")
        return False
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]
    esc_id = f"SESICE-{count + 250001}"
    sentiment, priority, escalation_flag, urgency, category = analyze_issue(issue)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")

    insert_escalation((
        esc_id, customer, issue[:500], now, "Open",
        sentiment, priority, escalation_flag, urgency, category, "", "", now
    ))
    st.sidebar.success(f"Added escalation {esc_id}")
    if escalation_flag == 1:
        send_alerts(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {customer}\nIssue: {issue[:200]}...")
    return True

# ------------------- Kanban Board Display -------------------

def display_kanban_card(row):
    """
    Display a single escalation card with inline editing fields.
    Defensive get() used to avoid KeyError if columns missing.
    """
    esc_id = row.get('escalation_id', "UnknownID")
    sentiment = row.get('sentiment', "Unknown")
    priority = row.get('priority', "Unknown")
    status = row.get('status', "Unknown")
    urgency = row.get('urgency', "Unknown")
    category = row.get('category', "Unknown")
    action_taken = row.get('action_taken', "")
    action_owner = row.get('action_owner', "")

    sentiment_colors = {"Positive": "#27ae60", "Negative": "#e74c3c", "Neutral": "#f39c12", "Unknown": "#7f8c8d"}
    priority_colors = {"High": "#c0392b", "Low": "#27ae60", "Unknown": "#34495e"}
    status_colors = {"Open": "#f1c40f", "In Progress": "#2980b9", "Resolved": "#2ecc71", "Unknown": "#95a5a6"}

    border_color = priority_colors.get(priority, "#000000")
    status_color = status_colors.get(status, "#bdc3c7")
    sentiment_color = sentiment_colors.get(sentiment, "#7f8c8d")

    header_html = f"""
    <div style="
        border-left: 6px solid {border_color};
        padding-left: 
