# EscalateAI – Complete Escalation Management System
# =================================================
# Features:
# - Parse Gmail emails + Excel sheets for customer issues
# - Analyze sentiment, urgency, and escalation triggers using VADER + keywords
# - Tag issues with severity, criticality, category
# - Unique escalation IDs SESICE-XXXXX (incremental)
# - Kanban board UI: Open, In Progress, Resolved with editable action/status/owner
# - SLA breach detection (>10 minutes open high-priority) with MS Teams + email alerts
# - Predictive ML model (stub) to forecast escalation risk
# - Continuous feedback loop for real-time ML retraining (stub)
# - Comprehensive comments and error handling

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
import smtplib
from email.mime.text import MIMEText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load environment variables for credentials & endpoints
load_dotenv()
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
IMAP_SERVER = "imap.gmail.com"
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
ALERT_EMAIL = os.getenv("ALERT_EMAIL")  # Email to send SLA alerts

DB_PATH = "escalations.db"

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# --- Keywords and categories ---
NEGATIVE_KEYWORDS = [
    # Technical Failures & Product Malfunction
    "fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip", "malfunction", "blank", "shutdown", "discharge",
    # Customer Dissatisfaction & Escalations
    "dissatisfy", "frustrate", "complain", "reject", "delay", "ignore", "escalate", "displease", "noncompliance", "neglect",
    # Support Gaps & Operational Delays
    "wait", "pending", "slow", "incomplete", "miss", "omit", "unresolved", "shortage", "no response",
    # Hazardous Conditions & Safety Risks
    "fire", "burn", "flashover", "arc", "explode", "unsafe", "leak", "corrode", "alarm", "incident",
    # Business Risk & Impact
    "impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate", "penalty"
]

URGENCY_PHRASES = ["asap", "urgent", "immediately", "right away", "critical"]

CATEGORY_KEYWORDS = {
    "Safety": ["fire", "burn", "leak", "unsafe", "explosion", "arc", "flashover"],
    "Performance": ["slow", "crash", "malfunction", "degrade", "damage"],
    "Delay": ["delay", "pending", "wait", "unresolved", "shortage"],
    "Compliance": ["noncompliance", "violation", "penalty"],
    "Service": ["ignore", "unavailable", "neglect", "reject"],
    "Quality": ["defect", "fault", "break", "fail", "trip", "shutdown"],
    "Risk": ["impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate"]
}

# --- Database Setup ---
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
    predicted_risk REAL
)
""")
conn.commit()

# --- Utility functions ---

def generate_id():
    """Generate unique ID SESICE-XXXXX starting from 250001"""
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0] + 250001
    return f"SESICE-{count}"

def classify_sentiment(text):
    """Use VADER to classify sentiment"""
    score = analyzer.polarity_scores(text)['compound']
    if score < -0.05:
        return "Negative"
    elif score > 0.05:
        return "Positive"
    else:
        return "Neutral"

def detect_urgency(text):
    """Detect urgency by keywords"""
    text_lower = text.lower()
    return "High" if any(p in text_lower for p in URGENCY_PHRASES) else "Normal"

def detect_category(text):
    """Tag category based on keyword dictionary"""
    text_lower = text.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return cat
    return "General"

def is_escalation(text):
    """Check if text contains escalation keywords"""
    text_lower = text.lower()
    return any(word in text_lower for word in NEGATIVE_KEYWORDS)

def insert_to_db(data):
    """Insert a new escalation record"""
    cursor.execute("""
        INSERT INTO escalations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()

def fetch_cases():
    """Fetch all cases as pandas DataFrame"""
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    return df

def send_teams_alert(msg):
    """Send alert message to MS Teams via webhook"""
    if MS_TEAMS_WEBHOOK_URL:
        try:
            response = requests.post(MS_TEAMS_WEBHOOK_URL, json={"text": msg})
            if response.status_code != 200:
                st.warning(f"MS Teams alert failed: {response.status_code}")
        except Exception as e:
            st.warning(f"MS Teams alert exception: {e}")

def send_email_alert(subject, body, to_email):
    """Send SLA alert email"""
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL
        msg['To'] = to_email

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL, PASSWORD)
        server.sendmail(EMAIL, to_email, msg.as_string())
        server.quit()
    except Exception as e:
        st.warning(f"Error sending email alert: {e}")

def parse_date_flexible(date_str):
    """Try parsing date in multiple formats, return datetime or None"""
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d"
    ]
    for fmt in formats:
        try:
            return datetime.datetime.strptime(date_str, fmt)
        except:
            continue
    return None

# --- SLA Breach Detection ---
def detect_sla_breach():
    """Check open high priority cases > 10 mins, alert MS Teams and email"""
    now = datetime.datetime.now()
    cursor.execute("SELECT escalation_id, date, customer, issue FROM escalations WHERE priority='High' AND status='Open'")
    rows = cursor.fetchall()
    for eid, date_str, customer, issue in rows:
        created_dt = parse_date_flexible(date_str)
        if not created_dt:
            st.warning(f"Cannot parse date for escalation {eid}: {date_str}")
            continue
        elapsed = now - created_dt
        if elapsed.total_seconds() > 600:  # 10 minutes SLA
            msg = f"⚠️ SLA breach detected:\nID: {eid}\nCustomer: {customer}\nOpen for {int(elapsed.total_seconds() // 60)} minutes\nIssue: {issue}"
            send_teams_alert(msg)
            if ALERT_EMAIL:
                send_email_alert(f"SLA Breach: {eid}", msg, ALERT_EMAIL)

# --- Email Parsing ---
def parse_email():
    """Connect to Gmail, fetch unseen emails and process"""
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL, PASSWORD)
        mail.select("inbox")
        status, messages = mail.search(None, 'UNSEEN')
        if status != "OK":
            st.warning("Failed to fetch emails")
            return
        for num in messages[0].split():
            _, data = mail.fetch(num, '(RFC822)')
            msg = email.message_from_bytes(data[0][1])
            from_ = msg.get("From")
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode(errors='ignore')
                        break
            else:
                body = msg.get_payload(decode=True).decode(errors='ignore')
            process_case(from_, body)
        mail.logout()
    except Exception as e:
        st.warning(f"Error parsing emails: {e}")

# --- Excel Processing ---
def process_excel(uploaded_file):
    """Process uploaded Excel file with 'Customer' and 'Issue' columns"""
    try:
        df = pd.read_excel(uploaded_file)
        for _, row in df.iterrows():
            customer = row.get('Customer', 'Unknown')
            issue = row.get('Issue', '')
            if pd.isna(issue) or issue.strip() == "":
                continue
            process_case(customer, issue)
    except Exception as e:
        st.warning(f"Error processing Excel file: {e}")

# --- Core Case Processing ---
def process_case(customer, issue):
    """Analyze and insert new escalation record with ML prediction"""
    eid = generate_id()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sentiment = classify_sentiment(issue)
    urgency = detect_urgency(issue)
    category = detect_category(issue)
    flag = 1 if is_escalation(issue) else 0
    priority = "High" if sentiment == "Negative" or urgency == "High" else "Normal"

    # ML Predictive risk score (stub model, extend with real model)
    predicted_risk = predict_escalation(issue)

    data = (eid, customer, issue, now, "Open", sentiment, priority, flag, urgency, category, "", "", now, predicted_risk)
    insert_to_db(data)

    # Send alert if escalation
    if flag and priority == "High":
        alert_msg = f"🚨 New High Priority Escalation: {eid}\nCustomer: {customer}\nIssue: {issue}\nPredicted Risk: {predicted_risk:.2f}"
        send_teams_alert(alert_msg)
        if ALERT_EMAIL:
            send_email_alert(f"New High Priority Escalation {eid}", alert_msg, ALERT_EMAIL)

# --- Kanban Board UI ---
def display_kanban_card(row):
    st.markdown(f"### {row['escalation_id']} - {row['customer']}")
    st.markdown(f"**Issue:** {row['issue']}")
    st.markdown(f"**Sentiment:** {row['sentiment']} | **Urgency:** {row['urgency']} | **Category:** {row['category']}")
    st.markdown(f"**Predicted Risk:** {row['predicted_risk']:.2f}")
    action_taken = st.text_input("Action Taken", value=row['action_taken'] or "", key=f"{row['escalation_id']}_action")
    action_owner = st.text_input("Action Owner", value=row['action_owner'] or "", key=f"{row['escalation_id']}_owner")
    statuses = ["Open", "In Progress", "Resolved"]
    current_status = row['status'] if row['status'] in statuses else "Open"
    new_status = st.selectbox("Status", statuses, index=statuses.index(current_status), key=f"{row['escalation_id']}_status")
    if st.button("Save", key=f"save_{row['escalation_id']}"):
        cursor.execute("""
            UPDATE escalations SET status=?, action_taken=?, action_owner=?, status_update_date=?
            WHERE escalation_id=?
        """, (new_status, action_taken, action_owner, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), row['escalation_id']))
        conn.commit()
        st.success(f"Updated {row['escalation_id']}")

def render_kanban():
    df = fetch_cases()
    statuses = ["Open", "In Progress", "Resolved"]
    cols = st.columns(len(statuses))
    for idx, status in enumerate(statuses):
        with cols[idx]:
            st.header(status)
            filtered = df[df['status'] == status]
            if filtered.empty:
                st.write("No cases")
            for _, row in filtered.iterrows():
                with st.expander(f"{row['escalation_id']} - {row['customer']}"):
                    display_kanban_card(row)

# --- Predictive ML model stub ---

# For demonstration, train a very simple logistic regression on existing data
def train_predictive_model():
    """Train a logistic regression model to predict escalation risk (1 or 0)"""
    df = fetch_cases()
    if df.empty or len(df) < 10:
        return None, None

    # Text vectorizer
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(df['issue'].astype(str))
    y = df['escalation_flag']

    model = LogisticRegression()
    model.fit(X, y)

    return model, vectorizer

def predict_escalation(issue_text):
    """Predict escalation risk using trained model, return probability 0-1"""
    global predictive_model, predictive_vectorizer
    if predictive_model is None or predictive_vectorizer is None:
        return 0.0
    X = predictive_vectorizer.transform([issue_text])
    prob = predictive_model.predict_proba(X)[0][1]
    return prob

# --- Feedback loop stub ---

def feedback_loop():
    """Placeholder for feedback collection and model retraining"""
    # In practice: collect user edits, retrain model, update DB predictions
    pass

# Initialize ML model globally
predictive_model, predictive_vectorizer = None, None

# --- Main Streamlit Application ---

def main():
    st.title("🚨 EscalateAI - AI-powered Escalation Management")

    # Sidebar: Email parse button + Excel upload + manual entry
    with st.sidebar:
        st.subheader("Inbox Parsing")
        if st.button("Parse new Emails"):
            parse_email()
            st.success("Email parsing complete.")

        st.subheader("Upload Excel")
        uploaded_file = st.file_uploader("Excel file (.xlsx)", type=["xlsx"])
        if uploaded_file:
            process_excel(uploaded_file)
            st.success("Excel processed.")

        st.subheader("Manual Escalation Entry")
        cust = st.text_input("Customer Name")
        issue = st.text_area("Issue Description")
        if st.button("Add Escalation"):
            if not cust.strip() or not issue.strip():
                st.warning("Both fields are required!")
            else:
                process_case(cust.strip(), issue.strip())
                st.success("Escalation added manually.")

    # Train/update predictive model once per app run
    global predictive_model, predictive_vectorizer
    predictive_model, predictive_vectorizer = train_predictive_model()

    # Render Kanban board
    render_kanban()

    # Check SLA breaches and send alerts if needed
    detect_sla_breach()

    # Feedback loop placeholder
    feedback_loop()

if __name__ == "__main__":
    main()
