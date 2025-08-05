import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import re
import time
import datetime
import base64
import imaplib
import email
from email.header import decode_header
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import smtplib
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import threading
from dotenv import load_dotenv
from twilio.rest import Client  # For WhatsApp
import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
score = sia.polarity_scores("This is great!")
print(score)


# Load environment variables
load_dotenv()

# Email & Alert Configs
EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
SMTP_EMAIL = EMAIL_USER
SMTP_PASS = EMAIL_PASS
ALERT_RECIPIENT = os.getenv("EMAIL_RECEIVER")
TEAMS_WEBHOOK = os.getenv("MS_TEAMS_WEBHOOK_URL")

# Twilio for WhatsApp
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE")

# Escalation & Database
DB_PATH = "escalations.db"
ESCALATION_PREFIX = "SESICE-25"

# NLP Analyzer
analyzer = SentimentIntensityAnalyzer()

# Negative Keywords
NEGATIVE_KEYWORDS = {
    "technical": ["fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip", "malfunction", "blank", "shutdown", "discharge","leak"],
    "dissatisfaction": ["dissatisfy", "frustrate", "complain", "reject", "delay", "ignore", "escalate", "displease", "noncompliance", "neglect"],
    "support": ["wait", "pending", "slow", "incomplete", "miss", "omit", "unresolved", "shortage", "no response"],
    "safety": ["fire", "burn", "flashover", "arc", "explode", "unsafe", "leak", "corrode", "alarm", "incident"],
    "business": ["impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate", "penalty"]
}

# For email polling
processed_email_uids = set()
processed_email_uids_lock = threading.Lock()

def ensure_schema():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS escalations (
            id TEXT PRIMARY KEY,
            customer TEXT,
            issue TEXT,
            sentiment TEXT,
            urgency TEXT,
            severity TEXT,
            criticality TEXT,
            category TEXT,
            status TEXT,
            timestamp TEXT,
            action_taken TEXT,
            owner TEXT,
            escalated TEXT,
            priority TEXT,
            escalation_flag TEXT,
            action_owner TEXT,
            status_update_date TEXT,
            user_feedback TEXT,
            customer_contact TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            escalation_id TEXT,
            field_changed TEXT,
            old_value TEXT,
            new_value TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_change(escalation_id, field, old, new):
    if old == new:
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO audit_log (escalation_id, field_changed, old_value, new_value, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (escalation_id, field, str(old), str(new), datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_next_escalation_id():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id FROM escalations WHERE id LIKE '{ESCALATION_PREFIX}%' ORDER BY id DESC LIMIT 1")
    last = cursor.fetchone()
    conn.close()
    last_num = int(last[0].replace(ESCALATION_PREFIX, "")) if last else 0
    return f"{ESCALATION_PREFIX}{str(last_num + 1).zfill(5)}"

def send_alert(message, via="email"):
    try:
        if via == "email":
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_EMAIL, SMTP_PASS)
                server.sendmail(SMTP_EMAIL, ALERT_RECIPIENT, message)
        elif via == "teams":
            requests.post(TEAMS_WEBHOOK, json={"text": message})
    except Exception as e:
        st.error(f"{via.capitalize()} alert failed: {e}")

def notify_customer(contact, message, channel="email"):
    if channel == "email":
        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_EMAIL, SMTP_PASS)
                server.sendmail(SMTP_EMAIL, contact, message)
        except Exception as e:
            st.error(f"Customer email notification failed: {e}")
    elif channel == "whatsapp" and TWILIO_SID:
        try:
            client = Client(TWILIO_SID, TWILIO_TOKEN)
            client.messages.create(
                body=message,
                from_=f"whatsapp:{TWILIO_PHONE}",
                to=f"whatsapp:{contact}"
            )
        except Exception as e:
            st.error(f"WhatsApp notification failed: {e}")

# --- NLP & Escalation Logic ---

def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def detect_urgency(text):
    # Urgency detection via keywords and sentiment
    urgency = "Low"
    negative_words_found = sum(text.lower().count(word) for cat in NEGATIVE_KEYWORDS.values() for word in cat)
    sentiment = analyzer.polarity_scores(text)['compound']
    if negative_words_found > 2 or sentiment < -0.4:
        urgency = "High"
    elif negative_words_found > 0 or sentiment < -0.1:
        urgency = "Medium"
    return urgency

def classify_severity(text):
    # Simple keyword-based severity
    for word in NEGATIVE_KEYWORDS["safety"]:
        if word in text.lower():
            return "Critical"
    for word in NEGATIVE_KEYWORDS["business"]:
        if word in text.lower():
            return "High"
    for word in NEGATIVE_KEYWORDS["technical"]:
        if word in text.lower():
            return "Medium"
    return "Low"

def classify_criticality(text):
    # Placeholder logic, could be extended
    if "shutdown" in text.lower() or "fire" in text.lower():
        return "High"
    elif "delay" in text.lower() or "reject" in text.lower():
        return "Medium"
    return "Low"

def detect_category(text):
    text_lower = text.lower()
    for cat, keywords in NEGATIVE_KEYWORDS.items():
        if any(word in text_lower for word in keywords):
            return cat.capitalize()
    return "General"

def escalation_flag(urgency, severity):
    # Escalate if high urgency and high severity
    if urgency == "High" and severity in ("Critical", "High"):
        return "Yes"
    return "No"


# --- ML Model ---

def prepare_features(df):
    # Convert text features to numeric (dummy example)
    urgency_map = {"Low": 0, "Medium": 1, "High": 2}
    severity_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    criticality_map = {"Low": 0, "Medium": 1, "High": 2}
    df["urgency_num"] = df["urgency"].map(urgency_map).fillna(0)
    df["severity_num"] = df["severity"].map(severity_map).fillna(0)
    df["criticality_num"] = df["criticality"].map(criticality_map).fillna(0)
    return df[["urgency_num", "severity_num", "criticality_num"]], df["escalation_flag"].map({"Yes":1, "No":0})

def train_ml_model():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT urgency, severity, criticality, escalation_flag FROM escalations", conn)
    conn.close()
    if df.empty or df.shape[0]<20:
        return None  # Insufficient data
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return model, {"accuracy": acc, "precision": precision, "recall": recall}

def predict_escalation(model, urgency, severity, criticality):
    urgency_map = {"Low": 0, "Medium": 1, "High": 2}
    severity_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    criticality_map = {"Low": 0, "Medium": 1, "High": 2}
    features = [[urgency_map.get(urgency,0), severity_map.get(severity,0), criticality_map.get(criticality,0)]]
    return "Yes" if model and model.predict(features)[0] == 1 else "No"


# --- Email Parsing ---

def clean_text(text):
    # Remove unwanted characters
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def fetch_emails():
    global processed_email_uids
    try:
        mail = imaplib.IMAP4_SSL(EMAIL_SERVER)
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select("inbox")
        status, messages = mail.search(None, 'UNSEEN')
        mail_ids = messages[0].split()
        new_records = []
        for mail_id in mail_ids:
            with processed_email_uids_lock:
                if mail_id in processed_email_uids:
                    continue
                processed_email_uids.add(mail_id)
            status, msg_data = mail.fetch(mail_id, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding if encoding else "utf-8")
                    from_ = msg.get("From")
                    date_ = msg.get("Date")
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            content_dispo = str(part.get("Content-Disposition"))
                            if content_type == "text/plain" and "attachment" not in content_dispo:
                                body = part.get_payload(decode=True).decode()
                                break
                    else:
                        body = msg.get_payload(decode=True).decode()
                    full_text = f"{subject} {body}"
                    full_text = clean_text(full_text)
                    # Extract customer from From header or subject if possible
                    customer = from_
                    # NLP tagging
                    sentiment = analyze_sentiment(full_text)
                    urgency = detect_urgency(full_text)
                    severity = classify_severity(full_text)
                    criticality = classify_criticality(full_text)
                    category = detect_category(full_text)
                    flag = escalation_flag(urgency, severity)
                    new_records.append({
                        "customer": customer,
                        "issue": full_text,
                        "sentiment": sentiment,
                        "urgency": urgency,
                        "severity": severity,
                        "criticality": criticality,
                        "category": category,
                        "status": "Open",
                        "timestamp": date_,
                        "action_taken": "",
                        "owner": "",
                        "escalated": flag,
                        "priority": urgency,
                        "escalation_flag": flag,
                        "action_owner": "",
                        "status_update_date": datetime.datetime.now().isoformat(),
                        "user_feedback": "",
                        "customer_contact": customer,
                    })
        mail.logout()
        return new_records
    except Exception as e:
        st.error(f"Failed to fetch emails: {e}")
        return []
import streamlit as st
import pandas as pd
import datetime
import smtplib
from email.message import EmailMessage
import threading
import json
import sqlite3
import logging

# === Database and Logging Setup ===

DB_PATH = "escalate_ai.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS escalations (
        id TEXT PRIMARY KEY,
        customer TEXT,
        issue TEXT,
        sentiment TEXT,
        urgency TEXT,
        severity TEXT,
        criticality TEXT,
        category TEXT,
        status TEXT,
        timestamp TEXT,
        action_taken TEXT,
        owner TEXT,
        escalated TEXT,
        priority TEXT,
        escalation_flag TEXT,
        action_owner TEXT,
        status_update_date TEXT,
        user_feedback TEXT,
        customer_contact TEXT
    )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS audit_log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        escalation_id TEXT,
        timestamp TEXT,
        user TEXT,
        action TEXT,
        old_value TEXT,
        new_value TEXT
    )
    ''')
    conn.commit()
    conn.close()

init_db()

def log_audit(escalation_id, user, action, old_value, new_value):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO audit_log (escalation_id, timestamp, user, action, old_value, new_value) VALUES (?, ?, ?, ?, ?, ?)",
        (escalation_id, datetime.datetime.now().isoformat(), user, action, old_value, new_value)
    )
    conn.commit()
    conn.close()

# === Data Insertion/Update ===

def generate_new_id():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM escalations ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    if row:
        last_id = row[0]
        last_num = int(last_id.split("-")[-1])
        new_num = last_num + 1
    else:
        new_num = 2500001
    return f"SESICE-{new_num:07d}"

def insert_escalation(record):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    new_id = generate_new_id()
    record['id'] = new_id
    c.execute('''
        INSERT INTO escalations (id, customer, issue, sentiment, urgency, severity, criticality,
        category, status, timestamp, action_taken, owner, escalated, priority, escalation_flag,
        action_owner, status_update_date, user_feedback, customer_contact)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        record['id'], record['customer'], record['issue'], record['sentiment'], record['urgency'],
        record['severity'], record['criticality'], record['category'], record['status'],
        record['timestamp'], record['action_taken'], record['owner'], record['escalated'],
        record['priority'], record['escalation_flag'], record['action_owner'],
        record['status_update_date'], record['user_feedback'], record['customer_contact']
    ))
    conn.commit()
    conn.close()
    return new_id

def update_escalation_status(escalation_id, new_status, user):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT status FROM escalations WHERE id = ?", (escalation_id,))
    old_status = c.fetchone()[0]
    c.execute("UPDATE escalations SET status = ?, status_update_date = ? WHERE id = ?", (new_status, datetime.datetime.now().isoformat(), escalation_id))
    conn.commit()
    conn.close()
    log_audit(escalation_id, user, "Status Change", old_status, new_status)

def update_escalation_field(escalation_id, field, new_value, user):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"SELECT {field} FROM escalations WHERE id = ?", (escalation_id,))
    old_value = c.fetchone()[0]
    c.execute(f"UPDATE escalations SET {field} = ?, status_update_date = ? WHERE id = ?", (new_value, datetime.datetime.now().isoformat(), escalation_id))
    conn.commit()
    conn.close()
    log_audit(escalation_id, user, f"Field Update: {field}", old_value, new_value)

def get_all_escalations():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    conn.close()
    return df

# === MS Teams & Email Alerts ===

def send_teams_alert(message: str):
    import requests
    payload = {"text": message}
    try:
        response = requests.post(MS_TEAMS_WEBHOOK_URL, json=payload)
        if response.status_code == 200:
            st.success("MS Teams alert sent.")
        else:
            st.error(f"Failed to send Teams alert: {response.status_code}")
    except Exception as e:
        st.error(f"Teams alert error: {e}")

def send_email_alert(to_email: str, subject: str, body: str):
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_USER
        msg['To'] = to_email
        with smtplib.SMTP_SSL(EMAIL_SERVER, 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)
        st.success("Email alert sent.")
    except Exception as e:
        st.error(f"Email alert error: {e}")

# === Status Change Notification (Email / WhatsApp) ===

def notify_customer(escalation_id, new_status):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT customer_contact, issue FROM escalations WHERE id = ?", (escalation_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return
    contact, issue = row
    message = f"Dear Customer,\n\nYour issue with ID {escalation_id} has changed status to '{new_status}'.\nIssue summary:\n{issue}\n\nThank you."
    # Send email notification
    send_email_alert(contact, f"Status Update for {escalation_id}", message)
    # Optionally, implement WhatsApp notification (needs 3rd party API)
    # Placeholder: st.info("WhatsApp notification sent (placeholder).")

# === Feedback and Retraining ===

def retrain_model_on_feedback():
    model_metrics = None
    model = None
    try:
        model, model_metrics = train_ml_model()
        if model_metrics:
            st.success(f"Retrained model with Accuracy: {model_metrics['accuracy']:.2f}, Precision: {model_metrics['precision']:.2f}, Recall: {model_metrics['recall']:.2f}")
        else:
            st.info("Insufficient data for retraining.")
    except Exception as e:
        st.error(f"Retraining error: {e}")
    return model, model_metrics

# === Search & Filter ===

def filter_escalations(df, filters):
    # Filters dict keys: status, customer, category, urgency, severity, text_search
    if filters.get("status") and filters["status"] != "All":
        df = df[df["status"] == filters["status"]]
    if filters.get("customer") and filters["customer"]:
        df = df[df["customer"].str.contains(filters["customer"], case=False, na=False)]
    if filters.get("category") and filters["category"] != "All":
        df = df[df["category"] == filters["category"]]
    if filters.get("urgency") and filters["urgency"] != "All":
        df = df[df["urgency"] == filters["urgency"]]
    if filters.get("severity") and filters["severity"] != "All":
        df = df[df["severity"] == filters["severity"]]
    if filters.get("text_search") and filters["text_search"]:
        df = df[df["issue"].str.contains(filters["text_search"], case=False, na=False)]
    return df

# === Dashboard ===

def render_dashboard(df):
    st.title("EscalateAI Dashboard")

    total = len(df)
    open_issues = len(df[df["status"] == "Open"])
    escalated = len(df[df["escalated"] == "Yes"])
    critical = len(df[df["severity"] == "Critical"])
    high_urgency = len(df[df["urgency"] == "High"])

    st.metric("Total Issues", total)
    st.metric("Open Issues", open_issues)
    st.metric("Escalated Issues", escalated)
    st.metric("Critical Severity", critical)
    st.metric("High Urgency", high_urgency)

    st.subheader("Issues by Status")
    st.bar_chart(df["status"].value_counts())

    st.subheader("Issues by Category")
    st.bar_chart(df["category"].value_counts())

    st.subheader("Issues by Severity")
    st.bar_chart(df["severity"].value_counts())

# === NLP Feedback Parsing ===

def parse_nlp_feedback(feedback_text):
    # Basic sentiment on feedback text and classify feedback type
    sentiment = analyze_sentiment(feedback_text)
    if "resolved" in feedback_text.lower() or "thank" in feedback_text.lower():
        feedback_type = "Positive"
    elif "delay" in feedback_text.lower() or "not fixed" in feedback_text.lower():
        feedback_type = "Negative"
    else:
        feedback_type = "Neutral"
    return sentiment, feedback_type

# === Streamlit UI Integration ===

def escalate_ai_app():
    st.sidebar.title("EscalateAI Controls")

    # Buttons for MS Teams and Email alerting
    if st.sidebar.button("Send MS Teams Alert for Escalations"):
        df = get_all_escalations()
        escalated_df = df[df["escalated"] == "Yes"]
        if not escalated_df.empty:
            message = f"Escalated cases count: {len(escalated_df)}"
            send_teams_alert(message)
        else:
            st.info("No escalated cases to alert.")

    if st.sidebar.button("Send Email Alert for Escalations"):
        df = get_all_escalations()
        escalated_df = df[df["escalated"] == "Yes"]
        if not escalated_df.empty:
            # Sending to admin email for example
            send_email_alert(EMAIL_USER, "Escalated Cases Alert", f"Escalated cases count: {len(escalated_df)}")
        else:
            st.info("No escalated cases to alert.")

    # Load data and filter UI
    df = get_all_escalations()
    status_options = ["All"] + sorted(df["status"].dropna().unique().tolist())
    category_options = ["All"] + sorted(df["category"].dropna().unique().tolist())
    urgency_options = ["All", "Low", "Medium", "High"]
    severity_options = ["All", "Low", "Medium", "High", "Critical"]

    filters = {
        "status": st.sidebar.selectbox("Filter by Status", status_options, index=0),
        "customer": st.sidebar.text_input("Filter by Customer"),
        "category": st.sidebar.selectbox("Filter by Category", category_options, index=0),
        "urgency": st.sidebar.selectbox("Filter by Urgency", urgency_options, index=0),
        "severity": st.sidebar.selectbox("Filter by Severity", severity_options, index=0),
        "text_search": st.sidebar.text_input("Search in Issues")
    }

    filtered_df = filter_escalations(df, filters)

    tab1, tab2, tab3 = st.tabs(["Dashboard", "Kanban / Table View", "Audit Log"])

    with tab1:
        render_dashboard(filtered_df)

    with tab2:
        # Show Kanban-like status summary for escalated cases
        escalated_df = filtered_df[filtered_df["escalated"] == "Yes"]
        status_counts = escalated_df["status"].value_counts().to_dict()
        st.write("### Escalated Cases Status Summary")
        st.write(pd.DataFrame.from_dict(status_counts, orient='index', columns=["Count"]))

        # Table view with edit options
        edited_index = None
        st.write("### Escalated Cases Detail")
        for idx, row in escalated_df.iterrows():
            with st.expander(f"{row['id']} - {row['customer']} - Status: {row['status']}"):
                st.write(f"Issue: {row['issue']}")
                st.write(f"Urgency: {row['urgency']} | Severity: {row['severity']} | Category: {row['category']}")
                new_status = st.selectbox("Change Status", options=status_options[1:], index=status_options.index(row['status'])-1, key=f"status_{row['id']}")
                new_owner = st.text_input("Action Owner", value=row.get('owner',''), key=f"owner_{row['id']}")
                new_action = st.text_area("Action Taken", value=row.get('action_taken',''), key=f"action_{row['id']}")
                new_feedback = st.text_area("Customer Feedback", value=row.get('user_feedback',''), key=f"feedback_{row['id']}")

                if st.button("Save Changes", key=f"save_{row['id']}"):
                    # Update DB if changed
                    if new_status != row['status']:
                        update_escalation_status(row['id'], new_status, "User")
                        notify_customer(row['id'], new_status)
                        st.success(f"Status updated and customer notified for {row['id']}.")
                    if new_owner != row.get('owner',''):
                        update_escalation_field(row['id'], "owner", new_owner, "User")
                    if new_action != row.get('action_taken',''):
                        update_escalation_field(row['id'], "action_taken", new_action, "User")
                    if new_feedback != row.get('user_feedback',''):
                        update_escalation_field(row['id'], "user_feedback", new_feedback, "User")
                        # NLP parse feedback and retrain model
                        sentiment, fb_type = parse_nlp_feedback(new_feedback)
                        st.info(f"Feedback Sentiment: {sentiment}, Type: {fb_type}")
                        retrain_model_on_feedback()

    with tab3:
        st.write("### Audit Log")
        conn = sqlite3.connect(DB_PATH)
        audit_df = pd.read_sql_query("SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT 50", conn)
        conn.close()
        st.dataframe(audit_df)

if __name__ == "__main__":
    # Initialize global vars, e.g. NLP analyzer, config, etc.
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    # You need to define your EMAIL_USER, EMAIL_PASS, EMAIL_SERVER, MS_TEAMS_WEBHOOK_URL above or in config
    # Example placeholders:
    EMAIL_USER = "your_email@example.com"
    EMAIL_PASS = "your_password"
    EMAIL_SERVER = "smtp.example.com"
    MS_TEAMS_WEBHOOK_URL = "https://outlook.office.com/webhook/your_webhook_url"

    # Run the app
    escalate_ai_app()
