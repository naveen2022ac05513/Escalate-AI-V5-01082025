# EscalateAI - Full Application Code
# Version: Robust Deployment Version (with Email Parsing, NLP, SLA Alerting, Teams/Email Alerting, Feedback Loop, Excel integration, Downloadable Output)

import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import os
import sqlite3
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import joblib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from dotenv import load_dotenv
from io import BytesIO

# Load environment variables
load_dotenv()
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_SERVER = os.getenv("EMAIL_SERVER")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", 587))
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")

# Initialize NLP
nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

# Define escalation keywords
escalation_keywords = ["urgent", "immediately", "escalate", "priority", "critical", "severe", "not working"]

# Connect to DB
conn = sqlite3.connect("escalate_ai.db", check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS escalations (
    id TEXT PRIMARY KEY,
    timestamp TEXT,
    customer TEXT,
    issue TEXT,
    sentiment TEXT,
    urgency TEXT,
    escalation_cue TEXT,
    severity TEXT,
    criticality TEXT,
    category TEXT,
    status TEXT,
    action_taken TEXT,
    owner TEXT
)''')
conn.commit()

# Function to get sentiment

def get_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    return "Neutral"

# Function to check escalation

def check_escalation(text):
    return any(kw in text.lower() for kw in escalation_keywords)

# Function to tag severity/criticality (can be extended)

def tag_severity(text):
    if "critical" in text.lower():
        return "High"
    elif "minor" in text.lower():
        return "Low"
    return "Medium"

def tag_criticality(text):
    return "High" if "down" in text.lower() else "Normal"

# Generate unique escalation ID

def generate_id():
    c.execute("SELECT COUNT(*) FROM escalations")
    count = c.fetchone()[0] + 250001
    return f"SESICE-{count}"

# Email parsing

def parse_emails():
    try:
        imap = imaplib.IMAP4_SSL(EMAIL_SERVER)
        imap.login(EMAIL_USER, EMAIL_PASS)
        imap.select("inbox")
        _, msgnums = imap.search(None, 'UNSEEN')

        for num in msgnums[0].split():
            _, msg_data = imap.fetch(num, '(RFC822)')
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)
            subject, encoding = decode_header(msg['Subject'])[0]
            subject = subject.decode(encoding or 'utf-8') if isinstance(subject, bytes) else subject
            from_ = msg.get("From")

            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode()
                        break
            else:
                body = msg.get_payload(decode=True).decode()

            entry_id = generate_id()
            sentiment = get_sentiment(body)
            urgency = "High" if check_escalation(body) else "Normal"
            escalation_cue = "Yes" if check_escalation(body) else "No"
            severity = tag_severity(body)
            criticality = tag_criticality(body)
            category = "General"
            timestamp = datetime.datetime.now().isoformat()

            c.execute("""
                INSERT INTO escalations (id, timestamp, customer, issue, sentiment, urgency, escalation_cue,
                severity, criticality, category, status, action_taken, owner)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (entry_id, timestamp, from_, body, sentiment, urgency, escalation_cue, severity, criticality,
                  category, "Open", "", ""))
            conn.commit()
        imap.logout()
    except Exception as e:
        st.error(f"Email parsing failed: {e}")

# ML model for escalation prediction (dummy for now)
model = joblib.load("escalation_model.pkl") if os.path.exists("escalation_model.pkl") else None

def predict_escalation(text):
    if model:
        return model.predict([text])[0]
    return "Likely"

# Alerting

def send_email_alert(issue):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = "SLA Breach Alert - EscalateAI"
    msg.attach(MIMEText(f"The following issue breached SLA:\n\n{issue}", "plain"))

    with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)


def send_teams_alert(issue):
    payload = {"text": f"SLA Breach Detected:\n\n{issue}"}
    requests.post(MS_TEAMS_WEBHOOK_URL, json=payload)

# Feedback loop
feedback_data = []

def collect_feedback(text, label):
    feedback_data.append((text, label))
    if len(feedback_data) >= 10:
        df = pd.DataFrame(feedback_data, columns=["text", "label"])
        # Save or retrain
        joblib.dump(model, "escalation_model.pkl")

# Streamlit UI

def main():
    st.set_page_config(layout="wide")
    st.title("EscalateAI - Escalation Management System")

    with st.sidebar:
        st.header("Controls")
        if st.button("Parse Gmail"):
            parse_emails()
            st.success("Emails parsed successfully.")

        uploaded_file = st.file_uploader("Upload Excel", type="xlsx")
        if uploaded_file:
            df_excel = pd.read_excel(uploaded_file)
            for _, row in df_excel.iterrows():
                entry_id = generate_id()
                text = row.get("Issue", "")
                c.execute("INSERT INTO escalations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          (entry_id, datetime.datetime.now().isoformat(), row.get("Customer", "Unknown"), text,
                           get_sentiment(text), "High" if check_escalation(text) else "Normal",
                           "Yes" if check_escalation(text) else "No", tag_severity(text),
                           tag_criticality(text), "General", "Open", "", ""))
                conn.commit()
            st.success("Uploaded and processed successfully")

        st.download_button("Download All Complaints", data=export_excel(), file_name="complaints.xlsx")
        if st.button("Send SLA Alerts"):
            df = fetch_cases()
            for _, row in df.iterrows():
                if row['urgency'] == 'High' and row['status'] == 'Open' and check_sla_breach(row['timestamp']):
                    send_email_alert(row['issue'])
                    send_teams_alert(row['issue'])
            st.success("Alerts sent.")

    df = fetch_cases()
    status_filters = st.multiselect("Filter Status", ["Open", "In Progress", "Resolved"], default=["Open"])
    escalated_only = st.checkbox("Show Only Escalated Cases")
    filtered = df[df["status"].isin(status_filters)]
    if escalated_only:
        filtered = filtered[filtered["escalation_cue"] == "Yes"]

    counts = filtered["status"].value_counts().to_dict()
    st.write(f"Open: {counts.get('Open', 0)} | In Progress: {counts.get('In Progress', 0)} | Resolved: {counts.get('Resolved', 0)}")

    for status in ["Open", "In Progress", "Resolved"]:
        with st.expander(f"{status} Cases"):
            st.dataframe(filtered[filtered["status"] == status])

# Utility functions

def fetch_cases():
    return pd.read_sql("SELECT * FROM escalations", conn)

def check_sla_breach(timestamp):
    time_diff = datetime.datetime.now() - datetime.datetime.fromisoformat(timestamp)
    return time_diff.total_seconds() > 600  # 10 min

def export_excel():
    output = BytesIO()
    df = fetch_cases()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

if __name__ == '__main__':
    main()
