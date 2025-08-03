# EscalateAI - Robust Customer Escalation Tracker & Predictor

import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
import re
import smtplib
from email.mime.text import MIMEText
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh
import spacy
from transformers import pipeline
import threading
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")  # Default admin password

DB_PATH = "escalations.db"

# Initialize NLP models
analyzer = SentimentIntensityAnalyzer()
nlp_spacy = spacy.load("en_core_web_sm")
bert_classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Negative keywords
NEGATIVE_KEYWORDS = [
    "fail", "break", "crash", "defect", "fault", "degrade", "damage",
    "trip", "malfunction", "blank", "shutdown", "discharge",
    "dissatisfy", "frustrate", "complain", "reject", "delay", "ignore",
    "escalate", "displease", "noncompliance", "neglect",
    "wait", "pending", "slow", "incomplete", "miss", "omit",
    "unresolved", "shortage", "no response",
    "fire", "burn", "flashover", "arc", "explode", "unsafe", "leak",
    "corrode", "alarm", "incident",
    "impact", "loss", "risk", "downtime", "interrupt", "cancel",
    "terminate", "penalty"
]

URGENCY_PHRASES = ["asap", "urgent", "immediately", "right away", "critical"]

CATEGORY_KEYWORDS = {
    "Safety": ["fire", "burn", "leak", "unsafe", "alarm", "incident", "explode", "flashover", "arc", "corrode"],
    "Performance": ["slow", "crash", "malfunction", "degrade", "fault", "blank", "shutdown"],
    "Delay": ["delay", "pending", "wait", "unresolved", "shortage", "no response", "incomplete", "miss", "omit"],
    "Compliance": ["noncompliance", "violation", "penalty"],
    "Service": ["ignore", "unavailable", "reject", "complain", "frustrate", "dissatisfy", "displease"],
    "Quality": ["defect", "fault", "break", "damage", "fail", "trip"],
    "Business Risk": ["impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate"]
}

# Database Setup
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
    predicted_risk REAL,
    feedback TEXT
)
""")
conn.commit()

# Utility Functions

def generate_id():
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0] + 250001
    return f"SESICE-{count}"

def classify_sentiment_vader(text):
    score = analyzer.polarity_scores(text)['compound']
    if score < -0.05:
        return "Negative"
    elif score > 0.05:
        return "Positive"
    else:
        return "Neutral"

def classify_sentiment_bert(text):
    try:
        res = bert_classifier(text[:512])[0]  # Truncate long texts
        label = res['label'].lower()
        if "negative" in label:
            return "Negative"
        elif "positive" in label:
            return "Positive"
        else:
            return "Neutral"
    except Exception as e:
        logging.warning(f"BERT classification error: {e}")
        return classify_sentiment_vader(text)  # fallback

def detect_urgency(text):
    text_lower = text.lower()
    return "High" if any(phrase in text_lower for phrase in URGENCY_PHRASES) else "Normal"

def detect_category(text):
    text_lower = text.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return cat
    return "General"

def is_escalation(text):
    text_lower = text.lower()
    return any(word in text_lower for word in NEGATIVE_KEYWORDS)

def insert_to_db(data):
    try:
        cursor.execute("""
            INSERT INTO escalations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        conn.commit()
    except sqlite3.IntegrityError:
        logging.info("Duplicate escalation ID - skipping insert")

def fetch_cases():
    return pd.read_sql_query("SELECT * FROM escalations ORDER BY date DESC", conn)

def send_teams_alert(msg):
    if not MS_TEAMS_WEBHOOK_URL:
        return
    try:
        response = requests.post(MS_TEAMS_WEBHOOK_URL, json={"text": msg})
        if response.status_code != 200:
            logging.warning(f"Teams alert failed with status {response.status_code}")
    except Exception as e:
        logging.warning(f"Teams alert exception: {e}")

def send_email_alert(subject, body):
    if not (EMAIL_USER and EMAIL_PASS and EMAIL_RECEIVER):
        logging.warning("Email credentials or receiver not set")
        return
    try:
        msg = MIMEText(body)
        msg['From'] = EMAIL_USER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = subject

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
    except Exception as e:
        logging.warning(f"Failed to send email alert: {e}")

def detect_sla_breach(alert_method="teams"):
    now = datetime.datetime.now()
    df = fetch_cases()
    breached = []
    for _, row in df.iterrows():
        try:
            created_dt = datetime.datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
        if row["priority"] == "High" and row["status"] != "Resolved":
            if (now - created_dt).total_seconds() > 600:  # 10 min SLA
                breached.append(row)
    if breached:
        for case in breached:
            msg = (f"SLA Breach Alert!\nEscalation: {case['escalation_id']}\n"
                   f"Customer: {case['customer']}\nIssue: {case['issue']}\n"
                   f"Status: {case['status']}\nDate: {case['date']}")
            if alert_method == "teams":
                send_teams_alert(msg)
            elif alert_method == "email":
                send_email_alert("SLA Breach Alert", msg)
        return True
    return False

def parse_email():
    if not (EMAIL_USER and EMAIL_PASS):
        st.warning("Email credentials not set in environment variables.")
        return 0
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select("inbox")
        status, messages = mail.search(None, 'UNSEEN')
        count = 0
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
            if from_ and body:
                process_case(from_, body)
                count += 1
        mail.logout()
        return count
    except Exception as e:
        st.warning(f"Error parsing emails: {e}")
        return 0

def process_excel(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        processed = 0
        for _, row in df.iterrows():
            customer = str(row.get('Customer', '')).strip()
            issue = str(row.get('Issue', '')).strip()
            if customer and issue:
                process_case(customer, issue)
                processed += 1
        return processed
    except Exception as e:
        st.error(f"Error processing Excel file: {e}")
        return 0

def process_case(customer, issue):
    if not customer or not issue:
        return
    eid = generate_id()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # NLP processing
    sentiment_vader = classify_sentiment_vader(issue)
    sentiment_bert = classify_sentiment_bert(issue)
    sentiment = sentiment_bert if sentiment_bert else sentiment_vader

    priority = "High" if sentiment == "Negative" else "Normal"
    urgency = detect_urgency(issue)
    category = detect_category(issue)
    flag = 1 if is_escalation(issue) else 0
    predicted_risk = predict_risk(issue)  # ML stub

    data = (eid, customer, issue, now_str, "Open", sentiment, priority, flag,
            urgency, category, "", "", now_str, predicted_risk, "")
    insert_to_db(data)

    if flag and priority == "High":
        alert_msg = f"New Escalation: {eid} from {customer}\nIssue: {issue[:300]}"
        send_teams_alert(alert_msg)
        send_email_alert("New Escalation Alert", alert_msg)

def predict_risk(issue_text):
    # TODO: Integrate your real ML model here
    # For now, dummy heuristic: 0.7 risk if negative sentiment else 0.2
    sentiment = classify_sentiment_vader(issue_text)
    return 0.7 if sentiment == "Negative" else 0.2

def update_predictions():
    df = fetch_cases()
    updated = 0
    for idx, row in df.iterrows():
        if pd.isna(row['predicted_risk']):
            risk = predict_risk(row['issue'])
            cursor.execute("UPDATE escalations SET predicted_risk=? WHERE escalation_id=?", (risk, row['escalation_id']))
            updated += 1
    conn.commit()
    return updated

def display_kanban_card(row):
    st.markdown(f"**ID:** {row['escalation_id']} | **Customer:** {row['customer']}")
    st.markdown(f"**Sentiment:** {row['sentiment']} | **Urgency:** {row['urgency']} | **Category:** {row['category']}")
    st.write(row['issue'])

    action_taken = st.text_input("Action Taken", value=row['action_taken'], key=f"action_{row['escalation_id']}")
    action_owner = st.text_input("Action Owner", value=row['action_owner'], key=f"owner_{row['escalation_id']}")
    statuses = ["Open", "In Progress", "Resolved"]
    try:
        current_idx = statuses.index(row['status'])
    except ValueError:
        current_idx = 0
    new_status = st.selectbox("Status", statuses, index=current_idx, key=f"status_{row['escalation_id']}")

    if st.button("Save Changes", key=f"save_{row['escalation_id']}"):
        cursor.execute("""
            UPDATE escalations SET status=?, action_taken=?, action_owner=?, status_update_date=?
            WHERE escalation_id=?
        """, (new_status, action_taken, action_owner, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), row['escalation_id']))
        conn.commit()
        st.success(f"Updated escalation {row['escalation_id']}")

def render_kanban(filter_escalated=False):
    df = fetch_cases()
    if filter_escalated:
        df = df[df['escalation_flag'] == 1]
    statuses = ["Open", "In Progress", "Resolved"]
    cols = st.columns(len(statuses))
    for i, status in enumerate(statuses):
        with cols[i]:
            count = len(df[df['status'] == status])
            st.markdown(f"### {status} ({count})")
            filtered = df[df['status'] == status]
            if filtered.empty:
                st.write("No cases.")
            for _, row in filtered.iterrows():
                with st.expander(f"{row['escalation_id']} - {row['customer']}"):
                    display_kanban_card(row)

def get_complaints_df():
    df = fetch_cases()
    return df[[
        "escalation_id", "customer", "issue", "date", "status", "priority",
        "category", "predicted_risk"
    ]]

def save_df_to_excel(df):
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# Feedback loop for model improvement
def feedback_loop():
    st.subheader("Escalation Prediction Feedback")
    pwd = st.text_input("Enter Admin Password", type="password")
    if pwd != ADMIN_PASSWORD:
        st.warning("Invalid password!")
        return

    df = fetch_cases()
    pending_feedback = df[df["feedback"] == ""]
    if pending_feedback.empty:
        st.info("No cases pending feedback.")
        return

    for _, row in pending_feedback.iterrows():
        st.markdown(f"**{row['escalation_id']} - {row['customer']}**")
        st.write(row['issue'])
        feedback = st.radio("Is the escalation prediction correct?", ("Yes", "No"), key=row['escalation_id'])
        if st.button("Submit Feedback", key=f"fb_{row['escalation_id']}"):
            cursor.execute("UPDATE escalations SET feedback=? WHERE escalation_id=?", (feedback, row['escalation_id']))
            conn.commit()
            st.success("Feedback submitted.")
            # Optionally trigger retraining async here
            threading.Thread(target=async_retrain_model).start()

def async_retrain_model():
    # Placeholder for model retraining logic
    logging.info("Started async model retraining...")
    import time
    time.sleep(5)  # simulate retraining delay
    logging.info("Retraining complete.")

# Main Streamlit App

def main():
    st.set_page_config(page_title="EscalateAI", layout="wide")
    st.title("🚨 EscalateAI – Customer Escalation Tracker & Predictor")

    with st.sidebar:
        st.header("Upload & Alerts")

        uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])
        if uploaded_file is not None:
            count = process_excel(uploaded_file)
            st.success(f"Processed {count} records from Excel.")

        st.markdown("---")

        cust = st.text_input("Manual Entry - Customer")
        iss = st.text_area("Manual Entry - Issue")
        if st.button("Add Manual Escalation"):
            if cust.strip() and iss.strip():
                process_case(cust.strip(), iss.strip())
                st.success("Manual escalation added.")
            else:
                st.warning("Please enter customer and issue.")

        st.markdown("---")

        if st.button("Parse New Emails"):
            count = parse_email()
            st.success(f"Parsed {count} new email(s).")

        st.markdown("---")

        if st.button("Update ML Predictions"):
            updated = update_predictions()
            st.success(f"Updated {updated} predictions.")

        st.markdown("---")

        alert_method = st.radio("SLA Breach Alert Method", ["teams", "email"])

        if st.button("Trigger SLA Breach Alerts"):
            breached = detect_sla_breach(alert_method)
            if breached:
                st.success("SLA breach alerts sent.")
            else:
                st.info("No SLA breaches detected.")

        st.markdown("---")

        st.subheader("Download Complaints")
        df_download = get_complaints_df()
        excel_data = save_df_to_excel(df_download)
        st.download_button(label="Download All Complaints as Excel", data=excel_data, file_name="complaints.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.markdown("---")

        if st.checkbox("Show Feedback Loop (Admin Only)"):
            feedback_loop()

    st.markdown("---")

    filter_escalated = st.checkbox("Show Only Escalated Cases (High Priority)", value=False)
    render_kanban(filter_escalated)

    st_autorefresh(interval=60 * 1000, key="refresh")  # Refresh every 60 sec for live updates

if __name__ == "__main__":
    main()
