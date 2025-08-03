import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh
import smtplib
from email.mime.text import MIMEText
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib
from io import BytesIO

# Load environment variables for email and Teams webhook credentials
load_dotenv()
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
IMAP_SERVER = "imap.gmail.com"
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER", EMAIL)
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", PASSWORD)
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO")  # comma separated emails to alert on SLA breach

DB_PATH = "escalations.db"
MODEL_PATH = "escalation_model.pkl"

analyzer = SentimentIntensityAnalyzer()

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

# Database setup
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
    feedback INTEGER DEFAULT 0 -- 0 = no feedback, 1 = confirmed escalation, -1 = false positive
)
""")
conn.commit()

# Utility functions
def generate_id():
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0] + 250001
    return f"SESICE-{count}"

def classify_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score < -0.05:
        return "Negative"
    elif score > 0.05:
        return "Positive"
    else:
        return "Neutral"

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
    cursor.execute("""
        INSERT INTO escalations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()

def fetch_cases():
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    return df

def update_case(escalation_id, status, action_taken, action_owner):
    cursor.execute("""
        UPDATE escalations SET status=?, action_taken=?, action_owner=?, status_update_date=?
        WHERE escalation_id=?
    """, (status, action_taken, action_owner, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), escalation_id))
    conn.commit()

def update_feedback(escalation_id, feedback_value):
    cursor.execute("""
        UPDATE escalations SET feedback=?
        WHERE escalation_id=?
    """, (feedback_value, escalation_id))
    conn.commit()

# Teams alert
def send_teams_alert(msg):
    if MS_TEAMS_WEBHOOK_URL:
        try:
            response = requests.post(MS_TEAMS_WEBHOOK_URL, json={"text": msg})
            if response.status_code != 200:
                st.warning(f"Failed to send Teams alert: {response.status_code}")
        except Exception as e:
            st.warning(f"Exception while sending Teams alert: {e}")

# Email alert (for SLA breach)
def send_email_alert(subject, body):
    if not ALERT_EMAIL_TO:
        return
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = SMTP_USER
        msg['To'] = ALERT_EMAIL_TO
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, ALERT_EMAIL_TO.split(","), msg.as_string())
    except Exception as e:
        st.warning(f"Failed to send email alert: {e}")

# SLA breach detection with email or Teams alert
def detect_sla_breach():
    now = datetime.datetime.now()
    df = fetch_cases()
    for idx, row in df.iterrows():
        try:
            created_dt = datetime.datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
        if row.get("priority", "") == "High" and row.get("status", "") != "Resolved":
            if (now - created_dt).total_seconds() > 600:
                alert_msg = f"SLA Breach: Escalation {row['escalation_id']} still open for over 10 minutes"
                send_teams_alert(alert_msg)
                send_email_alert("SLA Breach Notification", alert_msg)

# Email Parsing - unchanged from your code
def parse_email():
    if not (EMAIL and PASSWORD):
        st.warning("Email credentials not set in environment variables.")
        return
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL, PASSWORD)
        mail.select("inbox")
        status, messages = mail.search(None, 'UNSEEN')
        for num in messages[0].split():
            _, data = mail.fetch(num, '(RFC822)')
            msg = email.message_from_bytes(data[0][1])
            subject = decode_header(msg["Subject"])[0][0]
            from_ = msg.get("From")
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode()
                        break
            else:
                body = msg.get_payload(decode=True).decode()
            process_case(from_, body)
        mail.logout()
    except Exception as e:
        st.warning(f"Error parsing emails: {e}")

def process_excel(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        for _, row in df.iterrows():
            customer = str(row.get('Customer', '')).strip()
            issue = str(row.get('Issue', '')).strip()
            if customer and issue:
                process_case(customer, issue)
    except Exception as e:
        st.error(f"Error processing Excel file: {e}")

# Your existing core processing + ML prediction integrated
def process_case(customer, issue):
    if not customer or not issue:
        return
    eid = generate_id()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sentiment = classify_sentiment(issue)
    urgency = detect_urgency(issue)
    category = detect_category(issue)
    flag = 1 if is_escalation(issue) else 0
    
    # Predict risk with ML model if available
    predicted_risk = predict_risk(issue)

    priority = "High" if predicted_risk > 0.5 else ("High" if sentiment == "Negative" else "Normal")

    data = (eid, customer, issue, now_str, "Open", sentiment, priority, flag,
            urgency, category, "", "", now_str, predicted_risk, 0)
    insert_to_db(data)

    if flag and priority == "High":
        send_teams_alert(f"New Escalation: {eid} from {customer}\nIssue: {issue}")
        send_email_alert("New Escalation Alert", f"Escalation {eid} from {customer}\nIssue: {issue}")

# ML model - logistic regression with simple feature vectorization for demonstration
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        # Train a dummy model on existing data if available
        df = fetch_cases()
        if df.empty:
            return None
        X, y = [], []
        for _, row in df.iterrows():
            features = text_to_features(row['issue'])
            X.append(features)
            y.append(row['escalation_flag'])
        if not X:
            return None
        model = LogisticRegression()
        model.fit(np.array(X), np.array(y))
        joblib.dump(model, MODEL_PATH)
        return model

def text_to_features(text):
    # Basic features: sentiment compound, length, presence of urgency keywords, negative keywords count
    sentiment_score = analyzer.polarity_scores(text)['compound']
    length = len(text)
    urgency_flag = 1 if any(u in text.lower() for u in URGENCY_PHRASES) else 0
    negative_count = sum(text.lower().count(kw) for kw in NEGATIVE_KEYWORDS)
    return [sentiment_score, length, urgency_flag, negative_count]

model = load_model()

def predict_risk(issue_text):
    if model:
        features = np.array(text_to_features(issue_text)).reshape(1, -1)
        prob = model.predict_proba(features)[0][1]
        return prob
    return 0.5  # default

def retrain_model():
    df = fetch_cases()
    X, y = [], []
    for _, row in df.iterrows():
        if row['feedback'] != 0:  # Use only feedback confirmed labels
            features = text_to_features(row['issue'])
            X.append(features)
            y.append(row['feedback'] if row['feedback'] == 1 else 0)
    if not X:
        return False
    new_model = LogisticRegression()
    new_model.fit(np.array(X), np.array(y))
    joblib.dump(new_model, MODEL_PATH)
    global model
    model = new_model
    return True

# Kanban UI with filters, escalated highlight, and counts
def display_kanban_card(row):
    escalated_style = "background-color:#ffcccc" if row['escalation_flag'] == 1 else ""
    st.markdown(f"<div style='padding:8px;{escalated_style}'>", unsafe_allow_html=True)
    st.markdown(f"**Sentiment:** {row['sentiment']} | **Urgency:** {row['urgency']} | **Category:** {row['category']}")
    st.write(row['issue'])
    action_taken = st.text_input("Action Taken", value=row['action_taken'], key=f"{row['escalation_id']}_action")
    action_owner = st.text_input("Action Owner", value=row['action_owner'], key=f"{row['escalation_id']}_owner")
    statuses = ["Open", "In Progress", "Resolved"]
    try:
        current_status_index = statuses.index(row['status'])
    except ValueError:
        current_status_index = 0
    new_status = st.selectbox("Update Status", statuses, index=current_status_index, key=f"{row['escalation_id']}_status")

    if st.button("Save", key=f"{row['escalation_id']}_save"):
        update_case(row['escalation_id'], new_status, action_taken, action_owner)
        st.success("Escalation updated")

    # Feedback buttons for user to improve model
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Confirm Escalation", key=f"{row['escalation_id']}_feedback_pos"):
            update_feedback(row['escalation_id'], 1)
            st.success("Feedback recorded: Confirmed Escalation")
    with col2:
        if st.button("Mark as False Positive", key=f"{row['escalation_id']}_feedback_neg"):
            update_feedback(row['escalation_id'], -1)
            st.success("Feedback recorded: False Positive")
    st.markdown("</div>", unsafe_allow_html=True)

def render_kanban(filter_escalated=False):
    df = fetch_cases()
    if filter_escalated:
        df = df[df['escalation_flag'] == 1]

    statuses = ["Open", "In Progress", "Resolved"]
    cols = st.columns(len(statuses))

    for i, status in enumerate(statuses):
        with cols[i]:
            filtered = df[df['status'] == status]
            count = filtered.shape[0]
            st.markdown(f"### {status} ({count})")
            if count == 0:
                st.write("No escalations")
            for _, row in filtered.iterrows():
                with st.expander(f"{row['escalation_id']} - {row['customer']}"):
                    display_kanban_card(row)

def get_all_complaints_df():
    # Get all complaints from DB for download
    df = fetch_cases()
    return df[['escalation_id', 'customer', 'issue', 'date', 'status', 'priority', 'category', 'predicted_risk']]

def main():
    st.title("🚨 EscalateAI – Customer Escalation Tracker")

    with st.sidebar:
        st.subheader("Upload Customer Issues")
        uploaded_file = st.file_uploader("Excel File (.xlsx)", type=["xlsx"])
        if uploaded_file is not None:
            process_excel(uploaded_file)
            st.success("Uploaded and processed Excel file.")

        st.subheader("Manual Entry")
        cust = st.text_input("Customer")
        iss = st.text_area("Issue")
        if st.button("Add"):
            process_case(cust.strip(), iss.strip())
            st.success("Manual escalation added.")

        st.markdown("---")
        if st.button("Parse New Emails"):
            parse_email()
            st.success("Email parsing completed.")

        st.markdown("---")
        if st.button("Update ML Predictions"):
            # Update predicted risk for all cases without prediction
            df = fetch_cases()
            for idx, row in df.iterrows():
                if row['predicted_risk'] is None:
                    risk = predict_risk(row['issue'])
                    cursor.execute("UPDATE escalations SET predicted_risk=? WHERE escalation_id=?", (risk, row['escalation_id']))
            conn.commit()
            st.success("Predictions updated.")

        st.markdown("---")
        if st.button("Retrain Model (using feedback)"):
            success = retrain_model()
            if success:
                st.success("Model retrained with user feedback.")
            else:
                st.info("No feedback data available for retraining.")

        st.markdown("---")
        st.subheader("Download Consolidated Complaints")
        complaints_df = get_all_complaints_df()
        towrite = BytesIO()
        complaints_df.to_excel(towrite, index=False, engine='openpyxl')
        towrite.seek(0)
        st.download_button(label="Download Complaints Excel", data=towrite, file_name="consolidated_complaints.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st_autorefresh(interval=60000, limit=None)  # Refresh every 60 seconds

    # SLA monitoring and alerting
    detect_sla_breach()

    # Filter toggle for escalated/all
    filter_escalated = st.checkbox("Show only escalated cases", value=False)

    st.subheader("Kanban Board")
    render_kanban(filter_escalated)

if __name__ == "__main__":
    main()
