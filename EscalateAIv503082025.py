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

# Load environment variables
load_dotenv()
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
IMAP_SERVER = "imap.gmail.com"
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")
DB_PATH = "escalations.db"
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

# Sentiment analyzer init
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

# DB init
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

# Utils
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
        INSERT INTO escalations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()

def fetch_cases():
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    return df

# Alerts
def send_teams_alert(msg):
    if MS_TEAMS_WEBHOOK_URL:
        try:
            response = requests.post(MS_TEAMS_WEBHOOK_URL, json={"text": msg})
            if response.status_code != 200:
                st.warning(f"Failed to send Teams alert: {response.status_code}")
        except Exception as e:
            st.warning(f"Exception while sending Teams alert: {e}")

def send_email_alert(subject, body, to_emails):
    if not all([SMTP_SERVER, SMTP_PORT, SMTP_EMAIL, SMTP_PASSWORD]):
        st.warning("SMTP Email credentials not fully configured.")
        return
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = SMTP_EMAIL
        msg['To'] = ", ".join(to_emails)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.sendmail(SMTP_EMAIL, to_emails, msg.as_string())
    except Exception as e:
        st.warning(f"Error sending email alert: {e}")

# SLA breach detection & alert
def detect_sla_breach(send_alerts=True):
    now = datetime.datetime.now()
    df = fetch_cases()
    breaches = []
    for idx, row in df.iterrows():
        try:
            created_dt = datetime.datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
        if row.get("priority", "") == "High" and row.get("status", "") != "Resolved":
            if (now - created_dt).total_seconds() > 600:
                breaches.append(row)
                if send_alerts:
                    msg = f"SLA Breach: Escalation {row['escalation_id']} still open for over 10 minutes\nCustomer: {row['customer']}\nIssue: {row['issue']}"
                    send_teams_alert(msg)
                    alert_emails = [SMTP_EMAIL]  # update to your recipients
                    send_email_alert("EscalateAI SLA Breach Alert", msg, alert_emails)
    return breaches

# Email parsing
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

# Excel processing
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

# Core processing
def process_case(customer, issue):
    if not customer or not issue:
        return
    eid = generate_id()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sentiment = classify_sentiment(issue)
    priority = "High" if sentiment == "Negative" else "Normal"
    urgency = detect_urgency(issue)
    category = detect_category(issue)
    flag = 1 if is_escalation(issue) else 0
    predicted_risk = None
    
    data = (eid, customer, issue, now_str, "Open", sentiment, priority, flag,
            urgency, category, "", "", now_str, predicted_risk)
    insert_to_db(data)

    if flag and priority == "High":
        send_teams_alert(f"New Escalation: {eid} from {customer}\nIssue: {issue}")

# Dummy ML prediction stub
def predict_risk(issue_text):
    return 0.5

def update_predictions():
    df = fetch_cases()
    for idx, row in df.iterrows():
        if pd.isna(row['predicted_risk']):
            risk = predict_risk(row['issue'])
            cursor.execute("UPDATE escalations SET predicted_risk=? WHERE escalation_id=?", (risk, row['escalation_id']))
    conn.commit()

# Kanban card UI
def display_kanban_card(row):
    sentiment = row.get('sentiment', 'N/A')
    urgency = row.get('urgency', 'N/A')
    category = row.get('category', 'N/A')

    st.markdown(f"**Sentiment:** {sentiment} | **Urgency:** {urgency} | **Category:** {category}")
    st.write(row.get('issue', ''))
    action_taken = st.text_input("Action Taken", value=row.get('action_taken', ''), key=f"{row['escalation_id']}_action")
    action_owner = st.text_input("Action Owner", value=row.get('action_owner', ''), key=f"{row['escalation_id']}_owner")
    statuses = ["Open", "In Progress", "Resolved"]
    try:
        current_status_index = statuses.index(row.get('status', 'Open'))
    except ValueError:
        current_status_index = 0
    new_status = st.selectbox("Update Status", statuses, index=current_status_index, key=f"{row['escalation_id']}_status")

    if st.button("Save", key=f"{row['escalation_id']}_save"):
        cursor.execute("""
            UPDATE escalations SET status=?, action_taken=?, action_owner=?, status_update_date=?
            WHERE escalation_id=?
        """, (new_status, action_taken, action_owner, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), row['escalation_id']))
        conn.commit()
        st.success("Escalation updated")

# Kanban board UI
def render_kanban(filter_escalated=False):
    df = fetch_cases()
    if filter_escalated:
        df = df[df['escalation_flag'] == 1]
    statuses = ["Open", "In Progress", "Resolved"]
    cols = st.columns(len(statuses))

    for i, status in enumerate(statuses):
        with cols[i]:
            count = df[df['status'] == status].shape[0]
            st.markdown(f"### {status} ({count})")
            filtered = df[df['status'] == status]
            if filtered.empty:
                st.write("No escalations")
            for _, row in filtered.iterrows():
                with st.expander(f"{row['escalation_id']} - {row['customer']}"):
                    display_kanban_card(row)

# Concatenate all complaints dataframe for download
def get_all_complaints_df():
    df = fetch_cases()
    return df[['escalation_id', 'customer', 'issue', 'date', 'status', 'priority', 'category', 'predicted_risk']]

# Main app
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
            update_predictions()
            st.success("Predictions updated.")

        st.markdown("---")
        if st.button("Trigger SLA Breach Alerts (Email & Teams)"):
            breaches = detect_sla_breach(send_alerts=True)
            st.success(f"SLA breach alerts sent for {len(breaches)} case(s).")

        st.markdown("---")
        st.subheader("Download Consolidated Complaints")
        complaints_df = get_all_complaints_df()
        st.download_button(
            label="Download Complaints as Excel",
            data=complaints_df.to_excel(index=False, engine='openpyxl'),
            file_name='consolidated_complaints.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

        st.markdown("---")
        filter_escalated = st.checkbox("Show only escalated cases")

    st_autorefresh(interval=60000, limit=None)

    detect_sla_breach(send_alerts=False)

    st.subheader("Kanban Board")
    render_kanban(filter_escalated=filter_escalated)

if __name__ == "__main__":
    main()
