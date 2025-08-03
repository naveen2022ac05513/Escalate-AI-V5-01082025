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

EMAIL = os.getenv("EMAIL_USER")           # Your email login
PASSWORD = os.getenv("EMAIL_PASS")        # Your email password or app password
IMAP_SERVER = os.getenv("EMAIL_SERVER")  # e.g. imap.gmail.com
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")

SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER")  # e.g. smtp.gmail.com
SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
ALERT_EMAIL_TO = os.getenv("EMAIL_RECEIVER")  # Recipient of alert emails

DB_PATH = "escalations.db"

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Keywords and categories
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

# Setup SQLite database connection and schema
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
        INSERT INTO escalations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()

def fetch_cases():
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    return df

# Email alert function
def send_email_alert(subject, message):
    if not all([SMTP_SERVER, SMTP_PORT, EMAIL, PASSWORD, ALERT_EMAIL_TO]):
        st.warning("Email alert settings incomplete in environment variables.")
        return
    try:
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = EMAIL
        msg['To'] = ALERT_EMAIL_TO

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL, PASSWORD)
        server.sendmail(EMAIL, ALERT_EMAIL_TO, msg.as_string())
        server.quit()
    except Exception as e:
        st.warning(f"Failed to send email alert: {e}")

# Teams alert helper
def send_teams_alert(msg):
    if MS_TEAMS_WEBHOOK_URL:
        try:
            response = requests.post(MS_TEAMS_WEBHOOK_URL, json={"text": msg})
            if response.status_code != 200:
                st.warning(f"Failed to send Teams alert: {response.status_code}")
        except Exception as e:
            st.warning(f"Exception while sending Teams alert: {e}")

# SLA breach detection & alerts
def detect_sla_breach(send_email=False, send_teams=False):
    now = datetime.datetime.now()
    df = fetch_cases()
    breaches = []
    for _, row in df.iterrows():
        try:
            created_dt = datetime.datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
        if row.get("priority", "") == "High" and row.get("status", "") != "Resolved":
            if (now - created_dt).total_seconds() > 600:  # 10 minutes
                breaches.append(row)
    # Send alerts if any breaches
    for breach in breaches:
        msg = f"SLA Breach: Escalation {breach['escalation_id']} from {breach['customer']} still open over 10 minutes.\nIssue: {breach['issue']}"
        if send_teams:
            send_teams_alert(msg)
        if send_email:
            send_email_alert("EscalateAI SLA Breach Alert", msg)

# Email Parsing function
def parse_email():
    if not (EMAIL and PASSWORD and IMAP_SERVER):
        st.warning("Email credentials or server not set in environment variables.")
        return
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL, PASSWORD)
        mail.select("inbox")
        status, messages = mail.search(None, 'UNSEEN')
        if status != "OK":
            st.warning("No new emails found or could not fetch emails.")
            return
        for num in messages[0].split():
            _, data = mail.fetch(num, '(RFC822)')
            msg = email.message_from_bytes(data[0][1])
            subject = decode_header(msg["Subject"])[0][0]
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

# Process Excel upload
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

# Process each case
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
    predicted_risk = None  # Placeholder for ML prediction

    data = (eid, customer, issue, now_str, "Open", sentiment, priority, flag,
            urgency, category, "", "", now_str, predicted_risk)
    insert_to_db(data)

    if flag and priority == "High":
        send_teams_alert(f"New Escalation: {eid} from {customer}\nIssue: {issue}")

# ML prediction stub placeholder
def predict_risk(issue_text):
    return 0.5  # Placeholder

def update_predictions():
    df = fetch_cases()
    for _, row in df.iterrows():
        if pd.isna(row['predicted_risk']):
            risk = predict_risk(row['issue'])
            cursor.execute("UPDATE escalations SET predicted_risk=? WHERE escalation_id=?", (risk, row['escalation_id']))
    conn.commit()

# Render Kanban cards
def display_kanban_card(row):
    st.markdown(f"**Sentiment:** {row.get('sentiment', 'N/A')} | **Urgency:** {row.get('urgency', 'N/A')} | **Category:** {row.get('category', 'N/A')}")
    st.write(row.get('issue', ''))

    action_taken = st.text_input("Action Taken", value=row.get('action_taken', ''), key=f"{row['escalation_id']}_action")
    action_owner = st.text_input("Action Owner", value=row.get('action_owner', ''), key=f"{row['escalation_id']}_owner")

    statuses = ["Open", "In Progress", "Resolved"]
    try:
        current_index = statuses.index(row.get('status', 'Open'))
    except ValueError:
        current_index = 0

    new_status = st.selectbox("Update Status", statuses, index=current_index, key=f"{row['escalation_id']}_status")

    if st.button("Save", key=f"{row['escalation_id']}_save"):
        cursor.execute("""
            UPDATE escalations SET status=?, action_taken=?, action_owner=?, status_update_date=?
            WHERE escalation_id=?
        """, (new_status, action_taken, action_owner, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), row['escalation_id']))
        conn.commit()
        st.success("Escalation updated")

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
                st.write("No escalations")
            for _, row in filtered.iterrows():
                with st.expander(f"{row['escalation_id']} - {row['customer']}"):
                    display_kanban_card(row)

# Prepare consolidated complaints dataframe
def get_all_complaints_df():
    df = fetch_cases()
    # Select relevant columns
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
        st.subheader("SLA Breach Alerts")
        send_teams = st.checkbox("Send MS Teams Alert on SLA Breach")
        send_email = st.checkbox("Send Email Alert on SLA Breach")
        if st.button("Check SLA Breach Now"):
            detect_sla_breach(send_email=send_email, send_teams=send_teams)
            st.success("SLA breach alerts sent if any.")

        st.markdown("---")
        st.subheader("Download Consolidated Complaints")
        complaints_df = get_all_complaints_df()
        if not complaints_df.empty:
            towrite = pd.ExcelWriter("complaints.xlsx", engine='openpyxl')
            complaints_df.to_excel(towrite, index=False)
            towrite.save()
            with open("complaints.xlsx", "rb") as f:
                btn = st.download_button(
                    label="Download Complaints Excel",
                    data=f,
                    file_name="complaints.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    st_autorefresh(interval=60000, limit=None)  # refresh every 60s

    st.subheader("Kanban Board")
    filter_opt = st.radio("Filter escalations to show:", options=["All", "Escalated Only"], index=0)
    render_kanban(filter_escalated=(filter_opt == "Escalated Only"))

if __name__ == "__main__":
    main()
