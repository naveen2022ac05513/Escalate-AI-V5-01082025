import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import requests
import base64
import smtplib
from email.mime.text import MIMEText
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load environment variables
load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", 587))
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

DB_FILE = "escalations.db"

NEGATIVE_WORDS = [
    "fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip", "malfunction", "blank",
    "shutdown", "discharge", "dissatisfy", "frustrate", "complain", "reject", "delay", "ignore", "escalate",
    "displease", "noncompliance", "neglect", "wait", "pending", "slow", "incomplete", "miss", "omit",
    "unresolved", "shortage", "no response", "fire", "burn", "flashover", "arc", "explode", "unsafe", "leak",
    "corrode", "alarm", "incident", "impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate", "penalty"
]

analyzer = SentimentIntensityAnalyzer()

# Database setup and initialization
def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute('''CREATE TABLE IF NOT EXISTS escalations (
        id TEXT PRIMARY KEY,
        customer TEXT,
        issue TEXT,
        sentiment REAL,
        urgency TEXT,
        severity TEXT,
        criticality TEXT,
        category TEXT,
        status TEXT,
        timestamp TEXT,
        action_taken TEXT,
        owner TEXT,
        escalated INTEGER
    )''')
    conn.close()

def generate_id():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM escalations")
    count = cur.fetchone()[0]
    conn.close()
    return f"SESICE-{250000 + count + 1}"

# Parse unseen emails from inbox
def fetch_emails():
    try:
        mail = imaplib.IMAP4_SSL(EMAIL_SERVER)
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select("inbox")
        _, search_data = mail.search(None, "UNSEEN")
        messages = []
        for num in search_data[0].split():
            _, data = mail.fetch(num, "(RFC822)")
            msg = email.message_from_bytes(data[0][1])
            subject = decode_header(msg["Subject"])[0][0]
            if isinstance(subject, bytes):
                subject = subject.decode(errors='ignore')
            from_ = msg.get("From")
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition")):
                        body = part.get_payload(decode=True).decode(errors='ignore')
                        break
            else:
                body = msg.get_payload(decode=True).decode(errors='ignore')
            messages.append((from_, subject, body))
        mail.logout()
        return messages
    except Exception as e:
        st.error(f"Error fetching emails: {e}")
        return []

def analyze_text(issue):
    sentiment_score = analyzer.polarity_scores(issue)["compound"]
    urgency = "High" if any(word in issue.lower() for word in NEGATIVE_WORDS) else "Normal"
    severity = "High" if sentiment_score < -0.5 else "Medium" if sentiment_score < 0 else "Low"
    criticality = "Critical" if urgency == "High" and severity == "High" else "Moderate"
    category = "Technical" if any(w in issue.lower() for w in ["fail", "crash", "malfunction"]) else "General"
    return sentiment_score, urgency, severity, criticality, category

def save_escalation(customer, issue):
    sentiment, urgency, severity, criticality, category = analyze_text(issue)
    entry_id = generate_id()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    escalated = 1 if urgency == "High" and severity == "High" else 0
    conn = sqlite3.connect(DB_FILE)
    conn.execute("INSERT OR IGNORE INTO escalations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (entry_id, customer, issue, sentiment, urgency, severity, criticality, category,
         "Open", timestamp, "", "", escalated))
    conn.commit()
    conn.close()
    return entry_id, urgency, severity

def send_teams_alert(message):
    if not MS_TEAMS_WEBHOOK_URL:
        st.warning("MS Teams webhook URL not configured.")
        return
    try:
        res = requests.post(MS_TEAMS_WEBHOOK_URL, json={"text": message})
        if res.status_code != 200:
            st.error(f"Failed to send Teams alert: {res.status_code}")
    except Exception as e:
        st.error(f"Error sending Teams alert: {e}")

def send_email_alert(subject, message):
    if not (EMAIL_SMTP_SERVER and EMAIL_RECEIVER):
        st.warning("SMTP server or email receiver not configured.")
        return
    try:
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = EMAIL_USER
        msg['To'] = EMAIL_RECEIVER
        with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, EMAIL_RECEIVER, msg.as_string())
    except Exception as e:
        st.error(f"Error sending email alert: {e}")

def load_data():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    conn.close()
    return df

def update_escalation_status(id_, status, action_taken, owner):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE escalations SET status=?, action_taken=?, owner=? WHERE id=?",
                 (status, action_taken, owner, id_))
    conn.commit()
    conn.close()

def process_email_fetch_and_save():
    emails = fetch_emails()
    count = 0
    for from_, subject, body in emails:
        esc_id, urgency, severity = save_escalation(from_, body)
        if urgency == "High":
            send_teams_alert(f"🚨 Escalation Alert\nID: {esc_id}\nCustomer: {from_}\nIssue: {body[:200]}...")
            send_email_alert("Escalation Alert", f"ID: {esc_id}\nCustomer: {from_}\nIssue: {body}")
        count += 1
    return count

def process_excel_upload(file):
    try:
        df = pd.read_excel(file)
        df.columns = [c.lower().strip() for c in df.columns]
        cust_col = next((c for c in df.columns if "customer" in c or "email" in c), None)
        issue_col = next((c for c in df.columns if "issue" in c or "complaint" in c or "text" in c), None)
        if not cust_col or not issue_col:
            st.error("Excel must contain Customer and Issue columns.")
            return 0
        count = 0
        for _, row in df.iterrows():
            customer = str(row[cust_col])
            issue = str(row[issue_col])
            esc_id, urgency, severity = save_escalation(customer, issue)
            if urgency == "High":
                send_teams_alert(f"🚨 Escalation Alert\nID: {esc_id}\nCustomer: {customer}\nIssue: {issue[:200]}...")
                send_email_alert("Escalation Alert", f"ID: {esc_id}\nCustomer: {customer}\nIssue: {issue}")
            count += 1
        return count
    except Exception as e:
        st.error(f"Error processing Excel upload: {e}")
        return 0

def save_excel_download(df):
    filename = "EscalateAI_Combined_Complaints.xlsx"
    df.to_excel(filename, index=False)
    return filename

def check_sla_and_alert():
    df = load_data()
    now = datetime.datetime.now()
    alert_count = 0
    for _, row in df.iterrows():
        if row['urgency'] == "High" and row['status'] == "Open":
            timestamp = datetime.datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S")
            diff = now - timestamp
            if diff.total_seconds() > 600:  # 10 minutes
                msg = f"⚠️ SLA Breach Alert:\nID: {row['id']}\nCustomer: {row['customer']}\nOpen for {int(diff.total_seconds()//60)} minutes\nIssue: {row['issue'][:200]}..."
                send_teams_alert(msg)
                send_email_alert("SLA Breach Alert", msg)
                alert_count += 1
    return alert_count

# Basic ML model for escalation prediction (dummy example)
def train_predictive_model():
    df = load_data()
    if df.empty or df.shape[0] < 20:
        return None
    le_status = LabelEncoder()
    df['status_encoded'] = le_status.fit_transform(df['status'])
    df['urgency_bin'] = df['urgency'].apply(lambda x: 1 if x == "High" else 0)
    X = df[['sentiment', 'urgency_bin']]
    y = df['escalated']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    return clf

def predict_escalation(issue, model):
    if not model:
        return 0
    sentiment_score = analyzer.polarity_scores(issue)["compound"]
    urgency_bin = 1 if any(word in issue.lower() for word in NEGATIVE_WORDS) else 0
    pred = model.predict([[sentiment_score, urgency_bin]])
    return int(pred[0])

def display_kanban_card(row):
    priority_color = "#c0392b" if row['urgency'] == "High" else "#27ae60"
    status_color_map = {"Open": "#f1c40f", "In Progress": "#2980b9", "Resolved": "#2ecc71"}
    status_color = status_color_map.get(row['status'], "#7f8c8d")
    escalated_mark = "🔥" if row['escalated'] else ""
    st.markdown(f"""
        <div style="border-left: 6px solid {priority_color}; padding-left: 8px; margin-bottom:10px;">
            <b>{row['id']} {escalated_mark}</b><br>
            <small>Status: <span style='color:{status_color}'>{row['status']}</span> | Severity: {row['severity']} | Criticality: {row['criticality']}</small><br>
            <small>Customer: {row['customer']}</small><br>
            <small>Issue: {row['issue'][:250]}...</small>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("Update Details"):
        new_status = st.selectbox("Status", ["Open", "In Progress", "Resolved"], index=["Open", "In Progress", "Resolved"].index(row['status']), key=f"status_{row['id']}")
        new_action = st.text_area("Action Taken", value=row['action_taken'] or "", key=f"action_{row['id']}")
        new_owner = st.text_input("Owner", value=row['owner'] or "", key=f"owner_{row['id']}")
        if st.button("Save Updates", key=f"save_{row['id']}"):
            update_escalation_status(row['id'], new_status, new_action, new_owner)
            st.experimental_rerun()

def main():
    st.title("🚀 EscalateAI - AI-Powered Escalation Management")
    init_db()

    # Sidebar
    st.sidebar.header("Controls & Filters")

    # Manual entry
    st.sidebar.subheader("Manual Escalation Entry")
    manual_customer = st.sidebar.text_input("Customer Email/Name")
    manual_issue = st.sidebar.text_area("Issue / Complaint")
    if st.sidebar.button("Add Manual Escalation"):
        if manual_customer.strip() == "" or manual_issue.strip() == "":
            st.sidebar.error("Both customer and issue must be filled.")
        else:
            esc_id, urgency, severity = save_escalation(manual_customer, manual_issue)
            if urgency == "High":
                send_teams_alert(f"🚨 Escalation Alert\nID: {esc_id}\nCustomer: {manual_customer}\nIssue: {manual_issue[:200]}...")
                send_email_alert("Escalation Alert", f"ID: {esc_id}\nCustomer: {manual_customer}\nIssue: {manual_issue}")
            st.sidebar.success(f"Escalation {esc_id} added.")

    # Excel upload
    st.sidebar.subheader("Upload Complaints Excel")
    uploaded_file = st.sidebar.file_uploader("Select Excel file", type=["xlsx", "xls"])
    if uploaded_file is not None:
        count = process_excel_upload(uploaded_file)
        st.sidebar.success(f"Processed {count} escalations from upload.")

    # Fetch emails button
    if st.sidebar.button("Fetch & Process New Emails"):
        count = process_email_fetch_and_save()
        st.sidebar.success(f"Fetched and processed {count} new emails.")

    # Download all complaints
    if st.sidebar.button("Download All Complaints as Excel"):
        df_all = load_data()
        filename = save_excel_download(df_all)
        with open(filename, "rb") as f:
            btn = st.sidebar.download_button(
                label="Download Excel",
                data=f,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # SLA alert button
    if st.sidebar.button("Check SLA Breaches and Send Alerts"):
        alert_count = check_sla_and_alert()
        st.sidebar.success(f"Sent alerts for {alert_count} SLA breaches.")

    # Filters
    st.sidebar.subheader("Filter Escalations")
    status_filter = st.sidebar.multiselect("Status", ["Open", "In Progress", "Resolved"], default=["Open", "In Progress", "Resolved"])
    show_only_escalated = st.sidebar.checkbox("Show Only Escalated")

    # Load and filter data
    df = load_data()
    if show_only_escalated:
        df = df[df['escalated'] == 1]
    if status_filter:
        df = df[df['status'].isin(status_filter)]
    else:
        df = df.iloc[0:0]

    # Show counts
    counts = load_data()['status'].value_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("Open", counts.get("Open", 0))
    col2.metric("In Progress", counts.get("In Progress", 0))
    col3.metric("Resolved", counts.get("Resolved", 0))

    # Predictive Model Section
    st.subheader("Escalation Prediction (using ML)")
    model = train_predictive_model()
    new_issue = st.text_area("Enter new issue text to predict escalation risk")
    if st.button("Predict Escalation Risk"):
        if new_issue.strip() == "":
            st.warning("Please enter issue text.")
        else:
            prediction = predict_escalation(new_issue, model)
            st.info("Predicted Escalation Risk: 🔥 High" if prediction == 1 else "Low")

    # Display Kanban board
    st.subheader("Kanban Board")
    for status in ["Open", "In Progress", "Resolved"]:
        st.markdown(f"### {status} ({counts.get(status,0)})")
        sub_df = df[df['status'] == status]
        if sub_df.empty:
            st.write("No escalations.")
        else:
            for _, row in sub_df.iterrows():
                display_kanban_card(row)

if __name__ == "__main__":
    main()
