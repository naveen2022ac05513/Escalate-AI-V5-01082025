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
from sklearn.ensemble import RandomForestClassifier
import pickle
import base64

# Load environment variables
load_dotenv()
EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# Database file and model file
DB_FILE = "escalations.db"
MODEL_FILE = "escalation_model.pkl"

NEGATIVE_KEYWORDS = [
    # ⚙️ Technical Failures & Product Malfunction
    "fail", "break", "crash", "defect", "fault", "degrade", "damage",
    "trip", "malfunction", "blank", "shutdown", "discharge",
    # 💢 Customer Dissatisfaction & Escalations
    "dissatisfy", "frustrate", "complain", "reject", "delay", "ignore",
    "escalate", "displease", "noncompliance", "neglect",
    # ⏳ Support Gaps & Operational Delays
    "wait", "pending", "slow", "incomplete", "miss", "omit",
    "unresolved", "shortage", "no response",
    # 💥 Hazardous Conditions & Safety Risks
    "fire", "burn", "flashover", "arc", "explode", "unsafe",
    "leak", "corrode", "alarm", "incident",
    # 📉 Business Risk & Impact
    "impact", "loss", "risk", "downtime", "interrupt", "cancel",
    "terminate", "penalty"
]

# Initialize DB connection and cursor (thread-safe for Streamlit)
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

def init_db():
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
        severity TEXT,
        criticality TEXT,
        category TEXT,
        action_taken TEXT,
        action_owner TEXT,
        status_update_date TEXT,
        user_feedback TEXT
    )
    """)
    conn.commit()

# Generate escalation ID
def generate_escalation_id():
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]
    return f"SESICE-{250000 + count + 1}"

# NLP analysis (sentiment, urgency, severity, criticality, category)
def analyze_issue(issue_text):
    text_lower = issue_text.lower()
    vs = analyzer.polarity_scores(issue_text)
    sentiment = "Positive" if vs["compound"] >= 0 else "Negative"
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    priority = "High" if sentiment == "Negative" and neg_count >= 2 else "Low"
    escalation_flag = 1 if priority == "High" else 0

    # Tagging severity, criticality, category
    severity = "Critical" if priority == "High" else "Medium"
    criticality = "Urgent" if priority == "High" else "Routine"
    category = "Complaint" if sentiment == "Negative" else "Feedback"

    return sentiment, priority, escalation_flag, severity, criticality, category

# Insert new escalation
def insert_escalation(customer, issue, date=None):
    if date is None:
        date = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
    # Avoid duplicates
    cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (customer, issue[:500]))
    if cursor.fetchone():
        return None, None  # Already exists

    esc_id = generate_escalation_id()
    sentiment, priority, escalation_flag, severity, criticality, category = analyze_issue(issue)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")

    cursor.execute("""
        INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, severity, criticality, category, action_taken, action_owner, status_update_date, user_feedback)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (esc_id, customer, issue[:500], date, "Open", sentiment, priority, escalation_flag, severity, criticality, category, "", "", now, ""))
    conn.commit()
    return esc_id, escalation_flag

# Fetch all escalations as DataFrame
def load_escalations_df():
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    return df

# Fetch unseen emails from Gmail and parse
def fetch_gmail_emails():
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

# Process new emails and save to DB + send alerts if needed
def process_new_emails():
    emails = fetch_gmail_emails()
    new_count = 0
    for e in emails:
        esc_id, esc_flag = insert_escalation(e['customer'], e['issue'], e['date'])
        if esc_id:
            new_count += 1
            if esc_flag == 1:
                send_ms_teams_alert(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {e['customer']}\nIssue: {e['issue'][:200]}...")
    return new_count

# Upload Excel file and parse, save escalations
def upload_excel_and_analyze(file):
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
        for idx, row in df.iterrows():
            customer = str(row[customer_col])
            issue = str(row[issue_col])
            date = str(row[date_col]) if date_col and pd.notna(row[date_col]) else None
            esc_id, esc_flag = insert_escalation(customer, issue, date)
            if esc_id:
                count += 1
                if esc_flag == 1:
                    send_ms_teams_alert(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {customer}\nIssue: {issue[:200]}...")
        return count
    except Exception as e:
        st.error(f"Error processing uploaded Excel: {e}")
        return 0

# Send alert to MS Teams webhook
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

# Optional: send email alerts (simple SMTP)
def send_email_alert(subject, body):
    import smtplib
    from email.mime.text import MIMEText
    sender = EMAIL
    receiver = EMAIL_RECEIVER
    if not sender or not receiver:
        st.warning("Email sender or receiver not configured.")
        return
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = receiver
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL, APP_PASSWORD)
            server.sendmail(sender, receiver, msg.as_string())
    except Exception as e:
        st.error(f"Error sending email alert: {e}")

# Save complaints to Excel for download
def save_complaints_excel():
    df = load_escalations_df()
    filename = "EscalateAI_Complaints.xlsx"
    df.to_excel(filename, index=False)
    return filename

# Check SLA breaches for High priority Open escalations >10 minutes
def check_sla_and_alert():
    df = load_escalations_df()
    now = datetime.datetime.now(datetime.timezone.utc)
    breached = df[
        (df['priority'] == "High") &
        (df['status'] == "Open")
    ]

    alerts_sent = 0
    for _, row in breached.iterrows():
        try:
            last_update = datetime.datetime.strptime(row['status_update_date'], "%a, %d %b %Y %H:%M:%S %z")
        except Exception:
            continue
        elapsed = now - last_update
        if elapsed.total_seconds() > 10 * 60:
            alert_msg = (
                f"⚠️ SLA breach detected:\nID: {row['escalation_id']}\n"
                f"Customer: {row['customer']}\nOpen for: {elapsed.seconds // 60} minutes\n"
                f"Issue: {row['issue'][:200]}..."
            )
            send_ms_teams_alert(alert_msg)
            send_email_alert("EscalateAI SLA Breach Alert", alert_msg)
            alerts_sent += 1
    return alerts_sent

# ML model training
def train_ml_model():
    df = load_escalations_df()
    if df.empty or 'escalation_flag' not in df.columns:
        st.warning("Not enough data to train model.")
        return None
    # Simple feature: length of issue, sentiment encoded, priority encoded
    df = df.copy()
    df['issue_len'] = df['issue'].apply(len)
    df['sentiment_enc'] = df['sentiment'].map({"Positive":0, "Negative":1}).fillna(0)
    df['priority_enc'] = df['priority'].map({"Low":0, "High":1}).fillna(0)
    X = df[['issue_len', 'sentiment_enc', 'priority_enc']]
    y = df['escalation_flag']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    st.success("ML Model trained and saved.")
    return model

# Load ML model
def load_ml_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    return None

# Predict escalation with ML model
def predict_escalation(issue_text, model=None):
    if not model:
        model = load_ml_model()
    if not model:
        return 0
    sentiment_score = analyzer.polarity_scores(issue_text)['compound']
    sentiment_enc = 0 if sentiment_score >= 0 else 1
    issue_len = len(issue_text)
    priority = "High" if sentiment_enc == 1 and any(kw in issue_text.lower() for kw in NEGATIVE_KEYWORDS) else "Low"
    priority_enc = 1 if priority == "High" else 0
    X_test = [[issue_len, sentiment_enc, priority_enc]]
    pred = model.predict(X_test)[0]
    return int(pred)

# Download button helper
def get_table_download_link(df, filename="data.xlsx"):
    towrite = pd.ExcelWriter(filename, engine='openpyxl')
    df.to_excel(towrite, index=False)
    towrite.save()
    with open(filename, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Display Kanban card for a single escalation
def display_kanban_card(row):
    esc_id = row.get('escalation_id') or row.get('id') or "Unknown ID"
    status = row.get('status', "Open")
    priority = row.get('priority', "Low")
    escalation_flag = row.get('escalation_flag', 0)
    sentiment = row.get('sentiment', "Unknown")
    customer = row.get('customer', "Unknown")
    issue = row.get('issue', "")
    severity = row.get('severity', "Medium")
    criticality = row.get('criticality', "Routine")
    category = row.get('category', "Feedback")
    action_taken = row.get('action_taken', "")
    action_owner = row.get('action_owner', "")
    status_update_date = row.get('status_update_date', "")
    user_feedback = row.get('user_feedback', "")

    header = f"{esc_id} - {customer} - {status}"
    if escalation_flag:
        header += " 🚩Escalated"

    with st.expander(header):
        st.markdown(f"**Issue:** {issue}")
        st.markdown(f"**Sentiment:** {sentiment}")
        st.markdown(f"**Priority:** {priority}")
        st.markdown(f"**Severity:** {severity}")
        st.markdown(f"**Criticality:** {criticality}")
        st.markdown(f"**Category:** {category}")
        st.markdown(f"**Last Update:** {status_update_date}")
        st.markdown(f"**Action Taken:**")
        new_action = st.text_area(f"Action Taken for {esc_id}", value=action_taken, key=f"action_{esc_id}")
        st.markdown(f"**Action Owner:**")
        new_owner = st.text_input(f"Action Owner for {esc_id}", value=action_owner, key=f"owner_{esc_id}")
        st.markdown(f"**User Feedback:**")
        new_feedback = st.text_area(f"Feedback for {esc_id}", value=user_feedback, key=f"feedback_{esc_id}")

        # Status update selectbox
        new_status = st.selectbox(f"Status for {esc_id}", options=["Open", "In Progress", "Resolved", "Escalated"], index=["Open", "In Progress", "Resolved", "Escalated"].index(status), key=f"status_{esc_id}")

        # Save updates
        if st.button(f"Save Updates for {esc_id}", key=f"save_{esc_id}"):
            cursor.execute("""
                UPDATE escalations SET
                    action_taken = ?,
                    action_owner = ?,
                    user_feedback = ?,
                    status = ?,
                    status_update_date = ?
                WHERE escalation_id = ?
            """, (new_action, new_owner, new_feedback, new_status, datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z"), esc_id))
            conn.commit()
            st.success(f"Updated escalation {esc_id}")
            st.experimental_rerun()

# Main UI
def main():
    st.title("EscalateAI - Customer Escalation Management")

    init_db()

    st.sidebar.title("EscalateAI Controls & Upload")

    # Email processing button
    if st.sidebar.button("Fetch and Process New Emails"):
        new_emails = process_new_emails()
        st.sidebar.success(f"Processed {new_emails} new emails.")

    # Excel Upload
    uploaded_file = st.sidebar.file_uploader("Upload Customer Issues Excel", type=["xls", "xlsx"])
    if uploaded_file is not None:
        count = upload_excel_and_analyze(uploaded_file)
        st.sidebar.success(f"Processed {count} records from Excel.")

    # Download consolidated complaints
    if st.sidebar.button("Download Consolidated Complaints"):
        filename = save_complaints_excel()
        with open(filename, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download Complaints Excel</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)

    # SLA Alert button
    if st.sidebar.button("Check SLA Breaches & Send Alerts"):
        alert_count = check_sla_and_alert()
        st.sidebar.success(f"Sent {alert_count} SLA breach alerts.")

    # Train ML model
    if st.sidebar.button("Train ML Escalation Prediction Model"):
        train_ml_model()

    # Filters
    df = load_escalations_df()

    status_filter = st.sidebar.multiselect("Filter by Status", options=["Open", "In Progress", "Resolved", "Escalated"], default=["Open", "In Progress", "Resolved", "Escalated"])
    priority_filter = st.sidebar.multiselect("Filter by Priority", options=["High", "Low"], default=["High", "Low"])
    escalated_only = st.sidebar.checkbox("Show Only Escalated Cases")

    filtered_df = df[df['status'].isin(status_filter) & df['priority'].isin(priority_filter)]
    if escalated_only:
        filtered_df = filtered_df[filtered_df['escalation_flag'] == 1]

    # Show counts
    counts = df.groupby("status").size().to_dict()
    st.sidebar.markdown("### Status Counts")
    for status in ["Open", "In Progress", "Resolved", "Escalated"]:
        st.sidebar.markdown(f"{status}: {counts.get(status, 0)}")

    # Show Kanban cards
    st.header("Escalations Kanban Board")
    if filtered_df.empty:
        st.info("No escalations found for selected filters.")
    else:
        for idx, row in filtered_df.iterrows():
            display_kanban_card(row)

    # Show model prediction for a test input
    st.sidebar.markdown("---")
    st.sidebar.header("Predict Escalation (Test)")
    test_issue = st.sidebar.text_area("Enter Issue Text for Prediction", value="", height=80)
    if st.sidebar.button("Predict Escalation"):
        model = load_ml_model()
        pred = predict_escalation(test_issue, model)
        st.sidebar.markdown(f"Predicted Escalation Flag: {'Yes' if pred == 1 else 'No'}")

if __name__ == "__main__":
    main()
