import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
import re
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import requests
import smtplib
from email.mime.text import MIMEText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import threading

# Load env vars
load_dotenv()
EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASS = os.getenv("SMTP_PASS")
ALERT_EMAIL_RECEIVER = os.getenv("ALERT_EMAIL_RECEIVER")

DB_FILE = "escalations.db"
MODEL_FILE = "escalate_predictor.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

# Load SpaCy model (small English)
nlp = spacy.load("en_core_web_sm")

# Sentiment Analyzer
vader = SentimentIntensityAnalyzer()

NEGATIVE_KEYWORDS = [
    "fail", "break", "crash", "defect", "fault", "degrade", "damage",
    "trip", "malfunction", "blank", "shutdown", "discharge",
    "dissatisfy", "frustrate", "complain", "reject", "delay", "ignore",
    "escalate", "displease", "noncompliance", "neglect",
    "wait", "pending", "slow", "incomplete", "miss", "omit",
    "unresolved", "shortage", "no response",
    "fire", "burn", "flashover", "arc", "explode", "unsafe",
    "leak", "corrode", "alarm", "incident",
    "impact", "loss", "risk", "downtime", "interrupt", "cancel",
    "terminate", "penalty"
]

# Initialize DB
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS escalations (
    escalation_id TEXT PRIMARY KEY,
    customer TEXT,
    issue TEXT,
    date TEXT,
    status TEXT,
    sentiment TEXT,
    urgency TEXT,
    severity TEXT,
    criticality TEXT,
    category TEXT,
    priority TEXT,
    escalation_flag INTEGER,
    action_taken TEXT,
    action_owner TEXT,
    status_update_date TEXT,
    user_feedback TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    escalation_id TEXT PRIMARY KEY,
    correct_priority TEXT,
    correct_severity TEXT,
    correct_criticality TEXT,
    correct_category TEXT,
    timestamp TEXT
)
""")

conn.commit()

# Helper to generate unique escalation IDs
def generate_escalation_id():
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]
    return f"SESICE-{250000 + count + 1}"

# NLP Tagging functions
def spacy_classify_issue(issue):
    """ Use simple rule-based or ML classification to assign severity, criticality, category """
    doc = nlp(issue.lower())

    # Simple rules: just example heuristics - you can replace with custom ML model
    severity = "Medium"
    criticality = "Routine"
    category = "General"

    # Severity
    if any(tok.lemma_ in ["urgent", "immediate", "critical", "fail", "error", "break"] for tok in doc):
        severity = "Critical"
    elif any(tok.lemma_ in ["delay", "slow", "issue"] for tok in doc):
        severity = "High"

    # Criticality
    if any(tok.lemma_ in ["shutdown", "down", "stop", "fail", "fault"] for tok in doc):
        criticality = "High"
    elif any(tok.lemma_ in ["warning", "alert"] for tok in doc):
        criticality = "Medium"

    # Category
    if "billing" in issue.lower():
        category = "Billing"
    elif "technical" in issue.lower() or "error" in issue.lower() or "bug" in issue.lower():
        category = "Technical"
    elif "service" in issue.lower() or "support" in issue.lower():
        category = "Service"
    elif "delivery" in issue.lower():
        category = "Delivery"

    return severity, criticality, category

def analyze_issue(issue):
    # VADER sentiment
    vs = vader.polarity_scores(issue)
    sentiment = "Positive" if vs["compound"] >= 0 else "Negative"

    # Urgency based on keywords count
    text_lower = issue.lower()
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    urgency = "High" if neg_count >= 2 else "Normal"

    # SpaCy tags
    severity, criticality, category = spacy_classify_issue(issue)

    # Priority logic
    priority = "High" if sentiment == "Negative" and urgency == "High" else "Low"
    escalation_flag = 1 if priority == "High" else 0

    return sentiment, urgency, severity, criticality, category, priority, escalation_flag

# Save escalation to DB
def save_escalation(customer, issue, date=None):
    esc_id = generate_escalation_id()
    sentiment, urgency, severity, criticality, category, priority, escalation_flag = analyze_issue(issue)
    date = date or datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
    now = date
    cursor.execute("""
        INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, urgency, severity, criticality, category, priority, escalation_flag,
                                action_taken, action_owner, status_update_date, user_feedback)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (esc_id, customer, issue[:1000], date, "Open", sentiment, urgency, severity, criticality, category, priority, escalation_flag,
          "", "", now, ""))
    conn.commit()
    # Alert if high priority
    if escalation_flag:
        send_ms_teams_alert(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {customer}\nIssue: {issue[:200]}...")
    return esc_id

# Email Parsing
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

        for eid in email_ids[-10:]:
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

            emails.append({"customer": from_, "issue": body.strip(), "subject": subject, "date": date})
            mail.store(eid, '+FLAGS', '\\Seen')

        mail.logout()
        return emails
    except Exception as e:
        st.error(f"Error fetching emails: {e}")
        return []

# Excel upload parsing
def upload_excel(file):
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
        for _, row in df.iterrows():
            customer = str(row[customer_col])
            issue = str(row[issue_col])
            date = str(row[date_col]) if date_col and pd.notna(row[date_col]) else None
            cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (customer, issue[:1000]))
            if cursor.fetchone():
                continue
            save_escalation(customer, issue, date)
            count += 1
        return count
    except Exception as e:
        st.error(f"Excel upload error: {e}")
        return 0

# MS Teams Alert
def send_ms_teams_alert(message):
    if not MS_TEAMS_WEBHOOK_URL:
        st.warning("MS Teams webhook URL not set; cannot send alerts.")
        return
    headers = {"Content-Type": "application/json"}
    payload = {"text": message}
    try:
        resp = requests.post(MS_TEAMS_WEBHOOK_URL, json=payload, headers=headers)
        if resp.status_code != 200:
            st.error(f"MS Teams alert failed: {resp.status_code} {resp.text}")
    except Exception as e:
        st.error(f"Error sending MS Teams alert: {e}")

# Email Alert
def send_email_alert(subject, body):
    if not (SMTP_SERVER and SMTP_EMAIL and SMTP_PASS and ALERT_EMAIL_RECEIVER):
        st.warning("SMTP email config not set; cannot send email alerts.")
        return
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = SMTP_EMAIL
        msg["To"] = ALERT_EMAIL_RECEIVER
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASS)
            server.sendmail(SMTP_EMAIL, ALERT_EMAIL_RECEIVER, msg.as_string())
    except Exception as e:
        st.error(f"Error sending email alert: {e}")

# Load Escalations as dataframe
@st.cache_data(ttl=60)
def load_escalations_df():
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    return df

# Display kanban card
def display_kanban_card(row):
    esc_id = row['escalation_id']
    sentiment = row['sentiment']
    priority = row['priority']
    status = row['status']
    severity = row['severity']
    criticality = row['criticality']
    category = row['category']

    # Colors
    sentiment_colors = {"Positive": "#27ae60", "Negative": "#e74c3c"}
    priority_colors = {"High": "#c0392b", "Low": "#27ae60"}
    status_colors = {"Open": "#f1c40f", "In Progress": "#2980b9", "Resolved": "#2ecc71"}

    border_color = priority_colors.get(priority, "#000")
    status_color = status_colors.get(status, "#bdc3c7")
    sentiment_color = sentiment_colors.get(sentiment, "#7f8c8d")

    header_html = f"""
    <div style="
        border-left: 6px solid {border_color};
        padding-left: 10px;
        margin-bottom: 10px;
        font-weight:bold;">
        {esc_id} &nbsp; 
        <span style='color:{sentiment_color}; font-weight:bold;'>● {sentiment}</span> / 
        <span style='color:{priority_colors.get(priority, '#000')}; font-weight:bold;'>■ {priority}</span> / 
        <span style='color:{status_color}; font-weight:bold;'>{status}</span>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    with st.expander("Details", expanded=False):
        st.markdown(f"**Customer:** {row['customer']}")
        st.markdown(f"**Issue:** {row['issue']}")
        st.markdown(f"**Date:** {row['date']}")
        st.markdown(f"**Severity:** {severity}")
        st.markdown(f"**Criticality:** {criticality}")
        st.markdown(f"**Category:** {category}")

        new_status = st.selectbox(
            "Update Status",
            ["Open", "In Progress", "Resolved"],
            index=["Open", "In Progress", "Resolved"].index(status),
            key=f"{esc_id}_status"
        )
        new_action_taken = st.text_area(
            "Action Taken",
            value=row['action_taken'] or "",
            key=f"{esc_id}_action"
        )
        new_action_owner = st.text_input(
            "Action Owner",
            value=row['action_owner'] or "",
            key=f"{esc_id}_owner"
        )

        if st.button("Save Updates", key=f"save_{esc_id}"):
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("""
                UPDATE escalations SET status=?, action_taken=?, action_owner=?, status_update_date=?
                WHERE escalation_id=?
            """, (new_status, new_action_taken, new_action_owner, now, esc_id))
            conn.commit()
            st.success("Updated successfully!")
            st.experimental_rerun()

# Save escalations Excel for download
def save_escalations_excel():
    df = load_escalations_df()
    filename = "EscalateAI_Complaints.xlsx"
    df.to_excel(filename, index=False)
    return filename

# SLA breach checker + alert sender
def check_sla_and_alert():
    df = load_escalations_df()
    now = datetime.datetime.now(datetime.timezone.utc)
    breached = df[(df['priority'] == "High") & (df['status'] == "Open")]

    alerts_sent = 0
    for _, row in breached.iterrows():
        try:
            last_update = datetime.datetime.strptime(row['status_update_date'], "%a, %d %b %Y %H:%M:%S %z")
        except:
            continue
        elapsed = now - last_update
        if elapsed.total_seconds() > 10 * 60:  # 10 minutes SLA breach
            msg = f"⚠️ SLA breach detected:\nID: {row['escalation_id']}\nCustomer: {row['customer']}\nOpen for: {elapsed.seconds // 60} minutes\nIssue: {row['issue'][:200]}..."
            send_ms_teams_alert(msg)
            send_email_alert(f"SLA Breach Alert: {row['escalation_id']}", msg)
            alerts_sent += 1
    return alerts_sent

# ML Model for predicting escalation likelihood
def train_predictor():
    df = load_escalations_df()
    if df.empty:
        st.warning("No data to train ML model yet.")
        return None, None
    # Use issue text as feature, priority as label (High=1, Low=0)
    X = df['issue']
    y = df['priority'].apply(lambda x: 1 if x == "High" else 0)
    vectorizer = TfidfVectorizer(max_features=1000)
    X_vect = vectorizer.fit_transform(X)
    model = LogisticRegression(max_iter=500)
    model.fit(X_vect, y)
    # Save model and vectorizer
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)
    st.success("ML predictor trained successfully.")
    return model, vectorizer

def load_predictor():
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_FILE, "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    return None, None

def predict_escalation(issue, model, vectorizer):
    if not model or not vectorizer:
        return None
    X_vect = vectorizer.transform([issue])
    pred_prob = model.predict_proba(X_vect)[0][1]
    return pred_prob

# Feedback saving
def save_feedback(escalation_id, priority, severity, criticality, category):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
    cursor.execute("""
        INSERT OR REPLACE INTO feedback (escalation_id, correct_priority, correct_severity, correct_criticality, correct_category, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (escalation_id, priority, severity, criticality, category, now))
    # Also update main table to reflect corrected values
    cursor.execute("""
        UPDATE escalations SET priority=?, severity=?, criticality=?, category=?, user_feedback=?
        WHERE escalation_id=?
    """, (priority, severity, criticality, category, "corrected by user", escalation_id))
    conn.commit()
    st.success("Feedback saved!")

# Streamlit app UI
def main():
    st.set_page_config(page_title="EscalateAI", layout="wide")

    st.title("EscalateAI — Customer Escalation Management")

    with st.sidebar:
        st.header("Add New Data")

        # Email fetch and add
        if st.button("Fetch New Emails"):
            emails = fetch_gmail_emails()
            count = 0
            for e in emails:
                cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (e['customer'], e['issue'][:1000]))
                if not cursor.fetchone():
                    save_escalation(e['customer'], e['issue'], e['date'])
                    count += 1
            st.success(f"Fetched and added {count} new emails.")

        # Excel upload
        uploaded_file = st.file_uploader("Upload Excel complaints file", type=["xls", "xlsx"])
        if uploaded_file:
            count = upload_excel(uploaded_file)
            st.success(f"Added {count} new complaints from Excel.")

        # Download all complaints button
        if st.button("Download Consolidated Complaints Excel"):
            filename = save_escalations_excel()
            with open(filename, "rb") as f:
                st.download_button("Download Excel", f, file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # SLA Alerts
        st.header("SLA Alerts")
        if st.button("Check and Send SLA Alerts (Teams + Email)"):
            count = check_sla_and_alert()
            if count == 0:
                st.info("No SLA breaches detected.")
            else:
                st.success(f"Sent {count} SLA breach alerts.")

        # ML Model training
        st.header("Machine Learning")
        if st.button("Train Escalation Predictor Model"):
            with st.spinner("Training ML model..."):
                train_predictor()

    # Load escalations and display Kanban board
    df = load_escalations_df()

    # Filters
    filter_status = st.multiselect("Filter by Status", options=["Open", "In Progress", "Resolved"], default=["Open", "In Progress", "Resolved"])
    filter_escalated = st.selectbox("Show Escalated Cases Only?", options=["All", "Escalated Only"])
    if filter_escalated == "Escalated Only":
        df = df[df["priority"] == "High"]

    df = df[df["status"].isin(filter_status)]

    # Kanban columns
    st.subheader("Escalation Kanban Board")
    cols = st.columns(3)
    status_map = ["Open", "In Progress", "Resolved"]

    for i, status in enumerate(status_map):
        with cols[i]:
            st.markdown(f"### {status} ({len(df[df['status']==status])})")
            for _, row in df[df['status'] == status].iterrows():
                display_kanban_card(row)

    # Feedback section
    st.header("User Feedback and Corrections")
    esc_ids = df['escalation_id'].tolist()
    selected_id = st.selectbox("Select Escalation ID to give feedback", options=[""] + esc_ids)
    if selected_id:
        sel_row = df[df['escalation_id'] == selected_id].iloc[0]
        st.markdown(f"**Customer:** {sel_row['customer']}")
        st.markdown(f"**Issue:** {sel_row['issue']}")
        # Show current tags
        st.markdown(f"Current Priority: {sel_row['priority']}")
        st.markdown(f"Current Severity: {sel_row['severity']}")
        st.markdown(f"Current Criticality: {sel_row['criticality']}")
        st.markdown(f"Current Category: {sel_row['category']}")

        new_priority = st.selectbox("Correct Priority", ["High", "Low"], index=["High","Low"].index(sel_row['priority']))
        new_severity = st.selectbox("Correct Severity", ["Critical", "High", "Medium", "Low"], index=["Critical", "High", "Medium", "Low"].index(sel_row['severity']))
        new_criticality = st.selectbox("Correct Criticality", ["High", "Medium", "Routine"], index=["High", "Medium", "Routine"].index(sel_row['criticality']))
        new_category = st.selectbox("Correct Category", ["Technical", "Billing", "Service", "Delivery", "General"], index=["Technical", "Billing", "Service", "Delivery", "General"].index(sel_row['category']))

        if st.button("Submit Feedback"):
            save_feedback(selected_id, new_priority, new_severity, new_criticality, new_category)

    # Predict escalation probability on new text
    st.header("Predict Escalation Likelihood on New Issue")
    input_text = st.text_area("Enter issue description")
    if st.button("Predict Escalation Probability"):
        model, vectorizer = load_predictor()
        if model and vectorizer:
            prob = predict_escalation(input_text, model, vectorizer)
            st.info(f"Predicted probability of escalation: {prob*100:.2f}%")
        else:
            st.warning("ML model not trained yet. Please train from sidebar.")

if __name__ == "__main__":
    main()
