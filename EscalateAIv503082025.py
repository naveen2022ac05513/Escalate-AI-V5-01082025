# EscalateAI – Full Functionality with Detailed Negative Keywords
# ===============================================================
# Features:
# • Parsing emails + Excel for customer issues
# • NLP analysis (sentiment, urgency, category tagging)
# • Unique ID: SESICE-XXXXX
# • Kanban Board: Open, In Progress, Resolved
# • Action Taken + Owner inline editing
# • SLA Breach detection (10 mins)
# • Teams/email alerts
# • Predictive ML + Feedback + Retraining

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
from streamlit_autorefresh import st_autorefresh
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import smtplib
from email.mime.text import MIMEText

# Load environment variables
load_dotenv()
EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")

EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", 587))
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

# Updated NEGATIVE_KEYWORDS list as per your categories
NEGATIVE_KEYWORDS = [
    # Technical Failures & Product Malfunction
    "fail", "break", "crash", "defect", "fault", "degrade", "damage",
    "trip", "malfunction", "blank", "shutdown", "discharge",
    # Customer Dissatisfaction & Escalations
    "dissatisfy", "frustrate", "complain", "reject", "delay", "ignore",
    "escalate", "displease", "noncompliance", "neglect",
    # Support Gaps & Operational Delays
    "wait", "pending", "slow", "incomplete", "miss", "omit",
    "unresolved", "shortage", "no response",
    # Hazardous Conditions & Safety Risks
    "fire", "burn", "flashover", "arc", "explode", "unsafe",
    "leak", "corrode", "alarm", "incident",
    # Business Risk & Impact
    "impact", "loss", "risk", "downtime", "interrupt", "cancel",
    "terminate", "penalty"
]

URGENCY_PHRASES = ["asap", "urgent", "immediately", "right away", "critical"]

CATEGORY_KEYWORDS = {
    "Safety": ["fire", "burn", "leak", "unsafe", "flashover", "arc", "explode", "corrode", "alarm", "incident"],
    "Performance": ["slow", "crash", "malfunction", "fail", "break", "degrade", "damage", "trip", "blank", "shutdown", "discharge"],
    "Delay": ["delay", "pending", "wait", "incomplete", "miss", "omit", "unresolved", "shortage", "no response"],
    "Compliance": ["noncompliance", "violation", "neglect", "reject", "cancel", "terminate", "penalty"],
    "Service": ["ignore", "unavailable", "complain", "dissatisfy", "frustrate", "escalate", "displease"],
    "Quality": ["defect", "fault"]
}

# Database setup
conn = sqlite3.connect("escalations.db", check_same_thread=False)
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
    user_feedback INTEGER DEFAULT 0
)
""")
conn.commit()

analyzer = SentimentIntensityAnalyzer()

# Utility Functions

def generate_escalation_id():
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]
    return f"SESICE-{count + 250001}"

def classify_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def detect_urgency(text):
    text_lower = text.lower()
    for phrase in URGENCY_PHRASES:
        if phrase in text_lower:
            return "High"
    return "Normal"

def detect_category(text):
    text_lower = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    return "General"

def detect_escalation_flag(text):
    text_lower = text.lower()
    for keyword in NEGATIVE_KEYWORDS:
        if keyword in text_lower:
            return 1
    return 0

# Send Alerts

def send_ms_teams_alert(message):
    if not MS_TEAMS_WEBHOOK_URL:
        return
    try:
        requests.post(MS_TEAMS_WEBHOOK_URL, json={"text": message})
    except Exception as e:
        st.error(f"Failed to send MS Teams alert: {e}")

def send_email_alert(subject, message):
    if not (EMAIL_SENDER and EMAIL_PASSWORD and EMAIL_RECEIVER):
        return
    try:
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER

        server = smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, [EMAIL_RECEIVER], msg.as_string())
        server.quit()
    except Exception as e:
        st.error(f"Failed to send email alert: {e}")

def send_alerts(message):
    send_ms_teams_alert(message)
    send_email_alert("EscalateAI Alert", message)

# Email Parsing

def fetch_gmail_emails():
    if not EMAIL or not APP_PASSWORD:
        st.error("Gmail credentials not configured in environment variables.")
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

# Save parsed emails to DB

def save_emails_to_db(emails):
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]
    new_entries = 0
    for e in emails:
        # Avoid duplicates based on customer+issue
        cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (e['customer'], e['issue'][:500]))
        if cursor.fetchone():
            continue
        count += 1
        esc_id = f"SESICE-{count+250000}"
        sentiment = classify_sentiment(e['issue'])
        priority = "High" if sentiment == "Negative" else "Low"
        escalation_flag = detect_escalation_flag(e['issue'])
        urgency = detect_urgency(e['issue'])
        category = detect_category(e['issue'])
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
        cursor.execute("""
            INSERT INTO escalations 
            (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, urgency, category, action_taken, action_owner, status_update_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (esc_id, e['customer'], e['issue'][:500], e['date'], "Open", sentiment, priority, escalation_flag, urgency, category, "", "", now))
        new_entries += 1
        if escalation_flag == 1 and priority == "High":
            send_alerts(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {e['customer']}\nIssue: {e['issue'][:200]}...")
    conn.commit()
    return new_entries

# Excel Upload and processing

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
        cursor.execute("SELECT COUNT(*) FROM escalations")
        existing_count = cursor.fetchone()[0]

        for idx, row in df.iterrows():
            customer = str(row[customer_col])
            issue = str(row[issue_col])
            date = str(row[date_col]) if date_col and pd.notna(row[date_col]) else datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (customer, issue[:500]))
            if cursor.fetchone():
                continue
            existing_count += 1
            esc_id = f"SESICE-{existing_count+250000}"
            sentiment = classify_sentiment(issue)
            priority = "High" if sentiment == "Negative" else "Low"
            escalation_flag = detect_escalation_flag(issue)
            urgency = detect_urgency(issue)
            category = detect_category(issue)
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("""
                INSERT INTO escalations 
                (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, urgency, category, action_taken, action_owner, status_update_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (esc_id, customer, issue[:500], date, "Open", sentiment, priority, escalation_flag, urgency, category, "", "", now))
            count += 1
            if escalation_flag == 1 and priority == "High":
                send_alerts(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {customer}\nIssue: {issue[:200]}...")
        conn.commit()
        return count
    except Exception as e:
        st.error(f"Error processing uploaded Excel: {e}")
        return 0

# Manual entry

def manual_entry_process(customer, issue):
    if not customer or not issue:
        st.sidebar.error("Please fill customer and issue.")
        return False
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]
    esc_id = f"SESICE-{count+250001}"
    sentiment = classify_sentiment(issue)
    priority = "High" if sentiment == "Negative" else "Low"
    escalation_flag = detect_escalation_flag(issue)
    urgency = detect_urgency(issue)
    category = detect_category(issue)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
    cursor.execute("""
        INSERT INTO escalations 
        (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, urgency, category, action_taken, action_owner, status_update_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (esc_id, customer, issue[:500], now, "Open", sentiment, priority, escalation_flag, urgency, category, "", "", now))
    conn.commit()
    st.sidebar.success(f"Added escalation {esc_id}")
    if escalation_flag == 1 and priority == "High":
        send_alerts(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {customer}\nIssue: {issue[:200]}...")
    return True

# Load DB into dataframe

def load_escalations_df():
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    return df

# Kanban card display

def display_kanban_card(row):
    esc_id = row['escalation_id']
    sentiment = row['sentiment']
    priority = row['priority']
    status = row['status']

    sentiment_colors = {"Positive": "#27ae60", "Negative": "#e74c3c", "Neutral": "#f39c12"}
    priority_colors = {"High": "#c0392b", "Low": "#27ae60"}
    status_colors = {"Open": "#f1c40f", "In Progress": "#2980b9", "Resolved": "#2ecc71"}

    border_color = priority_colors.get(priority, "#000000")
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
        st.markdown(f"**Urgency:** {row['urgency']}")
        st.markdown(f"**Category:** {row['category']}")

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

        # User feedback checkbox for continuous learning
        feedback_key = f"{esc_id}_feedback"
        feedback_correct = st.checkbox("Is the priority classification correct?", key=feedback_key)
        if feedback_correct:
            cursor.execute("UPDATE escalations SET user_feedback=1 WHERE escalation_id=?", (esc_id,))
            conn.commit()
            st.success("Thanks for your feedback!")

# Kanban rendering

def render_kanban():
    st.title("🚀 EscalateAI - Escalations & Complaints Kanban Board")

    search_text = st.text_input("Search by customer or issue text")

    df = load_escalations_df()

    if search_text:
        df = df[df.apply(lambda row: search_text.lower() in str(row['customer']).lower() or search_text.lower() in str(row['issue']).lower(), axis=1)]

    filter_choice = st.radio("Filter Escalations:", ["All", "Escalated Only"])

    if filter_choice == "Escalated Only":
        df = df[df['escalation_flag'] == 1]

    cols = st.columns(3)
    statuses = ["Open", "In Progress", "Resolved"]

    for idx, status in enumerate(statuses):
        with cols[idx]:
            st.header(status)
            df_status = df[df['status'] == status]
            if df_status.empty:
                st.info(f"No {status} escalations")
            else:
                for _, row in df_status.iterrows():
                    display_kanban_card(row)

# SLA Monitoring

def check_sla_breach():
    now = datetime.datetime.now(datetime.timezone.utc)
    cursor.execute("SELECT escalation_id, date, status, priority FROM escalations WHERE status != 'Resolved'")
    rows = cursor.fetchall()
    for esc_id, date_str, status, priority in rows:
        try:
            issue_date = datetime.datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
        except Exception:
            # fallback if no tzinfo
            issue_date = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            issue_date = issue_date.replace(tzinfo=datetime.timezone.utc)
        delta = now - issue_date
        if priority == "High" and status != "Resolved" and delta.total_seconds() > 600:  # 10 mins breach
            send_alerts(f"⚠️ SLA Breach Alert: Escalation {esc_id} has been open for more than 10 minutes.")
            st.warning(f"SLA Breach Alert: Escalation {esc_id} open >10 minutes!")

# Predictive ML Model (simple example)

def train_predictive_model():
    df = load_escalations_df()
    # We only train if sufficient data is there
    if len(df) < 10:
        return None, None

    df['label'] = df['priority'].apply(lambda x: 1 if x == "High" else 0)
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(df['issue'])
    y = df['label']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, "escalate_model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")
    return model, vectorizer

def load_model_and_vectorizer():
    try:
        model = joblib.load("escalate_model.joblib")
        vectorizer = joblib.load("vectorizer.joblib")
        return model, vectorizer
    except:
        return None, None

def predict_escalation(issue, model, vectorizer):
    if not model or not vectorizer:
        return "Unknown"
    X = vectorizer.transform([issue])
    pred = model.predict(X)[0]
    return "High" if pred == 1 else "Low"

# Retrain feedback loop

def retrain_model_with_feedback():
    # Only retrain on user feedback = 1 rows
    df = pd.read_sql_query("SELECT * FROM escalations WHERE user_feedback=1", conn)
    if len(df) < 5:
        return
    df['label'] = df['priority'].apply(lambda x: 1 if x == "High" else 0)
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(df['issue'])
    y = df['label']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, "escalate_model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")

    # Reset user feedback flags
    cursor.execute("UPDATE escalations SET user_feedback=0 WHERE user_feedback=1")
    conn.commit()

# Main Streamlit App

def main():
    st.set_page_config(page_title="EscalateAI Customer Escalation Management", layout="wide")
    st.title("🚨 EscalateAI - Customer Escalation Tracker")

    # Sidebar inputs
    st.sidebar.header("Add New Escalation")

    with st.sidebar.expander("Manual Entry"):
        customer = st.text_input("Customer / Email")
        issue = st.text_area("Issue Description")
        if st.button("Add Escalation"):
            if manual_entry_process(customer, issue):
                st.success("Escalation added successfully!")
            else:
                st.error("Failed to add escalation.")

    with st.sidebar.expander("Upload Excel"):
        uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
        if uploaded_file:
            added_count = upload_excel_and_analyze(uploaded_file)
            st.success(f"Processed {added_count} new escalations from Excel.")

    with st.sidebar.expander("Fetch Latest Emails"):
        if st.button("Fetch Emails"):
            emails = fetch_gmail_emails()
            new_added = save_emails_to_db(emails)
            st.success(f"Fetched {len(emails)} emails and added {new_added} new escalations.")

    # Show Kanban Board
    render_kanban()

    # SLA Checks and Alerts
    check_sla_breach()

    # Model training and retraining
    if st.sidebar.checkbox("Retrain Model with Feedback Data"):
        with st.spinner("Retraining model..."):
            retrain_model_with_feedback()
            st.success("Model retrained with user feedback data.")

    if st.sidebar.checkbox("Train New Model from Data"):
        with st.spinner("Training model..."):
            model, vectorizer = train_predictive_model()
            if model:
                st.success("Model trained successfully.")
            else:
                st.warning("Not enough data to train model.")

    # Prediction on new text input
    st.sidebar.header("Predict Escalation Priority")
    model, vectorizer = load_model_and_vectorizer()
    input_text = st.sidebar.text_area("Enter issue text for prediction")
    if st.sidebar.button("Predict Priority"):
        if not model or not vectorizer:
            st.sidebar.warning("Model not found. Train or retrain the model first.")
        else:
            pred = predict_escalation(input_text, model, vectorizer)
            st.sidebar.info(f"Predicted priority: {pred}")

if __name__ == "__main__":
    # Auto-refresh every 60 seconds for new emails and SLA check
    st_autorefresh(interval=60000, limit=None, key="auto_refresh")
    main()
