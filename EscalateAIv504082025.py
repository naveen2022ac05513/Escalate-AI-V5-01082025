# escalate_ai.py – Full EscalateAI with production-ready ML, email parsing,
# escalation tracking, alerts, feedback loop, and Streamlit UI.
# All key functions have detailed inline explanations for easy understanding.

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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import threading
from dotenv import load_dotenv

# Load environment variables from .env file (credentials, config)
load_dotenv()

# Email & alert configuration variables read from environment
EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
SMTP_EMAIL = EMAIL_USER
SMTP_PASS = EMAIL_PASS
ALERT_RECIPIENT = os.getenv("EMAIL_RECEIVER")
TEAMS_WEBHOOK = os.getenv("MS_TEAMS_WEBHOOK_URL")

# Database filename for storing escalations
DB_PATH = "escalations.db"
# Escalation ID prefix for unique IDs
ESCALATION_PREFIX = "SESICE-25"

# Initialize VADER sentiment analyzer for quick NLP sentiment scoring
analyzer = SentimentIntensityAnalyzer()

# Negative keywords organized by categories for urgency and escalation detection
NEGATIVE_KEYWORDS = {
    "technical": ["fail", "break", "crash", "defect", "fault", "degrade", "damage",
                  "trip", "malfunction", "blank", "shutdown", "discharge"],
    "dissatisfaction": ["dissatisfy", "frustrate", "complain", "reject", "delay", "ignore",
                       "escalate", "displease", "noncompliance", "neglect"],
    "support": ["wait", "pending", "slow", "incomplete", "miss", "omit",
                "unresolved", "shortage", "no response"],
    "safety": ["fire", "burn", "flashover", "arc", "explode", "unsafe",
               "leak", "corrode", "alarm", "incident"],
    "business": ["impact", "loss", "risk", "downtime", "interrupt", "cancel",
                 "terminate", "penalty"]
}

# Thread-safe set of processed email UIDs to avoid duplicates in background polling
processed_email_uids = set()
processed_email_uids_lock = threading.Lock()

# -------------------------------
# ID Generation for Escalations
# -------------------------------

def get_next_escalation_id():
    """
    Generate a new unique escalation ID in the format SESICE-25XXXXX
    by querying the latest ID in the database and incrementing it.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f'''
        SELECT id FROM escalations WHERE id LIKE '{ESCALATION_PREFIX}%'
        ORDER BY id DESC LIMIT 1
    ''')
    last = cursor.fetchone()
    conn.close()

    if last:
        last_id = last[0]
        last_num_str = last_id.replace(ESCALATION_PREFIX, "")
        try:
            last_num = int(last_num_str)
        except ValueError:
            last_num = 0
        next_num = last_num + 1
    else:
        next_num = 1

    # Zero pad to 5 digits, e.g., SESICE-2500001
    return f"{ESCALATION_PREFIX}{str(next_num).zfill(5)}"

# -------------------------------
# Database Initialization & Access
# -------------------------------

def ensure_schema():
    """
    Creates the escalations table in SQLite DB if it does not exist.
    Stores all case details and status.
    """
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
            user_feedback INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag):
    """
    Inserts a new escalation record into the DB with all NLP tags.
    Initial status is 'Open' and timestamps are set to current time.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    new_id = get_next_escalation_id()
    now = datetime.datetime.now().isoformat()
    cursor.execute('''
        INSERT INTO escalations (
            id, customer, issue, sentiment, urgency, severity, criticality, category,
            status, timestamp, escalated, priority, escalation_flag,
            action_taken, owner, action_owner, status_update_date, user_feedback
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        new_id, customer, issue, sentiment, urgency, severity, criticality, category,
        "Open", now, escalation_flag, "normal", escalation_flag,
        "", "", "", "", None
    ))
    conn.commit()
    conn.close()

def fetch_escalations():
    """
    Fetch all escalation records from DB as a Pandas DataFrame.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM escalations", conn)
    except Exception as e:
        st.error(f"Error reading escalations table: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def update_escalation_status(esc_id, status, action_taken, action_owner, feedback=None):
    """
    Update status, action taken, action owner, last update timestamp, and optional user feedback
    for a given escalation ID.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE escalations
        SET status = ?, action_taken = ?, action_owner = ?, status_update_date = ?, user_feedback = ?
        WHERE id = ?
    ''', (status, action_taken, action_owner, datetime.datetime.now().isoformat(), feedback, esc_id))
    conn.commit()
    conn.close()

# -------------------------------
# Email Parsing (IMAP)
# -------------------------------

def parse_emails(imap_server, email_user, email_pass):
    """
    Connects to IMAP server, fetches unread emails, extracts subject, sender, and body text.
    Returns a list of dicts with keys: 'customer' (email from), 'issue' (subject + body snippet).
    """
    try:
        conn = imaplib.IMAP4_SSL(imap_server)
        conn.login(email_user, email_pass)
        conn.select("inbox")
        _, messages = conn.search(None, "UNSEEN")  # Fetch only unseen emails
        emails = []
        for num in messages[0].split():
            _, msg_data = conn.fetch(num, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject = decode_header(msg["Subject"])[0][0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(errors='ignore')
                    from_ = msg.get("From")
                    body = ""
                    if msg.is_multipart():
                        # Walk through parts to find plain text body
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                body = part.get_payload(decode=True).decode(errors='ignore')
                                break
                    else:
                        body = msg.get_payload(decode=True).decode(errors='ignore')

                    emails.append({
                        "customer": from_,
                        "issue": f"{subject} - {body[:200]}"  # first 200 chars
                    })
        conn.logout()
        return emails
    except Exception as e:
        st.error(f"Failed to parse emails: {e}")
        return []

# -------------------------------
# NLP & Escalation Tagging
# -------------------------------

def analyze_issue(issue_text):
    """
    Given a text of issue, analyzes sentiment using VADER, detects urgency based on negative keywords,
    categorizes severity and criticality, and returns escalation flag.
    """
    # Sentiment scoring (compound score)
    sentiment_score = analyzer.polarity_scores(issue_text)
    if sentiment_score["compound"] < -0.05:
        sentiment = "negative"
    elif sentiment_score["compound"] > 0.05:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    # Urgency detection if any negative keyword is present in the text
    urgency = "high" if any(word in issue_text.lower() for category in NEGATIVE_KEYWORDS.values() for word in category) else "normal"

    # Identify category by checking which keyword set has matches
    category = None
    for cat, keywords in NEGATIVE_KEYWORDS.items():
        if any(k in issue_text.lower() for k in keywords):
            category = cat
            break

    # Severity mapping based on category
    if category in ["safety", "technical"]:
        severity = "critical"
    elif category in ["support", "business"]:
        severity = "major"
    else:
        severity = "minor"

    # Criticality depends on sentiment and urgency
    criticality = "high" if sentiment == "negative" and urgency == "high" else "medium"

    # Escalation flag set if urgent or negative sentiment
    escalation_flag = "Yes" if urgency == "high" or sentiment == "negative" else "No"

    return sentiment, urgency, severity, criticality, category, escalation_flag

# -------------------------------
# Machine Learning Model
# -------------------------------

MODEL_PATH = "escalate_rf_model.joblib"

def train_model():
    """
    Trains a Random Forest classifier to predict if an issue will be escalated or not.
    Uses existing labeled escalation data with feedback if available.
    Saves trained model to disk.
    """
    df = fetch_escalations()
    # Require at least 30 labeled rows to train meaningfully
    if df.shape[0] < 30:
        return None

    # Drop rows missing essential columns or feedback (feedback is label)
    df_train = df.dropna(subset=['sentiment', 'urgency', 'severity', 'criticality', 'escalated', 'user_feedback'])

    # Use user_feedback as label: 1=Correct escalation, 0=Incorrect or no
    df_train = df_train[df_train['user_feedback'].notnull()]
    if df_train.empty or df_train['user_feedback'].nunique() < 2:
        # Not enough variation in labels
        return None

    # Prepare features - one-hot encode categorical variables and add numerical features
    X_cat = pd.get_dummies(df_train[['sentiment', 'urgency', 'severity', 'criticality', 'category']].fillna("unknown"))
    X_num = pd.DataFrame({
        'sentiment_compound': df_train['issue'].apply(lambda x: analyzer.polarity_scores(x)['compound']),
        'issue_len': df_train['issue'].apply(len)
    })

    X = pd.concat([X_cat.reset_index(drop=True), X_num.reset_index(drop=True)], axis=1)
    y = df_train['user_feedback'].astype(int)

    # Split data into training and test sets to evaluate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate performance on test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Save model and feature columns for prediction use
    joblib.dump({'model': model, 'columns': X.columns.tolist()}, MODEL_PATH)

    # Return performance metrics for display
    return model, acc, f1, y_test, y_pred

def load_model():
    """
    Load the trained model and expected columns from disk.
    Returns None if model does not exist.
    """
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        return data['model'], data['columns']
    return None, None

def predict_escalation(model, feature_columns, issue_text, sentiment, urgency, severity, criticality, category):
    """
    Predict if a new issue will be escalated using trained model.
    Prepares feature vector from NLP tags and text features.
    Returns predicted label and probability.
    """
    # Build feature vector dict with all feature columns set to 0 by default
    features = dict.fromkeys(feature_columns, 0)

    # Set categorical one-hot features to 1 if present
    keys = {
        f"sentiment_{sentiment}",
        f"urgency_{urgency}",
        f"severity_{severity}",
        f"criticality_{criticality}",
        f"category_{category if category else 'unknown'}"
    }
    for k in keys:
        if k in features:
            features[k] = 1

    # Add numerical features
    features['sentiment_compound'] = analyzer.polarity_scores(issue_text)['compound']
    features['issue_len'] = len(issue_text)

    # Convert to DataFrame for prediction
    X_pred = pd.DataFrame([features], columns=feature_columns)

    pred = model.predict(X_pred)[0]
    proba = model.predict_proba(X_pred)[0][1]  # Probability for class 1 (escalated)

    return "Yes" if pred == 1 else "No", proba

# -------------------------------
# Alerting Functions (Email & Teams)
# -------------------------------

def send_email_alert(subject, body):
    """
    Sends alert email with given subject and body to ALERT_RECIPIENT.
    Uses SMTP with TLS.
    """
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASS)
            msg = f"Subject: {subject}\n\n{body}"
            server.sendmail(SMTP_EMAIL, ALERT_RECIPIENT, msg)
    except Exception as e:
        st.error(f"Failed to send email alert: {e}")

def send_teams_alert(message):
    """
    Sends alert message to MS Teams channel using webhook URL.
    """
    try:
        resp = requests.post(TEAMS_WEBHOOK, json={"text": message})
        if resp.status_code != 200:
            st.error(f"Teams alert failed with status code {resp.status_code}")
    except Exception as e:
        st.error(f"Failed to send Teams alert: {e}")

def send_alert(message, via="email"):
    """
    Sends alert via specified channel: 'email' or 'teams'.
    """
    if via == "email":
        send_email_alert("EscalateAI Alert", message)
    elif via == "teams":
        send_teams_alert(message)

# -------------------------------
# Background Email Polling
# -------------------------------

def email_polling_job():
    """
    Background thread function that periodically fetches new emails,
    analyzes them, inserts new escalation records if any.
    Runs every 60 seconds.
    """
    while True:
        emails = parse_emails(EMAIL_SERVER, EMAIL_USER, EMAIL_PASS)
        with processed_email_uids_lock:
            for e in emails:
                issue = e["issue"]
                customer = e["customer"]
                sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)

                # Before inserting, optionally check if already exists (optional enhancement)
                insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag)
        time.sleep(60)

# -------------------------------
# UI Color Maps for better visuals
# -------------------------------

STATUS_COLORS = {
    "Open": "#FFA500",
    "In Progress": "#1E90FF",
    "Resolved": "#32CD32"
}

SEVERITY_COLORS = {
    "critical": "#FF4500",
    "major": "#FF8C00",
    "minor": "#228B22"
}

URGENCY_COLORS = {
    "high": "#DC143C",
    "normal": "#008000"
}

def colored_text(text, color):
    """
    Utility to return colored HTML span for Streamlit markdown.
    """
    return f'<span style="color:{color};font-weight:bold;">{text}</span>'

# -------------------------------
# Main Streamlit UI Application
# -------------------------------

def main():
    """
    Main Streamlit app function that provides:
    - Upload Excel with issues
    - Fetch emails from IMAP
    - Display Kanban board
    - Allow status/action updates
    - Show escalated issues
    - Feedback & Retraining interface
    - Alerts on SLA breach
    """
    ensure_schema()

    st.set_page_config(layout="wide")
    st.title("🚨 EscalateAI – Customer Escalation Management")

    st.sidebar.header("⚙️ Controls")

    # --- Excel Bulk Upload ---
    uploaded_file = st.sidebar.file_uploader("📥 Upload Excel (Customer complaints)", type=["xlsx"])
    if uploaded_file:
        try:
            df_excel = pd.read_excel(uploaded_file)
            for _, row in df_excel.iterrows():
                issue = str(row.get("issue", ""))
                customer = str(row.get("customer", "Unknown"))
                sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)
                insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag)
            st.sidebar.success("Uploaded and processed Excel file successfully.")
        except Exception as e:
            st.sidebar.error(f"Failed to process Excel upload: {e}")

    # --- Download CSV of all complaints ---
    if st.sidebar.button("📤 Download All Complaints (CSV)"):
        df = fetch_escalations()
        csv = df.to_csv(index=False)
        st.sidebar.download_button("Download CSV", csv, file_name="escalations.csv", mime="text/csv")

    # --- Download Excel of escalated cases ---
    if st.sidebar.button("📥 Download Escalated Cases (Excel)"):
        df = fetch_escalations()
        df_esc = df[df["escalated"] == "Yes"]
        if df_esc.empty:
            st.sidebar.warning("No escalated cases to export.")
        else:
            output = pd.ExcelWriter("escalated_cases.xlsx", engine="xlsxwriter")
            df_esc.to_excel(output, index=False)
            output.save()
            with open("escalated_cases.xlsx", "rb") as f:
                st.sidebar.download_button("Download Excel", f, file_name="escalated_cases.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # --- Fetch Emails Button ---
    if st.sidebar.button("📧 Fetch Emails Now"):
        emails = parse_emails(EMAIL_SERVER, EMAIL_USER, EMAIL_PASS)
        count = 0
        for e in emails:
            issue = e["issue"]
            customer = e["customer"]
            sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)
            insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag)
            count += 1
        st.sidebar.success(f"Fetched and inserted {count} new emails.")

    # --- Trigger SLA Alert Button ---
    if st.sidebar.button("🚨 Trigger SLA Alert Check"):
        sla_breaches = check_sla_breach()
        if sla_breaches:
            alert_msg = f"SLA Breach detected for {len(sla_breaches)} issues:\n"
            for esc in sla_breaches:
                alert_msg += f"{esc['id']} - {esc['issue'][:50]}...\n"
            send_alert(alert_msg, via="teams")
            send_alert(alert_msg, via="email")
            st.sidebar.success("SLA breach alert sent!")
        else:
            st.sidebar.info("No SLA breaches detected.")

    # --- Main Kanban View ---
    st.header("🗂️ Escalation Kanban Board")

    df = fetch_escalations()
    if df.empty:
        st.info("No escalations found. Upload or fetch emails to start.")
        return

    status_filter = st.selectbox("Filter by Status", options=["All", "Open", "In Progress", "Resolved"], index=0)
    if status_filter != "All":
        df = df[df["status"] == status_filter]

    # Show counts summary
    total = len(df)
    escalated_count = len(df[df["escalated"] == "Yes"])
    st.markdown(f"**Total issues:** {total} | **Escalated:** {escalated_count}")

    # Display issues as cards grouped by status
    for status in ["Open", "In Progress", "Resolved"]:
        if status_filter != "All" and status != status_filter:
            continue
        st.subheader(f"{status} ({len(df[df['status'] == status])})")
        for _, row in df[df["status"] == status].iterrows():
            st.markdown("---")
            st.markdown(
                f"**ID:** {row['id']} | "
                f"Customer: {row['customer']} | "
                f"Issue: {row['issue'][:100]}... | "
                f"Sentiment: {colored_text(row['sentiment'], 'red' if row['sentiment']=='negative' else 'green')} | "
                f"Urgency: {colored_text(row['urgency'], 'red' if row['urgency']=='high' else 'green')} | "
                f"Severity: {colored_text(row['severity'], 'red' if row['severity']=='critical' else 'orange')} | "
                f"Category: {row['category']}"
                , unsafe_allow_html=True)

            # Allow status and action owner update
            new_status = st.selectbox(f"Update Status for {row['id']}", options=["Open", "In Progress", "Resolved"], index=["Open", "In Progress", "Resolved"].index(row["status"]), key=row["id"])
            action_taken = st.text_input(f"Action Taken for {row['id']}", value=row["action_taken"], key=row["id"] + "_action")
            action_owner = st.text_input(f"Action Owner for {row['id']}", value=row["action_owner"], key=row["id"] + "_owner")

            if st.button(f"Save Updates for {row['id']}", key=row["id"] + "_save"):
                update_escalation_status(row["id"], new_status, action_taken, action_owner)
                st.success(f"Updated {row['id']}")

    # --- Feedback and Retrain ML ---
    st.header("🤖 Feedback and Model Retraining")

    model, feature_cols = load_model()
    if model:
        st.success("ML Model Loaded.")
    else:
        st.warning("No ML model found. Please train model with feedback data.")

    # Show feedback form for open or escalated cases
    st.subheader("Submit Feedback on Escalations")

    feedback_df = df[(df["status"] != "Resolved") & (df["user_feedback"].isnull())]
    for _, row in feedback_df.iterrows():
        st.markdown(f"**{row['id']}** - {row['issue'][:100]}...")
        feedback = st.radio("Was this escalation prediction correct?", ("Yes", "No"), key=row['id'])
        if st.button(f"Submit Feedback for {row['id']}", key="fb_"+row['id']):
            val = 1 if feedback == "Yes" else 0
            update_escalation_status(row['id'], row['status'], row['action_taken'], row['action_owner'], feedback=val)
            st.success(f"Feedback saved for {row['id']}")

    # Retrain button
    if st.button("🔄 Retrain ML Model"):
        result = train_model()
        if result is None:
            st.warning("Not enough labeled data for training (minimum 30 rows with feedback).")
        else:
            model, acc, f1, y_test, y_pred = result
            st.success(f"Model retrained successfully! Accuracy: {acc:.2f}, F1-score: {f1:.2f}")
            # Show confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
    st.markdown("---")

# -------------------------------
# SLA Breach Checker
# -------------------------------

def check_sla_breach():
    """
    Checks for any open, high-priority escalations that have been open > 10 minutes.
    Returns a list of such escalation dicts.
    """
    df = fetch_escalations()
    breaches = []
    now = datetime.datetime.now()

    for _, row in df.iterrows():
        if row["status"] == "Open" and row["priority"] == "high":
            ts = datetime.datetime.fromisoformat(row["timestamp"])
            diff = now - ts
            if diff.total_seconds() > 600:  # 10 minutes
                breaches.append(row)
    return breaches

# -------------------------------
# Run background email polling thread
# -------------------------------

def start_background_thread():
    t = threading.Thread(target=email_polling_job, daemon=True)
    t.start()

# -------------------------------
# Entrypoint
# -------------------------------

if __name__ == "__main__":
    start_background_thread()
    main()
