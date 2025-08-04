# escalate_ai.py – Full EscalateAI with sequential IDs, polished UI, expanded ML, explanations

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
import threading
from dotenv import load_dotenv

# Load environment variables from .env file (for credentials & config)
load_dotenv()

# --- Configuration from environment variables ---
EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
SMTP_EMAIL = EMAIL_USER
SMTP_PASS = EMAIL_PASS
ALERT_RECIPIENT = os.getenv("EMAIL_RECEIVER")
TEAMS_WEBHOOK = os.getenv("MS_TEAMS_WEBHOOK_URL")

# SQLite database file path
DB_PATH = "escalations.db"

# Prefix for escalation IDs (fixed "SESICE-25" + 5-digit number)
ESCALATION_PREFIX = "SESICE-25"

# Initialize VADER sentiment analyzer (pretrained lexicon for sentiment scoring)
analyzer = SentimentIntensityAnalyzer()

# Expanded negative keywords list categorized by type of issue,
# used for keyword matching to detect urgency and category of escalation
NEGATIVE_KEYWORDS = {
    "technical": ["fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip", "malfunction", "blank", "shutdown", "discharge"],
    "dissatisfaction": ["dissatisfy", "frustrate", "complain", "reject", "delay", "ignore", "escalate", "displease", "noncompliance", "neglect"],
    "support": ["wait", "pending", "slow", "incomplete", "miss", "omit", "unresolved", "shortage", "no response"],
    "safety": ["fire", "burn", "flashover", "arc", "explode", "unsafe", "leak", "corrode", "alarm", "incident"],
    "business": ["impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate", "penalty"]
}

# Used for tracking processed email UIDs in the background email polling thread
processed_email_uids = set()
processed_email_uids_lock = threading.Lock()  # Ensure thread-safe access to processed_email_uids


# ---------------------
# --- Helper Functions
# ---------------------

def get_next_escalation_id():
    """
    Generate a sequential escalation ID in the format SESICE-25XXXXX
    by querying the database for the last inserted ID and incrementing.
    Ensures unique and sequential IDs for traceability.
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
        # If no previous IDs, start numbering at 1
        next_num = 1

    # Zero-pad number to 5 digits (e.g., SESICE-2500001)
    return f"{ESCALATION_PREFIX}{str(next_num).zfill(5)}"


def ensure_schema():
    """
    Ensure the SQLite database and escalations table exist.
    Creates the table with all necessary columns for tracking escalations.
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
            user_feedback TEXT
        )
    ''')
    conn.commit()
    conn.close()


def insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag):
    """
    Insert a new escalation record into the SQLite database.
    Fields like status default to "Open", timestamps set to now.
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
        "", "", "", "", ""
    ))
    conn.commit()
    conn.close()


def fetch_escalations():
    """
    Retrieve all escalation records from the database as a pandas DataFrame.
    Provides the basis for display in the UI and model training.
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
    Update an escalation’s status, action taken, owner, and optionally user feedback.
    This is used when users update details on the Kanban board or provide feedback.
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


# --------------------
# --- Email Parsing ---
# --------------------

def parse_emails(imap_server, email_user, email_pass):
    """
    Connect to the IMAP email server, fetch unseen emails from the inbox,
    extract customer email and issue text (subject + body snippet).
    Returns a list of dicts with 'customer' and 'issue' keys.
    """
    try:
        conn = imaplib.IMAP4_SSL(imap_server)
        conn.login(email_user, email_pass)
        conn.select("inbox")
        _, messages = conn.search(None, "UNSEEN")
        emails = []
        for num in messages[0].split():
            _, msg_data = conn.fetch(num, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    # Decode email subject properly
                    subject = decode_header(msg["Subject"])[0][0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(errors='ignore')
                    from_ = msg.get("From")
                    body = ""
                    if msg.is_multipart():
                        # Iterate over parts to find plain text body
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                body = part.get_payload(decode=True).decode(errors='ignore')
                                break
                    else:
                        body = msg.get_payload(decode=True).decode(errors='ignore')
                    emails.append({
                        "customer": from_,
                        "issue": f"{subject} - {body[:200]}"  # Truncate body snippet for brevity
                    })
        conn.logout()
        return emails
    except Exception as e:
        st.error(f"Failed to parse emails: {e}")
        return []


# -----------------------
# --- NLP & Tagging ---
# -----------------------

def analyze_issue(issue_text):
    """
    Analyze the issue text using VADER sentiment analysis and keyword matching.
    Determine sentiment polarity, urgency, severity, criticality, category, and escalation flag.
    """
    # Get sentiment scores from VADER
    sentiment_score = analyzer.polarity_scores(issue_text)
    compound = sentiment_score["compound"]
    # Classify sentiment based on compound score thresholds
    if compound < -0.05:
        sentiment = "negative"
    elif compound > 0.05:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    # Determine urgency: high if any negative keyword detected
    urgency = "high" if any(word in issue_text.lower() for category in NEGATIVE_KEYWORDS.values() for word in category) else "normal"

    # Assign category based on which negative keywords matched
    category = None
    for cat, keywords in NEGATIVE_KEYWORDS.items():
        if any(k in issue_text.lower() for k in keywords):
            category = cat
            break

    # Assign severity: critical for safety or technical issues, major for support/business, else minor
    if category in ["safety", "technical"]:
        severity = "critical"
    elif category in ["support", "business"]:
        severity = "major"
    else:
        severity = "minor"

    # Criticality is high if sentiment is negative and urgency high, else medium
    criticality = "high" if sentiment == "negative" and urgency == "high" else "medium"

    # Escalation flag set to "Yes" if urgency high or sentiment negative
    escalation_flag = "Yes" if urgency == "high" or sentiment == "negative" else "No"

    return sentiment, urgency, severity, criticality, category, escalation_flag


# -------------------------
# --- ML MODEL FUNCTIONS ---
# -------------------------

def train_model():
    """
    Train a RandomForestClassifier to predict whether an issue should escalate,
    based on historical escalation data in the database.
    Returns the trained model, or None if not enough data.
    """
    df = fetch_escalations()
    if df.shape[0] < 20:
        # Not enough data for meaningful model training
        return None

    # Remove rows with missing critical info
    df = df.dropna(subset=['sentiment', 'urgency', 'severity', 'criticality', 'escalated'])
    if df.empty:
        return None

    # Prepare categorical features for modeling via one-hot encoding
    X = pd.get_dummies(df[['sentiment', 'urgency', 'severity', 'criticality']])
    # Label: escalated = Yes -> 1, else 0
    y = df['escalated'].apply(lambda x: 1 if x == 'Yes' else 0)

    if y.nunique() < 2:
        # Not enough class variety to train
        return None

    # Split into train and test sets (20% test for evaluation if needed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train RandomForest classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # (Optional) Evaluate model accuracy here if needed

    return model


def predict_escalation(model, sentiment, urgency, severity, criticality):
    """
    Use trained model to predict if a new issue should be escalated.
    Returns "Yes" if predicted to escalate, otherwise "No".
    """
    # Build feature vector for prediction; initialize zeros for all expected columns
    X_pred = pd.DataFrame([{
        f"sentiment_{sentiment}": 1,
        f"urgency_{urgency}": 1,
        f"severity_{severity}": 1,
        f"criticality_{criticality}": 1
    }])
    # Reindex to model’s expected features, fill missing with 0
    X_pred = X_pred.reindex(columns=model.feature_names_in_, fill_value=0)

    pred = model.predict(X_pred)
    return "Yes" if pred[0] == 1 else "No"


# -------------------
# --- ALERTING ------
# -------------------

def send_alert(message, via="email"):
    """
    Send an alert message either via email or Microsoft Teams webhook.
    Handles errors and displays them in Streamlit UI.
    """
    if via == "email":
        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_EMAIL, SMTP_PASS)
                # Send plain text email to alert recipient
                server.sendmail(SMTP_EMAIL, ALERT_RECIPIENT, message)
        except Exception as e:
            st.error(f"Email alert failed: {e}")
    elif via == "teams":
        try:
            # Post JSON payload with alert text to MS Teams webhook URL
            requests.post(TEAMS_WEBHOOK, json={"text": message})
        except Exception as e:
            st.error(f"Teams alert failed: {e}")


# ------------------------------
# --- BACKGROUND EMAIL POLLING -
# ------------------------------

def email_polling_job():
    """
    Background thread function that runs indefinitely,
    fetching new unseen emails every 60 seconds,
    analyzing them, and inserting new escalations into the DB.
    """
    while True:
        emails = parse_emails(EMAIL_SERVER, EMAIL_USER, EMAIL_PASS)
        with processed_email_uids_lock:
            for e in emails:
                issue = e["issue"]
                customer = e["customer"]
                sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)
                insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag)
        time.sleep(60)


# -------------------
# --- UI COLORS -----
# -------------------

STATUS_COLORS = {
    "Open": "#FFA500",        # Orange
    "In Progress": "#1E90FF", # Dodger Blue
    "Resolved": "#32CD32"     # Lime Green
}

SEVERITY_COLORS = {
    "critical": "#FF4500",    # OrangeRed
    "major": "#FF8C00",       # DarkOrange
    "minor": "#228B22"        # ForestGreen
}

URGENCY_COLORS = {
    "high": "#DC143C",        # Crimson
    "normal": "#008000"       # Green
}

def colored_text(text, color):
    """
    Utility to format colored HTML text (used in markdown with unsafe_allow_html).
    """
    return f'<span style="color:{color};font-weight:bold;">{text}</span>'


# -------------------
# --- STREAMLIT UI ---
# -------------------

# Ensure DB schema exists before starting
ensure_schema()

st.set_page_config(layout="wide")
#st.title("🚨 EscalateAI – Customer Escalation Management System")
st.markdown(
    """
    <style>
    /* Your CSS from above */
    </style>
    <header>
        <div>
            <h1 style="margin: 0; padding-left: 20px;">🚨 EscalateAI – Customer Escalation Management System</h1>
        </div>
    </header>
    """,
    unsafe_allow_html=True
)

# Sidebar controls
st.sidebar.header("⚙️ Controls")

# Excel bulk upload for customer complaints
uploaded_file = st.sidebar.file_uploader("📥 Upload Excel (Customer complaints)", type=["xlsx"])
if uploaded_file:
    df_excel = pd.read_excel(uploaded_file)
    for _, row in df_excel.iterrows():
        issue = str(row.get("issue", ""))
        customer = str(row.get("customer", "Unknown"))
        sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)
        insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag)
    st.sidebar.success("Uploaded and processed Excel file successfully.")

# Download all complaints CSV button
if st.sidebar.button("📤 Download All Complaints (CSV)"):
    df = fetch_escalations()
    csv = df.to_csv(index=False)
    st.sidebar.download_button("Download CSV", csv, file_name="escalations.csv", mime="text/csv")

# Download escalated cases Excel button with proper context manager fix
if st.sidebar.button("📥 Download Escalated Cases (Excel)"):
    df = fetch_escalations()
    df_esc = df[df["escalated"] == "Yes"]
    if df_esc.empty:
        st.sidebar.info("No escalated cases to download.")
    else:
        with pd.ExcelWriter("escalated_cases.xlsx", engine='xlsxwriter') as writer:
            df_esc.to_excel(writer, index=False, sheet_name='EscalatedCases')
        with open("escalated_cases.xlsx", "rb") as file:
            st.sidebar.download_button(
                label="Download Escalated Cases Excel",
                data=file,
                file_name="escalated_cases.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# Fetch emails manually from sidebar
if st.sidebar.button("📩 Fetch Emails (IMAP)"):
    emails = parse_emails(EMAIL_SERVER, EMAIL_USER, EMAIL_PASS)
    count = len(emails)
    for e in emails:
        issue = e["issue"]
        customer = e["customer"]
        sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)
        insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag)
    st.sidebar.success(f"Fetched and processed {count} new emails.")

# Trigger SLA alert manually
if st.sidebar.button("📣 Trigger SLA Alert"):
    df = fetch_escalations()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    breaches = df[(df['status'] != 'Resolved') & (df['priority'] == 'high') & 
                  ((datetime.datetime.now() - df['timestamp']) > datetime.timedelta(minutes=10))]
    if not breaches.empty:
        alert_msg = f"🚨 SLA breach detected for {len(breaches)} case(s)!"
        send_alert(alert_msg, via="teams")
        send_alert(alert_msg, via="email")
        st.sidebar.success("SLA breach alert sent.")
    else:
        st.sidebar.info("No SLA breaches detected.")

# Show SLA breach warning on sidebar if any
df_all = fetch_escalations()
df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], errors='coerce')
breaches = df_all[(df_all['status'] != 'Resolved') & (df_all['priority'] == 'high') & 
                  ((datetime.datetime.now() - df_all['timestamp']) > datetime.timedelta(minutes=10))]
if not breaches.empty:
    st.sidebar.markdown(
        f"<div style='background-color:#FF6347;color:white;padding:10px;border-radius:5px;margin-bottom:10px;text-align:center;'>"
        f"🚨 SLA breach detected for {len(breaches)} case(s)!"
        f"</div>", unsafe_allow_html=True
    )

# --- Main Tabs ---
tabs = st.tabs(["🗃️ All", "🚩 Escalated", "🔁 Feedback & Retraining"])

# --- All escalations tab with Kanban board ---
with tabs[0]:
    st.subheader("📊 Escalation Kanban Board")

    df = fetch_escalations()
    counts = df['status'].value_counts()
    open_count = counts.get('Open', 0)
    inprogress_count = counts.get('In Progress', 0)
    resolved_count = counts.get('Resolved', 0)
    st.markdown(f"**Open:** {open_count} | **In Progress:** {inprogress_count} | **Resolved:** {resolved_count}")

    col1, col2, col3 = st.columns(3)
    for status, col in zip(["Open", "In Progress", "Resolved"], [col1, col2, col3]):
        with col:
            # Column header with color
            col.markdown(f"<h3 style='background-color:{STATUS_COLORS[status]};color:white;padding:8px;border-radius:5px;text-align:center;'>{status}</h3>", unsafe_allow_html=True)
            bucket = df[df["status"] == status]
            for i, row in bucket.iterrows():
                flag = "🚩" if row['escalated'] == 'Yes' else ""
                header_color = SEVERITY_COLORS.get(row['severity'], "#000000")
                urgency_color = URGENCY_COLORS.get(row['urgency'], "#000000")
                expander_label = f"{row['id']} - {row['customer']} {flag}"
                with st.expander(expander_label, expanded=False):
                    st.markdown(f"**Issue:** {row['issue']}")
                    st.markdown(f"**Severity:** <span style='color:{header_color};font-weight:bold;'>{row['severity']}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Criticality:** {row['criticality']}")
                    st.markdown(f"**Category:** {row['category']}")
                    st.markdown(f"**Sentiment:** {row['sentiment']}")
                    st.markdown(f"**Urgency:** <span style='color:{urgency_color};font-weight:bold;'>{row['urgency']}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Escalated:** {row['escalated']}")
                    # Editable status, action taken, owner
                    new_status = st.selectbox("Update Status", ["Open", "In Progress", "Resolved"], index=["Open", "In Progress", "Resolved"].index(row["status"]), key=f"status_{row['id']}")
                    new_action = st.text_input("Action Taken", row.get("action_taken", ""), key=f"action_{row['id']}")
                    new_owner = st.text_input("Owner", row.get("owner", ""), key=f"owner_{row['id']}")
                    if st.button("💾 Save Changes", key=f"save_{row['id']}"):
                        update_escalation_status(row['id'], new_status, new_action, new_owner)
                        st.success("Escalation updated.")

# --- Escalated issues tab ---
with tabs[1]:
    st.subheader("🚩 Escalated Issues")
    df = fetch_escalations()
    df_esc = df[df["escalated"] == "Yes"]
    st.dataframe(df_esc)

# --- Feedback and Retraining tab ---
with tabs[2]:
    st.subheader("🔁 Feedback & Retraining")
    df = fetch_escalations()
    df_feedback = df[df["escalated"].notnull()]
    feedback_map = {"Correct": 1, "Incorrect": 0}

    # Feedback form per escalation
    for i, row in df_feedback.iterrows():
        feedback = st.selectbox(f"Is escalation for {row['id']} correct?", ["Correct", "Incorrect"], key=f"fb_{row['id']}")
        if st.button(f"Submit Feedback for {row['id']}", key=f"fb_btn_{row['id']}"):
            update_escalation_status(row['id'], row['status'], row.get('action_taken',''), row.get('owner',''), feedback_map[feedback])
            st.success("Feedback saved.")

    # Retrain model button
    if st.button("🔁 Retrain Model"):
        st.info("Retraining model with feedback (may take a few seconds)...")
        model = train_model()
        if model:
            st.success("Model retrained successfully.")
        else:
            st.warning("Not enough data to retrain model.")


# ------------------------------
# --- BACKGROUND EMAIL THREAD ---
# ------------------------------

if 'email_thread' not in st.session_state:
    email_thread = threading.Thread(target=email_polling_job, daemon=True)
    email_thread.start()
    st.session_state['email_thread'] = email_thread


# -----------------------
# --- DEV OPTIONS -------
# -----------------------

if st.sidebar.checkbox("🧪 View Raw Database"):
    df = fetch_escalations()
    st.sidebar.dataframe(df)

if st.sidebar.button("🗑️ Reset Database (Dev Only)"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS escalations")
    conn.commit()
    conn.close()
    st.sidebar.warning("Database reset. Please restart the app.")

# -----------------------
# --- NOTES -------------
# -----------------------
# - Update .env file with correct credentials:
#   EMAIL_USER, EMAIL_PASS, EMAIL_SERVER, EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT, EMAIL_RECEIVER, MS_TEAMS_WEBHOOK_URL
# - Run app with Streamlit >=1.10 for best support
# - ML model is RandomForest; can be replaced or enhanced as needed
# - Background email polling fetches every 60 seconds automatically
# - Excel export fixed with context manager, no deprecated save()

def send_alerts_for_all_escalated():
    df = load_escalations_df()
    if df.empty:
        return 0
    escalated_cases = df[(df['priority'] == "High") & (df['status'] == "Open")]

    alerts_sent = 0
    for _, row in escalated_cases.iterrows():
        msg = (
            f"🚨 Escalated Case Alert (Manual Override):\n"
            f"ID: {row['escalation_id']}\n"
            f"Customer: {row['customer']}\n"
            f"Issue: {row['issue'][:200]}..."
        )
        send_alerts(msg)
        alerts_sent += 1
    return alerts_sent

def sidebar_alert_controls():
    st.sidebar.header("Alerts")

    if st.sidebar.button("Send Test Alert"):
        test_msg = "🚨 This is a test alert from EscalateAI!"
        send_alerts(test_msg)
        st.sidebar.success("Test alert sent!")

    # Manual override button available to all users
    if st.sidebar.button("Send Alerts for ALL Escalated Cases (Manual Override)"):
        with st.spinner("Sending alerts for all escalated cases..."):
            count = send_alerts_for_all_escalated()
        if count > 0:
            st.sidebar.success(f"Sent {count} alert(s) for escalated cases!")
        else:
            st.sidebar.info("No escalated cases found.")

def main():
    st.sidebar.title("EscalateAI Controls")

    # Other sidebar controls here (manual entry, upload, fetch emails, etc.)

    sidebar_alert_controls()

    # Auto check SLA and alert every page load
    alert_count = check_sla_and_alert()
    if alert_count > 0:
        st.sidebar.warning(f"{alert_count} SLA breaches detected and alerted.")

    # Rest of your main app code (kanban board, data display, etc.)

if __name__ == "__main__":
    main()

