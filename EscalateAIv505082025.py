# escalate_ai.py – Full EscalateAI with sequential IDs, polished UI, expanded ML, explanations
#https://github.com/naveen2022ac05513/Escalate-AI-V5-01082025/blob/main/EscalateAIv504082025.py
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
import hashlib
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
    "technical": ["fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip", "malfunction", "blank", "shutdown", "discharge","leak"],
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

def summarize_issue_text(issue_text):
    """
    Generate a concise issue summary for the Kanban board.
    Trims verbosity and keeps it within ~120 chars.
    """
    clean_text = re.sub(r'\s+', ' ', issue_text).strip()
    return clean_text[:120] + "..." if len(clean_text) > 120 else clean_text
    
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
    Adds 'owner_email' if missing.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table with new column if doesn't exist
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
            owner_email TEXT,
            escalated TEXT,
            priority TEXT,
            escalation_flag TEXT,
            action_owner TEXT,
            status_update_date TEXT,
            user_feedback TEXT
        )
    ''')

    # Check if 'owner_email' column exists, add if not
    try:
        cursor.execute("SELECT owner_email FROM escalations LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE escalations ADD COLUMN owner_email TEXT")

    conn.commit()
    conn.close()

def generate_issue_hash(issue_text):
    """
    Cleans and extracts core text from email (stripping forwards, quotes, excessive metadata).
    Produces a hash that's more tolerant to formatting noise.
    """
    # Remove common forwarding markers
    patterns_to_remove = [
        r"[-]+[ ]*Forwarded message[ ]*[-]+",
        r"From:.*", r"Sent:.*", r"To:.*", r"Subject:.*",
        r">.*",                     # Quoted lines
        r"On .* wrote:",            # Replies
        r"\n\s*\n"                  # Excess whitespace blocks
    ]
    for pat in patterns_to_remove:
        issue_text = re.sub(pat, "", issue_text, flags=re.IGNORECASE)

    # Normalize text and trim
    clean_text = re.sub(r'\s+', ' ', issue_text.lower().strip())
    return hashlib.md5(clean_text.encode()).hexdigest() 
    
def insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag, owner_email=""):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    new_id = get_next_escalation_id()
    now = datetime.datetime.now().isoformat()

    cursor.execute('''
        INSERT INTO escalations (
            id, customer, issue, sentiment, urgency, severity, criticality, category,
            status, timestamp, escalated, priority, escalation_flag,
            action_taken, owner, action_owner, status_update_date, user_feedback, owner_email
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        new_id, customer, issue, sentiment, urgency, severity, criticality, category,
        "Open", now, escalation_flag, "normal", escalation_flag,
        "", "", "", "", "", owner_email
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


def update_escalation_status(esc_id, status, action_taken, action_owner, owner_email=None, feedback=None, sentiment=None, criticality=None, notes=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE escalations
        SET status = ?, action_taken = ?, action_owner = ?, status_update_date = ?, 
            owner_email = ?, user_feedback = ?, sentiment = ?, criticality = ?
        WHERE id = ?
    ''', (
        status,
        action_taken,
        action_owner,
        datetime.datetime.now().isoformat(),
        owner_email,
        notes,
        sentiment,
        criticality,
        esc_id
    ))
    conn.commit()
    conn.close()




# --------------------
# --- Email Parsing ---
# --------------------

global_seen_hashes = set()
# REMOVE or COMMENT OUT this invalid block:
# for email in email_list:
#     body = email.get("body", "")
#     email_hash = hashlib.md5(body.encode()).hexdigest()
#     if email_hash in seen_hashes:
#         continue  # Skip duplicate
#     seen_hashes.add(email_hash)
#     # Process email...

    # Process email...

def parse_emails():
    """
    Parses unseen emails, extracts summaries,
    and filters forwarded/repeated content via normalized hashing.
    """
    from dotenv import load_dotenv
    load_dotenv()

    imap_server = os.getenv("EMAIL_SERVER", "imap.gmail.com")
    email_user = os.getenv("EMAIL_USER")
    email_pass = os.getenv("EMAIL_PASS")

    emails = []

    try:
        conn = imaplib.IMAP4_SSL(imap_server)
        conn.login(email_user, email_pass)
        conn.select("inbox")
        _, messages = conn.search(None, "UNSEEN")

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
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                body = part.get_payload(decode=True).decode(errors='ignore')
                                break
                    else:
                        body = msg.get_payload(decode=True).decode(errors='ignore')

                    full_text = f"{subject} - {body}"
                    hash_val = generate_issue_hash(full_text)

                    if hash_val not in global_seen_hashes:
                        global_seen_hashes.add(hash_val)
                        summary = summarize_issue_text(full_text)
                        emails.append({
                            "customer": from_,
                            "issue": summary
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

from dotenv import load_dotenv
import os
import smtplib
import requests
import streamlit as st
from email.mime.text import MIMEText

# ✅ Load environment variables
load_dotenv()

SMTP_EMAIL = os.getenv("EMAIL_USER")
SMTP_PASS = os.getenv("EMAIL_PASS") or ""  # Handles blank password
SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER")
SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
ALERT_RECIPIENT = os.getenv("EMAIL_RECEIVER")
TEAMS_WEBHOOK = os.getenv("MS_TEAMS_WEBHOOK_URL")
EMAIL_SUBJECT = os.getenv("EMAIL_SUBJECT", "🚨 EscalateAI Alert")

def send_alert(message, via="email"):
    """
    Send alert via email or Microsoft Teams webhook using environment variables.
    Properly encodes Unicode and handles errors gracefully.
    """
    if via == "email":
        try:
            msg = MIMEText(message, 'plain', 'utf-8')
            msg['Subject'] = EMAIL_SUBJECT
            msg['From'] = SMTP_EMAIL
            msg['To'] = ALERT_RECIPIENT

            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_EMAIL, SMTP_PASS)
                server.sendmail(SMTP_EMAIL, ALERT_RECIPIENT, msg.as_string())
        except Exception as e:
            st.error(f"Email alert failed: {e}")
    elif via == "teams":
        try:
            response = requests.post(
                TEAMS_WEBHOOK,
                json={"text": message},
                headers={"Content-Type": "application/json"}
            )
            if response.status_code != 200:
                st.error(f"Teams alert failed: {response.status_code} - {response.text}")
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
            <h1 style="margin: 0; padding-left: 20px;">🚨 EscalateAI – AI Based Customer Escalation Prediction & Management Tool </h1>
        </div>
    </header>
    """,
    unsafe_allow_html=True
)


import streamlit as st
import datetime
import pandas as pd

st.sidebar.markdown("""
    <style>
    .sidebar-title h2 {
        font-size: 20px;
        margin-bottom: 4px;
        font-weight: 600;
        color: #2c3e50;  /* Deep Slate */
        text-align: center;
    }
    .sidebar-subtext {
        font-size: 13px;
        color: #7f8c8d;  /* Cool Gray */
        text-align: center;
        margin-bottom: 10px;
    }
    .sidebar-content {
        background-color: #ecf0f1; /* Soft Light Gray */
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
    }
    .sidebar-content label, .sidebar-content select, .sidebar-content input {
        font-size: 12px;
        color: #34495e;
    }
    </style>
    <div class="sidebar-title">
        <h2>⚙️ EscalateAI Controls</h2>
    </div>
    <div class="sidebar-subtext">
        Manage, monitor & respond with agility.
    </div>
    <div class="sidebar-content">
        <!-- 👇 Place your Streamlit widgets here -->
        <!-- Example:
        st.selectbox("Choose escalation level", ["Low", "Medium", "High"])
        -->
    </div>
""", unsafe_allow_html=True)

# 📥 Upload Section

import pandas as pd
import streamlit as st

import pandas as pd
import streamlit as st

# === 1. Upload Excel File from Sidebar ===
st.sidebar.header("📁 Upload Escalation Sheet")
uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx"])

if uploaded_file:
    try:
        df_excel = pd.read_excel(uploaded_file)
        st.success("✅ Excel file loaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to read Excel file: {e}")
        st.stop()

    # === 2. Validate Required Columns ===
    required_columns = ["Customer", "Issue"]
    missing_cols = [col for col in required_columns if col not in df_excel.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        st.stop()

    # === 3. Process Each Row ===
    for idx, row in df_excel.iterrows():
        issue = str(row.get("Issue", "")).strip()
        customer = str(row.get("Customer", "Unknown")).strip()

        if not issue:
            st.warning(f"⚠️ Row {idx + 1} skipped: empty issue text.")
            continue

        # === 4. Summarize and Classify ===
        issue_summary = summarize_issue_text(issue)
        sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)

        # === 5. Insert into Escalation Tracker ===
        insert_escalation(
            customer,
            issue_summary,
            sentiment,
            urgency,
            severity,
            criticality,
            category,
            escalation_flag
        )

    st.sidebar.success("🎯 All valid rows processed successfully.")
# 📤 Download Section
st.sidebar.markdown("### 📤 Downloads")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("⬇️ All Complaints"):
        csv = fetch_escalations().to_csv(index=False)
        st.download_button("Download CSV", csv, file_name="escalations.csv", mime="text/csv")
with col2:
    if st.button("⬇️ Escalated Only"):
        df_esc = fetch_escalations()
        df_esc = df_esc[df_esc["escalated"] == "Yes"]
        if df_esc.empty:
            st.info("No escalated cases.")
        else:
            with pd.ExcelWriter("escalated_cases.xlsx") as writer:
                df_esc.to_excel(writer, index=False)
            with open("escalated_cases.xlsx", "rb") as f:
                st.download_button("Download Excel", f, file_name="escalated_cases.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# 📩 Email Fetching
st.sidebar.markdown("### 📩 Email Integration")
if st.sidebar.button("Fetch Emails"):
    emails = parse_emails()
    for e in emails:
        issue, customer = e["issue"], e["customer"]
        sentiment, urgency, severity, criticality, category, escalation_flag = analyze_issue(issue)
        insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag)
    st.sidebar.success(f"✅ {len(emails)} emails processed")
    #st.info(f"Fetched {len(messages[0].split())} unseen message(s)")

# ⏰ SLA Monitoring
st.sidebar.markdown("### ⏰ SLA Monitor")
if st.sidebar.button("Trigger SLA Check"):
    df = fetch_escalations()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    breaches = df[(df['status'] != 'Resolved') & (df['priority'] == 'high') &
                  ((datetime.datetime.now() - df['timestamp']) > datetime.timedelta(minutes=10))]
    if not breaches.empty:
        alert_msg = f"🚨 SLA breach for {len(breaches)} case(s)!"
        send_alert(alert_msg, via="teams")
        send_alert(alert_msg, via="email")
        st.sidebar.success("✅ Alerts sent")
    else:
        st.sidebar.info("All SLAs healthy")

# 🚨 SLA Banner
df_all = fetch_escalations()
df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], errors='coerce')
breaches = df_all[(df_all['status'] != 'Resolved') & (df_all['priority'] == 'high') &
                  ((datetime.datetime.now() - df_all['timestamp']) > datetime.timedelta(minutes=10))]
if not breaches.empty:
    st.sidebar.markdown(
        f"<div style='background:#dc3545;padding:8px;border-radius:5px;color:white;text-align:center;'>"
        f"<strong>🚨 {len(breaches)} SLA Breach(s) Detected</strong></div>",
        unsafe_allow_html=True
    )

# 🔍 Filters
st.sidebar.markdown("### 🔍 Escalation Filters")
df = fetch_escalations()
status = st.sidebar.selectbox("Status", ["All"] + sorted(df["status"].dropna().unique()))
severity = st.sidebar.selectbox("Severity", ["All"] + sorted(df["severity"].dropna().unique()))
sentiment = st.sidebar.selectbox("Sentiment", ["All"] + sorted(df["sentiment"].dropna().unique()))
category = st.sidebar.selectbox("Category", ["All"] + sorted(df["category"].dropna().unique()))
view = st.sidebar.radio("Escalation View", ["All", "Escalated", "Non-Escalated"])

filtered_df = df.copy()
if status != "All":
    filtered_df = filtered_df[filtered_df["status"] == status]
if severity != "All":
    filtered_df = filtered_df[filtered_df["severity"] == severity]
if sentiment != "All":
    filtered_df = filtered_df[filtered_df["sentiment"] == sentiment]
if category != "All":
    filtered_df = filtered_df[filtered_df["category"] == category]
if view == "Escalated":
    filtered_df = filtered_df[filtered_df["escalated"] == "Yes"]
elif view == "Non-Escalated":
    filtered_df = filtered_df[filtered_df["escalated"] != "Yes"]

# 🔔 Manual Alerts
st.sidebar.markdown("### 🔔 Manual Notifications")
msg = st.sidebar.text_area("Compose Alert", "🚨 Test alert from EscalateAI")
if st.sidebar.button("Send MS Teams"):
    send_alert(msg, via="teams")
    st.sidebar.success("✅ MS Teams alert sent")
if st.sidebar.button("Send Email"):
    send_alert(msg, via="email")
    st.sidebar.success("✅ Email alert sent")

# 📲 WhatsApp Notification
st.sidebar.markdown("### 📲 WhatsApp Alerts")
status_check = st.sidebar.selectbox("Case Status", ["Open", "In Progress", "Resolved"])
if status_check == "Resolved":
    phone = st.sidebar.text_input("Phone Number", "+91")
    msg = st.sidebar.text_area("Message", "Your issue has been resolved.")
    if st.sidebar.button("Send WhatsApp"):
        # send_whatsapp_message(phone, msg)
        st.sidebar.success(f"✅ WhatsApp sent to {phone}")
else:
    st.sidebar.info("Available only for 'Resolved' cases.")

import os
import smtplib
import requests
from email.message import EmailMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ENV values
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT"))
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")

def send_alert(message, via="email", recipient=None):
    if via == "email":
        try:
            msg = EmailMessage()
            msg['Subject'] = "🔔 Escalation Update Notification"
            msg['From'] = EMAIL_USER
            msg['To'] = recipient if recipient else EMAIL_USER  # fallback to sender
            msg.set_content(message)

            with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_USER, EMAIL_PASS)
                server.send_message(msg)
            print(f"✅ Email alert sent to {msg['To']}")
        except Exception as e:
            print(f"❌ Email sending failed: {e}")

    elif via == "teams":
        try:
            payload = {"text": message}
            response = requests.post(MS_TEAMS_WEBHOOK_URL, json=payload)
            if response.status_code == 200:
                print("✅ Teams alert sent successfully")
            else:
                print(f"❌ Teams alert failed with status {response.status_code}: {response.text}")
        except Exception as e:
            print(f"❌ Teams sending failed: {e}")
            
# --- Main Tabs ---
tabs = st.tabs(["🗃️ All", "🚩 Escalated", "🔁 Feedback & Retraining"])

# --- All escalations tab with Kanban board ---
with tabs[0]:
    st.subheader("📊 Escalation Kanban Board")

    df = filtered_df
    counts = df['status'].value_counts()
    open_count = counts.get('Open', 0)
    inprogress_count = counts.get('In Progress', 0)
    resolved_count = counts.get('Resolved', 0)
    st.markdown(f"**Open:** {open_count} | **In Progress:** {inprogress_count} | **Resolved:** {resolved_count}")

    col1, col2, col3 = st.columns(3)
for status, col in zip(["Open", "In Progress", "Resolved"], [col1, col2, col3]):
    with col:
        col.markdown(f"<h3 style='background-color:{STATUS_COLORS[status]};color:white;padding:8px;border-radius:5px;text-align:center;'>{status}</h3>", unsafe_allow_html=True)
        bucket = df[df["status"] == status]

        for i, row in bucket.iterrows():
            flag = "🚩" if row['escalated'] == 'Yes' else ""
            header_color = SEVERITY_COLORS.get(row['severity'], "#000000")
            urgency_color = URGENCY_COLORS.get(row['urgency'], "#000000")
            summary = summarize_issue_text(row['issue'])
            expander_label = f"{row['id']} - {row['customer']} {flag} – {summary}"

            with st.expander(expander_label, expanded=False):
                colA, colB, colC = st.columns(3)

                try:
                    timestamp = pd.to_datetime(row["timestamp"])
                    now = datetime.datetime.now()
                    ageing_timedelta = now - timestamp
                    days = ageing_timedelta.days
                    hours, remainder = divmod(ageing_timedelta.seconds, 3600)
                    minutes, _ = divmod(remainder, 60)
                    ageing_str = f"{days}d {hours}h {minutes}m"
                    
                    # Convert total age to hours for color coding
                    total_hours = ageing_timedelta.total_seconds() / 3600
                
                    if total_hours < 12:
                        ageing_color = "#2ecc71"  # Green
                    elif 12 <= total_hours < 24:
                        ageing_color = "#e67e22"  # Orange
                    else:
                        ageing_color = "#e74c3c"  # Red
                
                except:
                    ageing_str = "N/A"
                    ageing_color = "#7f8c8d"  # Grey if error
                
                # Display ageing with color
                st.markdown(f"**⏱️ Ageing:** <span style='color:{ageing_color}; font-weight:bold;'>{ageing_str}</span>", unsafe_allow_html=True)

                if colA.button("✔️ Mark as Resolved", key=f"resolved_{row['id']}"):
                    owner_email = row.get("owner_email", EMAIL_USER)
                    update_escalation_status(row['id'], "Resolved", row["action_taken"], row["owner"], owner_email)
                    send_alert("Case marked as resolved.", via="email", recipient=owner_email)
                    send_alert("Case marked as resolved.", via="email", recipient=row["owner_email"])
                    send_alert("Case marked as resolved.", via="teams", recipient=row["owner_email"])

                if colB.button("🚀 Escalate to Tier 2", key=f"tier2_{row['id']}"):
                    escalate_to_tier_2(row["id"])
                    send_alert("Case escalated to Tier 2.", via="teams", recipient="tier2@company.com")

                if colC.button("📣 Notify Manager", key=f"notify_mgr_{row['id']}"):
                    send_alert(f"Escalation #{row['id']} needs review", via="email", recipient="manager@company.com")

                st.markdown(f"**Issue:** {row['issue']}")
                st.markdown(f"**Severity:** <span style='color:{header_color};font-weight:bold;'>{row['severity']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Criticality:** {row['criticality']}")
                st.markdown(f"**Category:** {row['category']}")
                st.markdown(f"**Sentiment:** {row['sentiment']}")
                st.markdown(f"**Urgency:** <span style='color:{urgency_color};font-weight:bold;'>{row['urgency']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Escalated:** {row['escalated']}")

                new_status = st.selectbox("Update Status", ["Open", "In Progress", "Resolved"],
                                          index=["Open", "In Progress", "Resolved"].index(row["status"]),
                                          key=f"status_{row['id']}")
                new_action = st.text_input("Action Taken", row.get("action_taken", ""), key=f"action_{row['id']}")
                new_owner = st.text_input("Owner", row.get("owner", ""), key=f"owner_{row['id']}")
                new_owner_email = st.text_input("Owner Email", row.get("owner_email", ""), key=f"email_{row['id']}")

                if st.button("💾 Save Changes", key=f"save_{row['id']}"):
                    owner_email = row.get("owner_email", EMAIL_USER)
                    update_escalation_status(row['id'], "Resolved", row["action_taken"], row["owner"], owner_email)
                    send_alert("Case marked as resolved.", via="email", recipient=owner_email)
                    st.success("Escalation updated.")

                    notification_message = f"""
                    🔔 Hello {new_owner},

                    The escalation case #{row['id']} assigned to you has been updated:

                    • Status: {new_status}
                    • Action Taken: {new_action}
                    • Category: {row['category']}
                    • Severity: {row['severity']}
                    • Urgency: {row['urgency']}
                    • Sentiment: {row['sentiment']}

                    Please review the updates on the EscalateAI dashboard.
                    """

                    send_alert(notification_message.strip(), via="email", recipient=new_owner_email)
                    send_alert(notification_message.strip(), via="teams", recipient=new_owner_email)
    
# --- Escalated issues tab ---
with tabs[1]:
    st.subheader("🚩 Escalated Issues")
    df = filtered_df
    df_esc = df[df["escalated"] == "Yes"]
    st.dataframe(df_esc)

with tabs[2]:
    st.subheader("🔁 Feedback & Retraining")
    df = fetch_escalations()
    df_feedback = df[df["escalated"].notnull()]
    fb_map = {"Correct": 1, "Incorrect": 0}

    for _, row in df_feedback.iterrows():
        with st.expander(f"🆔 {row['id']}"):
            fb = st.selectbox("Escalation Accuracy", ["Correct", "Incorrect"], key=f"fb_{row['id']}")
            sent = st.selectbox("Sentiment", ["Positive", "Neutral", "Negative"], key=f"sent_{row['id']}")
            crit = st.selectbox("Criticality", ["Low", "Medium", "High", "Urgent"], key=f"crit_{row['id']}")
            notes = st.text_area("Notes", key=f"note_{row['id']}")
            if st.button("Submit", key=f"btn_{row['id']}"):
                owner_email = row.get("owner_email", EMAIL_USER)
                update_escalation_status(row['id'], "Resolved", row["action_taken"], row["owner"], owner_email)
                send_alert("Case marked as resolved.", via="email", recipient=owner_email)
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


