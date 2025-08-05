# 🚨 EscalateAI 2.0 – Customer Escalation System
# Includes all enhancements: alerts, NLP, dashboard, audit logs, WhatsApp, ML metrics

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import time
import datetime
import threading
import smtplib
import requests
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load environment variables
load_dotenv()
EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
SMTP_EMAIL = EMAIL_USER
SMTP_PASS = EMAIL_PASS
ALERT_RECIPIENT = os.getenv("EMAIL_RECEIVER")
TEAMS_WEBHOOK = os.getenv("MS_TEAMS_WEBHOOK_URL")
DB_PATH = "escalations.db"
ESCALATION_PREFIX = "SESICE-25"

# Initialize locks & analyzer
id_lock = threading.Lock()
analyzer = SentimentIntensityAnalyzer()
NEGATIVE_KEYWORDS = {
    "technical": ["fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip", "malfunction", "blank", "shutdown", "discharge", "leak"],
    "dissatisfaction": ["dissatisfy", "frustrate", "complain", "reject", "delay", "ignore", "escalate", "displease", "noncompliance", "neglect"],
    "support": ["wait", "pending", "slow", "incomplete", "miss", "omit", "unresolved", "shortage", "no response"],
    "safety": ["fire", "burn", "flashover", "arc", "explode", "unsafe", "leak", "corrode", "alarm", "incident"],
    "business": ["impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate", "penalty"]
}

def ensure_schema():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS escalations (
            id TEXT PRIMARY KEY, customer TEXT, issue TEXT, sentiment TEXT, urgency TEXT,
            severity TEXT, criticality TEXT, category TEXT, status TEXT, timestamp TEXT,
            action_taken TEXT, owner TEXT, escalated TEXT, priority TEXT, escalation_flag TEXT,
            action_owner TEXT, status_update_date TEXT, user_feedback TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS audit_log (
            escalation_id TEXT, field TEXT, old_value TEXT, new_value TEXT, timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def get_next_escalation_id():
    with id_lock:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(f'''
            SELECT id FROM escalations WHERE id LIKE '{ESCALATION_PREFIX}%' ORDER BY id DESC LIMIT 1
        ''')
        last = c.fetchone()
        conn.close()
        next_num = int(last[0].replace(ESCALATION_PREFIX, "")) + 1 if last else 1
        return f"{ESCALATION_PREFIX}{str(next_num).zfill(5)}"

def analyze_issue(issue):
    score = analyzer.polarity_scores(issue)
    sentiment = "positive" if score["compound"] > 0.05 else "negative" if score["compound"] < -0.05 else "neutral"
    urgency = "high" if any(w in issue.lower() for kws in NEGATIVE_KEYWORDS.values() for w in kws) else "normal"
    category = next((cat for cat, kws in NEGATIVE_KEYWORDS.items() if any(k in issue.lower() for k in kws)), None)
    severity = "critical" if category in ["technical", "safety"] else "major" if category in ["support", "business"] else "minor"
    criticality = "high" if sentiment == "negative" and urgency == "high" else "medium"
    escalation_flag = "Yes" if urgency == "high" or sentiment == "negative" else "No"
    return sentiment, urgency, severity, criticality, category, escalation_flag

def insert_escalation(customer, issue, sentiment, urgency, severity, criticality, category, escalation_flag):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    new_id = get_next_escalation_id()
    now = datetime.datetime.now().isoformat()
    c.execute('''
        INSERT INTO escalations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (new_id, customer, issue, sentiment, urgency, severity, criticality, category,
          "Open", now, "", "", escalation_flag, "normal", escalation_flag, "", "", ""))
    conn.commit()
    conn.close()

def send_alert(message, via="email"):
    try:
        if via == "email":
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_EMAIL, SMTP_PASS)
                server.sendmail(SMTP_EMAIL, ALERT_RECIPIENT, message)
        elif via == "teams":
            requests.post(TEAMS_WEBHOOK, json={"text": message})
    except Exception as e:
        st.error(f"Alert error: {e}")

def send_status_email(escalation_id, new_status):
    message = f"Dear customer,\n\nYour escalation {escalation_id} status has been updated to: {new_status}\nRegards,\nSupport Team"
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASS)
            server.sendmail(SMTP_EMAIL, ALERT_RECIPIENT, message)
    except: pass

def update_escalation_status(escalation_id, new_status, action_taken, action_owner, feedback=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT status FROM escalations WHERE id = ?", (escalation_id,))
    old_status = c.fetchone()[0]
    c.execute('''
        UPDATE escalations SET status=?, action_taken=?, action_owner=?, status_update_date=?, user_feedback=? WHERE id=?
    ''', (new_status, action_taken, action_owner, datetime.datetime.now().isoformat(), feedback, escalation_id))
    c.execute('''
        INSERT INTO audit_log VALUES (?, ?, ?, ?, ?)
    ''', (escalation_id, "status", old_status, new_status, datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()
    send_status_email(escalation_id, new_status)
    # ML Model Training
def train_model():
    df = fetch_escalations()
    df = df.dropna(subset=['sentiment', 'urgency', 'severity', 'criticality', 'escalated'])
    if df.shape[0] < 20 or df['escalated'].nunique() < 2:
        st.warning("Not enough data to train model.")
        return None
    X = pd.get_dummies(df[['sentiment', 'urgency', 'severity', 'criticality']])
    y = df['escalated'].apply(lambda x: 1 if x == 'Yes' else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    st.success(f"Model Accuracy: {acc:.2f}")
    return model

def predict_escalation(model, sentiment, urgency, severity, criticality):
    X_pred = pd.DataFrame([{
        f"sentiment_{sentiment}": 1,
        f"urgency_{urgency}": 1,
        f"severity_{severity}": 1,
        f"criticality_{criticality}": 1
    }])
    X_pred = X_pred.reindex(columns=model.feature_names_in_, fill_value=0)
    pred = model.predict(X_pred)
    return "Yes" if pred[0] == 1 else "No"

# Feedback parsing
def parse_feedback(feedback_text):
    text = feedback_text.lower()
    if "wrong" in text or "not needed" in text: return 0
    elif "correct" in text or "appropriate" in text: return 1
    else: return None

# Streamlit UI
ensure_schema()
st.set_page_config(layout="wide")
st.title("🚨 EscalateAI 2.0 – Customer Escalation Platform")

# Sidebar controls
st.sidebar.header("⚙️ Controls")
if st.sidebar.button("📣 Send Manual Alert to MS Teams"):
    send_alert("Manual alert triggered from dashboard", via="teams")
if st.sidebar.button("📧 Send Manual Alert via Email"):
    send_alert("Manual alert triggered from dashboard", via="email")

# Main tabs
tabs = st.tabs(["📊 Dashboard", "🗃️ Kanban", "🚩 Escalated", "🔁 Feedback & Retrain", "🧪 Dev Panel"])

# Dashboard Tab
with tabs[0]:
    st.subheader("📊 Summary Dashboard")
    df = fetch_escalations()
    st.metric("Total Escalations", len(df))
    st.metric("Resolved", df[df["status"] == "Resolved"].shape[0])
    st.metric("Negative Sentiment", df[df["sentiment"] == "negative"].shape[0])
    st.bar_chart(df["category"].value_counts())
    st.bar_chart(df["urgency"].value_counts())
    st.bar_chart(df["severity"].value_counts())

# Kanban Board
with tabs[1]:
    st.subheader("🗃️ Escalation Kanban")
    df = fetch_escalations()
    statuses = ["Open", "In Progress", "Resolved"]
    cols = st.columns(len(statuses))
    for i, status in enumerate(statuses):
        with cols[i]:
            st.markdown(f"<h4 style='color:{STATUS_COLORS[status]}'>{status}</h4>", unsafe_allow_html=True)
            bucket = df[df["status"] == status]
            for _, row in bucket.iterrows():
                with st.expander(f"{row['id']} - {row['customer']}"):
                    st.markdown(f"**Issue**: {row['issue']}")
                    st.markdown(f"**Sentiment**: {row['sentiment']}")
                    st.markdown(f"**Urgency**: {row['urgency']}")
                    st.markdown(f"**Severity**: {row['severity']}")
                    st.markdown(f"**Criticality**: {row['criticality']}")
                    st.markdown(f"**Escalated**: {row['escalated']}")
                    new_status = st.selectbox("Update Status", statuses, index=statuses.index(row["status"]), key=f"s_{row['id']}")
                    new_action = st.text_input("Action Taken", row.get("action_taken",""), key=f"a_{row['id']}")
                    new_owner = st.text_input("Owner", row.get("owner",""), key=f"o_{row['id']}")
                    if st.button("💾 Save", key=f"save_{row['id']}"):
                        update_escalation_status(row['id'], new_status, new_action, new_owner)
                        if new_status == "Resolved":
                            send_whatsapp_message("whatsapp:+911234567890", f"Your escalation {row['id']} has been resolved.")
                        st.success("Saved successfully.")

# Escalated Cases Tab
with tabs[2]:
    st.subheader("🚩 Escalated Issues")
    df = fetch_escalations()
    st.dataframe(df[df["escalated"] == "Yes"])

# Feedback & Retrain Tab
with tabs[3]:
    st.subheader("🔁 Feedback & Retraining")
    df = fetch_escalations()
    for _, row in df.iterrows():
        feedback = st.text_input(f"Feedback on {row['id']}", key=f"fb_{row['id']}")
        if st.button(f"Submit Feedback for {row['id']}", key=f"fb_btn_{row['id']}"):
            score = parse_feedback(feedback)
            if score is not None:
                update_escalation_status(row['id'], row["status"], row.get("action_taken",""), row.get("owner",""), score)
                st.success("Feedback recorded.")

    if st.button("🔁 Retrain ML Model"):
        model = train_model()
        if model:
            st.success("Model retrained.")
        else:
            st.warning("Model training failed.")

# Dev Panel Tab
with tabs[4]:
    st.subheader("🧪 Developer Utilities")
    st.write("Raw Escalation Data")
    st.dataframe(fetch_escalations())
    if st.button("🧨 Reset Database (Dev Only)"):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS escalations")
        c.execute("DROP TABLE IF EXISTS audit_log")
        conn.commit()
        conn.close()
        st.warning("Database wiped. Refresh app to reinitialize.")
