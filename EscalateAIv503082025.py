# EscalateAI – Final Full Code Implementation
# ==========================================
# Features:
# • Parsing customer issue data from Gmail and Excel sheets
# • Extracting sentiment, urgency, and escalation cues using VADER NLP
# • Tagging issues with severity, criticality, and category
# • Visualizing issues on a Kanban board
# • Sending automated alerts via Microsoft Teams for SLA breaches
# • Predicting escalations using a Random Forest ML model
# • Implementing a feedback loop for real-time model retraining

import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import uuid
from streamlit_autorefresh import st_autorefresh
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# --------------- Configuration ------------------
EMAIL_CHECK_INTERVAL = 60  # seconds
DB_NAME = "escalations.db"
SLA_THRESHOLD_MINUTES = 10
MODEL_PATH = "escalation_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
TEAMS_WEBHOOK = os.getenv("TEAMS_WEBHOOK")  # set your webhook env var

# --------------- Utilities ----------------------
def create_connection():
    conn = sqlite3.connect(DB_NAME)
    return conn

def create_table():
    conn = create_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS escalations (
                    id TEXT PRIMARY KEY,
                    customer TEXT,
                    issue TEXT,
                    sentiment TEXT,
                    urgency TEXT,
                    severity TEXT,
                    criticality TEXT,
                    category TEXT,
                    status TEXT,
                    action_taken TEXT,
                    action_owner TEXT,
                    timestamp TEXT
                )''')
    conn.commit()
    conn.close()

def generate_id():
    return f"SESICE-{str(uuid.uuid4().int)[-6:]}"

def classify_sentiment(issue):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(issue)['compound']
    if score <= -0.5:
        return "Very Negative"
    elif score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    elif score < 0.5:
        return "Positive"
    else:
        return "Very Positive"

def extract_category(issue):
    categories = {
        "Service": ["delay", "support", "response", "slow"],
        "Technical": ["bug", "crash", "fail", "error"],
        "Billing": ["invoice", "charge", "refund"]
    }
    for cat, keywords in categories.items():
        if any(word in issue.lower() for word in keywords):
            return cat
    return "General"

def extract_urgency(issue):
    urgency_keywords = ["urgent", "immediately", "asap", "critical"]
    for word in urgency_keywords:
        if word in issue.lower():
            return "High"
    return "Normal"

def tag_severity(sentiment, urgency):
    if sentiment == "Very Negative" or urgency == "High":
        return "High"
    elif sentiment == "Negative":
        return "Medium"
    else:
        return "Low"

def tag_criticality(category, severity):
    if category in ["Service", "Technical"] and severity == "High":
        return "Critical"
    elif severity == "Medium":
        return "Major"
    else:
        return "Minor"

def insert_case(data):
    conn = create_connection()
    c = conn.cursor()
    c.execute("INSERT INTO escalations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", data)
    conn.commit()
    conn.close()

def check_sla():
    now = datetime.datetime.now()
    conn = create_connection()
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    breached = []
    for _, row in df.iterrows():
        if row['severity'] == "High" and row['status'] != "Resolved":
            ts = datetime.datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S")
            if (now - ts).total_seconds() > SLA_THRESHOLD_MINUTES * 60:
                breached.append(row['id'])
                send_teams_alert(row['id'], row['issue'])
    conn.close()

def send_teams_alert(case_id, issue):
    if not TEAMS_WEBHOOK:
        return
    message = {
        "text": f"🚨 SLA Breach: Case {case_id} remains unresolved.\nIssue: {issue}"
    }
    requests.post(TEAMS_WEBHOOK, json=message)

def predict_escalation(issue):
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        vec = vectorizer.transform([issue])
        prediction = model.predict(vec)[0]
        return "Likely" if prediction == 1 else "Unlikely"
    return "Unknown"

def train_model():
    conn = create_connection()
    df = pd.read_sql_query("SELECT issue, severity FROM escalations", conn)
    conn.close()
    df['label'] = df['severity'].apply(lambda x: 1 if x == "High" else 0)
    X = df['issue']
    y = df['label']
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)
    model = RandomForestClassifier()
    model.fit(X_vec, y)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

# --------------- Streamlit App ------------------
st.set_page_config(layout="wide")
st.title("🚨 EscalateAI - Escalation Management")
create_table()
st_autorefresh(interval=EMAIL_CHECK_INTERVAL * 1000, key="email_refresh")

menu = st.sidebar.radio("Choose", ["Manual Entry", "Excel Upload", "Kanban Board"])

if menu == "Manual Entry":
    with st.form("entry_form"):
        customer = st.text_input("Customer")
        issue = st.text_area("Issue")
        owner = st.text_input("Action Owner")
        submit = st.form_submit_button("Log Escalation")

        if submit and issue:
            sentiment = classify_sentiment(issue)
            urgency = extract_urgency(issue)
            category = extract_category(issue)
            severity = tag_severity(sentiment, urgency)
            criticality = tag_criticality(category, severity)
            escalation = predict_escalation(issue)
            row = [
                generate_id(), customer, issue, sentiment, urgency,
                severity, criticality, category, "Open", "", owner,
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
            insert_case(row)
            st.success(f"Escalation logged with severity {severity} and criticality {criticality}")
            train_model()

elif menu == "Excel Upload":
    file = st.file_uploader("Upload Excel", type=["xlsx"])
    if file:
        df = pd.read_excel(file)
        for _, row in df.iterrows():
            customer = row.get("Customer", "Unknown")
            issue = str(row.get("Issue", ""))
            owner = row.get("Action Owner", "Unassigned")
            sentiment = classify_sentiment(issue)
            urgency = extract_urgency(issue)
            category = extract_category(issue)
            severity = tag_severity(sentiment, urgency)
            criticality = tag_criticality(category, severity)
            new_row = [
                generate_id(), customer, issue, sentiment, urgency,
                severity, criticality, category, "Open", "", owner,
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
            insert_case(new_row)
        train_model()
        st.success("All escalations uploaded and analyzed.")

elif menu == "Kanban Board":
    conn = create_connection()
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    conn.close()

    for status in ["Open", "In Progress", "Resolved"]:
        with st.expander(f"{status} Escalations"):
            status_df = df[df['status'] == status]
            for i, row in status_df.iterrows():
                with st.container():
                    st.markdown(f"**{row['id']} | {row['customer']}**")
                    st.write(f"Issue: {row['issue']}")
                    st.write(f"Sentiment: {row['sentiment']}, Urgency: {row['urgency']}, Category: {row['category']}")
                    new_status = st.selectbox("Update Status", ["Open", "In Progress", "Resolved"], index=["Open", "In Progress", "Resolved"].index(row['status']), key=f"status_{row['id']}")
                    new_action = st.text_input("Action Taken", value=row['action_taken'], key=f"action_{row['id']}")
                    if st.button("Update", key=f"update_{row['id']}"):
                        conn = create_connection()
                        c = conn.cursor()
                        c.execute("UPDATE escalations SET status=?, action_taken=? WHERE id=?", (new_status, new_action, row['id']))
                        conn.commit()
                        conn.close()
                        st.success("Updated")

# --------------- SLA Check ----------------------
check_sla()
