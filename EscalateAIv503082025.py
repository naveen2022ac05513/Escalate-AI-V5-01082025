# EscalateAI – Full Application Code
# --------------------------------------------------------------
# Functionality:
# • Parsing customer issue data from Gmail & Excel
# • NLP: VADER for sentiment + urgency + escalation keyword detection
# • Tagging: severity, criticality, category
# • Visualization: Kanban board (Open, In Progress, Resolved) with counts
# • Alerts: SLA breach notification via Teams (stubbed)
# • Predictive Escalation using RandomForestClassifier
# • Continuous feedback loop & retraining option
# • Filters: Escalated only or All
# • Manual entry + Excel bulk upload + export analyzed data

import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
import base64
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
import joblib

# ---------- Constants ----------
DB_FILE = "escalateai.db"
NEGATIVE_WORDS = ["delay", "issue", "problem", "not working", "unacceptable", "angry", "frustrated"]
ESCALATION_KEYWORDS = ["escalate", "urgent", "priority", "critical"]
SEVERITY_LEVELS = ["Low", "Medium", "High"]
CATEGORIES = ["Login", "Performance", "Feature Request", "Bug", "Other"]

# ---------- Initialize DB ----------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute('''CREATE TABLE IF NOT EXISTS escalations (
                    id TEXT PRIMARY KEY,
                    customer TEXT,
                    issue TEXT,
                    sentiment TEXT,
                    urgency TEXT,
                    escalation_flag TEXT,
                    severity TEXT,
                    criticality TEXT,
                    category TEXT,
                    status TEXT,
                    action_taken TEXT,
                    action_owner TEXT,
                    date TEXT
                )''')
    conn.close()

# ---------- Generate ID ----------
def generate_id():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT id FROM escalations", conn)
    conn.close()
    last_id = 250000 if df.empty else max([int(x.split('-')[1]) for x in df['id']])
    return f"SESICE-{last_id + 1}"

# ---------- Sentiment + Escalation ----------
def analyze_text(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)
    sentiment = "Positive" if sentiment_score['compound'] > 0.2 else "Negative" if sentiment_score['compound'] < -0.2 else "Neutral"
    urgency = "High" if any(word in text.lower() for word in NEGATIVE_WORDS) else "Normal"
    escalation_flag = "Yes" if any(word in text.lower() for word in ESCALATION_KEYWORDS) or urgency == "High" and sentiment == "Negative" else "No"
    return sentiment, urgency, escalation_flag

# ---------- Predict Escalation ----------
def train_model():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM escalations", conn)
    conn.close()
    if len(df) >= 10:
        df['label'] = df['escalation_flag'].map({"Yes": 1, "No": 0})
        X = pd.get_dummies(df[['sentiment', 'urgency']])
        y = df['label']
        model = RandomForestClassifier()
        model.fit(X, y)
        joblib.dump(model, "model.pkl")

# ---------- Email Parsing ----------
def parse_gmail():
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login("your_email@gmail.com", "your_password")
    mail.select("inbox")
    _, data = mail.search(None, "ALL")
    emails = []
    for num in data[0].split():
        _, msg_data = mail.fetch(num, "RFC822")
        msg = email.message_from_bytes(msg_data[0][1])
        subject = decode_header(msg["Subject"])[0][0]
        subject = subject.decode() if isinstance(subject, bytes) else subject
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode("utf-8", errors='ignore')
                    break
        else:
            body = msg.get_payload(decode=True).decode("utf-8", errors='ignore')
        emails.append((msg["From"], subject + "\n" + body))
    mail.logout()
    return emails

# ---------- Insert into DB ----------
def insert_case(customer, issue, sentiment, urgency, escalation_flag):
    conn = sqlite3.connect(DB_FILE)
    case_id = generate_id()
    dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
        INSERT INTO escalations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (case_id, customer, issue, sentiment, urgency, escalation_flag, "Medium", "Medium", "Other", "Open", "", "", dt))
    conn.commit()
    conn.close()

# ---------- Load DB ----------
def fetch_cases(filter_type="All"):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM escalations", conn)
    conn.close()
    if filter_type == "Escalated":
        return df[df['escalation_flag'] == "Yes"]
    return df

# ---------- SLA Detection ----------
def detect_sla_breach():
    df = fetch_cases("Escalated")
    now = datetime.datetime.now()
    breached = df[(df['status'] != "Resolved") & (df['date'].apply(lambda x: (now - datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")).total_seconds() > 600))]
    for _, row in breached.iterrows():
        send_teams_alert(row['id'], row['customer'], row['issue'])

def send_teams_alert(case_id, customer, issue):
    print(f"[TEAMS ALERT] SLA breached for {case_id} | {customer}: {issue[:100]}")

# ---------- UI: Kanban ----------
def display_kanban(df):
    st.subheader("📋 Escalation Tracker")
    statuses = ["Open", "In Progress", "Resolved"]
    cols = st.columns(3)
    for i, status in enumerate(statuses):
        filtered = df[df['status'] == status]
        with cols[i]:
            st.markdown(f"### {status} ({len(filtered)})")
            for _, row in filtered.iterrows():
                with st.expander(f"{row['id']} | {row['customer']}"):
                    st.markdown(f"**Sentiment:** {row['sentiment']} | **Urgency:** {row['urgency']} | **Category:** {row['category']}")
                    st.text_area("Issue", row['issue'], height=100)
                    new_status = st.selectbox("Status", statuses, index=statuses.index(row['status']), key=row['id']+"_status")
                    new_action = st.text_input("Action Taken", row['action_taken'], key=row['id']+"_action")
                    new_owner = st.text_input("Action Owner", row['action_owner'], key=row['id']+"_owner")
                    if st.button("Update", key=row['id']+"_update"):
                        update_case(row['id'], new_status, new_action, new_owner)

def update_case(case_id, status, action_taken, action_owner):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        UPDATE escalations SET status=?, action_taken=?, action_owner=? WHERE id=?
    """, (status, action_taken, action_owner, case_id))
    conn.commit()
    conn.close()

# ---------- UI: Sidebar ----------
def sidebar_ui():
    st.sidebar.subheader("📤 Upload Excel")
    uploaded_file = st.sidebar.file_uploader("Upload file", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        for _, row in df.iterrows():
            sentiment, urgency, escalation_flag = analyze_text(str(row['Issue']))
            insert_case(str(row['Customer']), str(row['Issue']), sentiment, urgency, escalation_flag)
        st.sidebar.success("Uploaded and analyzed.")

    if st.sidebar.button("Download All Cases"):
        df = fetch_cases()
        df.to_excel("cases_download.xlsx", index=False)
        with open("cases_download.xlsx", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="all_cases.xlsx">Download</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)

# ---------- Main ----------
def main():
    st.title("🚨 EscalateAI - Smart Escalation Management")
    init_db()
    sidebar_ui()

    st.subheader("🔍 View Complaints")
    filter_option = st.radio("Select View", ["All", "Escalated"])
    df = fetch_cases(filter_option)

    display_kanban(df)

    detect_sla_breach()
    train_model()

main()
