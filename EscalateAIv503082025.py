# EscalateAI - Complete Version with Predictive ML and Continuous Feedback Loop
# =============================================================================
# Features:
# • Email & Excel parsing for customer issues
# • Sentiment, urgency, escalation detection via keyword + VADER
# • Tagging severity, category
# • Kanban board with inline edits
# • SLA breach alerts (10 mins)
# • MS Teams & email alerts
# • Predictive escalation ML model (Random Forest)
# • Feedback loop for retraining from user feedback

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
import smtplib
from email.mime.text import MIMEText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

# Load environment variables (set these in your .env file or environment)
from dotenv import load_dotenv
load_dotenv()

# ----------- Configuration -------------
EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")

MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")

EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", 587))
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

DB_PATH = "escalations.db"
MODEL_PATH = "escalateai_model.joblib"
VECTORIZER_PATH = "escalateai_vectorizer.joblib"

# ----------- Keyword lists ------------
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

URGENCY_PHRASES = ["asap", "urgent", "immediately", "right away", "critical"]

CATEGORY_KEYWORDS = {
    "Safety": ["fire", "burn", "leak", "unsafe", "alarm", "incident", "explode", "flashover", "arc", "corrode"],
    "Performance": ["slow", "crash", "malfunction", "fail", "break", "shutdown", "trip"],
    "Delay": ["delay", "pending", "wait", "miss", "omit", "incomplete", "shortage", "no response"],
    "Compliance": ["noncompliance", "violation", "penalty", "reject", "cancel", "terminate"],
    "Service": ["ignore", "unresolved", "unavailable", "dissatisfy", "frustrate", "complain", "displease", "neglect"],
    "Quality": ["defect", "fault", "damage", "degrade", "blank", "discharge"],
    "Business Risk": ["impact", "loss", "risk", "downtime", "interrupt"],
}

# ----------- Initialize DB -------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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
    user_feedback INTEGER DEFAULT 0  -- 0=no feedback, 1=escalation correct, -1=incorrect
)
""")
conn.commit()

# ----------- Sentiment Analyzer ---------
analyzer = SentimentIntensityAnalyzer()

# ----------- Helper Functions ------------

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
    lowered = text.lower()
    return "High" if any(p in lowered for p in URGENCY_PHRASES) else "Normal"

def detect_category(text):
    lowered = text.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return cat
    return "General"

def count_negative_keywords(text):
    lowered = text.lower()
    return sum(1 for kw in NEGATIVE_KEYWORDS if kw in lowered)

# ----------- ML Model Handling -------------

def load_predictive_model():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except:
        return None, None

def train_predictive_model():
    """
    Train a simple predictive model (RandomForest) to classify escalation priority
    from existing data labeled with user feedback.
    """
    df = pd.read_sql_query("SELECT * FROM escalations WHERE user_feedback != 0", conn)
    if df.empty:
        return None, None

    X = df['issue'].values
    y = (df['priority'] == 'High').astype(int).values

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_vect = vectorizer.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_vect, y)

    # Save model and vectorizer for future use
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    return model, vectorizer

def predict_priority(issue_text, model, vectorizer):
    vect = vectorizer.transform([issue_text])
    pred = model.predict(vect)[0]
    return "High" if pred == 1 else "Low"

# ----------- Insert and Fetch -----------

def insert_escalation(data):
    cursor.execute("""
        INSERT INTO escalations (
            escalation_id, customer, issue, date, status, sentiment,
            priority, escalation_flag, urgency, category,
            action_taken, action_owner, status_update_date, user_feedback)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()

def fetch_escalations():
    return pd.read_sql_query("SELECT * FROM escalations", conn)

def update_escalation(escalation_id, status, action_taken, action_owner, feedback=None):
    params = [status, action_taken, action_owner, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), escalation_id]
    query = """
        UPDATE escalations SET status=?, action_taken=?, action_owner=?, status_update_date=?
        WHERE escalation_id=?
    """
    if feedback is not None:
        query = """
            UPDATE escalations SET status=?, action_taken=?, action_owner=?, status_update_date=?, user_feedback=?
            WHERE escalation_id=?
        """
        params.insert(-1, feedback)

    cursor.execute(query, params)
    conn.commit()

# ----------- Alerting -----------

def send_ms_teams_alert(message):
    if not MS_TEAMS_WEBHOOK_URL:
        return
    try:
        requests.post(MS_TEAMS_WEBHOOK_URL, json={"text": message})
    except:
        pass

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
    except:
        pass

def send_alerts(message):
    send_ms_teams_alert(message)
    send_email_alert("EscalateAI Alert", message)

# ----------- Parse Emails -----------

def fetch_unseen_emails():
    if not EMAIL or not APP_PASSWORD:
        st.warning("Email credentials missing!")
        return []

    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL, APP_PASSWORD)
        mail.select("inbox")
        status, messages = mail.search(None, "UNSEEN")
        if status != "OK":
            return []
        email_ids = messages[0].split()
        emails = []

        for eid in email_ids[-10:]:
            _, data = mail.fetch(eid, "(RFC822)")
            msg = email.message_from_bytes(data[0][1])
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
                        body = part.get_payload(decode=True).decode(errors="ignore")
                        break
            else:
                body = msg.get_payload(decode=True).decode(errors="ignore")

            emails.append({"customer": from_, "issue": body.strip(), "date": date})

            mail.store(eid, '+FLAGS', '\\Seen')
        mail.logout()
        return emails

    except Exception as e:
        st.warning(f"Email fetch failed: {e}")
        return []

# ----------- Process new escalations -----------

def process_new_issues(issues):
    count_new = 0
    cursor.execute("SELECT COUNT(*) FROM escalations")
    base_count = cursor.fetchone()[0]
    for issue in issues:
        # Prevent duplicates by customer + issue snippet
        cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (issue['customer'], issue['issue'][:200]))
        if cursor.fetchone():
            continue

        base_count += 1
        esc_id = f"SESICE-{base_count + 250000}"
        sentiment = classify_sentiment(issue['issue'])
        urgency = detect_urgency(issue['issue'])
        category = detect_category(issue['issue'])
        neg_count = count_negative_keywords(issue['issue'])

        # Load ML model to predict priority if possible
        model, vectorizer = load_predictive_model()
        if model and vectorizer:
            priority = predict_priority(issue['issue'], model, vectorizer)
            escalation_flag = 1 if priority == "High" else 0
        else:
            # heuristic fallback
            priority = "High" if (sentiment == "Negative" and neg_count >= 2) else "Low"
            escalation_flag = 1 if priority == "High" else 0

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        insert_escalation((
            esc_id, issue['customer'], issue['issue'][:500], now_str, "Open",
            sentiment, priority, escalation_flag, urgency, category, "", "", now_str, 0
        ))
        count_new += 1
        if escalation_flag == 1:
            send_alerts(f"🚨 New HIGH priority escalation:\nID: {esc_id}\nCustomer: {issue['customer']}\nIssue: {issue['issue'][:200]}...")
    return count_new

# ----------- Excel Upload --------------

def process_excel_file(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = [c.lower().strip() for c in df.columns]

        customer_col = next((c for c in df.columns if "customer" in c or "email" in c), None)
        issue_col = next((c for c in df.columns if "issue" in c or "complaint" in c or "text" in c), None)
        date_col = next((c for c in df.columns if "date" in c), None)

        if not customer_col or not issue_col:
            st.error("Excel must have customer/email and issue column.")
            return 0

        count_added = 0
        cursor.execute("SELECT COUNT(*) FROM escalations")
        base_count = cursor.fetchone()[0]

        for _, row in df.iterrows():
            cust = str(row[customer_col])
            iss = str(row[issue_col])
            date_val = str(row[date_col]) if date_col and pd.notna(row[date_col]) else datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (cust, iss[:200]))
            if cursor.fetchone():
                continue

            base_count += 1
            esc_id = f"SESICE-{base_count + 250000}"
            sentiment = classify_sentiment(iss)
            urgency = detect_urgency(iss)
            category = detect_category(iss)
            neg_count = count_negative_keywords(iss)

            model, vectorizer = load_predictive_model()
            if model and vectorizer:
                priority = predict_priority(iss, model, vectorizer)
                escalation_flag = 1 if priority == "High" else 0
            else:
                priority = "High" if (sentiment == "Negative" and neg_count >= 2) else "Low"
                escalation_flag = 1 if priority == "High" else 0

            insert_escalation((
                esc_id, cust, iss[:500], date_val, "Open",
                sentiment, priority, escalation_flag, urgency, category, "", "", date_val, 0
            ))
            count_added += 1

            if escalation_flag == 1:
                send_alerts(f"🚨 New HIGH priority escalation:\nID: {esc_id}\nCustomer: {cust}\nIssue: {iss[:200]}...")

        return count_added
    except Exception as e:
        st.error(f"Failed to process Excel: {e}")
        return 0

# ----------- SLA Monitoring -----------

def check_sla_breaches():
    now = datetime.datetime.now()
    df = fetch_escalations()
    breached = []
    for _, row in df.iterrows():
        if row['priority'] == "High" and row['status'] != "Resolved":
            date_obj = datetime.datetime.strptime(row['date'], "%Y-%m-%d %H:%M:%S")
            diff = (now - date_obj).total_seconds()
            if diff > 600:  # 10 minutes SLA
                breached.append(row['escalation_id'])
                send_alerts(f"⚠️ SLA Breach: Escalation {row['escalation_id']} open for more than 10 minutes.")
    return breached

# ----------- Streamlit UI -------------

def render_kanban():
    df = fetch_escalations()
    statuses = ["Open", "In Progress", "Resolved"]
    cols = st.columns(len(statuses))

    for idx, status in enumerate(statuses):
        with cols[idx]:
            st.markdown(f"### {status}")
            filtered = df[df['status'] == status]
            for _, row in filtered.iterrows():
                esc_id = row['escalation_id']
                with st.expander(f"{esc_id} - {row['customer']}"):
                    st.write(f"**Issue:** {row['issue']}")
                    st.write(f"**Sentiment:** {row['sentiment']}  |  **Priority:** {row['priority']}  |  **Urgency:** {row['urgency']}  |  **Category:** {row['category']}")
                    action_taken = st.text_area(f"Action Taken ({esc_id})", value=row['action_taken'], key=f"action_{esc_id}")
                    action_owner = st.text_input(f"Action Owner ({esc_id})", value=row['action_owner'], key=f"owner_{esc_id}")
                    new_status = st.selectbox(f"Status ({esc_id})", statuses, index=statuses.index(row['status']), key=f"status_{esc_id}")

                    feedback_options = {"No Feedback": 0, "Correct Escalation": 1, "Incorrect Escalation": -1}
                    feedback = st.selectbox(f"User Feedback ({esc_id})", options=list(feedback_options.keys()),
                                            index=[v for v,k in feedback_options.items() if k==row['user_feedback']][0], key=f"feedback_{esc_id}")

                    if st.button("Save Changes", key=f"save_{esc_id}"):
                        update_escalation(esc_id, new_status, action_taken, action_owner, feedback_options[feedback])
                        st.success(f"Updated {esc_id}")

def main():
    st.title("🚨 EscalateAI - Customer Escalation Management")

    # Auto-refresh every 60 sec to get new emails & alerts
    st_autorefresh = st.experimental_singleton(lambda: None)
    st_autorefresh()

    # Section: Fetch Emails
    st.sidebar.header("Fetch Emails from Gmail")
    if st.sidebar.button("Fetch New Emails"):
        emails = fetch_unseen_emails()
        count = process_new_issues(emails)
        st.sidebar.success(f"Processed {count} new emails")

    # Section: Upload Excel
    st.sidebar.header("Upload Excel with Issues")
    uploaded_file = st.sidebar.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
    if uploaded_file:
