import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import datetime
import threading
import time
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import logging
import imaplib
import email as pyemail
from email.header import decode_header

# Load environment variables from .env
load_dotenv()

# Constants and config
DB_PATH = "escalate_ai.db"
SLA_THRESHOLD_MINUTES = 10
ID_START = 2500001
NEGATIVE_WORDS = {
    "technical": ["fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip", "malfunction", "blank", "shutdown", "discharge"],
    "customer": ["dissatisfy", "frustrate", "complain", "reject", "delay", "ignore", "escalate", "displease", "noncompliance", "neglect"],
    "support": ["wait", "pending", "slow", "incomplete", "miss", "omit", "unresolved", "shortage", "no response"],
    "hazard": ["fire", "burn", "flashover", "arc", "explode", "unsafe", "leak", "corrode", "alarm", "incident"],
    "business": ["impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate", "penalty"]
}

STATUS_OPTIONS = ["Open", "In Progress", "Resolved", "Escalated"]

# Initialize logger
logging.basicConfig(filename='escalate_ai.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# -- DB Setup --

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS escalations (
        id TEXT PRIMARY KEY,
        customer TEXT,
        account TEXT,
        email TEXT,
        description TEXT,
        severity TEXT,
        criticality TEXT,
        category TEXT,
        sentiment REAL,
        urgency INTEGER,
        escalation_flag INTEGER,
        status TEXT,
        action_taken TEXT,
        action_owner TEXT,
        created_at TEXT,
        updated_at TEXT,
        predicted_escalation INTEGER,
        sla_breached INTEGER
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS audit_log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        escalation_id TEXT,
        changed_field TEXT,
        old_value TEXT,
        new_value TEXT,
        changed_by TEXT,
        changed_at TEXT
    )
    """)
    conn.commit()
    conn.close()

def get_next_id():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM escalations ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    if row:
        last_id = row[0]
        num = int(last_id.split('-')[1])
        next_num = max(num + 1, ID_START)
    else:
        next_num = ID_START
    return f"SESICE-{next_num}"

# -- NLP & Utility --

analyzer = SentimentIntensityAnalyzer()

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip().lower()

def check_negative_keywords(text):
    text = clean_text(text)
    found_categories = []
    for cat, words in NEGATIVE_WORDS.items():
        for w in words:
            if w in text:
                found_categories.append(cat)
                break
    return found_categories

def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    return scores['compound']

def infer_severity(neg_cats):
    # Simple rules, could be improved
    if "hazard" in neg_cats:
        return "Critical"
    if "business" in neg_cats:
        return "High"
    if "technical" in neg_cats:
        return "Medium"
    if "customer" in neg_cats or "support" in neg_cats:
        return "Low"
    return "Low"

def infer_criticality(neg_cats):
    # Similar simple logic
    if "hazard" in neg_cats:
        return "High"
    if "business" in neg_cats or "technical" in neg_cats:
        return "Medium"
    if "customer" in neg_cats or "support" in neg_cats:
        return "Low"
    return "Low"

def infer_category(neg_cats):
    if not neg_cats:
        return "General"
    return ", ".join(sorted(set(neg_cats)))

def infer_urgency(sentiment, neg_cats):
    # Combine sentiment and negative keywords count
    urgency = 0
    if sentiment < -0.5:
        urgency += 2
    elif sentiment < 0:
        urgency += 1
    urgency += len(neg_cats)
    return min(urgency, 5)

def infer_escalation_flag(urgency, severity):
    # escalate if urgency >=3 or severity is Critical or High
    if urgency >= 3 or severity in ("Critical", "High"):
        return 1
    return 0

# -- DB Operations --

def insert_escalation(data: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    INSERT INTO escalations (id, customer, account, email, description, severity, criticality, category, sentiment, urgency, escalation_flag, status, action_taken, action_owner, created_at, updated_at, predicted_escalation, sla_breached)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data['id'], data['customer'], data['account'], data['email'], data['description'],
        data['severity'], data['criticality'], data['category'], data['sentiment'],
        data['urgency'], data['escalation_flag'], data['status'], data['action_taken'], data['action_owner'],
        data['created_at'], data['updated_at'], data.get('predicted_escalation', 0), data.get('sla_breached', 0)
    ))
    conn.commit()
    conn.close()

def update_escalation_field(escalation_id, field, new_value, changed_by="System"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"SELECT {field} FROM escalations WHERE id = ?", (escalation_id,))
    old_value = c.fetchone()
    old_value = old_value[0] if old_value else None
    c.execute(f"UPDATE escalations SET {field} = ?, updated_at = ? WHERE id = ?", (new_value, datetime.datetime.now().isoformat(), escalation_id))
    conn.commit()
    # Log change
    c.execute("""
    INSERT INTO audit_log (escalation_id, changed_field, old_value, new_value, changed_by, changed_at)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (escalation_id, field, str(old_value), str(new_value), changed_by, datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()

def fetch_all_escalations():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM escalations", conn, parse_dates=["created_at", "updated_at"])
    conn.close()
    return df

def fetch_audit_logs():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM audit_log ORDER BY changed_at DESC LIMIT 100", conn, parse_dates=["changed_at"])
    conn.close()
    return df

# -- ML Model --

class EscalationModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.features = ['sentiment', 'urgency']
        self.trained = False

    def prepare_data(self, df):
        # Filter rows with escalation_flag label available
        df = df.dropna(subset=['escalation_flag'])
        X = df[self.features]
        y = df['escalation_flag']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self, df):
        if len(df) < 10:
            return False  # Not enough data to train
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = accuracy_score(y_test, y_pred)
        self.trained = True
        return accuracy, report

    def predict(self, row):
        if not self.trained:
            return 0
        X = np.array([[row['sentiment'], row['urgency']]])
        pred = self.model.predict(X)
        return int(pred[0])

    def retrain(self, df):
        return self.train(df)

escalation_model = EscalationModel()

# -- Email & Teams Alerts --

def send_email_alert(subject, body, to_emails):
    try:
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER")
        smtp_pass = os.getenv("SMTP_PASS")
        from_email = smtp_user

        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = ", ".join(to_emails)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(from_email, to_emails, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        logging.error(f"Failed to send email alert: {e}")
        return False

def send_teams_alert(webhook_url, message):
    import requests
    try:
        payload = {
            "text": message
        }
        r = requests.post(webhook_url, json=payload)
        return r.status_code == 200
    except Exception as e:
        logging.error(f"Failed to send MS Teams alert: {e}")
        return False

# -- Gmail Email Fetching --

def fetch_gmail_emails():
    imap_server = os.getenv("GMAIL_IMAP_SERVER", "imap.gmail.com")
    email_user = os.getenv("GMAIL_USER")
    email_pass = os.getenv("GMAIL_PASS")
    mailbox = "INBOX"
    try:
        mail = imaplib.IMAP4_SSL(imap_server)
        mail.login(email_user, email_pass)
        mail.select(mailbox)
        status, messages = mail.search(None, 'UNSEEN')
        email_ids = messages[0].split()
        new_emails = []
        for eid in email_ids:
            res, msg_data = mail.fetch(eid, '(RFC822)')
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = pyemail.message_from_bytes(response_part[1])
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding or "utf-8")
                    from_ = msg.get("From")
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            ctype = part.get_content_type()
                            cdispo = str(part.get("Content-Disposition"))
                            if ctype == "text/plain" and "attachment" not in cdispo:
                                try:
                                    body = part.get_payload(decode=True).decode()
                                except:
                                    body = ""
                                break
                    else:
                        body = msg.get_payload(decode=True).decode()
                    new_emails.append({
                        "subject": subject,
                        "from": from_,
                        "body": body
                    })
            # Mark as seen/read
            mail.store(eid, '+FLAGS', '\\Seen')
        mail.logout()
        return new_emails
    except Exception as e:
        logging.error(f"Failed to fetch emails: {e}")
        return []

# -- SLA Monitoring Thread --

sla_stop_event = threading.Event()

def sla_monitor_thread():
    while not sla_stop_event.is_set():
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        threshold = (datetime.datetime.now() - datetime.timedelta(minutes=SLA_THRESHOLD_MINUTES)).isoformat()
        # Check for open escalations with high priority and created_at older than threshold and not already SLA breached
        c.execute("""
        SELECT id FROM escalations
        WHERE status != 'Resolved' AND escalation_flag=1 AND sla_breached=0 AND created_at < ?
        """, (threshold,))
        rows = c.fetchall()
        for (eid,) in rows:
            # Mark SLA breached
            update_escalation_field(eid, "sla_breached", 1, changed_by="SLA Monitor")
            # Send alert
            escalation = get_escalation_by_id(eid)
            alert_msg = f"SLA Breach Alert!\nEscalation ID: {eid}\nCustomer: {escalation['customer']}\nSeverity: {escalation['severity']}\nCreated At: {escalation['created_at']}"
            teams_webhook = os.getenv("MS_TEAMS_WEBHOOK")
            email_alert_recipients = os.getenv("ALERT_EMAILS", "").split(",")
            if teams_webhook:
                send_teams_alert(teams_webhook, alert_msg)
            if email_alert_recipients:
                send_email_alert("SLA Breach Alert - EscalateAI", alert_msg, email_alert_recipients)
        conn.close()
        time.sleep(60)  # Check every 1 min

def get_escalation_by_id(eid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM escalations WHERE id=?", (eid,))
    row = c.fetchone()
    if not row:
        return None
    columns = [desc[0] for desc in c.description]
    conn.close()
    return dict(zip(columns, row))

# -- Helper functions --

def add_escalation_from_input(customer, account, email, description):
    neg_cats = check_negative_keywords(description)
    sentiment = analyze_sentiment(description)
    severity = infer_severity(neg_cats)
    criticality = infer_criticality(neg_cats)
    category = infer_category(neg_cats)
    urgency = infer_urgency(sentiment, neg_cats)
    escalation_flag = infer_escalation_flag(urgency, severity)
    status = "Open"
    action_taken = ""
    action_owner = ""
    now = datetime.datetime.now().isoformat()
    eid = get_next_id()

    data = {
        "id": eid,
        "customer": customer,
        "account": account,
        "email": email,
        "description": description,
        "severity": severity,
        "criticality": criticality,
        "category": category,
        "sentiment": sentiment,
        "urgency": urgency,
        "escalation_flag": escalation_flag,
        "status": status,
        "action_taken": action_taken,
        "action_owner": action_owner,
        "created_at": now,
        "updated_at": now,
        "predicted_escalation": 0,
        "sla_breached": 0
    }
    insert_escalation(data)
    return eid

def update_escalation_status_and_notify(eid, new_status):
    old_status = get_escalation_by_id(eid)['status']
    if old_status != new_status:
        update_escalation_field(eid, "status", new_status, changed_by="User")
        # Send notification on status change to customer (Email/WhatsApp placeholder)
        escalation = get_escalation_by_id(eid)
        msg = f"Dear {escalation['customer']},\n\nYour issue (ID: {eid}) status has been updated to '{new_status}'. We are working on it.\n\nRegards,\nSupport Team"
        # Email send placeholder - could be extended to WhatsApp API
        send_email_alert(f"Update on your issue {eid}", msg, [escalation['email']])
        logging.info(f"Status change notification sent for {eid}")

def retrain_model_with_feedback(feedback_data):
    # feedback_data is DataFrame with updated fields
    try:
        # For simplicity, retrain entire model with current DB data
        df = fetch_all_escalations()
        acc_report = escalation_model.retrain(df)
        if acc_report:
            accuracy, report = acc_report
            st.success(f"Model retrained successfully. Accuracy: {accuracy:.2f}")
            logging.info(f"Model retrained with accuracy {accuracy}")
        else:
            st.warning("Not enough data to retrain model.")
    except Exception as e:
        st.error(f"Retrain failed: {e}")
        logging.error(f"Retrain failed: {e}")

def export_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Escalations')
        writer.save()
    processed_data = output.getvalue()
    b64 = base64.b64encode(processed_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="escalations_export.xlsx">Download Escalations Excel</a>'
    return href

def parse_excel_file(file):
    try:
        df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Failed to read Excel file: {e}")
        return pd.DataFrame()
        # --- Streamlit UI ---

st.set_page_config(page_title="EscalateAI - Customer Escalation Manager", layout="wide")

init_db()

# Start SLA monitor thread once
if 'sla_thread_started' not in st.session_state:
    st.session_state.sla_thread_started = True
    sla_thread = threading.Thread(target=sla_monitor_thread, daemon=True)
    sla_thread.start()

# Load all escalations
df_escalations = fetch_all_escalations()

# Train or retrain ML model if enough data
if len(df_escalations) > 10:
    acc_report = escalation_model.train(df_escalations)
else:
    acc_report = None

# Sidebar controls
st.sidebar.title("EscalateAI Controls")

# 1. Fetch Gmail emails and parse new escalations
if st.sidebar.button("Fetch Emails from Gmail"):
    with st.spinner("Fetching emails..."):
        new_emails = fetch_gmail_emails()
        new_count = 0
        for email_data in new_emails:
            desc = email_data["body"][:1000]
            # Basic parsing - use sender as customer and email, subject as description start
            cust = email_data["from"]
            account = "Unknown"
            email_addr = email_data["from"]
            eid = add_escalation_from_input(cust, account, email_addr, desc)
            new_count += 1
        st.sidebar.success(f"Fetched and added {new_count} new escalation(s).")

# 2. Alert buttons
if st.sidebar.button("Send MS Teams Alert for Open Escalations"):
    teams_webhook = os.getenv("MS_TEAMS_WEBHOOK")
    if not teams_webhook:
        st.sidebar.error("MS Teams webhook not configured in .env")
    else:
        open_esc = df_escalations[(df_escalations['status'] != 'Resolved') & (df_escalations['escalation_flag'] == 1)]
        msg = f"Open Escalations Count: {len(open_esc)}"
        send_teams_alert(teams_webhook, msg)
        st.sidebar.success("MS Teams alert sent.")

if st.sidebar.button("Send Email Alert for Open Escalations"):
    alert_emails = os.getenv("ALERT_EMAILS", "").split(",")
    if not alert_emails or alert_emails == ['']:
        st.sidebar.error("Alert email recipients not configured in .env")
    else:
        open_esc = df_escalations[(df_escalations['status'] != 'Resolved') & (df_escalations['escalation_flag'] == 1)]
        msg = f"Open Escalations Count: {len(open_esc)}"
        send_email_alert("Open Escalations Alert - EscalateAI", msg, alert_emails)
        st.sidebar.success("Email alert sent.")

# 3. Upload Excel file
uploaded_file = st.sidebar.file_uploader("Upload Escalations Excel", type=["xlsx"])
if uploaded_file:
    try:
        uploaded_df = pd.read_excel(uploaded_file)
        for _, row in uploaded_df.iterrows():
            desc = str(row.get("description", ""))
            cust = str(row.get("customer", "Unknown"))
            acc = str(row.get("account", "Unknown"))
            email_addr = str(row.get("email", "unknown@example.com"))
            add_escalation_from_input(cust, acc, email_addr, desc)
        st.sidebar.success(f"Uploaded {len(uploaded_df)} escalation(s) from Excel.")
        df_escalations = fetch_all_escalations()
    except Exception as e:
        st.sidebar.error(f"Failed to upload Excel: {e}")

# 4. Export current filtered escalations to Excel
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Escalations')
    processed_data = output.getvalue()
    return processed_data

if st.sidebar.button("Export Escalations to Excel"):
    filtered_df = df_escalations.copy()
    excel_data = to_excel(filtered_df)
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="escalations_export.xlsx">Download Exported Excel</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)

# --- Filters ---
st.title("EscalateAI - Customer Escalation Manager")

filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([2,2,2,3])

with filter_col1:
    status_filter = st.multiselect("Filter by Status", STATUS_OPTIONS, default=STATUS_OPTIONS)

with filter_col2:
    severity_filter = st.multiselect("Filter by Severity", ["Critical", "High", "Medium", "Low"], default=["Critical", "High", "Medium", "Low"])

with filter_col3:
    category_filter = st.text_input("Filter by Category (comma separated keywords)")

with filter_col4:
    search_text = st.text_input("Search Customer, Account, Description")

# Apply filters
filtered_df = df_escalations[
    df_escalations['status'].isin(status_filter) &
    df_escalations['severity'].isin(severity_filter)
]

if category_filter.strip():
    keywords = [k.strip().lower() for k in category_filter.split(",") if k.strip()]
    filtered_df = filtered_df[filtered_df['category'].apply(lambda c: any(kw in c.lower() for kw in keywords))]

if search_text.strip():
    search_lower = search_text.lower()
    filtered_df = filtered_df[
        filtered_df.apply(lambda row: search_lower in str(row['customer']).lower() or
                                       search_lower in str(row['account']).lower() or
                                       search_lower in str(row['description']).lower(), axis=1)
    ]

# --- Summary Dashboard ---
st.subheader("Summary Dashboard")
summary_cols = st.columns(5)

with summary_cols[0]:
    st.metric("Total Escalations", len(df_escalations))

with summary_cols[1]:
    open_count = len(df_escalations[df_escalations['status'] != "Resolved"])
    st.metric("Open Escalations", open_count)

with summary_cols[2]:
    sla_breach_count = len(df_escalations[df_escalations['sla_breached'] == 1])
    st.metric("SLA Breaches", sla_breach_count)

with summary_cols[3]:
    escalated_count = len(df_escalations[df_escalations['status'] == "Escalated"])
    st.metric("Escalated Cases", escalated_count)

with summary_cols[4]:
    predicted_pos = len(df_escalations[df_escalations['predicted_escalation'] == 1])
    st.metric("Predicted Escalations", predicted_pos)

# Plot Severity distribution
fig, ax = plt.subplots()
filtered_df['severity'].value_counts().plot(kind='bar', ax=ax, color='tomato')
ax.set_title("Severity Distribution")
ax.set_xlabel("Severity")
ax.set_ylabel("Count")
st.pyplot(fig)

# --- Kanban Board View ---

st.subheader("Escalation Cases Kanban Board")

# Group filtered by status
for status in STATUS_OPTIONS:
    st.markdown(f"### {status} ({len(filtered_df[filtered_df['status'] == status])})")
    esc_in_status = filtered_df[filtered_df['status'] == status]
    if esc_in_status.empty:
        st.write("_No cases in this status._")
        continue

    for idx, row in esc_in_status.iterrows():
        with st.expander(f"{row['id']} - {row['customer']} - Severity: {row['severity']} - Category: {row['category']}"):
            st.write(f"**Description:** {row['description']}")
            st.write(f"**Created At:** {row['created_at']}")
            st.write(f"**Sentiment Score:** {row['sentiment']:.2f}")
            st.write(f"**Urgency Level:** {row['urgency']}")
            st.write(f"**Escalation Flag:** {'Yes' if row['escalation_flag'] else 'No'}")
            st.write(f"**SLA Breached:** {'Yes' if row['sla_breached'] else 'No'}")

            # Editable fields: status, action_taken, action_owner
            new_status = st.selectbox(f"Change Status ({row['status']})", STATUS_OPTIONS, key=f"status_{row['id']}")
            if new_status != row['status']:
                if st.button(f"Update Status for {row['id']}", key=f"btn_status_{row['id']}"):
                    update_escalation_status_and_notify(row['id'], new_status)
                    st.success(f"Status updated to {new_status}")
                    st.experimental_rerun()

            new_action = st.text_area("Action Taken", value=row['action_taken'], key=f"action_{row['id']}")
            new_owner = st.text_input("Action Owner", value=row['action_owner'], key=f"owner_{row['id']}")

            if (new_action != row['action_taken']) or (new_owner != row['action_owner']):
                if st.button(f"Update Action for {row['id']}", key=f"btn_action_{row['id']}"):
                    update_action_taken(row['id'], new_action, new_owner)
                    st.success("Action updated")
                    st.experimental_rerun()

            # Feedback form for retraining
            st.markdown("---")
            st.write("### Provide Feedback & Retrain Model")
            feedback_sentiment = st.slider("Sentiment Score (VADER)", min_value=-1.0, max_value=1.0, value=row['sentiment'], step=0.01, key=f"fb_sent_{row['id']}")
            feedback_urgency = st.selectbox("Urgency Level", ["Low", "Medium", "High"], index=["Low", "Medium", "High"].index(row['urgency']), key=f"fb_urg_{row['id']}")
            feedback_escal_flag = st.checkbox("Escalation Flag", value=bool(row['escalation_flag']), key=f"fb_esc_{row['id']}")

            if st.button(f"Submit Feedback & Retrain {row['id']}", key=f"btn_fb_{row['id']}"):
                feedback_data = {
                    'sentiment': feedback_sentiment,
                    'urgency': feedback_urgency,
                    'escalation_flag': 1 if feedback_escal_flag else 0
                }
                retrain_on_feedback(row['id'], feedback_data)
                st.success("Feedback submitted and model retrained.")

# --- Audit Log Viewer ---

st.subheader("Audit Log")

audit_df = fetch_audit_logs()
if audit_df.empty:
    st.write("_No audit logs available._")
else:
    st.dataframe(audit_df.sort_values(by='timestamp', ascending=False).reset_index(drop=True), height=300)

# --- End of UI ---
