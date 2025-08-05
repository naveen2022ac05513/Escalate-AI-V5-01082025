# escalate_ai.py
import os
from dotenv import load_dotenv
import sqlite3
import streamlit as st
import pandas as pd
import requests
import json
import imaplib
import email
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import io

# Load environment variables
load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", 587))
MS_TEAMS_WEBHOOK = os.getenv("MS_TEAMS_WEBHOOK_URL")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# Database setup
DB_PATH = "escalate_ai.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS escalations (
            id TEXT PRIMARY KEY,
            source TEXT,
            timestamp TEXT,
            customer_email TEXT,
            subject TEXT,
            body TEXT,
            sentiment REAL,
            urgency TEXT,
            escalation_flag INTEGER,
            severity TEXT,
            criticality TEXT,
            category TEXT,
            status TEXT,
            action_taken TEXT,
            action_owner TEXT,
            last_updated TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS audit_log (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            escalation_id TEXT,
            timestamp TEXT,
            action TEXT,
            details TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

NEGATIVE_KEYWORDS = {
    "technical_failures": ["fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip", "malfunction", "blank", "shutdown", "discharge"],
    "customer_dissatisfaction": ["dissatisfy", "frustrate", "complain", "reject", "delay", "ignore", "escalate", "displease", "noncompliance", "neglect"],
    "support_gaps": ["wait", "pending", "slow", "incomplete", "miss", "omit", "unresolved", "shortage", "no response"],
    "hazardous_conditions": ["fire", "burn", "flashover", "arc", "explode", "unsafe", "leak", "corrode", "alarm", "incident"],
    "business_risks": ["impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate", "penalty"]
}

SEVERITY_MAP = {
    "technical_failures": "High",
    "hazardous_conditions": "High",
    "customer_dissatisfaction": "Medium",
    "support_gaps": "Medium",
    "business_risks": "High"
}

CRITICALITY_MAP = {
    "technical_failures": "Critical",
    "hazardous_conditions": "Critical",
    "customer_dissatisfaction": "Moderate",
    "support_gaps": "Moderate",
    "business_risks": "Critical"
}

CATEGORY_MAP = {
    "technical_failures": "Technical",
    "hazardous_conditions": "Safety",
    "customer_dissatisfaction": "Customer",
    "support_gaps": "Operations",
    "business_risks": "Business"
}

STATUS_BUCKETS = ["Open", "In Progress", "Resolved", "Escalated"]

analyzer = SentimentIntensityAnalyzer()

def log_audit(escalation_id, action, details):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO audit_log (escalation_id, timestamp, action, details) VALUES (?, ?, ?, ?)",
              (escalation_id, datetime.utcnow().isoformat(), action, details))
    conn.commit()
    conn.close()

def generate_escalation_id():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM escalations ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    if row:
        last_num = int(row[0].split("-")[-1])
        new_num = last_num + 1
    else:
        new_num = 2500001
    return f"SESICE-{new_num}"

def detect_category_and_severity(text):
    text_lower = text.lower()
    found_categories = []
    for key, keywords in NEGATIVE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                found_categories.append(key)
                break
    if not found_categories:
        return "General", "Low", "Low"
    cat_key = found_categories[0]
    return CATEGORY_MAP.get(cat_key, "General"), SEVERITY_MAP.get(cat_key, "Low"), CRITICALITY_MAP.get(cat_key, "Low")

def determine_urgency(sentiment, severity):
    if severity == "High" or sentiment < -0.4:
        return "High"
    elif severity == "Medium" or sentiment < -0.2:
        return "Medium"
    else:
        return "Low"

def is_escalation(text, sentiment, urgency):
    text_lower = text.lower()
    keyword_flag = any(kw in text_lower for kwlist in NEGATIVE_KEYWORDS.values() for kw in kwlist)
    if keyword_flag and sentiment < -0.3 and urgency == "High":
        return 1
    return 0

def parse_email_body(body):
    return body.strip()

def fetch_gmail_emails():
    st.info("Fetching emails from Gmail...")
    try:
        mail = imaplib.IMAP4_SSL(EMAIL_SERVER)
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select("inbox")
        typ, data = mail.search(None, '(UNSEEN)')
        email_ids = data[0].split()
        emails_data = []
        for num in email_ids:
            typ, msg_data = mail.fetch(num, '(RFC822)')
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject = msg["subject"]
                    from_ = msg["from"]
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            ctype = part.get_content_type()
                            if ctype == "text/plain" and not part.get("Content-Disposition"):
                                body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                                break
                    else:
                        body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
                    emails_data.append({
                        "subject": subject,
                        "from": from_,
                        "body": parse_email_body(body)
                    })
        mail.logout()
        st.success(f"Fetched {len(emails_data)} new emails")
        return emails_data
    except Exception as e:
        st.error(f"Error fetching emails: {e}")
        return []

def insert_email_issues(email_issues):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    new_entries = 0
    for email_issue in email_issues:
        text_to_analyze = (email_issue['subject'] or "") + " " + (email_issue['body'] or "")
        sentiment = analyzer.polarity_scores(text_to_analyze)["compound"]
        category, severity, criticality = detect_category_and_severity(text_to_analyze)
        urgency = determine_urgency(sentiment, severity)
        escalation_flag = is_escalation(text_to_analyze, sentiment, urgency)
        escalation_id = generate_escalation_id()
        timestamp = datetime.utcnow().isoformat()
        c.execute('''INSERT INTO escalations (id, source, timestamp, customer_email, subject, body,
            sentiment, urgency, escalation_flag, severity, criticality, category, status,
            action_taken, action_owner, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (escalation_id, "Email", timestamp, email_issue["from"], email_issue["subject"], email_issue["body"],
                   sentiment, urgency, escalation_flag, severity, criticality, category, "Open", "", "", timestamp))
        new_entries += 1
        log_audit(escalation_id, "Insert", f"Inserted escalation from email with subject '{email_issue['subject']}'")
    conn.commit()
    conn.close()
    return new_entries

def insert_excel_bulk(df):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    new_entries = 0
    for _, row in df.iterrows():
        subject = str(row.get("subject", ""))
        body = str(row.get("body", ""))
        customer_email = str(row.get("customer_email", ""))
        text_to_analyze = subject + " " + body
        sentiment = analyzer.polarity_scores(text_to_analyze)["compound"]
        category, severity, criticality = detect_category_and_severity(text_to_analyze)
        urgency = determine_urgency(sentiment, severity)
        escalation_flag = is_escalation(text_to_analyze, sentiment, urgency)
        escalation_id = generate_escalation_id()
        timestamp = datetime.utcnow().isoformat()
        c.execute('''INSERT INTO escalations (id, source, timestamp, customer_email, subject, body,
            sentiment, urgency, escalation_flag, severity, criticality, category, status,
            action_taken, action_owner, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (escalation_id, "Excel", timestamp, customer_email, subject, body,
                   sentiment, urgency, escalation_flag, severity, criticality, category, "Open", "", "", timestamp))
        new_entries += 1
        log_audit(escalation_id, "Insert", f"Inserted escalation from Excel upload, subject '{subject}'")
    conn.commit()
    conn.close()
    return new_entries

def fetch_all_escalations():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM escalations ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    columns = [desc[0] for desc in c.description]
    df = pd.DataFrame(rows, columns=columns)
    return df

def update_escalation_status(escalation_id, status, action_taken, action_owner):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE escalations SET status=?, action_taken=?, action_owner=?, last_updated=? WHERE id=?",
              (status, action_taken, action_owner, datetime.utcnow().isoformat(), escalation_id))
    conn.commit()
    conn.close()
    log_audit(escalation_id, "Status Update", f"Status set to {status} by {action_owner}")

def send_ms_teams_alert(message):
    if not MS_TEAMS_WEBHOOK:
        st.warning("MS Teams webhook URL not configured.")
        return False
    try:
        headers = {"Content-Type": "application/json"}
        data = {"text": message}
        response = requests.post(MS_TEAMS_WEBHOOK, headers=headers, json=data)
        if response.status_code == 200:
            return True
        else:
            st.error(f"MS Teams alert failed: {response.text}")
            return False
    except Exception as e:
        st.error(f"MS Teams alert exception: {e}")
        return False

def send_email_alert(subject, body):
    if not (EMAIL_USER and EMAIL_PASS and EMAIL_RECEIVER):
        st.warning("Email sender or receiver not configured.")
        return False
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_USER
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Email alert sending failed: {e}")
        return False

def check_sla_breaches():
    # High priority issues open >10 minutes
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    cutoff = datetime.utcnow() - timedelta(minutes=10)
    cutoff_str = cutoff.isoformat()
    c.execute("""
        SELECT id, subject, timestamp, urgency FROM escalations 
        WHERE urgency='High' AND status IN ('Open', 'In Progress') AND timestamp <= ?
    """, (cutoff_str,))
    rows = c.fetchall()
    conn.close()
    return rows

def retrain_model_stub():
    # Placeholder for ML retrain on feedback
    st.info("Retraining ML model with feedback (stub).")

def filter_escalations(df, status_filter, urgency_filter, search_term):
    if status_filter:
        df = df[df["status"].isin(status_filter)]
    if urgency_filter:
        df = df[df["urgency"].isin(urgency_filter)]
    if search_term:
        search_term_lower = search_term.lower()
        df = df[
            df["subject"].str.lower().str.contains(search_term_lower) |
            df["body"].str.lower().str.contains(search_term_lower) |
            df["id"].str.lower().str.contains(search_term_lower)
        ]
    return df

def kanban_board(df):
    st.header("Escalations Kanban Board")
    status_cols = st.columns(len(STATUS_BUCKETS))
    for idx, status in enumerate(STATUS_BUCKETS):
        with status_cols[idx]:
            st.subheader(f"{status} ({len(df[df['status']==status])})")
            subset = df[df["status"]==status]
            for _, row in subset.iterrows():
                with st.expander(f"{row['id']} | {row['subject'][:30]}..."):
                    st.markdown(f"**Customer Email:** {row['customer_email']}")
                    st.markdown(f"**Timestamp:** {row['timestamp']}")
                    st.markdown(f"**Sentiment:** {row['sentiment']:.2f}")
                    st.markdown(f"**Urgency:** {row['urgency']}")
                    st.markdown(f"**Severity:** {row['severity']}")
                    st.markdown(f"**Criticality:** {row['criticality']}")
                    st.markdown(f"**Category:** {row['category']}")
                    st.markdown(f"**Body:** {row['body'][:300]}...")
                    # Status update form
                    with st.form(f"update_{row['id']}"):
                        new_status = st.selectbox("Change Status", STATUS_BUCKETS, index=STATUS_BUCKETS.index(row["status"]))
                        action_taken = st.text_area("Action Taken", value=row["action_taken"])
                        action_owner = st.text_input("Action Owner", value=row["action_owner"])
                        submitted = st.form_submit_button("Update")
                        if submitted:
                            update_escalation_status(row["id"], new_status, action_taken, action_owner)
                            st.success(f"Updated {row['id']} status to {new_status}")

def summary_dashboard(df):
    st.header("Summary Dashboard")
    total = len(df)
    by_status = df["status"].value_counts()
    by_urgency = df["urgency"].value_counts()
    by_severity = df["severity"].value_counts()
    st.metric("Total Escalations", total)
    st.subheader("By Status")
    st.bar_chart(by_status)
    st.subheader("By Urgency")
    st.bar_chart(by_urgency)
    st.subheader("By Severity")
    st.bar_chart(by_severity)

def export_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Escalations")
        writer.save()
    processed_data = output.getvalue()
    return processed_data
    def create_tables():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS escalations (
        id TEXT PRIMARY KEY,
        source TEXT,
        timestamp TEXT,
        customer_email TEXT,
        subject TEXT,
        body TEXT,
        sentiment REAL,
        urgency TEXT,
        escalation_flag INTEGER,
        severity TEXT,
        criticality TEXT,
        category TEXT,
        status TEXT,
        action_taken TEXT,
        action_owner TEXT,
        last_updated TEXT
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        escalation_id TEXT,
        action TEXT,
        details TEXT,
        timestamp TEXT
    )
    """)
    conn.commit()
    conn.close()
def main():
    create_tables()
    # rest of your main() code

# Streamlit UI
def main():
    st.set_page_config(page_title="EscalateAI", layout="wide")
    st.title("EscalateAI - Customer Escalation Management System")

    # Sidebar controls
    st.sidebar.header("Actions")

    if st.sidebar.button("Fetch Emails from Gmail"):
        emails = fetch_gmail_emails()
        if emails:
            new_count = insert_email_issues(emails)
            st.sidebar.success(f"Inserted {new_count} new escalations from emails.")

    uploaded_file = st.sidebar.file_uploader("Upload Excel file (.xlsx) with escalations", type=["xlsx"])
    if uploaded_file:
        try:
            df_excel = pd.read_excel(uploaded_file)
            required_cols = {"subject", "body", "customer_email"}
            if not required_cols.issubset(df_excel.columns.str.lower()):
                st.sidebar.error(f"Excel missing required columns: {required_cols}")
            else:
                # Normalize columns to lowercase for safety
                df_excel.columns = df_excel.columns.str.lower()
                new_count = insert_excel_bulk(df_excel)
                st.sidebar.success(f"Inserted {new_count} new escalations from Excel.")
        except Exception as e:
            st.sidebar.error(f"Excel processing error: {e}")

    # Filters and search
    st.sidebar.header("Filters & Search")
    status_filter = st.sidebar.multiselect("Filter by Status", STATUS_BUCKETS, default=STATUS_BUCKETS)
    urgency_filter = st.sidebar.multiselect("Filter by Urgency", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    search_term = st.sidebar.text_input("Search (ID, subject, body)")

    df_escalations = fetch_all_escalations()
    df_filtered = filter_escalations(df_escalations, status_filter, urgency_filter, search_term)

    # SLA breach alerting
    st.sidebar.header("Alerts")
    if st.sidebar.button("Send MS Teams Alert for SLA breaches"):
        breaches = check_sla_breaches()
        if breaches:
            msg_lines = [f"Escalation {b[0]} - {b[1]} opened at {b[2]} - Urgency: {b[3]}" for b in breaches]
            message = "SLA Breach Alert:\n" + "\n".join(msg_lines)
            if send_ms_teams_alert(message):
                st.sidebar.success("MS Teams alert sent for SLA breaches.")
        else:
            st.sidebar.info("No SLA breaches detected.")

    if st.sidebar.button("Send Email Alert for SLA breaches"):
        breaches = check_sla_breaches()
        if breaches:
            msg_lines = [f"Escalation {b[0]} - {b[1]} opened at {b[2]} - Urgency: {b[3]}" for b in breaches]
            message = "SLA Breach Alert:\n" + "\n".join(msg_lines)
            if send_email_alert("SLA Breach Alert", message):
                st.sidebar.success("Email alert sent for SLA breaches.")
        else:
            st.sidebar.info("No SLA breaches detected.")

    if st.sidebar.button("Retrain ML Model (stub)"):
        retrain_model_stub()

    # Show summary dashboard
    summary_dashboard(df_filtered)

    # Show Kanban board
    kanban_board(df_filtered)

    # Export report
    st.header("Export Escalations")
    if st.button("Download Excel Report"):
        data = export_to_excel(df_filtered)
        st.download_button("Download Excel", data, "escalations_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Audit log viewer
    st.header("Audit Log")
    conn = sqlite3.connect(DB_PATH)
    audit_df = pd.read_sql_query("SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT 100", conn)
    conn.close()
    st.dataframe(audit_df)

if __name__ == "__main__":
    main()
