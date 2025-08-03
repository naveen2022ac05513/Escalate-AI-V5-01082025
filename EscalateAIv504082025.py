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

# Load environment variables
load_dotenv()
EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")
EMAIL_ALERT_ENABLED = os.getenv("EMAIL_ALERT_ENABLED", "false").lower() == "true"
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", 587))
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
EMAIL_SENDER_PASS = os.getenv("EMAIL_SENDER_PASS")

DB_FILE = "escalations.db"

# Negative keywords expanded as per your list
NEGATIVE_KEYWORDS = [
    # Technical Failures & Product Malfunction
    "fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip", "malfunction", "blank", "shutdown", "discharge",
    # Customer Dissatisfaction & Escalations
    "dissatisfy", "frustrate", "complain", "reject", "delay", "ignore", "escalate", "displease", "noncompliance", "neglect",
    # Support Gaps & Operational Delays
    "wait", "pending", "slow", "incomplete", "miss", "omit", "unresolved", "shortage", "no response",
    # Hazardous Conditions & Safety Risks
    "fire", "burn", "flashover", "arc", "explode", "unsafe", "leak", "corrode", "alarm", "incident",
    # Business Risk & Impact
    "impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate", "penalty"
]

analyzer = SentimentIntensityAnalyzer()

def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    # Create table if missing
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS escalations (
        escalation_id TEXT PRIMARY KEY,
        customer TEXT,
        issue TEXT,
        date TEXT,
        status TEXT DEFAULT 'Open'
    )
    """)
    # Columns to ensure present (add if missing)
    alter_columns = {
        "sentiment": "TEXT DEFAULT ''",
        "priority": "TEXT DEFAULT 'Low'",
        "escalation_flag": "INTEGER DEFAULT 0",
        "severity": "TEXT DEFAULT 'Medium'",
        "criticality": "TEXT DEFAULT 'Routine'",
        "category": "TEXT DEFAULT 'Feedback'",
        "action_taken": "TEXT DEFAULT ''",
        "action_owner": "TEXT DEFAULT ''",
        "status_update_date": "TEXT DEFAULT ''",
        "user_feedback": "TEXT DEFAULT ''"
    }
    cursor.execute("PRAGMA table_info(escalations)")
    existing_cols = [c[1] for c in cursor.fetchall()]
    for col, definition in alter_columns.items():
        if col not in existing_cols:
            try:
                cursor.execute(f"ALTER TABLE escalations ADD COLUMN {col} {definition}")
            except sqlite3.OperationalError:
                pass
    conn.commit()
    conn.close()

def fetch_gmail_emails():
    if not EMAIL or not APP_PASSWORD:
        st.error("Gmail credentials not set in environment variables.")
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
            emails.append({"customer": from_, "issue": body.strip(), "subject": subject, "date": date})
            mail.store(eid, '+FLAGS', '\\Seen')
        mail.logout()
        return emails
    except Exception as e:
        st.error(f"Error fetching emails: {e}")
        return []

def analyze_issue(issue_text):
    text_lower = issue_text.lower()
    vs = analyzer.polarity_scores(issue_text)
    sentiment = "Positive" if vs["compound"] >= 0 else "Negative"
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    priority = "High" if sentiment == "Negative" and neg_count >= 2 else "Low"
    escalation_flag = 1 if priority == "High" else 0
    # Tag severity, criticality, category based on priority & sentiment
    severity = "Critical" if priority == "High" else "Medium"
    criticality = "Urgent" if priority == "High" else "Routine"
    category = "Complaint" if sentiment == "Negative" else "Feedback"
    return sentiment, priority, escalation_flag, severity, criticality, category

def save_emails_to_db(emails):
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]
    new_entries = 0
    for e in emails:
        cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (e['customer'], e['issue'][:500]))
        if cursor.fetchone():
            continue
        count += 1
        esc_id = f"SESICE-{count + 250000}"
        sentiment, priority, escalation_flag, severity, criticality, category = analyze_issue(e['issue'])
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
        cursor.execute("""
            INSERT INTO escalations 
            (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, severity, criticality, category, action_taken, action_owner, status_update_date, user_feedback)
            VALUES (?, ?, ?, ?, 'Open', ?, ?, ?, ?, ?, ?, '', '', ?, '')
        """, (esc_id, e['customer'], e['issue'][:500], e['date'], sentiment, priority, escalation_flag, severity, criticality, category, now))
        new_entries += 1
        if escalation_flag == 1:
            send_ms_teams_alert(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {e['customer']}\nIssue: {e['issue'][:200]}...")
    conn.commit()
    conn.close()
    return new_entries

def send_ms_teams_alert(message):
    if not MS_TEAMS_WEBHOOK_URL:
        st.warning("MS Teams webhook URL not set; cannot send alerts.")
        return
    headers = {"Content-Type": "application/json"}
    payload = {"text": message}
    try:
        response = requests.post(MS_TEAMS_WEBHOOK_URL, json=payload, headers=headers)
        if response.status_code != 200:
            st.error(f"MS Teams alert failed: {response.status_code} {response.text}")
    except Exception as e:
        st.error(f"Error sending MS Teams alert: {e}")

# Placeholder for email alerts if needed
def send_email_alert(subject, body):
    import smtplib
    from email.mime.text import MIMEText
    if not (EMAIL_SMTP_SERVER and EMAIL_SENDER and EMAIL_SENDER_PASS and EMAIL_RECEIVER):
        st.warning("Email alert config missing, skipping email alert.")
        return
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    try:
        with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_SENDER_PASS)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
    except Exception as e:
        st.error(f"Error sending email alert: {e}")

def load_escalations_df():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    conn.close()
    return df

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

        conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM escalations")
        existing_count = cursor.fetchone()[0]

        count = 0
        for idx, row in df.iterrows():
            customer = str(row[customer_col])
            issue = str(row[issue_col])
            date = str(row[date_col]) if date_col and pd.notna(row[date_col]) else datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (customer, issue[:500]))
            if cursor.fetchone():
                continue
            existing_count += 1
            esc_id = f"SESICE-{existing_count + 250000}"
            sentiment, priority, escalation_flag, severity, criticality, category = analyze_issue(issue)
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("""
                INSERT INTO escalations 
                (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, severity, criticality, category, action_taken, action_owner, status_update_date, user_feedback)
                VALUES (?, ?, ?, ?, 'Open', ?, ?, ?, ?, ?, ?, '', '', ?, '')
            """, (esc_id, customer, issue[:500], date, sentiment, priority, escalation_flag, severity, criticality, category, now))
            count += 1
            if escalation_flag == 1:
                send_ms_teams_alert(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {customer}\nIssue: {issue[:200]}...")
        conn.commit()
        conn.close()
        return count
    except Exception as e:
        st.error(f"Error processing uploaded Excel: {e}")
        return 0

def manual_entry_process(customer, issue):
    if not customer or not issue:
        st.sidebar.error("Please fill customer and issue.")
        return False
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]
    esc_id = f"SESICE-{count + 250001}"
    sentiment, priority, escalation_flag, severity, criticality, category = analyze_issue(issue)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
    cursor.execute("""
        INSERT INTO escalations 
        (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, severity, criticality, category, action_taken, action_owner, status_update_date, user_feedback)
        VALUES (?, ?, ?, ?, 'Open', ?, ?, ?, ?, ?, ?, '', '', ?, '')
    """, (esc_id, customer, issue[:500], now, sentiment, priority, escalation_flag, severity, criticality, category, now))
    conn.commit()
    conn.close()
    st.sidebar.success(f"Added escalation {esc_id}")
    if escalation_flag == 1:
        send_ms_teams_alert(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {customer}\nIssue: {issue[:200]}...")
    return True

def update_case_action(esc_id, action_taken, action_owner, status):
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
    cursor.execute("""
        UPDATE escalations 
        SET action_taken = ?, action_owner = ?, status = ?, status_update_date = ?
        WHERE escalation_id = ?
    """, (action_taken, action_owner, status, now, esc_id))
    conn.commit()
    conn.close()

def sla_breach_check(row):
    # SLA: if high priority & open more than 10 minutes
    if row["priority"] == "High" and row["status"] in ["Open", "In Progress"]:
        try:
            status_time = datetime.datetime.strptime(row["status_update_date"], "%a, %d %b %Y %H:%M:%S %z")
        except:
            # If no update date, fallback to created date
            try:
                status_time = datetime.datetime.strptime(row["date"], "%a, %d %b %Y %H:%M:%S %z")
            except:
                return False
        now = datetime.datetime.now(datetime.timezone.utc)
        delta = now - status_time
        if delta.total_seconds() > 600:  # 10 minutes
            return True
    return False

def display_kanban_card(row):
    # Highlight card red if escalated & SLA breached
    sla_breached = sla_breach_check(row)
    if row["escalation_flag"] == 1:
        card_color = "background-color:#ffe6e6" if sla_breached else "background-color:#ffcccc"
    else:
        card_color = "background-color:#e6f2ff" if sla_breached else ""

    with st.expander(f"{row['escalation_id']} - {row['customer']} [{row['status']}]"):
        st.markdown(f"<div style='{card_color} padding: 10px; border-radius:5px;'>", unsafe_allow_html=True)
        st.write(f"**Issue:** {row['issue']}")
        st.write(f"**Date:** {row['date']}")
        st.write(f"**Sentiment:** {row['sentiment']}")
        st.write(f"**Priority:** {row['priority']}")
        st.write(f"**Severity:** {row['severity']}")
        st.write(f"**Criticality:** {row['criticality']}")
        st.write(f"**Category:** {row['category']}")
        st.write(f"**Action Taken:** {row['action_taken']}")
        st.write(f"**Action Owner:** {row['action_owner']}")
        # Allow status update
        status = st.selectbox("Status", ["Open", "In Progress", "Resolved", "Escalated"], index=["Open", "In Progress", "Resolved", "Escalated"].index(row['status']), key=f"status_{row['escalation_id']}")
        action_taken = st.text_area("Action Taken", row["action_taken"], key=f"action_{row['escalation_id']}")
        action_owner = st.text_input("Action Owner", row["action_owner"], key=f"owner_{row['escalation_id']}")
        if st.button("Update", key=f"update_{row['escalation_id']}"):
            update_case_action(row['escalation_id'], action_taken, action_owner, status)
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def main():
    st.title("EscalateAI - Customer Escalation Management")
    init_db()

    # Sidebar controls
    st.sidebar.title("EscalateAI Controls & Uploads")

    # Upload Excel complaints
    uploaded_file = st.sidebar.file_uploader("Upload Excel with complaints", type=["xlsx", "xls"])
    if uploaded_file:
        count = upload_excel_and_analyze(uploaded_file)
        st.sidebar.success(f"Uploaded and analyzed {count} complaints from Excel.")

    # Manual entry
    st.sidebar.subheader("Add Manual Escalation")
    customer_input = st.sidebar.text_input("Customer Email/Name")
    issue_input = st.sidebar.text_area("Issue Description")
    if st.sidebar.button("Add Escalation"):
        if manual_entry_process(customer_input, issue_input):
            st.experimental_rerun()

    # Load escalations dataframe
    df = load_escalations_df()

    # Download consolidated complaints
    st.sidebar.download_button(
        label="Download Consolidated Complaints",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="escalations_consolidated.csv",
        mime="text/csv"
    )

    # Filters
    status_filter = st.sidebar.multiselect("Filter by Status", ["Open", "In Progress", "Resolved", "Escalated"], default=["Open", "In Progress", "Resolved"])
    priority_filter = st.sidebar.multiselect("Filter by Priority", ["High", "Low"], default=["High", "Low"])

    # Show counts
    status_counts = df['status'].value_counts()
    open_count = status_counts.get("Open", 0)
    inprogress_count = status_counts.get("In Progress", 0)
    resolved_count = status_counts.get("Resolved", 0)
    escalated_count = status_counts.get("Escalated", 0)
    st.sidebar.markdown(f"""
    ### Status Counts
    - Open: {open_count}
    - In Progress: {inprogress_count}
    - Resolved: {resolved_count}
    - Escalated: {escalated_count}
    """)

    # Filter dataframe based on selection
    filtered_df = df[df['status'].isin(status_filter) & df['priority'].isin(priority_filter)]

    # Show Kanban columns
    columns = st.columns(4)
    buckets = ["Open", "In Progress", "Resolved", "Escalated"]

    for idx, bucket in enumerate(buckets):
        with columns[idx]:
            st.header(f"{bucket} ({status_counts.get(bucket,0)})")
            for _, row in filtered_df[filtered_df['status'] == bucket].iterrows():
                display_kanban_card(row)

    # SLA Alert Button
    if st.sidebar.button("Send SLA Breach Alert for High Priority Open Issues"):
        sla_issues = df[(df['priority'] == "High") & (df['status'].isin(["Open", "In Progress"])) & df.apply(sla_breach_check, axis=1)]
        if sla_issues.empty:
            st.sidebar.info("No SLA breaches detected.")
        else:
            for _, issue in sla_issues.iterrows():
                message = f"🚨 SLA Breach Alert for Escalation ID {issue['escalation_id']}!\nCustomer: {issue['customer']}\nIssue: {issue['issue'][:200]}...\nStatus: {issue['status']}\nPriority: {issue['priority']}"
                send_ms_teams_alert(message)
            st.sidebar.success(f"Sent {len(sla_issues)} SLA breach alerts.")

if __name__ == "__main__":
    main()
