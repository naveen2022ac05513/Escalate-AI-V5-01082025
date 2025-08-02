import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import re
import os
from dotenv import load_dotenv
import time
import threading

# Load environment variables
load_dotenv()

EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")

# DB setup
DB_FILE = "escalateai_escalations.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS escalations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        escalation_id TEXT UNIQUE,
        customer TEXT,
        issue TEXT,
        date_reported TEXT,
        status TEXT,
        sentiment TEXT,
        priority TEXT,
        action_taken TEXT,
        action_owner TEXT
    )
    """)
    conn.commit()
    conn.close()

def get_next_escalation_id():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM escalations")
    count = c.fetchone()[0]
    next_id = f"SESICE-{250001 + count}"
    conn.close()
    return next_id

def insert_escalation(data):
    # data: dict with keys customer, issue, date_reported, status, sentiment, priority
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Check duplicate: same customer + issue (ignoring case, trimming spaces)
    c.execute("""
    SELECT 1 FROM escalations WHERE lower(customer) = ? AND lower(issue) = ?
    """, (data['customer'].strip().lower(), data['issue'].strip().lower()))
    if c.fetchone():
        conn.close()
        return False  # Duplicate

    esc_id = get_next_escalation_id()
    c.execute("""
    INSERT INTO escalations (escalation_id, customer, issue, date_reported, status, sentiment, priority, action_taken, action_owner)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        esc_id,
        data['customer'].strip(),
        data['issue'].strip(),
        data['date_reported'],
        data.get('status', 'Open'),
        data['sentiment'],
        data['priority'],
        data.get('action_taken', ""),
        data.get('action_owner', "")
    ))
    conn.commit()
    conn.close()
    return True

def update_escalation_field(escalation_id, field, value):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    if field not in ['status', 'action_taken', 'action_owner']:
        conn.close()
        return False
    c.execute(f"UPDATE escalations SET {field}=? WHERE escalation_id=?", (value, escalation_id))
    conn.commit()
    conn.close()
    return True

def fetch_escalations():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT escalation_id, customer, issue, date_reported, status, sentiment, priority, action_taken, action_owner FROM escalations")
    rows = c.fetchall()
    conn.close()
    columns = ['escalation_id', 'customer', 'issue', 'date_reported', 'status', 'sentiment', 'priority', 'action_taken', 'action_owner']
    df = pd.DataFrame(rows, columns=columns)
    return df

def rule_sentiment(text):
    # Simple keyword based sentiment for demo
    neg_words = ['urgent', 'immediately', 'critical', 'fail', 'error', 'escalate', 'issue', 'problem', 'complaint', 'delay']
    text_l = text.lower()
    return "Negative" if any(word in text_l for word in neg_words) else "Positive"

def determine_priority(text):
    # Priority High if negative sentiment or contains strong urgency words
    urgency_words = ['urgent', 'immediately', 'critical', 'fail', 'escalate']
    text_l = text.lower()
    score = sum(word in text_l for word in urgency_words)
    return "High" if score > 0 else "Low"

def parse_email_body(msg):
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            cdisp = str(part.get("Content-Disposition"))
            if ctype == "text/plain" and "attachment" not in cdisp:
                try:
                    body = part.get_payload(decode=True).decode(errors="ignore")
                except:
                    pass
                break
    else:
        try:
            body = msg.get_payload(decode=True).decode(errors="ignore")
        except:
            pass
    return body

def fetch_emails_and_log():
    # Connect Gmail and fetch unseen emails, analyze and store escalations
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL, APP_PASSWORD)
    except Exception as e:
        st.error(f"Email login error: {e}")
        return 0

    status, _ = mail.select("INBOX")
    if status != 'OK':
        st.error("Cannot open mailbox")
        mail.logout()
        return 0

    # Search for unseen emails
    status, messages = mail.search(None, '(UNSEEN)')
    if status != 'OK':
        mail.logout()
        return 0

    email_ids = messages[0].split()
    new_logged = 0
    for num in email_ids[-10:]:  # process last 10 unseen only for performance
        res, msg_data = mail.fetch(num, '(RFC822)')
        if res != 'OK':
            continue
        msg = email.message_from_bytes(msg_data[0][1])

        subject, encoding = decode_header(msg.get("Subject"))[0]
        if isinstance(subject, bytes):
            subject = subject.decode(encoding or 'utf-8', errors='ignore')

        from_ = msg.get("From")
        date_ = msg.get("Date")
        body = parse_email_body(msg)

        sentiment = rule_sentiment(body)
        priority = determine_priority(body)

        # Insert escalation
        inserted = insert_escalation({
            "customer": from_,
            "issue": body[:1000],
            "date_reported": date_,
            "sentiment": sentiment,
            "priority": priority,
            "status": "Open"
        })
        if inserted:
            new_logged += 1

        # Mark as seen
        mail.store(num, '+FLAGS', '\\Seen')

    mail.logout()
    return new_logged

def create_excel_file():
    df = fetch_escalations()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"escalations_export_{timestamp}.xlsx"
    df.to_excel(filename, index=False)
    return filename

# --- Streamlit app start ---

st.set_page_config(page_title="EscalateAI - Customer Escalation Management", layout="wide")
st.title("🚀 EscalateAI - Escalation Management with Auto Escalation & Email Alerts")

init_db()

# Sidebar Inputs: Bulk upload, manual entry, filters
st.sidebar.header("📥 Bulk Upload Complaints via Excel")
uploaded_file = st.sidebar.file_uploader("Upload Excel with columns: customer, issue, date_reported", type=["xlsx"])
if uploaded_file is not None:
    try:
        df_upload = pd.read_excel(uploaded_file)
        required_cols = {"customer", "issue", "date_reported"}
        if not required_cols.issubset(set(c.lower() for c in df_upload.columns)):
            st.sidebar.error(f"Excel must contain columns: {required_cols}")
        else:
            df_upload.columns = [c.lower() for c in df_upload.columns]
            count_new = 0
            for _, row in df_upload.iterrows():
                cust = str(row["customer"]).strip()
                issue = str(row["issue"]).strip()
                date_rpt = str(row["date_reported"]).strip()
                if cust and issue:
                    sentiment = rule_sentiment(issue)
                    priority = determine_priority(issue)
                    inserted = insert_escalation({
                        "customer": cust,
                        "issue": issue[:1000],
                        "date_reported": date_rpt,
                        "sentiment": sentiment,
                        "priority": priority,
                        "status": "Open"
                    })
                    if inserted:
                        count_new += 1
            st.sidebar.success(f"✅ {count_new} escalations imported and logged.")
    except Exception as e:
        st.sidebar.error(f"Failed to process Excel file: {e}")

st.sidebar.header("✍️ Manually Add Escalation")
with st.sidebar.form("manual_form"):
    cust_email = st.text_input("Customer Email")
    issue_text = st.text_area("Issue Description")
    submit_manual = st.form_submit_button("Add Escalation")

    if submit_manual:
        if cust_email.strip() == "" or issue_text.strip() == "":
            st.sidebar.error("Please enter both customer email and issue description.")
        else:
            sentiment = rule_sentiment(issue_text)
            priority = determine_priority(issue_text)
            inserted = insert_escalation({
                "customer": cust_email,
                "issue": issue_text[:1000],
                "date_reported": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sentiment": sentiment,
                "priority": priority,
                "status": "Open"
            })
            if inserted:
                st.sidebar.success("Escalation logged successfully.")
                st.experimental_rerun()
            else:
                st.sidebar.warning("Duplicate escalation found, not added.")

# Filters in sidebar
st.sidebar.header("🔎 Filters")
filter_status = st.sidebar.multiselect("Filter by Status", options=["Open", "In Progress", "Resolved"], default=["Open", "In Progress", "Resolved"])
filter_priority = st.sidebar.multiselect("Filter by Priority", options=["High", "Low"], default=["High", "Low"])
filter_escalated_only = st.sidebar.checkbox("Show only escalated (High priority) cases", value=False)

# Auto-fetch emails every 60 seconds (simple workaround)
if 'last_fetch' not in st.session_state:
    st.session_state['last_fetch'] = 0

now = time.time()
if now - st.session_state['last_fetch'] > 60:
    new_count = fetch_emails_and_log()
    if new_count > 0:
        st.success(f"🔄 {new_count} new emails fetched and escalations logged.")
    st.session_state['last_fetch'] = now

# Fetch escalations dataframe
df_escalations = fetch_escalations()

# Apply filters
if filter_escalated_only:
    df_escalations = df_escalations[df_escalations['priority'] == "High"]
df_escalations = df_escalations[df_escalations['status'].isin(filter_status)]
df_escalations = df_escalations[df_escalations['priority'].isin(filter_priority)]

# Kanban board columns
status_columns = ["Open", "In Progress", "Resolved"]
cols = st.columns(len(status_columns))

def update_escalation_from_ui(escalation_id, field, new_val):
    update_escalation_field(escalation_id, field, new_val)
    st.experimental_rerun()

for idx, status in enumerate(status_columns):
    with cols[idx]:
        st.header(f"{status} ({len(df_escalations[df_escalations['status'] == status])})")
        df_status = df_escalations[df_escalations['status'] == status]

        if df_status.empty:
            st.write("No escalations")
        else:
            for i, row in df_status.iterrows():
                with st.expander(f"{row['escalation_id']} - {row['sentiment']} / {row['priority']}"):
                    st.write(f"**Customer:** {row['customer']}")
                    st.write(f"**Issue:** {row['issue']}")
                    st.write(f"**Reported on:** {row['date_reported']}")

                    new_status = st.selectbox(
                        "Update Status", status_columns,
                        index=status_columns.index(row['status']),
                        key=f"{row['escalation_id']}_status"
                    )
                    if new_status != row['status']:
                        update_escalation_from_ui(row['escalation_id'], 'status', new_status)

                    action_taken = st.text_area(
                        "Action Taken",
                        value=row.get('action_taken', ''),
                        key=f"{row['escalation_id']}_action_taken"
                    )
                    if action_taken != row.get('action_taken', ''):
                        update_escalation_from_ui(row['escalation_id'], 'action_taken', action_taken)

                    action_owner = st.text_input(
                        "Action Owner",
                        value=row.get('action_owner', ''),
                        key=f"{row['escalation_id']}_action_owner"
                    )
                    if action_owner != row.get('action_owner', ''):
                        update_escalation_from_ui(row['escalation_id'], 'action_owner', action_owner)

# Download button for all escalations Excel
st.markdown("---")
st.header("Download Escalations Data")
excel_file = create_excel_file()
with open(excel_file, "rb") as f:
    st.download_button(
        label="📥 Download Escalations Excel",
        data=f,
        file_name=excel_file,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
