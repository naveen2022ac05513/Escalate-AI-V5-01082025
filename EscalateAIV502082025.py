import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
import re
from dotenv import load_dotenv
import time
import threading

# Load environment variables
load_dotenv()

EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")

DB_FILE = "escalations.db"

# Sentiment and priority keywords (simple example)
URGENCY_KEYWORDS = ['urgent', 'immediately', 'critical', 'fail', 'escalate', 'issue', 'problem', 'complaint']
ESCALATION_TRIGGERS = ['escalate', 'not working', 'failure', 'down']

# Colors for Kanban board
STATUS_COLORS = {
    "Open": "#FFDDDD",
    "In Progress": "#FFF1CC",
    "Resolved": "#D0E9D0"
}
SENTIMENT_COLORS = {
    "Negative": "🔴",
    "Positive": "🟢"
}
PRIORITY_COLORS = {
    "High": "🔥",
    "Low": "⚪"
}

# Initialize DB connection
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()

def init_db():
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS escalations (
        escalation_id TEXT PRIMARY KEY,
        customer TEXT,
        issue TEXT,
        date_reported TEXT,
        status TEXT DEFAULT 'Open',
        sentiment TEXT,
        priority TEXT,
        action_taken TEXT DEFAULT '',
        action_owner TEXT DEFAULT ''
    )
    """)
    conn.commit()

def fetch_emails_from_gmail():
    if not EMAIL or not APP_PASSWORD:
        st.warning("Please set EMAIL_USER and EMAIL_PASS in your .env file.")
        return []
    try:
        mail = imaplib.IMAP4_SSL(EMAIL_SERVER)
        mail.login(EMAIL, APP_PASSWORD)
        mail.select("inbox")
        result, data = mail.search(None, '(UNSEEN)')
        if result != "OK":
            return []

        email_ids = data[0].split()
        fetched_emails = []

        # Limit to last 10 unread emails
        for num in email_ids[-10:]:
            result, msg_data = mail.fetch(num, '(RFC822)')
            if result != 'OK':
                continue
            msg = email.message_from_bytes(msg_data[0][1])

            subject, encoding = decode_header(msg["Subject"])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or 'utf-8', errors='ignore')

            from_ = msg.get("From")
            date = msg.get("Date")

            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    if content_type == "text/plain" and "attachment" not in content_disposition:
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

            fetched_emails.append({
                "from": from_,
                "subject": subject,
                "body": body,
                "date": date
            })

            # Mark as seen
            mail.store(num, '+FLAGS', '\\Seen')

        mail.logout()
        return fetched_emails
    except imaplib.IMAP4.error as e:
        st.error(f"Gmail login or fetch error: {e}")
        return []
    except Exception as e:
        st.error(f"Unexpected error fetching emails: {e}")
        return []

def analyze_and_log_emails(fetched_emails):
    if not fetched_emails:
        return 0
    count = 0
    for email_data in fetched_emails:
        from_email = email_data['from']
        body = email_data['body']
        date = email_data['date']

        # Check duplicate by customer + issue (basic)
        cursor.execute("SELECT COUNT(*) FROM escalations WHERE customer=? AND issue=?", (from_email, body))
        if cursor.fetchone()[0] > 0:
            continue

        sentiment_score = sum(1 for w in URGENCY_KEYWORDS if w in body.lower())
        sentiment = "Negative" if sentiment_score > 0 else "Positive"
        priority = "High" if sentiment_score >= 2 else "Low"

        # Escalation ID generation: SESICE-250001, incrementing
        cursor.execute("SELECT COUNT(*) FROM escalations")
        new_id_num = cursor.fetchone()[0] + 250001
        escalation_id = f"SESICE-{new_id_num}"

        cursor.execute("""
        INSERT INTO escalations (escalation_id, customer, issue, date_reported, status, sentiment, priority)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (escalation_id, from_email, body[:500], date, "Open", sentiment, priority))

        count += 1
    conn.commit()
    return count

def load_escalations():
    df = pd.read_sql_query("SELECT * FROM escalations ORDER BY date_reported DESC", conn)
    # Fill missing columns if any
    for col in ['action_taken', 'action_owner']:
        if col not in df.columns:
            df[col] = ''
        else:
            df[col] = df[col].fillna('')
    return df

def update_escalation(escalation_id, field, value):
    cursor.execute(f"UPDATE escalations SET {field}=? WHERE escalation_id=?", (value, escalation_id))
    conn.commit()

def upload_excel_to_db(uploaded_file):
    if uploaded_file is None:
        return 0
    try:
        df = pd.read_excel(uploaded_file)
        count = 0
        for _, row in df.iterrows():
            customer = str(row.get("customer") or row.get("from") or "")
            issue = str(row.get("issue") or row.get("body") or "")
            date_reported = str(row.get("date_reported") or datetime.datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z"))
            if not customer or not issue:
                continue
            # Check duplicate
            cursor.execute("SELECT COUNT(*) FROM escalations WHERE customer=? AND issue=?", (customer, issue))
            if cursor.fetchone()[0] > 0:
                continue
            sentiment_score = sum(1 for w in URGENCY_KEYWORDS if w in issue.lower())
            sentiment = "Negative" if sentiment_score > 0 else "Positive"
            priority = "High" if sentiment_score >= 2 else "Low"
            cursor.execute("SELECT COUNT(*) FROM escalations")
            new_id_num = cursor.fetchone()[0] + 250001
            escalation_id = f"SESICE-{new_id_num}"
            cursor.execute("""
            INSERT INTO escalations (escalation_id, customer, issue, date_reported, status, sentiment, priority)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (escalation_id, customer, issue[:500], date_reported, "Open", sentiment, priority))
            count += 1
        conn.commit()
        return count
    except Exception as e:
        st.error(f"Failed to process uploaded file: {e}")
        return 0

def manual_entry_form():
    st.sidebar.header("Manual Escalation Entry")
    customer = st.sidebar.text_input("Customer Email / Name")
    issue = st.sidebar.text_area("Issue / Complaint")
    date_reported = st.sidebar.date_input("Date Reported", datetime.datetime.now())
    if st.sidebar.button("Add Escalation"):
        if not customer or not issue:
            st.sidebar.error("Customer and Issue are required.")
            return
        sentiment_score = sum(1 for w in URGENCY_KEYWORDS if w in issue.lower())
        sentiment = "Negative" if sentiment_score > 0 else "Positive"
        priority = "High" if sentiment_score >= 2 else "Low"
        cursor.execute("SELECT COUNT(*) FROM escalations")
        new_id_num = cursor.fetchone()[0] + 250001
        escalation_id = f"SESICE-{new_id_num}"
        date_str = date_reported.strftime("%a, %d %b %Y %H:%M:%S %z")
        cursor.execute("""
        INSERT INTO escalations (escalation_id, customer, issue, date_reported, status, sentiment, priority)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (escalation_id, customer, issue[:500], date_str, "Open", sentiment, priority))
        conn.commit()
        st.sidebar.success(f"Escalation {escalation_id} added.")

def render_kanban():
    st.header("🚀 EscalateAI - Escalations Kanban Board")

    df = load_escalations()

    # Filters
    filter_option = st.selectbox("Filter escalations:", ["All", "Escalated (Negative Priority High)"])
    if filter_option == "Escalated (Negative Priority High)":
        df = df[(df['sentiment'] == "Negative") & (df['priority'] == "High")]

    status_columns = ["Open", "In Progress", "Resolved"]
    cols = st.columns(len(status_columns))

    for idx, status in enumerate(status_columns):
        with cols[idx]:
            st.subheader(f"{status} ({len(df[df['status'] == status])})")
            filtered = df[df['status'] == status]
            if filtered.empty:
                st.info("No escalations")
            for _, row in filtered.iterrows():
                header = f"{row.get('escalation_id','')} - {SENTIMENT_COLORS.get(row.get('sentiment',''), '')} {row.get('sentiment','')} / {PRIORITY_COLORS.get(row.get('priority',''), '')} {row.get('priority','')}"
                with st.expander(header):
                    st.write(f"**Customer:** {row.get('customer','')}")
                    st.write(f"**Issue:** {row.get('issue','')}")
                    st.write(f"**Date Reported:** {row.get('date_reported','')}")
                    st.write(f"**Status:** {row.get('status','')}")

                    # Editable fields
                    new_status = st.selectbox("Update Status", status_columns, index=status_columns.index(row.get('status','Open')), key=row.get('escalation_id','') + '_status')
                    new_action_taken = st.text_area("Action Taken", value=row.get('action_taken',''), key=row.get('escalation_id','') + '_action')
                    new_action_owner = st.text_input("Action Owner", value=row.get('action_owner',''), key=row.get('escalation_id','') + '_owner')

                    if (new_status != row.get('status','Open')):
                        update_escalation(row.get('escalation_id',''), 'status', new_status)
                        st.experimental_rerun()

                    if (new_action_taken != row.get('action_taken','')):
                        update_escalation(row.get('escalation_id',''), 'action_taken', new_action_taken)
                        st.experimental_rerun()

                    if (new_action_owner != row.get('action_owner','')):
                        update_escalation(row.get('escalation_id',''), 'action_owner', new_action_owner)
                        st.experimental_rerun()

    # Download all escalations as Excel
    if not df.empty:
        to_download = df.drop(columns=['action_taken', 'action_owner']).copy()
        to_download.rename(columns={
            'escalation_id': 'Escalation ID',
            'customer': 'Customer',
            'issue': 'Issue',
            'date_reported': 'Date Reported',
            'status': 'Status',
            'sentiment': 'Sentiment',
            'priority': 'Priority',
        }, inplace=True)
        excel_bytes = to_download.to_excel(index=False)
        st.download_button("⬇️ Download Escalations Excel", data=excel_bytes, file_name="escalations.xlsx", mime="application/vnd.ms-excel")

def background_email_fetcher():
    while True:
        fetched = fetch_emails_from_gmail()
        if fetched:
            analyze_and_log_emails(fetched)
        time.sleep(60)  # Run every 1 minute

def main():
    st.set_page_config(page_title="EscalateAI - Customer Escalation Manager", layout="wide")
    st.title("🚀 EscalateAI - Customer Escalation Manager")

    init_db()

    st.sidebar.header("Upload Complaints Excel")
    uploaded_file = st.sidebar.file_uploader("Upload Excel file with complaints", type=["xlsx"])
    if uploaded_file:
        count = upload_excel_to_db(uploaded_file)
        if count > 0:
            st.sidebar.success(f"Uploaded and logged {count} escalations from Excel.")
        else:
            st.sidebar.info("No new valid escalations found in uploaded file.")

    manual_entry_form()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Fetch Emails from Gmail (runs every minute automatically)")

    if st.sidebar.button("Fetch Emails Now"):
        fetched_emails = fetch_emails_from_gmail()
        if fetched_emails:
            count = analyze_and_log_emails(fetched_emails)
            st.sidebar.success(f"Fetched and logged {count} new escalations.")
        else:
            st.sidebar.info("No new emails or failed to fetch.")

    render_kanban()

if __name__ == "__main__":
    # Run background thread for automatic fetching without blocking UI
    threading.Thread(target=background_email_fetcher, daemon=True).start()
    main()
