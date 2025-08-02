import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")

# Initialize database
def init_db():
    conn = sqlite3.connect("escalations.db")
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
            action_taken TEXT DEFAULT ''
        )
    """)
    conn.commit()
    conn.close()

# Email fetching function
def connect_and_fetch_emails():
    if not EMAIL or not APP_PASSWORD:
        st.warning("🔐 Email or App Password missing. Please check .env file.")
        return []

    try:
        mail = imaplib.IMAP4_SSL(EMAIL_SERVER)
        mail.login(EMAIL, APP_PASSWORD)
        st.success("📡 Connected to Gmail.")
    except imaplib.IMAP4.error as e:
        st.error(f"❌ Gmail login failed: {str(e)}")
        return []

    status, _ = mail.select("inbox")
    if status != 'OK':
        st.error("❌ Could not select inbox.")
        return []

    result, data = mail.search(None, '(UNSEEN)')
    if result != 'OK':
        st.info("📭 No unread emails found.")
        return []

    fetched_emails = []
    for num in data[0].split()[-10:]:
        result, msg_data = mail.fetch(num, '(RFC822)')
        if result != 'OK':
            continue

        msg = email.message_from_bytes(msg_data[0][1])
        subject, encoding = decode_header(msg["Subject"])[0]
        subject = subject.decode(encoding or 'utf-8') if isinstance(subject, bytes) else subject
        from_ = msg.get("From")
        date = msg.get("Date")

        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition")):
                    try:
                        body = part.get_payload(decode=True).decode()
                        break
                    except:
                        pass
        else:
            try:
                body = msg.get_payload(decode=True).decode()
            except:
                pass

        fetched_emails.append({"from": from_, "subject": subject, "body": body, "date": date})
        mail.store(num, '+FLAGS', '\\Seen')

    mail.logout()
    return fetched_emails

# NLP and Logging function
def analyze_and_log_emails(fetched_emails):
    conn = sqlite3.connect("escalations.db")
    cursor = conn.cursor()
    for email_data in fetched_emails:
        from_email = email_data['from']
        subject = email_data['subject']
        body = email_data['body'][:500]
        date = email_data['date']

        cursor.execute("SELECT COUNT(*) FROM escalations WHERE customer=? AND issue=?", (from_email, body))
        if cursor.fetchone()[0] > 0:
            continue  # duplicate

        urgency_keywords = ['urgent', 'critical', 'immediately', 'fail', 'problem', 'escalate']
        sentiment_score = sum(1 for word in urgency_keywords if word in body.lower())
        sentiment = "Negative" if sentiment_score else "Positive"
        priority = "High" if sentiment_score >= 2 else "Low"

        cursor.execute("SELECT COUNT(*) FROM escalations")
        count = cursor.fetchone()[0] + 250001
        escalation_id = f"SESICE-{count}"

        cursor.execute("""
            INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (escalation_id, from_email, body, date, "Open", sentiment, priority))

    conn.commit()
    conn.close()

# Kanban UI
def render_kanban():
    conn = sqlite3.connect("escalations.db")
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    conn.close()

    status_columns = ["Open", "In Progress", "Resolved"]
    cols = st.columns(len(status_columns))
    for idx, status in enumerate(status_columns):
        with cols[idx]:
            st.subheader(f"{status} ({(df['status'] == status).sum()})")
            for _, row in df[df['status'] == status].iterrows():
                with st.expander(f"{row.get('escalation_id')} - {row.get('sentiment')}/{row.get('priority')}"):
                    st.write(f"**From:** {row.get('customer')}")
                    st.write(f"**Issue:** {row.get('issue')}")
                    new_status = st.selectbox("Update Status", status_columns, index=status_columns.index(status), key=row.get('escalation_id') + '_status')
                    action_taken = st.text_area("Action Taken", value=row.get('action_taken', ''), key=row.get('escalation_id') + '_action')
                    if st.button("Save", key=row.get('escalation_id') + '_save'):
                        conn = sqlite3.connect("escalations.db")
                        cursor = conn.cursor()
                        cursor.execute("UPDATE escalations SET status=?, action_taken=? WHERE escalation_id=?", (new_status, action_taken, row.get('escalation_id')))
                        conn.commit()
                        conn.close()
                        st.experimental_rerun()

# Main App
st.set_page_config(page_title="EscalateAI", layout="wide")
st.title("🚨 EscalateAI - Escalation Management")

init_db()

if st.button("📬 Fetch Emails"):
    emails = connect_and_fetch_emails()
    if emails:
        analyze_and_log_emails(emails)
        st.success(f"✅ {len(emails)} emails fetched and logged.")
    else:
        st.info("No new emails.")

st.markdown("---")
st.subheader("🧾 Escalation Kanban Board")
render_kanban()
