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
import base64

# Load environment variables
load_dotenv()

# Gmail credentials from environment variables (no user input needed)
EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER") or "imap.gmail.com"

# Initialize database
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
        action_taken TEXT
    )
""")
conn.commit()
conn.close()

def connect_and_fetch_emails():
    if not EMAIL or not APP_PASSWORD:
        st.warning("🔐 Gmail credentials not found in environment variables.")
        return []

    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL, APP_PASSWORD)
        st.success("📡 Connected to Gmail.")
    except imaplib.IMAP4.error as e:
        st.error(f"❌ Gmail login failed: {str(e)}")
        return []

    status, messages = mail.select("inbox")
    if status != 'OK':
        st.error("❌ Could not select the inbox.")
        return []

    result, data = mail.search(None, '(UNSEEN)')
    if result != 'OK' or not data or not data[0]:
        st.info("📪 No new emails or error in fetching.")
        return []

    email_ids = data[0].split()
    fetched_emails = []

    for num in email_ids[-10:]:  # Limit for performance
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

        mail.store(num, '+FLAGS', '\\Seen')  # Mark as read

    mail.logout()

    # Save emails to Excel
    df_emails = pd.DataFrame(fetched_emails)
    if not df_emails.empty:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fetched_emails_{timestamp}.xlsx"
        df_emails.to_excel(filename, index=False)
        st.success(f"📄 Emails saved to {filename}")

    return fetched_emails

def analyze_and_log_emails(fetched_emails):
    conn = sqlite3.connect("escalations.db")
    cursor = conn.cursor()

    for email_data in fetched_emails:
        from_email = email_data['from']
        subject = email_data['subject']
        body = email_data['body']
        date = email_data['date']

        cursor.execute("SELECT COUNT(*) FROM escalations WHERE customer=? AND issue=?", (from_email, body))
        if cursor.fetchone()[0] > 0:
            continue  # Skip duplicate

        urgency_keywords = ['urgent', 'immediately', 'critical', 'fail', 'escalate', 'issue', 'problem', 'complaint']
        sentiment_score = sum(1 for word in urgency_keywords if word in body.lower())
        sentiment = "Negative" if sentiment_score > 0 else "Positive"
        priority = "High" if sentiment_score >= 2 else "Low"

        cursor.execute("SELECT COUNT(*) FROM escalations")
        count = cursor.fetchone()[0] + 250001
        escalation_id = f"SESICE-{count}"

        cursor.execute("""
            INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, action_taken)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (escalation_id, from_email, body[:500], date, "Open", sentiment, priority, ""))

    conn.commit()
    conn.close()

def render_kanban():
    conn = sqlite3.connect("escalations.db")
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    conn.close()

    st.header("🚀 EscalateAI - Escalation Management with Auto Escalation & Email Alerts")
    st.subheader("Escalations Kanban Board")

    statuses = ["Open", "In Progress", "Resolved"]
    cols = st.columns(len(statuses))

    for i, status in enumerate(statuses):
        with cols[i]:
            st.markdown(f"### {status} ({df[df['status']==status].shape[0]})")
            for _, row in df[df['status'] == status].iterrows():
                with st.expander(f"{row['escalation_id']} - {row['sentiment']}/{row['priority']}"):
                    st.markdown(f"**Customer:** {row['customer']}")
                    st.markdown(f"**Issue:** {row['issue'][:300]}...")
                    st.markdown(f"**Date:** {row['date']}")
                    new_status = st.selectbox("Update Status", statuses, index=statuses.index(row['status']), key=f"status_{row['escalation_id']}")
                    new_action = st.text_input("Action Taken", value=row['action_taken'], key=f"action_{row['escalation_id']}")
                    if st.button("💾 Save", key=f"save_{row['escalation_id']}"):
                        conn = sqlite3.connect("escalations.db")
                        cursor = conn.cursor()
                        cursor.execute("UPDATE escalations SET status=?, action_taken=? WHERE escalation_id=?", (new_status, new_action, row['escalation_id']))
                        conn.commit()
                        conn.close()
                        st.success("✅ Updated")

st.set_page_config(page_title="EscalateAI", layout="wide")
st.title("📬 Gmail Escalation Fetcher")

if st.button("🔄 Fetch Emails"):
    emails = connect_and_fetch_emails()
    if emails:
        analyze_and_log_emails(emails)
        st.success(f"✅ {len(emails)} emails analyzed and logged.")
        st.dataframe(pd.DataFrame(emails))

render_kanban()
