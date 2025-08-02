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

# The error is caused by referencing keys in the DataFrame row that might not exist.
# Let's add debugging and fallback to avoid KeyError during rendering.

def render_kanban():
    conn = sqlite3.connect("escalations.db")
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    conn.close()

    statuses = ["Open", "In Progress", "Resolved"]
    st.subheader("📋 Escalation Kanban Board")
    
    cols = st.columns(len(statuses))
    for idx, status in enumerate(statuses):
        with cols[idx]:
            st.markdown(f"**{status}**")
            filtered = df[df["status"] == status]
            for _, row in filtered.iterrows():
                escalation_id = row.get("escalation_id", "UNKNOWN")
                sentiment = row.get("sentiment", "Unknown")
                priority = row.get("priority", "Unknown")
                customer = row.get("customer", "")
                issue = row.get("issue", "")
                date = row.get("date", "")
                
                with st.expander(f"{escalation_id} - {sentiment}/{priority}"):
                    st.write(f"**Customer:** {customer}")
                    st.write(f"**Issue:** {issue}")
                    st.write(f"**Date:** {date}")

                    new_status = st.selectbox("Update Status", statuses, index=statuses.index(status), key=f"status_{escalation_id}")
                    action_taken = st.text_area("Action Taken", key=f"action_{escalation_id}")

                    if st.button("Update", key=f"update_{escalation_id}"):
                        conn = sqlite3.connect("escalations.db")
                        cursor = conn.cursor()
                        cursor.execute("""
                            UPDATE escalations
                            SET status = ?, action_taken = ?
                            WHERE escalation_id = ?
                        """, (new_status, action_taken, escalation_id))
                        conn.commit()
                        conn.close()
                        st.success("Escalation updated successfully.")
                        st.experimental_rerun()
