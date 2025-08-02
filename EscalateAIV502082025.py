import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
from dotenv import load_dotenv
import time
import threading
import base64

# Load environment variables
load_dotenv()

# Gmail credentials from env
EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")

# DB setup
conn = sqlite3.connect("escalations.db", check_same_thread=False)
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
    action_taken TEXT,
    action_owner TEXT
)
""")
conn.commit()

# Colors for display
sentiment_colors = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
priority_colors = {"High": "🔥", "Low": "❄️", "Medium": "⚡"}

status_columns = ["Open", "In Progress", "Resolved"]

# NLP & urgency keywords
urgency_keywords = ['urgent', 'immediately', 'critical', 'fail', 'escalate', 'issue', 'problem', 'complaint']

# Function to fetch emails from Gmail
def fetch_gmail_emails():
    try:
        mail = imaplib.IMAP4_SSL(EMAIL_SERVER)
        mail.login(EMAIL, APP_PASSWORD)
        mail.select("inbox")
        status, messages = mail.search(None, '(UNSEEN)')
        if status != "OK":
            st.warning("Could not fetch emails")
            return []
        email_ids = messages[0].split()
        fetched_emails = []
        # Limit to last 10 unseen for performance
        for num in email_ids[-10:]:
            res, msg_data = mail.fetch(num, '(RFC822)')
            if res != 'OK':
                continue
            msg = email.message_from_bytes(msg_data[0][1])
            subject, encoding = decode_header(msg.get("Subject", ""))[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or 'utf-8', errors='ignore')
            from_ = msg.get("From", "")
            date = msg.get("Date", "")
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
            fetched_emails.append({
                "customer": from_,
                "issue": body.strip(),
                "date": date,
                "subject": subject
            })
            # Mark as seen
            mail.store(num, '+FLAGS', '\\Seen')
        mail.logout()
        return fetched_emails
    except Exception as e:
        st.error(f"Error fetching emails: {e}")
        return []

# Analyze emails for NLP, sentiment, priority and log into DB
def analyze_and_log_emails(fetched_emails):
    new_count = 0
    for email_data in fetched_emails:
        customer = email_data["customer"]
        issue = email_data["issue"]
        date = email_data["date"]

        # Check duplicates
        cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (customer, issue))
        if cursor.fetchone():
            continue  # Duplicate, skip

        # Sentiment and priority
        count_urgent = sum(word in issue.lower() for word in urgency_keywords)
        sentiment = "Negative" if count_urgent > 0 else "Positive"
        priority = "High" if count_urgent >= 2 else "Low"

        # Escalation ID
        cursor.execute("SELECT COUNT(*) FROM escalations")
        count = cursor.fetchone()[0] + 250001
        escalation_id = f"SESICE-{count}"

        cursor.execute("""
            INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, action_taken, action_owner)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (escalation_id, customer, issue[:1000], date, "Open", sentiment, priority, "", ""))
        new_count += 1
    conn.commit()
    return new_count

# Upload excel and analyze
def process_uploaded_file(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        # Expecting columns: customer, issue, date (optional)
        required_cols = {"customer", "issue"}
        if not required_cols.issubset(set(df.columns.str.lower())):
            st.error(f"Uploaded file must contain columns: {required_cols}")
            return 0
        # Normalize column names lowercase
        df.columns = [c.lower() for c in df.columns]
        new_escalations = []
        for _, row in df.iterrows():
            new_escalations.append({
                "customer": row.get("customer", ""),
                "issue": str(row.get("issue", "")),
                "date": row.get("date", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            })
        count = analyze_and_log_emails(new_escalations)
        return count
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        return 0

# Manual entry from sidebar
def manual_entry():
    st.sidebar.header("➕ Manual Escalation Entry")
    customer = st.sidebar.text_input("Customer Email or Name")
    issue = st.sidebar.text_area("Issue Description")
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if st.sidebar.button("Add Escalation"):
        if not customer or not issue:
            st.sidebar.error("Customer and Issue are required.")
        else:
            new_escalation = [{
                "customer": customer,
                "issue": issue,
                "date": date
            }]
            count = analyze_and_log_emails(new_escalation)
            if count > 0:
                st.sidebar.success("Escalation added successfully!")
            else:
                st.sidebar.warning("Escalation might be duplicate or invalid.")

# Download DB as Excel
def download_db_as_excel():
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    if df.empty:
        st.info("No escalations to download.")
        return
    excel_data = df.to_excel(index=False)
    b64 = base64.b64encode(excel_data.encode()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="escalations.xlsx">Download escalations Excel</a>'
    st.markdown(href, unsafe_allow_html=True)

# Render Kanban board with filters and update options
def render_kanban():
    st.header("🚀 EscalateAI - Escalations Kanban Board")

    # Filters
    filter_option = st.selectbox("Filter cases", ["All", "Escalated"])
    search_text = st.text_input("Search by customer or issue")

    # Fetch from DB
    df = pd.read_sql_query("SELECT * FROM escalations", conn)

    # Apply filter for escalated: priority == High or sentiment Negative
    if filter_option == "Escalated":
        df = df[(df['priority'] == "High") | (df['sentiment'] == "Negative")]

    # Apply search filter
    if search_text:
        df = df[df.apply(lambda r: search_text.lower() in str(r['customer']).lower() or search_text.lower() in str(r['issue']).lower(), axis=1)]

    if df.empty:
        st.info("No escalations found.")
        return

    # Split by status for Kanban
    for status in status_columns:
        st.subheader(f"{status} ({len(df[df['status'] == status])})")
        filtered = df[df['status'] == status]
        for idx, row in filtered.iterrows():
            esc_id = row.get('escalation_id', '') or f"noid_{idx}"
            sentiment = row.get('sentiment', 'Neutral')
            priority = row.get('priority', 'Low')
            # Colored header
            header = f"{esc_id} - {sentiment_colors.get(sentiment, '')} {sentiment} / {priority_colors.get(priority, '')} {priority}"
            with st.expander(header):
                st.markdown(f"**Customer:** {row.get('customer','')}")
                st.markdown(f"**Issue:** {row.get('issue','')}")
                st.markdown(f"**Date:** {row.get('date','')}")
                # Editable fields with unique keys
                new_status = st.selectbox(
                    "Update Status",
                    status_columns,
                    index=status_columns.index(row.get('status', 'Open')),
                    key=f"{esc_id}_status_{idx}"
                )
                new_action_taken = st.text_area(
                    "Action Taken",
                    value=row.get('action_taken', ''),
                    key=f"{esc_id}_action_{idx}"
                )
                new_action_owner = st.text_input(
                    "Action Owner",
                    value=row.get('action_owner', ''),
                    key=f"{esc_id}_owner_{idx}"
                )
                # Update DB on change
                if (new_status != row.get('status')) or (new_action_taken != row.get('action_taken')) or (new_action_owner != row.get('action_owner')):
                    cursor.execute("""
                        UPDATE escalations SET status=?, action_taken=?, action_owner=?
                        WHERE escalation_id=?
                    """, (new_status, new_action_taken, new_action_owner, esc_id))
                    conn.commit()
                    st.success(f"Updated escalation {esc_id}.")

    # Download option
    if st.button("⬇️ Download Escalations Excel"):
        df_download = pd.read_sql_query("SELECT * FROM escalations", conn)
        if df_download.empty:
            st.info("No escalations to download.")
        else:
            tmp_filename = "escalations_download.xlsx"
            df_download.to_excel(tmp_filename, index=False)
            with open(tmp_filename, "rb") as f:
                st.download_button("Click here to download Excel", f, file_name=tmp_filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Main app
def main():
    st.title("EscalateAI - AI-based Customer Escalation Management")

    # Sidebar features
    st.sidebar.header("Gmail Auto Fetch & Upload")
    if st.sidebar.button("Fetch new Emails from Gmail"):
        emails = fetch_gmail_emails()
        if emails:
            count = analyze_and_log_emails(emails)
            st.sidebar.success(f"Fetched and logged {count} new escalations.")
        else:
            st.sidebar.info("No new emails found or error.")

    uploaded_file = st.sidebar.file_uploader("Upload Excel file of complaints", type=["xlsx"])
    if uploaded_file:
        count = process_uploaded_file(uploaded_file)
        st.sidebar.success(f"Processed and logged {count} escalations from uploaded file.")

    manual_entry()

    # Main kanban board view
    render_kanban()

if __name__ == "__main__":
    main()
