import streamlit as st
import imaplib
import email
from email.header import decode_header
import pandas as pd
import sqlite3
import datetime
import os
import re
from dotenv import load_dotenv
from textblob import TextBlob  # For sentiment analysis

# Load environment variables for Gmail credentials
load_dotenv()

GMAIL_USER = os.getenv("EMAIL_USER")
GMAIL_APP_PASSWORD = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")

DB_FILE = "escalations.db"

# Status options for Kanban
STATUS_OPTIONS = ["Open", "In Progress", "Resolved"]
ESCALATION_FILTERS = ["All", "Escalated"]

# Colors for Kanban styling
status_colors = {
    "Open": "#FFDDDD",
    "In Progress": "#FFF0B3",
    "Resolved": "#DDFFDD"
}
sentiment_colors = {
    "Positive": "🟢",
    "Negative": "🔴",
    "Neutral": "🟡"
}
priority_colors = {
    "High": "🔥",
    "Low": "⚪"
}

# Initialize DB
def init_db():
    conn = sqlite3.connect(DB_FILE)
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
    conn.close()

# Generate next escalation ID
def get_next_escalation_id():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0] + 250001
    conn.close()
    return f"SESICE-{count}"

# Connect to Gmail and fetch unread emails (limit last 10)
def fetch_gmail_emails():
    if not GMAIL_USER or not GMAIL_APP_PASSWORD:
        st.error("Gmail credentials missing in .env file!")
        return []

    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        mail.select("inbox")
        result, data = mail.search(None, 'UNSEEN')
        if result != "OK":
            return []

        email_ids = data[0].split()
        emails = []

        # Fetch max 10 unread emails
        for num in email_ids[-10:]:
            res, msg_data = mail.fetch(num, "(RFC822)")
            if res != "OK":
                continue
            msg = email.message_from_bytes(msg_data[0][1])

            # Decode subject
            subject, encoding = decode_header(msg.get("Subject"))[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or "utf-8", errors="ignore")

            from_ = msg.get("From")
            date_ = msg.get("Date")

            # Extract plain text body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition")):
                        try:
                            body = part.get_payload(decode=True).decode()
                        except:
                            body = ""
                        break
            else:
                try:
                    body = msg.get_payload(decode=True).decode()
                except:
                    body = ""

            emails.append({
                "customer": from_,
                "issue": body.strip(),
                "date": date_,
                "subject": subject
            })

            # Mark as read
            mail.store(num, '+FLAGS', '\\Seen')

        mail.logout()
        return emails
    except Exception as e:
        st.error(f"Error fetching emails: {e}")
        return []

# Simple NLP sentiment and priority detection
def analyze_text(text):
    # Sentiment polarity: -1 to +1
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # Urgency keywords
    urgency_keywords = ['urgent', 'immediately', 'critical', 'fail', 'escalate', 'issue', 'problem', 'complaint', 'down', 'error']
    priority_count = sum(text.lower().count(word) for word in urgency_keywords)
    priority = "High" if priority_count >= 2 else "Low"

    return sentiment, priority

# Insert escalations into DB avoiding duplicates
def log_escalations(escalations):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    added = 0
    for esc in escalations:
        # Avoid duplicate by customer + issue (simple check)
        cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (esc['customer'], esc['issue'][:500]))
        if cursor.fetchone():
            continue

        sentiment, priority = analyze_text(esc['issue'])
        esc_id = get_next_escalation_id()

        cursor.execute("""
            INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, action_taken, action_owner)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (esc_id, esc['customer'], esc['issue'][:500], esc['date'], "Open", sentiment, priority, "", ""))

        added += 1
    conn.commit()
    conn.close()
    return added

# Load escalations from DB with optional filters
def load_escalations(status_filter=None, escalated_filter=None):
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT * FROM escalations"
    params = []
    conditions = []
    if status_filter and status_filter != "All":
        conditions.append("status=?")
        params.append(status_filter)
    if escalated_filter == "Escalated":
        conditions.append("sentiment='Negative'")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

# Update escalation entry in DB
def update_escalation(esc_id, status, action_taken, action_owner):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE escalations SET status=?, action_taken=?, action_owner=? WHERE escalation_id=?
    """, (status, action_taken, action_owner, esc_id))
    conn.commit()
    conn.close()

# Sidebar for manual entry and Excel upload
def sidebar_controls():
    st.sidebar.header("Add / Upload Escalations")

    with st.sidebar.expander("Manual Entry"):
        customer = st.text_input("Customer Email / Name")
        issue = st.text_area("Issue Description")
        date = st.date_input("Date", datetime.date.today())
        if st.button("Add Escalation"):
            if not customer or not issue:
                st.warning("Please enter customer and issue details.")
            else:
                esc = {
                    "customer": customer,
                    "issue": issue,
                    "date": date.strftime("%a, %d %b %Y %H:%M:%S"),
                }
                added = log_escalations([esc])
                if added:
                    st.success("Manual escalation added!")
                else:
                    st.info("Escalation may already exist.")

    with st.sidebar.expander("Upload Excel File"):
        uploaded_file = st.file_uploader("Upload Escalations Excel (.xlsx)", type=["xlsx"])
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                # Expect columns: customer, issue, date (case insensitive)
                required_cols = ["customer", "issue", "date"]
                lower_cols = [c.lower() for c in df.columns]
                if all(rc in lower_cols for rc in required_cols):
                    df.columns = lower_cols  # normalize
                    escalations = df[required_cols].to_dict(orient="records")
                    added = log_escalations(escalations)
                    st.success(f"{added} escalations added from uploaded file.")
                else:
                    st.error(f"Uploaded file must contain columns: {required_cols}")
            except Exception as e:
                st.error(f"Error reading file: {e}")

# Render Kanban Board
def render_kanban():
    st.header("🚀 EscalateAI - Escalations Kanban Board")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox("Filter by Status", options=["All"] + STATUS_OPTIONS)
    with col2:
        escalated_filter = st.selectbox("Filter by Escalation", options=ESCALATION_FILTERS)

    df = load_escalations(status_filter, escalated_filter)
    if df.empty:
        st.info("No escalations found.")
        return

    # Allow downloading escalations as Excel
    def to_excel(df):
        output = pd.ExcelWriter("escalations_export.xlsx", engine='xlsxwriter')
        df.to_excel(output, index=False, sheet_name='Escalations')
        output.save()
        return "escalations_export.xlsx"

    st.download_button("⬇️ Download Escalations Excel", data=df.to_csv(index=False), file_name="escalations.csv", mime="text/csv")

    # Display Kanban columns
    cols = st.columns(len(STATUS_OPTIONS))

    for idx, status in enumerate(STATUS_OPTIONS):
        with cols[idx]:
            st.subheader(f"{status} ({len(df[df['status'] == status])})")
            subset = df[df['status'] == status]

            for _, row in subset.iterrows():
                header = f"{row['escalation_id']} - {sentiment_colors.get(row['sentiment'],'')} {row['sentiment']} / {priority_colors.get(row['priority'],'')} {row['priority']}"
                with st.expander(header):
                    st.write(f"**Customer:** {row['customer']}")
                    st.write(f"**Issue:** {row['issue']}")
                    st.write(f"**Date:** {row['date']}")
                    # Editable fields
                    new_status = st.selectbox("Update Status", STATUS_OPTIONS, index=STATUS_OPTIONS.index(row['status']), key=f"status_{row['escalation_id']}")
                    new_action_taken = st.text_area("Action Taken", value=row['action_taken'] or "", key=f"action_{row['escalation_id']}")
                    new_action_owner = st.text_input("Action Owner", value=row['action_owner'] or "", key=f"owner_{row['escalation_id']}")

                    # Update DB on change
                    if (new_status != row['status'] or new_action_taken != (row['action_taken'] or "") or new_action_owner != (row['action_owner'] or "")):
                        update_escalation(row['escalation_id'], new_status, new_action_taken, new_action_owner)
                        st.success("Updated escalation details.")

# Main app
def main():
    st.title("EscalateAI - Automated Escalation Management")
    init_db()

    # Fetch new emails every minute (user can press button)
    if st.button("🔄 Fetch new emails from Gmail"):
        with st.spinner("Fetching emails..."):
            emails = fetch_gmail_emails()
            if emails:
                added = log_escalations(emails)
                st.success(f"Fetched {len(emails)} emails, added {added} new escalations.")
            else:
                st.info("No new emails fetched.")

    sidebar_controls()
    render_kanban()

if __name__ == "__main__":
    main()
