import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
EMAIL = os.getenv("GMAIL_USER")
APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
IMAP_SERVER = "imap.gmail.com"

# Database connection
conn = sqlite3.connect("escalateai.db", check_same_thread=False)
cursor = conn.cursor()

# Create escalations table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS escalations (
    escalation_id TEXT PRIMARY KEY,
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

# NLP basic keywords for urgency detection
URGENCY_KEYWORDS = ['urgent', 'immediately', 'critical', 'fail', 'escalate', 'issue', 'problem', 'complaint']

# Colors for Kanban statuses and sentiment/priority
STATUS_COLUMNS = ["Open", "In Progress", "Resolved"]
STATUS_COLORS = {
    "Open": "#FFCCCC",
    "In Progress": "#FFF0B3",
    "Resolved": "#CCFFCC"
}
SENTIMENT_COLORS = {
    "Negative": "❗",
    "Positive": "✅"
}
PRIORITY_COLORS = {
    "High": "🔴",
    "Low": "🟢"
}

# Utility function to generate next escalation ID
def generate_escalation_id():
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]
    return f"SESICE-{250001 + count + 1}"

# Function: fetch unseen emails from Gmail
def fetch_emails():
    if not EMAIL or not APP_PASSWORD:
        st.error("Gmail credentials missing in .env")
        return []

    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL, APP_PASSWORD)
        mail.select("inbox")
        status, data = mail.search(None, '(UNSEEN)')
        if status != "OK":
            st.warning("No unread emails found.")
            return []

        email_ids = data[0].split()
        emails = []
        for num in email_ids[-10:]:  # fetch last 10 unseen
            status, msg_data = mail.fetch(num, '(RFC822)')
            if status != "OK":
                continue
            msg = email.message_from_bytes(msg_data[0][1])

            # Decode subject
            subject, encoding = decode_header(msg.get("Subject"))[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or "utf-8", errors="ignore")

            from_ = msg.get("From")
            date = msg.get("Date")

            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain" and 'attachment' not in str(part.get("Content-Disposition")):
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

            emails.append({
                "customer": from_,
                "issue": body,
                "subject": subject,
                "date_reported": date
            })

            # Mark email as seen
            mail.store(num, '+FLAGS', '\\Seen')

        mail.logout()
        return emails
    except imaplib.IMAP4.error as e:
        st.error(f"IMAP error: {e}")
        return []

# NLP analysis function for sentiment, priority
def analyze_issue(issue_text):
    issue_lower = issue_text.lower()
    sentiment = "Positive"
    priority = "Low"

    score = sum(1 for kw in URGENCY_KEYWORDS if kw in issue_lower)
    if score > 0:
        sentiment = "Negative"
    if score >= 2:
        priority = "High"
    return sentiment, priority

# Log escalations into DB, avoid duplicates by (customer+issue)
def log_escalations(escalations):
    new_count = 0
    for esc in escalations:
        # Check duplicate
        cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (esc['customer'], esc['issue']))
        if cursor.fetchone():
            continue
        # Analyze NLP
        sentiment, priority = analyze_issue(esc['issue'])
        esc_id = generate_escalation_id()
        cursor.execute("""
            INSERT INTO escalations (escalation_id, customer, issue, date_reported, status, sentiment, priority, action_taken, action_owner)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (esc_id, esc['customer'], esc['issue'], esc['date_reported'], "Open", sentiment, priority, "", ""))
        new_count += 1
    conn.commit()
    return new_count

# Load escalations from DB into DataFrame
def load_escalations():
    df = pd.read_sql_query("SELECT * FROM escalations ORDER BY date_reported DESC", conn)
    return df

# Sidebar - Manual Upload Excel complaints
def upload_excel():
    uploaded_file = st.sidebar.file_uploader("Upload complaints Excel", type=["xlsx", "csv"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Expecting columns like customer, issue, date_reported (at least issue)
            if 'issue' not in df.columns:
                st.sidebar.error("Excel file must have 'issue' column.")
                return

            # Prepare escalation dicts
            escalations = []
            for _, row in df.iterrows():
                escalations.append({
                    "customer": row.get('customer', "Unknown"),
                    "issue": row['issue'],
                    "date_reported": str(row.get('date_reported', datetime.datetime.now()))
                })
            count = log_escalations(escalations)
            st.sidebar.success(f"{count} escalations logged from uploaded file.")
        except Exception as e:
            st.sidebar.error(f"Failed to read file: {e}")

# Sidebar - Manual escalation entry
def manual_entry():
    st.sidebar.header("Manual Escalation Entry")
    customer = st.sidebar.text_input("Customer Email/Name")
    issue = st.sidebar.text_area("Issue Description")
    date_reported = st.sidebar.date_input("Date Reported", datetime.date.today())
    if st.sidebar.button("Add Escalation"):
        if not issue.strip():
            st.sidebar.warning("Issue description cannot be empty.")
            return
        esc = {
            "customer": customer or "Unknown",
            "issue": issue,
            "date_reported": date_reported.strftime("%Y-%m-%d")
        }
        count = log_escalations([esc])
        if count > 0:
            st.sidebar.success("Escalation added successfully!")
        else:
            st.sidebar.info("Duplicate escalation ignored.")

# Render Kanban board with filters and editing
def render_kanban():
    st.header("🚀 EscalateAI - Escalations Kanban Board")

    filter_option = st.selectbox("Filter Escalations", options=["All", "Escalated (Negative)"])
    df = load_escalations()

    if filter_option == "Escalated (Negative)":
        df = df[df['sentiment'] == "Negative"]

    if df.empty:
        st.info("No escalations to display.")
        return

    # Display columns horizontally
    cols = st.columns(len(STATUS_COLUMNS))
    for col, status in zip(cols, STATUS_COLUMNS):
        col.markdown(f"### {status} ({len(df[df['status'] == status])})")
        df_status = df[df['status'] == status]

        for i, row in df_status.iterrows():
            header = f"{row['escalation_id']} - {SENTIMENT_COLORS.get(row['sentiment'], '')} {row['sentiment']} / {PRIORITY_COLORS.get(row['priority'], '')} {row['priority']}"
            with col.expander(header, expanded=False):
                col.write(f"**Customer:** {row['customer']}")
                col.write(f"**Issue:** {row['issue']}")
                col.write(f"**Date Reported:** {row['date_reported']}")

                # Editable status selectbox
                new_status = col.selectbox("Update Status", STATUS_COLUMNS, index=STATUS_COLUMNS.index(row['status']), key=f"status_{row['escalation_id']}")
                # Editable action taken and owner text inputs
                new_action_taken = col.text_area("Action Taken", value=row['action_taken'], key=f"action_{row['escalation_id']}")
                new_action_owner = col.text_input("Action Owner", value=row['action_owner'], key=f"owner_{row['escalation_id']}")

                # Update DB if changed
                if (new_status != row['status']) or (new_action_taken != row['action_taken']) or (new_action_owner != row['action_owner']):
                    cursor.execute("""
                        UPDATE escalations SET status=?, action_taken=?, action_owner=? WHERE escalation_id=?
                    """, (new_status, new_action_taken, new_action_owner, row['escalation_id']))
                    conn.commit()
                    st.experimental_rerun()

    # Download button for escalations
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Escalations CSV", data=csv, file_name="escalations.csv", mime='text/csv')

# Main app function with auto fetching every 1 min
def main():
    st.title("🚀 EscalateAI - AI Powered Escalation Management")

    # Sidebar upload and manual entry
    upload_excel()
    manual_entry()

    # Auto fetch emails every 60 sec
    if "last_fetch" not in st.session_state:
        st.session_state["last_fetch"] = 0

    now = time.time()
    if now - st.session_state["last_fetch"] > 60:
        st.session_state["last_fetch"] = now
        st.info("Fetching new emails from Gmail...")
        emails = fetch_emails()
        if emails:
            count = log_escalations(emails)
            if count > 0:
                st.success(f"{count} new escalations logged from email fetch.")
            else:
                st.info("No new escalations found in emails.")
        else:
            st.info("No new emails or failed to fetch.")

    render_kanban()

if __name__ == "__main__":
    main()
