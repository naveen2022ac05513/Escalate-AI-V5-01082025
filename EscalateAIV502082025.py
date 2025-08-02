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
from textblob import TextBlob
import time

# Load env vars
load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")

# DB setup
conn = sqlite3.connect("escalateai.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''
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
''')
conn.commit()

# Sentiment & priority helpers
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return "Positive"
    elif polarity < -0.2:
        return "Negative"
    else:
        return "Neutral"

def determine_priority(text):
    urgent_words = ['urgent', 'immediately', 'critical', 'fail', 'escalate', 'problem', 'complaint', 'down', 'error', 'fail']
    score = sum([text.lower().count(word) for word in urgent_words])
    if score >= 2:
        return "High"
    elif score == 1:
        return "Medium"
    else:
        return "Low"

# Generate unique escalation id
def generate_escalation_id():
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0] + 250001
    return f"SESICE-{count}"

# Fetch unseen emails from Gmail
def fetch_emails():
    try:
        mail = imaplib.IMAP4_SSL(EMAIL_SERVER)
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select("inbox")
        status, messages = mail.search(None, '(UNSEEN)')
        if status != "OK":
            return []
        email_ids = messages[0].split()
        fetched = []
        # Fetch last 10 unseen
        for num in email_ids[-10:]:
            _, msg_data = mail.fetch(num, "(RFC822)")
            msg = email.message_from_bytes(msg_data[0][1])
            # Decode subject
            subject, encoding = decode_header(msg["Subject"])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or "utf-8", errors='ignore')
            from_ = msg.get("From")
            date = msg.get("Date")
            # Get body text
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    ctype = part.get_content_type()
                    cdisp = str(part.get("Content-Disposition"))
                    if ctype == "text/plain" and "attachment" not in cdisp:
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
            # Mark as seen
            mail.store(num, '+FLAGS', '\\Seen')
            fetched.append({
                "from": from_,
                "subject": subject,
                "body": body,
                "date": date
            })
        mail.logout()
        return fetched
    except Exception as e:
        st.error(f"Error fetching emails: {e}")
        return []

# Insert escalations avoiding duplicates
def log_escalations(fetched_emails):
    new_logs = 0
    for email_data in fetched_emails:
        customer = email_data['from']
        issue = email_data['body'][:500]  # limit size
        date = email_data['date']
        cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (customer, issue))
        if cursor.fetchone():
            continue
        sentiment = analyze_sentiment(issue)
        priority = determine_priority(issue)
        escalation_id = generate_escalation_id()
        cursor.execute('''
            INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, action_taken, action_owner)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (escalation_id, customer, issue, date, "Open", sentiment, priority, "", ""))
        conn.commit()
        new_logs += 1
    if new_logs > 0:
        st.success(f"{new_logs} new escalations logged.")
    else:
        st.info("No new escalations to log.")

# Read escalations from DB
def get_escalations(filter_status=None, filter_priority=None):
    query = "SELECT * FROM escalations"
    params = []
    if filter_status and filter_status != "All":
        query += " WHERE status=?"
        params.append(filter_status)
    if filter_priority and filter_priority != "All":
        if "WHERE" in query:
            query += " AND priority=?"
        else:
            query += " WHERE priority=?"
        params.append(filter_priority)
    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])
    return df

# Update escalation fields
def update_escalation(esc_id, status=None, action_taken=None, action_owner=None):
    updates = []
    params = []
    if status is not None:
        updates.append("status=?")
        params.append(status)
    if action_taken is not None:
        updates.append("action_taken=?")
        params.append(action_taken)
    if action_owner is not None:
        updates.append("action_owner=?")
        params.append(action_owner)
    if not updates:
        return
    params.append(esc_id)
    query = f"UPDATE escalations SET {', '.join(updates)} WHERE escalation_id=?"
    cursor.execute(query, tuple(params))
    conn.commit()

# Save escalations to Excel for download
def save_escalations_to_excel(df):
    filename = f"escalations_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    df.to_excel(filename, index=False)
    return filename

# Analyze uploaded Excel for escalations
def analyze_excel(df):
    # Expect columns: customer, issue, date (or similar)
    required_cols = ['customer', 'issue']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Uploaded file must have columns: {required_cols}")
        return 0
    new_logs = 0
    for _, row in df.iterrows():
        customer = str(row['customer'])
        issue = str(row['issue'])[:500]
        date = str(row['date']) if 'date' in df.columns else datetime.datetime.now().strftime("%a, %d %b %Y %H:%M:%S")
        cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (customer, issue))
        if cursor.fetchone():
            continue
        sentiment = analyze_sentiment(issue)
        priority = determine_priority(issue)
        escalation_id = generate_escalation_id()
        cursor.execute('''
            INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, action_taken, action_owner)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (escalation_id, customer, issue, date, "Open", sentiment, priority, "", ""))
        conn.commit()
        new_logs += 1
    if new_logs > 0:
        st.success(f"{new_logs} escalations added from Excel.")
    else:
        st.info("No new escalations found in Excel.")
    return new_logs

# Manual entry
def manual_entry():
    st.sidebar.subheader("Add Manual Escalation")
    customer = st.sidebar.text_input("Customer Email/Name")
    issue = st.sidebar.text_area("Issue Description")
    date = st.sidebar.text_input("Date (leave blank for now)", datetime.datetime.now().strftime("%a, %d %b %Y %H:%M:%S"))
    if st.sidebar.button("Add Escalation"):
        if not customer or not issue:
            st.sidebar.error("Please enter Customer and Issue.")
            return
        sentiment = analyze_sentiment(issue)
        priority = determine_priority(issue)
        escalation_id = generate_escalation_id()
        cursor.execute('''
            INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, action_taken, action_owner)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (escalation_id, customer, issue, date, "Open", sentiment, priority, "", ""))
        conn.commit()
        st.sidebar.success(f"Escalation {escalation_id} added manually.")

# Main Kanban colors
sentiment_colors = {
    "Positive": "🟢",
    "Neutral": "🟡",
    "Negative": "🔴"
}
priority_colors = {
    "High": "🔥",
    "Medium": "⚠️",
    "Low": "ℹ️"
}
status_colors = {
    "Open": "🔵",
    "In Progress": "🟠",
    "Resolved": "✅"
}

def render_kanban():
    st.header("🚀 Escalations Kanban Board")

    filter_status = st.selectbox("Filter by Status", options=["All", "Open", "In Progress", "Resolved"])
    filter_priority = st.selectbox("Filter by Priority", options=["All", "High", "Medium", "Low"])
    filter_escalated = st.selectbox("Filter Escalated or All", options=["All", "Escalated Only"])

    df = get_escalations(filter_status if filter_status != "All" else None,
                         filter_priority if filter_priority != "All" else None)
    if df.empty:
        st.info("No escalations to display.")
        return

    if filter_escalated == "Escalated Only":
        df = df[df['priority'].isin(['High','Medium'])]

    # Columns for Kanban
    kanban_cols = st.columns(3)
    statuses = ["Open", "In Progress", "Resolved"]

    for i, status in enumerate(statuses):
        with kanban_cols[i]:
            st.markdown(f"### {status_colors[status]} {status} ({len(df[df['status'] == status])})")
            filtered_df = df[df['status'] == status]
            for idx, row in filtered_df.iterrows():
                esc_id = row['escalation_id']
                header = f"{esc_id} - {sentiment_colors.get(row['sentiment'], '')} {row['sentiment']} / {priority_colors.get(row['priority'], '')} {row['priority']}"
                with st.expander(header, expanded=False):
                    st.write(f"**Customer:** {row['customer']}")
                    st.write(f"**Date:** {row['date']}")
                    st.write(f"**Issue:** {row['issue']}")
                    new_status = st.selectbox("Update Status", options=statuses, index=statuses.index(row['status']), key=f"status_{esc_id}")
                    new_action_taken = st.text_area("Action Taken", value=row['action_taken'], key=f"action_{esc_id}")
                    new_action_owner = st.text_input("Action Owner", value=row['action_owner'], key=f"owner_{esc_id}")

                    if (new_status != row['status'] or new_action_taken != row['action_taken'] or new_action_owner != row['action_owner']):
                        update_escalation(esc_id, new_status, new_action_taken, new_action_owner)
                        st.success("Updated escalation.")

    # Download Excel
    if st.button("📥 Download All Escalations as Excel"):
        filename = save_escalations_to_excel(df)
        with open(filename, "rb") as f:
            st.download_button("Download Excel File", f, file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def upload_excel():
    st.sidebar.subheader("Upload Escalations Excel")
    uploaded_file = st.sidebar.file_uploader("Upload Excel file with columns: customer, issue, [date]", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        analyze_excel(df)

def periodic_email_fetch():
    if st.button("🔄 Fetch New Emails Now"):
        fetched = fetch_emails()
        if fetched:
            log_escalations(fetched)

    # Automatic fetch every 60 seconds with info
    if "last_fetch" not in st.session_state:
        st.session_state.last_fetch = 0

    now = time.time()
    if now - st.session_state.last_fetch > 60:
        fetched = fetch_emails()
        if fetched:
            log_escalations(fetched)
        st.session_state.last_fetch = now
        st.info(f"Emails auto-fetched at {datetime.datetime.now().strftime('%H:%M:%S')}")

def main():
    st.set_page_config(page_title="EscalateAI", layout="wide")
    st.title("🚀 EscalateAI - Customer Escalation Management")

    manual_entry()
    upload_excel()
    periodic_email_fetch()
    render_kanban()

if __name__ == "__main__":
    main()
