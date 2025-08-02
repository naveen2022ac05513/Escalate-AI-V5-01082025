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

# Load environment variables
load_dotenv()

EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")

DB_FILE = "escalations.db"

# Keywords for NLP heuristic
URGENCY_KEYWORDS = ['urgent', 'immediately', 'critical', 'fail', 'escalate', 'issue', 'problem', 'complaint']

# Initialize DB and table
def init_db():
    conn = sqlite3.connect(DB_FILE)
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
    conn.close()

# Generate next escalation ID
def next_escalation_id():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0] + 1
    conn.close()
    return f"SESICE-{250000 + count}"

# NLP heuristic: sentiment and priority
def analyze_text(text):
    score = sum(1 for kw in URGENCY_KEYWORDS if kw in text.lower())
    sentiment = "Negative" if score > 0 else "Positive"
    priority = "High" if score >= 2 else "Low"
    return sentiment, priority

# Fetch unseen emails from Gmail
def fetch_emails():
    if not EMAIL or not APP_PASSWORD:
        st.warning("Gmail credentials not set in .env.")
        return []
    try:
        mail = imaplib.IMAP4_SSL(EMAIL_SERVER)
        mail.login(EMAIL, APP_PASSWORD)
        mail.select("inbox")
        status, data = mail.search(None, 'UNSEEN')
        if status != 'OK':
            st.info("No new emails found.")
            mail.logout()
            return []

        email_ids = data[0].split()
        fetched = []

        # Limit to last 10 to avoid overload
        for num in email_ids[-10:]:
            res, msg_data = mail.fetch(num, '(RFC822)')
            if res != 'OK':
                continue
            msg = email.message_from_bytes(msg_data[0][1])

            # Decode subject
            subject, encoding = decode_header(msg["Subject"])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or 'utf-8', errors='ignore')

            from_ = msg.get("From", "Unknown sender")
            date = msg.get("Date", "")
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain" and 'attachment' not in str(part.get("Content-Disposition")):
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

            fetched.append({
                "customer": from_,
                "issue": body,
                "subject": subject,
                "date": date
            })

            # Mark email as seen
            mail.store(num, '+FLAGS', '\\Seen')

        mail.logout()
        return fetched
    except Exception as e:
        st.error(f"Failed to fetch emails: {e}")
        return []

# Save escalations to DB if not duplicate
def save_escalations(escalations):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    new_count = 0
    for esc in escalations:
        # Check duplicate: same customer + issue text (first 500 chars)
        cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue LIKE ?", (esc['customer'], esc['issue'][:500] + '%'))
        if cursor.fetchone():
            continue
        sentiment, priority = analyze_text(esc['issue'])
        esc_id = next_escalation_id()
        cursor.execute('''
            INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, action_taken, action_owner)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (esc_id, esc['customer'], esc['issue'][:1000], esc['date'], "Open", sentiment, priority, "", ""))
        new_count += 1
    conn.commit()
    conn.close()
    return new_count

# Load escalations with optional filter
def load_escalations(filter_status=None, filter_escalated=None):
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT * FROM escalations"
    params = []
    clauses = []
    if filter_status and filter_status != "All":
        clauses.append("status = ?")
        params.append(filter_status)
    if filter_escalated == "Yes":
        clauses.append("sentiment = 'Negative'")
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

# Update escalation fields in DB
def update_escalation(esc_id, status, action_taken, action_owner):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE escalations SET status=?, action_taken=?, action_owner=? WHERE escalation_id=?
    ''', (status, action_taken, action_owner, esc_id))
    conn.commit()
    conn.close()

# Sidebar UI for manual entry and bulk upload
def sidebar_ui():
    st.sidebar.header("Add New Escalation Manually")
    with st.sidebar.form("manual_entry"):
        customer = st.text_input("Customer Email / Name", key="man_customer")
        issue = st.text_area("Issue Description", key="man_issue")
        date = st.date_input("Date", datetime.date.today(), key="man_date")
        submitted = st.form_submit_button("Add Escalation")
        if submitted:
            if not customer or not issue:
                st.sidebar.error("Customer and Issue are required.")
            else:
                esc = {
                    "customer": customer,
                    "issue": issue,
                    "date": date.strftime("%a, %d %b %Y")
                }
                count = save_escalations([esc])
                if count > 0:
                    st.sidebar.success(f"Escalation added successfully.")
                else:
                    st.sidebar.warning("Duplicate escalation not added.")

    st.sidebar.header("Bulk Upload Escalations (Excel)")
    uploaded_file = st.sidebar.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            required_cols = {"customer", "issue", "date"}
            if not required_cols.issubset(set(df.columns.str.lower())):
                st.sidebar.error(f"Excel must contain columns: {required_cols}")
            else:
                # Normalize column names
                df.columns = [c.lower() for c in df.columns]
                escs = df.to_dict("records")
                count = save_escalations(escs)
                st.sidebar.success(f"{count} escalations added from upload.")
        except Exception as e:
            st.sidebar.error(f"Failed to process upload: {e}")

# Render Kanban board with colors and filters
def render_kanban():
    st.header("🚀 EscalateAI - Escalations Kanban Board")

    filter_escalated = st.selectbox("Show only escalated?", options=["All", "Yes"], index=0)
    filter_status = st.selectbox("Filter by Status:", options=["All", "Open", "In Progress", "Resolved"], index=0)

    df = load_escalations(filter_status if filter_status != "All" else None,
                          filter_escalated if filter_escalated != "All" else None)

    if df.empty:
        st.info("No escalations found matching filters.")
        return

    # Define status columns for Kanban
    status_columns = ["Open", "In Progress", "Resolved"]
    columns = st.columns(len(status_columns))

    # Colors for sentiment and priority
    sentiment_colors = {"Negative": "🟥", "Positive": "🟩"}
    priority_colors = {"High": "🔴", "Low": "🟢"}

    # Prepare dict of lists for each status
    kanban_data = {status: df[df['status'] == status] for status in status_columns}

    for idx, status in enumerate(status_columns):
        with columns[idx]:
            st.subheader(f"{status} ({len(kanban_data[status])})")
            for i, row in kanban_data[status].iterrows():
                esc_id = row['escalation_id']
                # Use emojis and color-coded labels in header
                header = f"{esc_id} - {sentiment_colors.get(row['sentiment'], '')} {row['sentiment']} / {priority_colors.get(row['priority'], '')} {row['priority']}"
                with st.expander(header, expanded=False):
                    st.markdown(f"**Customer:** {row['customer']}")
                    st.markdown(f"**Issue:** {row['issue']}")
                    st.markdown(f"**Date:** {row['date']}")

                    new_status = st.selectbox(
                        "Update Status",
                        options=status_columns,
                        index=status_columns.index(row['status']),
                        key=f"{esc_id}_status_{i}"
                    )
                    new_action_taken = st.text_area(
                        "Action Taken",
                        value=row['action_taken'] if row['action_taken'] else "",
                        key=f"{esc_id}_action_{i}"
                    )
                    new_action_owner = st.text_input(
                        "Action Owner",
                        value=row['action_owner'] if row['action_owner'] else "",
                        key=f"{esc_id}_owner_{i}"
                    )

                    # Update DB if changed
                    if (new_status != row['status'] or
                        new_action_taken != (row['action_taken'] or "") or
                        new_action_owner != (row['action_owner'] or "")):
                        update_escalation(esc_id, new_status, new_action_taken, new_action_owner)
                        st.success("Updated escalation.")

    # Download button for current filtered escalations
    if not df.empty:
        to_download = df.drop(columns=["action_taken", "action_owner"], errors='ignore')
        to_download = to_download[['escalation_id', 'customer', 'issue', 'date', 'status', 'sentiment', 'priority']]
        csv_data = to_download.to_csv(index=False)
        st.download_button("📥 Download Escalations CSV", csv_data, file_name="escalations.csv", mime="text/csv")

# Main app
def main():
    st.title("EscalateAI - AI Powered Escalation Management")

    # Initialize DB
    init_db()

    # Sidebar manual entry + upload
    sidebar_ui()

    # Button to fetch emails manually (can extend to schedule later)
    if st.button("🔄 Fetch New Emails from Gmail"):
        fetched_emails = fetch_emails()
        if fetched_emails:
            count = save_escalations(fetched_emails)
            st.success(f"Fetched and saved {count} new escalations from email.")
        else:
            st.info("No new emails fetched.")

    render_kanban()

if __name__ == "__main__":
    main()
