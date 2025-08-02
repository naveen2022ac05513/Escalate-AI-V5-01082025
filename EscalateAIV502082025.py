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

# Constants / Config
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
FETCH_INTERVAL_SEC = 60  # 1 minute

# Initialize DB connection and ensure table
conn = sqlite3.connect("escalations.db", check_same_thread=False)
cursor = conn.cursor()
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

# Utility functions ------------------------------------------

def parse_email_body(msg):
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            cdisp = str(part.get("Content-Disposition"))
            if ctype == "text/plain" and "attachment" not in cdisp:
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
    return body.strip()

def rule_sentiment(text):
    keywords = ['fail', 'error', 'issue', 'urgent', 'immediately', 'critical', 'problem', 'complaint', 'escalate']
    score = sum(word in text.lower() for word in keywords)
    if score >= 2:
        return "Negative"
    else:
        return "Positive"

def determine_priority(text):
    urgent_words = ['urgent', 'immediately', 'critical', 'fail', 'escalate']
    score = sum(word in text.lower() for word in urgent_words)
    if score >= 2:
        return "High"
    else:
        return "Low"

def generate_escalation_id():
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]
    return f"SESICE-{250001 + count + 1}"

def insert_escalation(data):
    # Avoid duplicates: customer + issue snippet
    cursor.execute("SELECT COUNT(*) FROM escalations WHERE customer=? AND issue=?", (data['customer'], data['issue'][:1000]))
    if cursor.fetchone()[0] > 0:
        return False
    escalation_id = generate_escalation_id()
    cursor.execute("""
    INSERT INTO escalations (escalation_id, customer, issue, date_reported, status, sentiment, priority, action_taken, action_owner)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        escalation_id,
        data['customer'],
        data['issue'][:1000],
        data['date_reported'],
        data.get('status', 'Open'),
        data.get('sentiment', 'Positive'),
        data.get('priority', 'Low'),
        data.get('action_taken', ''),
        data.get('action_owner', '')
    ))
    conn.commit()
    return True

def fetch_emails_and_log():
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL, APP_PASSWORD)
    except Exception as e:
        st.error(f"Email login error: {e}")
        return 0

    try:
        status, _ = mail.select("INBOX")
        if status != 'OK':
            st.error("Cannot open mailbox")
            mail.logout()
            return 0

        status, messages = mail.search(None, '(UNSEEN)')
        if status != 'OK':
            mail.logout()
            return 0

        email_ids = messages[0].split()
        new_logged = 0
        for num in email_ids[-10:]:  # last 10 unseen emails only
            try:
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

                inserted = insert_escalation({
                    "customer": from_,
                    "issue": body,
                    "date_reported": date_,
                    "sentiment": sentiment,
                    "priority": priority,
                    "status": "Open"
                })
                if inserted:
                    new_logged += 1

                mail.store(num, '+FLAGS', '\\Seen')
            except imaplib.IMAP4.abort:
                st.warning("IMAP abort error (too many connections). Fetch postponed.")
                break
            except Exception as ex:
                st.error(f"Error fetching one email: {ex}")
                continue
        mail.logout()
        return new_logged
    except Exception as e:
        st.error(f"Unexpected error during fetch: {e}")
        try:
            mail.logout()
        except:
            pass
        return 0

def get_all_escalations():
    df = pd.read_sql_query("SELECT * FROM escalations ORDER BY date_reported DESC", conn)
    return df

def update_escalation_field(escalation_id, field, value):
    if field not in ['status', 'action_taken', 'action_owner']:
        return
    cursor.execute(f"UPDATE escalations SET {field}=? WHERE escalation_id=?", (value, escalation_id))
    conn.commit()

# Streamlit UI -------------------------------------------

st.set_page_config(page_title="EscalateAI - Customer Escalations", layout="wide")

st.title("🚀 EscalateAI - Customer Escalation Management")

# Sidebar: Manual Entry + Excel Upload -----------------------------------

with st.sidebar.expander("➕ Manual Escalation Entry"):
    with st.form("manual_entry_form"):
        customer = st.text_input("Customer Email or Name")
        issue = st.text_area("Issue Description")
        date_reported = st.date_input("Date Reported", datetime.date.today())
        sentiment = st.selectbox("Sentiment", ["Positive", "Negative"])
        priority = st.selectbox("Priority", ["Low", "High"])
        submitted_manual = st.form_submit_button("Add Escalation")

        if submitted_manual:
            if customer.strip() == "" or issue.strip() == "":
                st.warning("Customer and Issue are required.")
            else:
                success = insert_escalation({
                    "customer": customer,
                    "issue": issue,
                    "date_reported": date_reported.strftime("%a, %d %b %Y"),
                    "sentiment": sentiment,
                    "priority": priority,
                    "status": "Open",
                    "action_taken": "",
                    "action_owner": ""
                })
                if success:
                    st.success("Manual escalation added successfully.")
                else:
                    st.info("Duplicate escalation detected, not added.")

with st.sidebar.expander("📂 Bulk Upload from Excel"):
    uploaded_file = st.file_uploader("Upload Excel file with escalations", type=["xlsx"])
    if uploaded_file:
        try:
            df_upload = pd.read_excel(uploaded_file)
            # Expect columns: customer, issue, date_reported, sentiment, priority, status, action_taken, action_owner (some optional)
            for idx, row in df_upload.iterrows():
                insert_escalation({
                    "customer": str(row.get("customer", "")),
                    "issue": str(row.get("issue", "")),
                    "date_reported": str(row.get("date_reported", datetime.datetime.now().strftime("%a, %d %b %Y"))),
                    "sentiment": str(row.get("sentiment", "Positive")),
                    "priority": str(row.get("priority", "Low")),
                    "status": str(row.get("status", "Open")),
                    "action_taken": str(row.get("action_taken", "")),
                    "action_owner": str(row.get("action_owner", ""))
                })
            st.success("Bulk upload completed.")
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")

# Main panel: Fetch emails and Kanban -----------------------------------

col1, col2 = st.columns([3,1])

with col2:
    if st.button("🔄 Fetch New Emails from Gmail"):
        new_count = fetch_emails_and_log()
        st.success(f"Fetched and logged {new_count} new emails.")

    st.markdown("### Auto Fetch Every 1 Minute (disable or adjust in code)")
    # Auto-refresh every FETCH_INTERVAL_SEC seconds
    if 'last_fetch' not in st.session_state:
        st.session_state['last_fetch'] = 0
    now = time.time()
    if now - st.session_state['last_fetch'] > FETCH_INTERVAL_SEC:
        new_count = fetch_emails_and_log()
        if new_count > 0:
            st.success(f"Auto fetched {new_count} new emails.")
        st.session_state['last_fetch'] = now
    else:
        remaining = int(FETCH_INTERVAL_SEC - (now - st.session_state['last_fetch']))
        st.info(f"Next auto-fetch in {remaining} seconds.")

with col1:
    # Load all escalations
    df_escalations = get_all_escalations()

    # Filters
    filter_type = st.selectbox("Filter Cases", ["All", "Escalated (High Priority & Negative)"])

    if filter_type == "Escalated (High Priority & Negative)":
        df_escalations = df_escalations[(df_escalations['priority'] == "High") & (df_escalations['sentiment'] == "Negative")]

    # Status columns for Kanban
    status_columns = ["Open", "In Progress", "Resolved"]

    # Colors / Emojis for visual cues
    status_colors = {
        "Open": "🔴",
        "In Progress": "🟠",
        "Resolved": "🟢"
    }
    priority_colors = {
        "High": "🔥",
        "Low": "🧊"
    }
    sentiment_colors = {
        "Negative": "⚠️",
        "Positive": "✅"
    }

    cols = st.columns(len(status_columns))
    for idx, status in enumerate(status_columns):
        with cols[idx]:
            st.markdown(f"### {status_colors.get(status)} **{status}** ({len(df_escalations[df_escalations['status'] == status])})")
            df_status = df_escalations[df_escalations['status'] == status]
            if df_status.empty:
                st.write("No escalations")
            else:
                for i, row in df_status.iterrows():
                    header = f"{row['escalation_id']} - {sentiment_colors.get(row['sentiment'], '')} {row['sentiment']} / {priority_colors.get(row['priority'], '')} {row['priority']}"
                    with st.expander(header):
                        st.markdown(f"**Customer:** {row['customer']}")
                        st.markdown(f"**Issue:** {row['issue']}")
                        st.markdown(f"**Reported on:** {row['date_reported']}")

                        # Editable status, action_taken, action_owner
                        status_options = ["Open", "In Progress", "Resolved"]
                        new_status = st.selectbox(
                            "Update Status",
                            status_options,
                            index=status_options.index(row['status']),
                            key=f"status_{row['escalation_id']}"
                        )
                        if new_status != row['status']:
                            update_escalation_field(row['escalation_id'], "status", new_status)
                            st.experimental_rerun()

                        action_taken = st.text_area(
                            "Action Taken",
                            value=row.get('action_taken', ''),
                            key=f"action_{row['escalation_id']}"
                        )
                        if action_taken != row.get('action_taken', ''):
                            update_escalation_field(row['escalation_id'], "action_taken", action_taken)
                            st.experimental_rerun()

                        action_owner = st.text_input(
                            "Action Owner",
                            value=row.get('action_owner', ''),
                            key=f"owner_{row['escalation_id']}"
                        )
                        if action_owner != row.get('action_owner', ''):
                            update_escalation_field(row['escalation_id'], "action_owner", action_owner)
                            st.experimental_rerun()

    # Download escalations as Excel
    def convert_df_to_excel(df):
        return df.to_excel(index=False)

    if st.button("📥 Download Escalations as Excel"):
        excel_bytes = convert_df_to_excel(df_escalations)
        st.download_button(
            label="Download Excel",
            data=excel_bytes,
            file_name=f"escalations_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
