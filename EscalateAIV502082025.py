import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load environment variables
load_dotenv()

# Email credentials from .env
EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Negative keywords grouped by categories
negative_keywords = {
    "technical": ['fail', 'break', 'crash', 'defect', 'fault', 'degrade', 'damage', 'trip',
                  'malfunction', 'blank', 'shutdown', 'discharge'],
    "customer": ['dissatisfy', 'frustrate', 'complain', 'reject', 'delay', 'ignore', 'escalate',
                 'displease', 'noncompliance', 'neglect'],
    "support": ['wait', 'pending', 'slow', 'incomplete', 'miss', 'omit', 'unresolved', 'shortage', 'no response'],
    "hazard": ['fire', 'burn', 'flashover', 'arc', 'explode', 'unsafe', 'leak', 'corrode', 'alarm', 'incident'],
    "business": ['impact', 'loss', 'risk', 'downtime', 'interrupt', 'cancel', 'terminate', 'penalty'],
}
# Flatten list of negative words
negative_words_flat = [word for sublist in negative_keywords.values() for word in sublist]

DB_NAME = "escalateai.db"
CSV_FILENAME = "complaints_log.csv"

def create_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
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

def fetch_emails():
    if not EMAIL or not APP_PASSWORD:
        st.error("Missing EMAIL_USER or EMAIL_PASS environment variables.")
        return []
    try:
        mail = imaplib.IMAP4_SSL(EMAIL_SERVER)
        mail.login(EMAIL, APP_PASSWORD)
        mail.select("inbox")
        result, data = mail.search(None, '(UNSEEN)')
        if result != "OK":
            st.info("No new emails found.")
            return []
        email_ids = data[0].split()
        emails = []
        for num in email_ids[-10:]:  # fetch last 10 unseen emails max
            res, msg_data = mail.fetch(num, '(RFC822)')
            if res != 'OK':
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
                    ctype = part.get_content_type()
                    cdisp = str(part.get("Content-Disposition"))
                    if ctype == "text/plain" and "attachment" not in cdisp:
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

            emails.append({"from": from_, "subject": subject, "body": body, "date": date})
            mail.store(num, '+FLAGS', '\\Seen')
        mail.logout()
        return emails
    except Exception as e:
        st.error(f"Failed to fetch emails: {e}")
        return []

def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    # Basic sentiment classification
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # Check negative keywords presence to adjust priority
    negative_hits = sum(word in text.lower() for word in negative_words_flat)
    priority = "High" if negative_hits >= 2 or sentiment == "Negative" else "Low"
    return sentiment, priority

def save_to_db(entries):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Get last escalation id number
    c.execute("SELECT escalation_id FROM escalations ORDER BY escalation_id DESC LIMIT 1")
    last_id = c.fetchone()
    last_num = int(last_id[0].split("-")[1]) if last_id else 250000

    for entry in entries:
        # Check for duplicates based on customer and issue substring
        c.execute("SELECT COUNT(*) FROM escalations WHERE customer=? AND issue=?", (entry['from'], entry['body'][:500]))
        if c.fetchone()[0] > 0:
            continue

        last_num += 1
        escalation_id = f"SESICE-{last_num}"
        sentiment, priority = analyze_sentiment(entry['body'])
        c.execute('''INSERT INTO escalations 
            (escalation_id, customer, issue, date, status, sentiment, priority, action_taken, action_owner)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (escalation_id, entry['from'], entry['body'][:500], entry['date'], "Open", sentiment, priority, "", "")
        )
    conn.commit()
    conn.close()

def load_escalations():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    conn.close()
    return df

def update_escalation_status(esc_id, status, action_taken, action_owner):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        UPDATE escalations SET status=?, action_taken=?, action_owner=? WHERE escalation_id=?
    ''', (status, action_taken, action_owner, esc_id))
    conn.commit()
    conn.close()

def download_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def main():
    st.set_page_config(page_title="🚀 EscalateAI - Escalations & Complaints Kanban Board", layout="wide")
    st.title("🚀 EscalateAI - Escalations & Complaints Kanban Board")

    create_db()

    # Sidebar for upload and manual entry
    st.sidebar.header("Options")

    # Upload complaints Excel file
    uploaded_file = st.sidebar.file_uploader("Upload complaints Excel file", type=["xlsx", "csv"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_uploaded = pd.read_csv(uploaded_file)
            else:
                df_uploaded = pd.read_excel(uploaded_file)
            # Analyze uploaded complaints and add to DB as escalations if priority high
            entries = []
            for idx, row in df_uploaded.iterrows():
                body_text = str(row.get('issue', row.get('body', '')))
                from_email = str(row.get('customer', row.get('from', 'Unknown')))
                date = str(row.get('date', datetime.datetime.now()))
                entries.append({'from': from_email, 'body': body_text, 'date': date})
            save_to_db(entries)
            st.sidebar.success("Uploaded complaints analyzed and added.")
        except Exception as e:
            st.sidebar.error(f"Failed to process uploaded file: {e}")

    # Manual escalation entry form
    with st.sidebar.expander("Manual Escalation Entry"):
        manual_customer = st.text_input("Customer Email")
        manual_issue = st.text_area("Issue/Complaint")
        manual_date = st.date_input("Date", datetime.date.today())
        if st.button("Add Escalation"):
            if manual_customer and manual_issue:
                entries = [{'from': manual_customer, 'body': manual_issue, 'date': manual_date.strftime("%a, %d %b %Y %H:%M:%S")}]
                save_to_db(entries)
                st.sidebar.success("Manual escalation added.")
            else:
                st.sidebar.error("Please fill Customer Email and Issue.")

    # Button to fetch and analyze new emails (calls Gmail, parse + analyze + save)
    if st.sidebar.button("Fetch & Analyze New Emails"):
        new_emails = fetch_emails()
        if new_emails:
            save_to_db(new_emails)
            st.sidebar.success(f"Fetched and analyzed {len(new_emails)} new emails.")
        else:
            st.sidebar.info("No new emails or error fetching.")

    # Load escalations to display
    df = load_escalations()

    # Download button for escalations data CSV
    st.sidebar.download_button("Download Escalations CSV", data=download_csv(df), file_name="escalations.csv")

    # Filters for Kanban board
    filter_status = st.sidebar.selectbox("Filter by Status", options=["All", "Open", "In Progress", "Resolved"])
    filter_priority = st.sidebar.selectbox("Filter by Priority", options=["All", "High", "Low"])

    # Filter DataFrame accordingly
    df_filtered = df.copy()
    if filter_status != "All":
        df_filtered = df_filtered[df_filtered['status'] == filter_status]
    if filter_priority != "All":
        df_filtered = df_filtered[df_filtered['priority'] == filter_priority]

    # Kanban Board Header
    st.header("📊 Complaints & Escalations Kanban Board")

    status_columns = ["Open", "In Progress", "Resolved"]
    sentiment_colors = {"Positive": "🟢", "Neutral": "🟡", "Negative": "🔴"}
    priority_colors = {"High": "🔥", "Low": "❄️"}

    cols = st.columns(len(status_columns))
    for idx, status in enumerate(status_columns):
        with cols[idx]:
            st.subheader(status)
            subset = df_filtered[df_filtered['status'] == status]
            if subset.empty:
                st.write("No escalations")
            else:
                for _, row in subset.iterrows():
                    esc_id = row['escalation_id']
                    sentiment = sentiment_colors.get(row['sentiment'], '')
                    priority = priority_colors.get(row['priority'], '')
                    header = f"{esc_id} - {sentiment} {row['sentiment']} / {priority} {row['priority']}"

                    with st.expander(header, expanded=False):
                        st.markdown(f"**Customer:** {row['customer']}")
                        st.markdown(f"**Issue:** {row['issue']}")
                        st.markdown(f"**Date:** {row['date']}")

                        # Status update
                        new_status = st.selectbox("Update Status", status_columns,
                                                  index=status_columns.index(row['status']),
                                                  key=f"status_{esc_id}")

                        # Action taken
                        new_action_taken = st.text_area("Action Taken",
                                                       value=row.get('action_taken', ''),
                                                       key=f"action_{esc_id}")

                        # Action owner
                        new_action_owner = st.text_input("Action Owner",
                                                        value=row.get('action_owner', ''),
                                                        key=f"owner_{esc_id}")

                        # Update DB on any change
                        if (new_status != row['status'] or
                            new_action_taken != row.get('action_taken', '') or
                            new_action_owner != row.get('action_owner', '')):
                            update_escalation_status(esc_id, new_status, new_action_taken, new_action_owner)
                            st.success(f"Escalation {esc_id} updated.")

if __name__ == "__main__":
    main()
