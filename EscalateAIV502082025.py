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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

# Email config from .env
EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")

# Initialize sentiment analyzer once
analyzer = SentimentIntensityAnalyzer()

# Negative keyword groups for escalation
NEGATIVE_KEYWORDS = {
    "technical": ['fail', 'break', 'crash', 'defect', 'fault', 'degrade', 'damage', 'trip', 'malfunction', 'blank', 'shutdown', 'discharge'],
    "customer": ['dissatisfy', 'frustrate', 'complain', 'reject', 'delay', 'ignore', 'escalate', 'displease', 'noncompliance', 'neglect'],
    "support": ['wait', 'pending', 'slow', 'incomplete', 'miss', 'omit', 'unresolved', 'shortage', 'no response'],
    "hazard": ['fire', 'burn', 'flashover', 'arc', 'explode', 'unsafe', 'leak', 'corrode', 'alarm', 'incident'],
    "business": ['impact', 'loss', 'risk', 'downtime', 'interrupt', 'cancel', 'terminate', 'penalty'],
}

ALL_NEGATIVE_WORDS = [w for words in NEGATIVE_KEYWORDS.values() for w in words]

# DB Setup
conn = sqlite3.connect("escalateai.db", check_same_thread=False)
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
    escalation_flag INTEGER,
    action_taken TEXT,
    action_owner TEXT
)
""")
conn.commit()

def clean_text(text):
    return re.sub(r'\s+', ' ', str(text)).strip()

def analyze_sentiment_and_escalation(text):
    text = clean_text(text).lower()
    vs = analyzer.polarity_scores(text)
    neg_score = vs['neg']
    # Check keyword presence count
    keyword_count = sum(text.count(word) for word in ALL_NEGATIVE_WORDS)
    
    # Determine sentiment
    sentiment = "Negative" if neg_score > 0.1 or keyword_count > 0 else "Positive"
    # Priority heuristic
    priority = "High" if keyword_count >= 2 or neg_score > 0.3 else "Low"
    # Escalation flag
    escalation_flag = 1 if keyword_count > 0 else 0
    
    return sentiment, priority, escalation_flag

def fetch_emails():
    if not EMAIL or not APP_PASSWORD:
        st.warning("Email credentials missing in .env")
        return []
    try:
        mail = imaplib.IMAP4_SSL(EMAIL_SERVER)
        mail.login(EMAIL, APP_PASSWORD)
        mail.select("inbox")
        status, data = mail.search(None, '(UNSEEN)')
        if status != 'OK':
            return []
        email_ids = data[0].split()
        fetched = []
        for eid in email_ids[-10:]:  # Last 10 unseen emails
            res, msg_data = mail.fetch(eid, "(RFC822)")
            if res != 'OK':
                continue
            msg = email.message_from_bytes(msg_data[0][1])
            subject, encoding = decode_header(msg.get("Subject", ""))[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or "utf-8", errors="ignore")
            from_ = msg.get("From", "")
            date = msg.get("Date", "")
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
            fetched.append({
                "from": from_,
                "subject": subject,
                "body": body,
                "date": date
            })
            mail.store(eid, '+FLAGS', '\\Seen')
        mail.logout()
        return fetched
    except Exception as e:
        st.error(f"Error fetching emails: {e}")
        return []

def save_complaints_to_csv(df):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"escalateai_complaints_{timestamp}.csv"
    df.to_csv(filename, index=False)
    return filename

def add_to_db(complaints):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM escalations")
    base_count = cursor.fetchone()[0]
    added = 0
    for comp in complaints:
        cust = comp["from"]
        issue = comp["body"]
        date = comp["date"]
        # Avoid duplicates: same customer & issue substring
        cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue LIKE ?", (cust, issue[:200] + '%'))
        if cursor.fetchone():
            continue
        base_count += 1
        esc_id = f"SESICE-{250000 + base_count}"
        sentiment, priority, esc_flag = analyze_sentiment_and_escalation(issue)
        cursor.execute("""
            INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, action_taken, action_owner)
            VALUES (?, ?, ?, ?, 'Open', ?, ?, ?, '', '')
        """, (esc_id, cust, issue[:500], date, sentiment, priority, esc_flag))
        added += 1
    conn.commit()
    return added

def load_escalations(filter_flag=None):
    df = pd.read_sql("SELECT * FROM escalations", conn)
    if filter_flag == "Escalated":
        df = df[df['escalation_flag'] == 1]
    return df

def main():
    st.set_page_config(page_title="🚀 EscalateAI - Escalations & Complaints Kanban Board", layout="wide")
    st.title("🚀 EscalateAI - Escalations & Complaints Kanban Board")
    
    # Sidebar: Upload Excel/CSV
    st.sidebar.header("Upload complaints (Excel/CSV)")
    uploaded_file = st.sidebar.file_uploader("", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_uploaded = pd.read_csv(uploaded_file)
            else:
                df_uploaded = pd.read_excel(uploaded_file)
            st.sidebar.success("File uploaded. Analyzing...")
            complaints = []
            for _, row in df_uploaded.iterrows():
                issue_text = row.get('issue') or row.get('complaint') or ""
                customer_email = row.get('customer_email') or row.get('customer') or "unknown"
                complaint_date = str(row.get('date') or datetime.datetime.now())
                complaints.append({
                    "from": customer_email,
                    "subject": row.get('subject') or "",
                    "body": issue_text,
                    "date": complaint_date
                })
            added = add_to_db(complaints)
            st.sidebar.success(f"Added {added} complaints to the database.")
        except Exception as e:
            st.sidebar.error(f"Failed to process file: {e}")

    # Sidebar: Manual entry
    st.sidebar.header("Manual Complaint Entry")
    manual_customer = st.sidebar.text_input("Customer Email/Name")
    manual_issue = st.sidebar.text_area("Issue / Complaint")
    manual_date = st.sidebar.date_input("Date", datetime.date.today())
    if st.sidebar.button("Add Manual Complaint"):
        if manual_customer and manual_issue:
            complaint = [{
                "from": manual_customer,
                "subject": "",
                "body": manual_issue,
                "date": str(manual_date)
            }]
            added = add_to_db(complaint)
            if added > 0:
                st.sidebar.success("Complaint added successfully!")
            else:
                st.sidebar.info("Complaint already exists or no new data added.")
        else:
            st.sidebar.warning("Please provide both Customer and Issue details.")

    # Sidebar: Fetch emails button
    st.sidebar.header("Fetch New Emails from Gmail")
    if st.sidebar.button("Fetch Emails Now"):
        emails = fetch_emails()
        if emails:
            added = add_to_db(emails)
            st.sidebar.success(f"Fetched and added {added} new complaints from email.")
        else:
            st.sidebar.info("No new emails found or fetch failed.")

    # Sidebar: Download all complaints CSV
    st.sidebar.header("Download Complaints CSV")
    all_complaints = pd.read_sql("SELECT * FROM escalations", conn)
    csv_data = all_complaints.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download Complaints CSV", csv_data, "escalateai_complaints.csv", "text/csv")

    # Filter on main page
    filter_option = st.selectbox("Filter cases", options=["All", "Escalated"])
    df = load_escalations(filter_option)
    
    # Count per status
    counts = df['status'].value_counts()
    open_count = counts.get('Open', 0)
    inprog_count = counts.get('In Progress', 0)
    resolved_count = counts.get('Resolved', 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader(f"Open ({open_count})")
        for _, row in df[df['status'] == 'Open'].iterrows():
            display_kanban_card(row)
    with col2:
        st.subheader(f"In Progress ({inprog_count})")
        for _, row in df[df['status'] == 'In Progress'].iterrows():
            display_kanban_card(row)
    with col3:
        st.subheader(f"Resolved ({resolved_count})")
        for _, row in df[df['status'] == 'Resolved'].iterrows():
            display_kanban_card(row)

def display_kanban_card(row):
    esc_id = row['escalation_id']
    sentiment = row['sentiment']
    priority = row['priority']
    sentiment_color = {"Positive": "green", "Negative": "red"}.get(sentiment, "black")
    priority_color = {"High": "red", "Low": "green"}.get(priority, "black")
    with st.expander(f"{esc_id} - [{sentiment_color}●] {sentiment} / [{priority_color}■] {priority}"):
        st.markdown(f"**Customer:** {row['customer']}")
        st.markdown(f"**Issue:** {row['issue']}")
        st.markdown(f"**Date:** {row['date']}")
        # Editable status
        new_status = st.selectbox("Update Status", ["Open", "In Progress", "Resolved"], index=["Open","In Progress","Resolved"].index(row['status']), key=f"{esc_id}_status")
        new_action_taken = st.text_area("Action Taken", value=row['action_taken'], key=f"{esc_id}_action")
        new_action_owner = st.text_input("Action Owner", value=row['action_owner'], key=f"{esc_id}_owner")
        if st.button("Save Updates", key=f"save_{esc_id}"):
            cursor.execute("""
                UPDATE escalations SET status=?, action_taken=?, action_owner=?
                WHERE escalation_id=?
            """, (new_status, new_action_taken, new_action_owner, esc_id))
            conn.commit()
            st.success("Updated successfully!")

if __name__ == "__main__":
    main()
