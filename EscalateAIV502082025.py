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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load environment variables from .env file
load_dotenv()

EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
EMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")

DB_PATH = "escalateai.db"
COMPLAINTS_CSV = "email_complaints.csv"

# VADER sentiment analyzer init
analyzer = SentimentIntensityAnalyzer()

# Negative keywords for escalation detection
NEGATIVE_KEYWORDS = {
    "technical_failures": ["fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip", "malfunction", "blank", "shutdown", "discharge"],
    "customer_dissatisfaction": ["dissatisfy", "frustrate", "complain", "reject", "delay", "ignore", "escalate", "displease", "noncompliance", "neglect"],
    "support_gaps": ["wait", "pending", "slow", "incomplete", "miss", "omit", "unresolved", "shortage", "no response"],
    "hazardous_conditions": ["fire", "burn", "flashover", "arc", "explode", "unsafe", "leak", "corrode", "alarm", "incident"],
    "business_risk": ["impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate", "penalty"]
}
ALL_NEGATIVE_WORDS = [word for sublist in NEGATIVE_KEYWORDS.values() for word in sublist]

# Initialize DB and table
def init_db():
    conn = sqlite3.connect(DB_PATH)
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
            escalation_flag INTEGER DEFAULT 0,
            action_taken TEXT DEFAULT '',
            action_owner TEXT DEFAULT ''
        )
    ''')
    conn.commit()
    conn.close()

def ensure_escalation_flag_column():
    # Add escalation_flag column if missing
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(escalations)")
    cols = [col[1] for col in cursor.fetchall()]
    if "escalation_flag" not in cols:
        cursor.execute("ALTER TABLE escalations ADD COLUMN escalation_flag INTEGER DEFAULT 0")
        conn.commit()
    conn.close()

def generate_escalation_id():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]
    conn.close()
    return f"SESICE-{250001 + count}"

def save_complaint_to_csv(df_new):
    if os.path.exists(COMPLAINTS_CSV):
        df_existing = pd.read_csv(COMPLAINTS_CSV)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates(subset=['from', 'subject', 'body', 'date'])
    else:
        df_combined = df_new
    df_combined.to_csv(COMPLAINTS_CSV, index=False)

def fetch_emails():
    if not EMAIL or not APP_PASSWORD:
        st.warning("Gmail credentials missing in environment variables.")
        return []

    try:
        mail = imaplib.IMAP4_SSL(EMAIL_SERVER)
        mail.login(EMAIL, APP_PASSWORD)
    except Exception as e:
        st.error(f"Gmail login failed: {e}")
        return []

    status, messages = mail.select("inbox")
    if status != 'OK':
        st.error("Failed to select inbox")
        return []

    # Search for unseen emails
    result, data = mail.search(None, '(UNSEEN)')
    if result != 'OK':
        st.info("No unread emails found.")
        return []

    email_ids = data[0].split()
    fetched_emails = []

    # Limit to last 10 for performance
    for num in email_ids[-10:]:
        result, msg_data = mail.fetch(num, '(RFC822)')
        if result != 'OK':
            continue

        msg = email.message_from_bytes(msg_data[0][1])

        # Decode subject
        subject, encoding = decode_header(msg["Subject"])[0]
        if isinstance(subject, bytes):
            subject = subject.decode(encoding or 'utf-8', errors='ignore')

        from_ = msg.get("From")
        date = msg.get("Date")

        # Get email body
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

        fetched_emails.append({
            "from": from_,
            "subject": subject,
            "body": body,
            "date": date
        })

        # Mark as seen so we don't fetch again
        mail.store(num, '+FLAGS', '\\Seen')

    mail.logout()
    return fetched_emails

def analyze_sentiment_and_escalation(text):
    # VADER sentiment
    vs = analyzer.polarity_scores(text)
    compound = vs["compound"]
    sentiment = "Positive" if compound >= 0 else "Negative"

    # Check for negative escalation keywords
    lower_text = text.lower()
    negative_count = sum(lower_text.count(word) for word in ALL_NEGATIVE_WORDS)
    escalation_flag = 1 if negative_count > 0 else 0

    # Priority determination
    if escalation_flag == 1:
        priority = "High" if negative_count >= 2 else "Medium"
    else:
        priority = "Low"

    return sentiment, priority, escalation_flag

def log_emails_to_db(fetched_emails):
    if not fetched_emails:
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    logged_count = 0
    for mail in fetched_emails:
        # Avoid duplicates (same from + body)
        cursor.execute("SELECT COUNT(*) FROM escalations WHERE customer=? AND issue=?", (mail['from'], mail['body']))
        if cursor.fetchone()[0] > 0:
            continue

        sentiment, priority, escalation_flag = analyze_sentiment_and_escalation(mail['body'])
        esc_id = generate_escalation_id()

        cursor.execute("""
            INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            esc_id,
            mail['from'],
            mail['body'][:500],  # limit text size
            mail['date'],
            "Open",
            sentiment,
            priority,
            escalation_flag
        ))
        logged_count += 1

    conn.commit()
    conn.close()
    return logged_count

def load_escalations():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    conn.close()
    return df

def update_escalation(esc_id, status, action_taken, action_owner):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE escalations SET status=?, action_taken=?, action_owner=? WHERE escalation_id=?
    """, (status, action_taken, action_owner, esc_id))
    conn.commit()
    conn.close()

def manual_entry():
    st.sidebar.subheader("➕ Manual Escalation Entry")
    with st.sidebar.form("manual_entry_form", clear_on_submit=True):
        customer = st.text_input("Customer Email")
        issue = st.text_area("Issue Description")
        date = st.date_input("Date", datetime.date.today())
        submitted = st.form_submit_button("Add Escalation")

        if submitted:
            if not customer or not issue:
                st.sidebar.error("Customer and Issue fields are required.")
                return

            sentiment, priority, escalation_flag = analyze_sentiment_and_escalation(issue)
            esc_id = generate_escalation_id()
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                esc_id,
                customer,
                issue[:500],
                date.strftime("%a, %d %b %Y"),
                "Open",
                sentiment,
                priority,
                escalation_flag
            ))
            conn.commit()
            conn.close()
            st.sidebar.success(f"Escalation {esc_id} added.")

def bulk_upload():
    st.sidebar.subheader("📁 Bulk Upload Complaints (Excel)")
    uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            # Expect columns: customer/email, issue, date (flexible)
            col_map = {}
            for col in df.columns:
                col_low = col.lower()
                if "customer" in col_low or "email" in col_low:
                    col_map['customer'] = col
                elif "issue" in col_low or "complaint" in col_low:
                    col_map['issue'] = col
                elif "date" in col_low:
                    col_map['date'] = col
            missing = [k for k in ['customer', 'issue', 'date'] if k not in col_map]
            if missing:
                st.sidebar.error(f"Missing required columns: {missing}")
                return

            added = 0
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            for _, row in df.iterrows():
                cust = row[col_map['customer']]
                iss = str(row[col_map['issue']])
                dt = row[col_map['date']]
                if pd.isna(cust) or pd.isna(iss) or pd.isna(dt):
                    continue
                # Convert date to string if datetime
                if isinstance(dt, (pd.Timestamp, datetime.datetime)):
                    dt_str = dt.strftime("%a, %d %b %Y")
                else:
                    dt_str = str(dt)

                # Check duplicates
                cursor.execute("SELECT COUNT(*) FROM escalations WHERE customer=? AND issue=?", (cust, iss))
                if cursor.fetchone()[0] > 0:
                    continue

                sentiment, priority, escalation_flag = analyze_sentiment_and_escalation(iss)
                esc_id = generate_escalation_id()
                cursor.execute("""
                    INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    esc_id,
                    cust,
                    iss[:500],
                    dt_str,
                    "Open",
                    sentiment,
                    priority,
                    escalation_flag
                ))
                added += 1
            conn.commit()
            conn.close()
            st.sidebar.success(f"{added} complaints uploaded and analyzed.")
        except Exception as e:
            st.sidebar.error(f"Failed to process uploaded file: {e}")

def render_kanban():
    df = load_escalations()
    if df.empty:
        st.info("No escalations/complaints found in the system.")
        return

    # Filter for escalated or all
    filter_opt = st.selectbox("Filter Escalations:", options=["All Cases", "Escalated Only"])
    if filter_opt == "Escalated Only":
        if "escalation_flag" not in df.columns:
            st.error("Escalation flag column missing in data.")
            return
        df = df[df["escalation_flag"] == 1]

    # Count by status
    counts = df['status'].value_counts().to_dict()
    statuses = ["Open", "In Progress", "Resolved"]

    cols = st.columns(len(statuses))

    # Colors
    sentiment_colors = {"Positive": "🟢", "Negative": "🔴"}
    priority_colors = {"High": "🔥", "Medium": "⚠️", "Low": "✅"}

    for i, status in enumerate(statuses):
        with cols[i]:
            count = counts.get(status, 0)
            st.markdown(f"### {status} ({count})")
            df_status = df[df['status'] == status]

            for idx, row in df_status.iterrows():
                # Safe access to keys
                esc_id = row.get('escalation_id', '')
                sentiment = row.get('sentiment', '')
                priority = row.get('priority', '')

                header = f"{esc_id} - {sentiment_colors.get(sentiment, '')} {sentiment} / {priority_colors.get(priority, '')} {priority}"

                with st.expander(header, expanded=False):
                    st.markdown(f"**Customer:** {row.get('customer', '')}")
                    st.markdown(f"**Issue:** {row.get('issue', '')}")
                    st.markdown(f"**Date:** {row.get('date', '')}")

                    # Editable status, action taken, action owner
                    new_status = st.selectbox(f"Update Status ({esc_id})", options=statuses,
                                             index=statuses.index(row.get('status', 'Open')),
                                             key=f"status_{esc_id}")
                    new_action_taken = st.text_area(f"Action Taken ({esc_id})", value=row.get('action_taken', ''),
                                                   key=f"action_{esc_id}")
                    new_action_owner = st.text_input(f"Action Owner ({esc_id})", value=row.get('action_owner', ''),
                                                    key=f"owner_{esc_id}")

                    if (new_status != row.get('status')) or (new_action_taken != row.get('action_taken')) or (new_action_owner != row.get('action_owner')):
                        update_escalation(esc_id, new_status, new_action_taken, new_action_owner)
                        st.success(f"Updated escalation {esc_id}.")

def main():
    st.title("🚀 EscalateAI - Escalations & Complaints Kanban Board")

    init_db()
    ensure_escalation_flag_column()

    manual_entry()
    bulk_upload()

    # Periodic email fetch every ~1 minute
    if 'last_email_fetch' not in st.session_state:
        st.session_state['last_email_fetch'] = 0
    now = time.time()
    if now - st.session_state['last_email_fetch'] > 60:
        fetched = fetch_emails()
        if fetched:
            logged = log_emails_to_db(fetched)
            if logged:
                st.success(f"Fetched & logged {logged} new emails.")
                # Save complaints to CSV for download and analysis
                df_new = pd.DataFrame(fetched)
                save_complaint_to_csv(df_new)
        st.session_state['last_email_fetch'] = now

    # Download complaints CSV
    if os.path.exists(COMPLAINTS_CSV):
        with open(COMPLAINTS_CSV, "rb") as f:
            st.sidebar.download_button(
                label="📥 Download Email Complaints CSV",
                data=f,
                file_name=COMPLAINTS_CSV,
                mime="text/csv"
            )

    render_kanban()

if __name__ == "__main__":
    main()
