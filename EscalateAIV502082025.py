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

# Constants & Globals
GMAIL_USER = os.getenv("EMAIL_USER")
GMAIL_PASS = os.getenv("EMAIL_PASS")
GMAIL_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
DB_PATH = "escalateai.db"
CSV_PATH = "complaints.csv"  # For sequential logging of parsed emails

# Negative keywords to detect escalation triggers
NEGATIVE_KEYWORDS = set([
    # Technical Failures & Product Malfunction
    "fail", "break", "crash", "defect", "fault", "degrade", "damage",
    "trip", "malfunction", "blank", "shutdown", "discharge",
    # Customer Dissatisfaction & Escalations
    "dissatisfy", "frustrate", "complain", "reject", "delay", "ignore",
    "escalate", "displease", "noncompliance", "neglect",
    # Support Gaps & Operational Delays
    "wait", "pending", "slow", "incomplete", "miss", "omit",
    "unresolved", "shortage", "no response",
    # Hazardous Conditions & Safety Risks
    "fire", "burn", "flashover", "arc", "explode", "unsafe",
    "leak", "corrode", "alarm", "incident",
    # Business Risk & Impact
    "impact", "loss", "risk", "downtime", "interrupt",
    "cancel", "terminate", "penalty"
])

analyzer = SentimentIntensityAnalyzer()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
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
    conn.close()

def get_next_escalation_id():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM escalations")
    count = c.fetchone()[0]
    conn.close()
    return f"SESICE-{250001 + count}"

def parse_emails():
    if not GMAIL_USER or not GMAIL_PASS:
        st.warning("Set EMAIL_USER and EMAIL_PASS in your .env file.")
        return []

    try:
        mail = imaplib.IMAP4_SSL(GMAIL_SERVER)
        mail.login(GMAIL_USER, GMAIL_PASS)
        mail.select("inbox")
        status, messages = mail.search(None, '(UNSEEN)')
        if status != 'OK':
            st.info("No new unread emails.")
            return []

        email_ids = messages[0].split()
        st.write(f"Found {len(email_ids)} unread emails. Fetching latest 10.")
        fetched_emails = []

        for num in email_ids[-10:]:
            res, msg_data = mail.fetch(num, "(RFC822)")
            if res != 'OK':
                continue
            msg = email.message_from_bytes(msg_data[0][1])
            subject, encoding = decode_header(msg["Subject"])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or "utf-8", errors="ignore")
            from_ = msg.get("From", "")
            date = msg.get("Date", "")
            body = ""

            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition", "")):
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

            fetched_emails.append({
                "customer": from_,
                "issue": body.strip(),
                "subject": subject,
                "date": date,
            })

            # Mark as seen
            mail.store(num, '+FLAGS', '\\Seen')

        mail.logout()

        # Append to CSV file for complaints tracking
        if fetched_emails:
            df_new = pd.DataFrame(fetched_emails)
            if os.path.exists(CSV_PATH):
                df_old = pd.read_csv(CSV_PATH)
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_combined = df_new
            df_combined.to_csv(CSV_PATH, index=False)
            st.success(f"Fetched and saved {len(fetched_emails)} new complaints to {CSV_PATH}")

        return fetched_emails

    except Exception as e:
        st.error(f"Failed to fetch emails: {e}")
        return []

def analyze_and_log_complaints(complaints):
    if not complaints:
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    for comp in complaints:
        customer = comp["customer"]
        issue = comp["issue"]
        date = comp["date"]
        # Deduplicate based on customer + issue
        c.execute("SELECT COUNT(*) FROM escalations WHERE customer=? AND issue=?", (customer, issue))
        if c.fetchone()[0] > 0:
            continue

        # Sentiment using VADER
        vs = analyzer.polarity_scores(issue)
        compound = vs["compound"]
        sentiment = "Negative" if compound < -0.05 else "Positive"
        # Priority logic based on keyword count
        keywords_found = sum(1 for w in NEGATIVE_KEYWORDS if w in issue.lower())
        priority = "High" if keywords_found >= 2 else "Low"
        escalation_flag = 1 if keywords_found > 0 else 0

        escalation_id = get_next_escalation_id()
        c.execute("""
            INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, action_taken, action_owner)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (escalation_id, customer, issue[:500], date, "Open", sentiment, priority, escalation_flag, "", ""))

    conn.commit()
    conn.close()

def upload_and_analyze_file():
    uploaded_file = st.sidebar.file_uploader("Upload Complaints Excel or CSV", type=["xlsx", "csv"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.sidebar.success(f"Uploaded {uploaded_file.name} with {len(df)} records.")
            # Expecting columns: customer, issue, date (case-insensitive)
            df.columns = [c.lower() for c in df.columns]
            if not all(col in df.columns for col in ['customer', 'issue']):
                st.sidebar.error("Uploaded file must contain 'customer' and 'issue' columns.")
                return

            # Normalize missing date
            if 'date' not in df.columns:
                df['date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            complaints = df.to_dict('records')
            analyze_and_log_complaints(complaints)
            st.sidebar.success("Uploaded complaints analyzed and logged.")
        except Exception as e:
            st.sidebar.error(f"Failed to process uploaded file: {e}")

def manual_entry():
    st.sidebar.header("Manual Escalation Entry")
    customer = st.sidebar.text_input("Customer Email or Name")
    issue = st.sidebar.text_area("Issue / Complaint")
    date = st.sidebar.text_input("Date (YYYY-MM-DD HH:MM:SS)", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if st.sidebar.button("Add Escalation"):
        if not customer or not issue:
            st.sidebar.warning("Please enter both customer and issue.")
            return
        escalation_flag = 1  # Manual entries are assumed escalations
        # Sentiment and priority
        vs = analyzer.polarity_scores(issue)
        compound = vs["compound"]
        sentiment = "Negative" if compound < -0.05 else "Positive"
        keywords_found = sum(1 for w in NEGATIVE_KEYWORDS if w in issue.lower())
        priority = "High" if keywords_found >= 2 else "Low"
        escalation_id = get_next_escalation_id()

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, action_taken, action_owner)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (escalation_id, customer, issue[:500], date, "Open", sentiment, priority, escalation_flag, "", ""))
        conn.commit()
        conn.close()
        st.sidebar.success(f"Escalation {escalation_id} added.")

def load_escalations():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    conn.close()
    return df

def update_escalation_field(escalation_id, field, value):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    query = f"UPDATE escalations SET {field}=? WHERE escalation_id=?"
    c.execute(query, (value, escalation_id))
    conn.commit()
    conn.close()

def render_kanban():
    st.title("🚀 EscalateAI - Escalations & Complaints Kanban Board")

    # Load escalations
    df = load_escalations()

    # Sidebar filter for viewing
    filter_view = st.sidebar.radio("View Cases:", options=["All", "Escalated Only"], index=0)

    if filter_view == "Escalated Only":
        df = df[df['escalation_flag'] == 1]

    # Show Download button for complaints CSV
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "rb") as f:
            st.sidebar.download_button("Download Email Complaints CSV", data=f, file_name="email_complaints.csv")

    # Status counts
    counts = df['status'].value_counts().to_dict()
    open_count = counts.get("Open", 0)
    inprogress_count = counts.get("In Progress", 0)
    resolved_count = counts.get("Resolved", 0)

    # Status categories for select box
    status_options = ["Open", "In Progress", "Resolved"]

    # Columns for Kanban board
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader(f"Open ({open_count})")
        open_rows = df[df['status'] == "Open"]
        if open_rows.empty:
            st.write("No escalations")
        for _, row in open_rows.iterrows():
            with st.expander(f"{row['escalation_id']} - {row['customer']} - {row['sentiment']}/{row['priority']}"):
                st.write(f"Issue: {row['issue']}")
                # Editable status
                new_status = st.selectbox("Update Status", status_options, index=status_options.index(row['status']), key=f"status_{row['escalation_id']}")
                new_action_taken = st.text_area("Action Taken", value=row['action_taken'] or "", key=f"action_{row['escalation_id']}")
                new_action_owner = st.text_input("Action Owner", value=row['action_owner'] or "", key=f"owner_{row['escalation_id']}")
                if new_status != row['status']:
                    update_escalation_field(row['escalation_id'], "status", new_status)
                    st.experimental_rerun()
                if new_action_taken != (row['action_taken'] or ""):
                    update_escalation_field(row['escalation_id'], "action_taken", new_action_taken)
                if new_action_owner != (row['action_owner'] or ""):
                    update_escalation_field(row['escalation_id'], "action_owner", new_action_owner)

    with col2:
        st.subheader(f"In Progress ({inprogress_count})")
        inprogress_rows = df[df['status'] == "In Progress"]
        if inprogress_rows.empty:
            st.write("No escalations")
        for _, row in inprogress_rows.iterrows():
            with st.expander(f"{row['escalation_id']} - {row['customer']} - {row['sentiment']}/{row['priority']}"):
                st.write(f"Issue: {row['issue']}")
                new_status = st.selectbox("Update Status", status_options, index=status_options.index(row['status']), key=f"status_{row['escalation_id']}")
                new_action_taken = st.text_area("Action Taken", value=row['action_taken'] or "", key=f"action_{row['escalation_id']}")
                new_action_owner = st.text_input("Action Owner", value=row['action_owner'] or "", key=f"owner_{row['escalation_id']}")
                if new_status != row['status']:
                    update_escalation_field(row['escalation_id'], "status", new_status)
                    st.experimental_rerun()
                if new_action_taken != (row['action_taken'] or ""):
                    update_escalation_field(row['escalation_id'], "action_taken", new_action_taken)
                if new_action_owner != (row['action_owner'] or ""):
                    update_escalation_field(row['escalation_id'], "action_owner", new_action_owner)

    with col3:
        st.subheader(f"Resolved ({resolved_count})")
        resolved_rows = df[df['status'] == "Resolved"]
        if resolved_rows.empty:
            st.write("No escalations")
        for _, row in resolved_rows.iterrows():
            with st.expander(f"{row['escalation_id']} - {row['customer']} - {row['sentiment']}/{row['priority']}"):
                st.write(f"Issue: {row['issue']}")
                new_status = st.selectbox("Update Status", status_options, index=status_options.index(row['status']), key=f"status_{row['escalation_id']}")
                new_action_taken = st.text_area("Action Taken", value=row['action_taken'] or "", key=f"action_{row['escalation_id']}")
                new_action_owner = st.text_input("Action Owner", value=row['action_owner'] or "", key=f"owner_{row['escalation_id']}")
                if new_status != row['status']:
                    update_escalation_field(row['escalation_id'], "status", new_status)
                    st.experimental_rerun()
                if new_action_taken != (row['action_taken'] or ""):
                    update_escalation_field(row['escalation_id'], "action_taken", new_action_taken)
                if new_action_owner != (row['action_owner'] or ""):
                    update_escalation_field(row['escalation_id'], "action_owner", new_action_owner)

def main():
    st.sidebar.title("EscalateAI Controls")

    init_db()

    manual_entry()

    upload_and_analyze_file()

    if st.sidebar.button("Fetch Latest Emails"):
        complaints = parse_emails()
        analyze_and_log_complaints(complaints)
        st.sidebar.success(f"Fetched and logged {len(complaints)} new complaints.")

    render_kanban()

if __name__ == "__main__":
    main()
