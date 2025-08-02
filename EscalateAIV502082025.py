import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")

NEGATIVE_KEYWORDS = [
    # Technical Failures & Product Malfunction
    "fail", "break", "crash", "defect", "fault", "degrade", "damage",
    "trip", "malfunction", "blank", "shutdown", "discharge",
    # Customer Dissatisfaction & Escalations
    "dissatisfy", "frustrate", "complain", "reject", "delay", "ignore",
    "escalate", "displease", "noncompliance", "neglect", "poor quality"
    # Support Gaps & Operational Delays
    "wait", "pending", "slow", "incomplete", "miss", "omit",
    "unresolved", "shortage", "no response", "delay"
    # Hazardous Conditions & Safety Risks
    "fire", "burn", "flashover", "arc", "explode", "unsafe",
    "leak", "corrode", "alarm", "incident",
    # Business Risk & Impact
    "impact", "loss", "risk", "downtime", "interrupt", "cancel",
    "terminate", "penalty"
]

# Connect to DB (create if not exists)
conn = sqlite3.connect("escalations.db", check_same_thread=False)
cursor = conn.cursor()

# Create table if not exists
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

analyzer = SentimentIntensityAnalyzer()

def fetch_gmail_emails():
    if not EMAIL or not APP_PASSWORD:
        st.error("Gmail credentials not set in environment variables.")
        return []
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL, APP_PASSWORD)
        mail.select("inbox")
        result, data = mail.search(None, "UNSEEN")
        if result != "OK":
            st.info("No new emails found.")
            mail.logout()
            return []

        email_ids = data[0].split()
        emails = []

        for eid in email_ids[-10:]:  # last 10 unseen emails
            res, msg_data = mail.fetch(eid, "(RFC822)")
            if res != "OK":
                continue
            msg = email.message_from_bytes(msg_data[0][1])

            # Decode subject
            subject, encoding = decode_header(msg.get("Subject"))[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or "utf-8", errors="ignore")

            from_ = msg.get("From")
            date = msg.get("Date")

            # Extract plain text body
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

            emails.append({
                "customer": from_,
                "issue": body.strip(),
                "subject": subject,
                "date": date
            })

            # Mark email as read
            mail.store(eid, '+FLAGS', '\\Seen')

        mail.logout()
        return emails
    except Exception as e:
        st.error(f"Error fetching emails: {e}")
        return []

def analyze_issue(issue_text):
    # Lowercase
    text_lower = issue_text.lower()

    # Sentiment using VADER
    vs = analyzer.polarity_scores(issue_text)
    sentiment = "Positive" if vs["compound"] >= 0 else "Negative"

    # Count how many negative keywords appear
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)

    # Priority: High if negative sentiment AND 2 or more neg keywords, else Low
    priority = "High" if sentiment == "Negative" and neg_count >= 2 else "Low"

    # Escalation flag 1 if high priority, else 0
    escalation_flag = 1 if priority == "High" else 0

    return sentiment, priority, escalation_flag

def save_emails_to_db(emails):
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]

    new_entries = 0
    for e in emails:
        # Check duplicate by customer + issue text
        cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (e['customer'], e['issue'][:500]))
        if cursor.fetchone():
            continue  # skip duplicate

        count += 1
        esc_id = f"SESICE-{count+250000}"

        sentiment, priority, escalation_flag = analyze_issue(e['issue'])

        cursor.execute("""
            INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, action_taken, action_owner)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (esc_id, e['customer'], e['issue'][:500], e['date'], "Open", sentiment, priority, escalation_flag, "", ""))

        new_entries += 1

    conn.commit()
    return new_entries

def load_escalations_df():
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    return df

def upload_excel_and_analyze(file):
    try:
        df = pd.read_excel(file)
        df.columns = [c.lower().strip() for c in df.columns]

        customer_col = next((c for c in df.columns if "customer" in c or "email" in c), None)
        issue_col = next((c for c in df.columns if "issue" in c or "text" in c or "complaint" in c), None)
        date_col = next((c for c in df.columns if "date" in c), None)

        if not customer_col or not issue_col:
            st.error("Excel must contain customer/email and issue/text columns.")
            return 0

        count = 0
        existing_count = pd.read_sql_query("SELECT COUNT(*) FROM escalations", conn).iloc[0,0]

        for idx, row in df.iterrows():
            customer = str(row[customer_col])
            issue = str(row[issue_col])
            date = str(row[date_col]) if date_col else datetime.datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z")

            cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (customer, issue[:500]))
            if cursor.fetchone():
                continue

            existing_count += 1
            esc_id = f"SESICE-{existing_count+250000}"
            sentiment, priority, escalation_flag = analyze_issue(issue)

            cursor.execute("""
                INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, action_taken, action_owner)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (esc_id, customer, issue[:500], date, "Open", sentiment, priority, escalation_flag, "", ""))

            count += 1
        conn.commit()
        return count
    except Exception as e:
        st.error(f"Error processing uploaded Excel: {e}")
        return 0

def manual_entry():
    st.sidebar.header("➕ Manual Escalation Entry")
    customer = st.sidebar.text_input("Customer Email/Name")
    issue = st.sidebar.text_area("Issue / Complaint")
    date = datetime.datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z")
    if st.sidebar.button("Add Escalation"):
        if not customer or not issue:
            st.sidebar.error("Please fill customer and issue.")
            return
        cursor.execute("SELECT COUNT(*) FROM escalations")
        count = cursor.fetchone()[0]
        esc_id = f"SESICE-{count+250001}"
        sentiment, priority, escalation_flag = analyze_issue(issue)

        cursor.execute("""
            INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, action_taken, action_owner)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (esc_id, customer, issue[:500], date, "Open", sentiment, priority, escalation_flag, "", ""))
        conn.commit()
        st.sidebar.success(f"Added escalation {esc_id}")

def display_kanban_card(row):
    esc_id = row['escalation_id']
    sentiment = row['sentiment']
    priority = row['priority']
    status = row['status']

    # Colors as per your snippet and standard
    sentiment_colors = {"Positive": "#27ae60", "Negative": "#e74c3c"}
    priority_colors = {"High": "#c0392b", "Low": "#27ae60"}
    status_colors = {"Open": "#f1c40f", "In Progress": "#2980b9", "Resolved": "#2ecc71"}

    border_color = priority_colors.get(priority, "#000000")
    status_color = status_colors.get(status, "#bdc3c7")
    sentiment_color = sentiment_colors.get(sentiment, "#7f8c8d")

    # Styled header matching your HTML snippet
    header_html = f"""
    <div style="
        border-left: 6px solid {border_color};
        padding-left: 10px;
        margin-bottom: 10px;
        font-weight:bold;">
        {esc_id} &nbsp; 
        <span style='color:{sentiment_color}; font-weight:bold;'>● {sentiment}</span> / 
        <span style='color:{priority_colors.get(priority, '#000')}; font-weight:bold;'>■ {priority}</span> / 
        <span style='color:{status_color}; font-weight:bold;'>{status}</span>
    </div>
    """

    with st.expander(header_html, expanded=False):
        st.markdown(f"**Customer:** {row['customer']}")
        st.markdown(f"**Issue:** {row['issue']}")
        st.markdown(f"**Date:** {row['date']}")

        new_status = st.selectbox(
            "Update Status",
            ["Open", "In Progress", "Resolved"],
            index=["Open", "In Progress", "Resolved"].index(status),
            key=f"{esc_id}_status"
        )
        new_action_taken = st.text_area(
            "Action Taken",
            value=row['action_taken'] or "",
            key=f"{esc_id}_action"
        )
        new_action_owner = st.text_input(
            "Action Owner",
            value=row['action_owner'] or "",
            key=f"{esc_id}_owner"
        )

        if st.button("Save Updates", key=f"save_{esc_id}"):
            cursor.execute("""
                UPDATE escalations SET status=?, action_taken=?, action_owner=?
                WHERE escalation_id=?
            """, (new_status, new_action_taken, new_action_owner, esc_id))
            conn.commit()
            st.success("Updated successfully!")
            st.experimental_rerun()

def render_kanban():
    st.title("🚀 EscalateAI - Escalations & Complaints Kanban Board")

    df = load_escalations_df()
    filter_choice = st.radio("Filter Escalations:", ["All", "Escalated Only"])

    if filter_choice == "Escalated Only":
        df = df[df['escalation_flag'] == 1]

    # Counts
    open_count = len(df[df['status'] == 'Open'])
    inprogress_count = len(df[df['status'] == 'In Progress'])
    resolved_count = len(df[df['status'] == 'Resolved'])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"<h3 style='color:#f1c40f;'>🟡 Open ({open_count})</h3>", unsafe_allow_html=True)
        for _, row in df[df['status'] == 'Open'].iterrows():
            display_kanban_card(row)

    with col2:
        st.markdown(f"<h3 style='color:#2980b9;'>🔵 In Progress ({inprogress_count})</h3>", unsafe_allow_html=True)
        for _, row in df[df['status'] == 'In Progress'].iterrows():
            display_kanban_card(row)

    with col3:
        st.markdown(f"<h3 style='color:#2ecc71;'>🟢 Resolved ({resolved_count})</h3>", unsafe_allow_html=True)
        for _, row in df[df['status'] == 'Resolved'].iterrows():
            display_kanban_card(row)

def save_complaints_csv():
    df = load_escalations_df()
    filename = "complaints_data.xlsx"
    df.to_excel(filename, index=False)
    return filename

def main():
    st.sidebar.header("📥 Upload Complaints Excel File")
    uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    if uploaded_file:
        count = upload_excel_and_analyze(uploaded_file)
        st.sidebar.success(f"Uploaded and analyzed {count} new complaints.")

    manual_entry()

    st.sidebar.markdown("---")
    if st.sidebar.button("Fetch New Emails from Gmail"):
        emails = fetch_gmail_emails()
        if emails:
            new_count = save_emails_to_db(emails)
            st.sidebar.success(f"Fetched and saved {new_count} new emails.")
        else:
            st.sidebar.info("No new emails or error.")

    if st.sidebar.button("Download Email Complaints Excel"):
        filepath = save_complaints_csv()
        with open(filepath, "rb") as f:
            st.sidebar.download_button(
                label="Download Complaints Data",
                data=f,
                file_name="EscalateAI_Complaints.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    render_kanban()

if __name__ == "__main__":
    main()
