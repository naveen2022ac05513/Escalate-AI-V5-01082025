import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import requests
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load environment variables
load_dotenv()
EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")
EMAIL_ALERT_ENABLED = os.getenv("EMAIL_ALERT_ENABLED", "false").lower() == "true"
EMAIL_ALERT_RECIPIENT = os.getenv("EMAIL_ALERT_RECIPIENT")

NEGATIVE_KEYWORDS = [
    # ⚙️ Technical Failures & Product Malfunction
    "fail", "break", "crash", "defect", "fault", "degrade", "damage",
    "trip", "malfunction", "blank", "shutdown", "discharge",
    # 💢 Customer Dissatisfaction & Escalations
    "dissatisfy", "frustrate", "complain", "reject", "delay", "ignore",
    "escalate", "displease", "noncompliance", "neglect",
    # ⏳ Support Gaps & Operational Delays
    "wait", "pending", "slow", "incomplete", "miss", "omit",
    "unresolved", "shortage", "no response",
    # 💥 Hazardous Conditions & Safety Risks
    "fire", "burn", "flashover", "arc", "explode", "unsafe",
    "leak", "corrode", "alarm", "incident",
    # 📉 Business Risk & Impact
    "impact", "loss", "risk", "downtime", "interrupt", "cancel",
    "terminate", "penalty"
]

# DB connection
conn = sqlite3.connect("escalations.db", check_same_thread=False)
cursor = conn.cursor()

# Ensure table with your provided columns
cursor.execute("""
CREATE TABLE IF NOT EXISTS escalations (
    id TEXT PRIMARY KEY,
    customer TEXT,
    issue TEXT,
    sentiment TEXT,
    urgency TEXT,
    severity TEXT,
    criticality TEXT,
    category TEXT,
    status TEXT,
    timestamp TEXT,
    action_taken TEXT,
    owner TEXT,
    escalated INTEGER,
    priority TEXT,
    escalation_flag INTEGER,
    action_owner TEXT,
    status_update_date TEXT,
    user_feedback TEXT
)
""")
conn.commit()

analyzer = SentimentIntensityAnalyzer()

# Placeholder ML model and training data (you should train and update this properly)
MODEL_PATH = "escalation_model.pkl"
def load_ml_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    else:
        # Return None or a dummy model if not available
        return None

def save_ml_model(model):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

model = load_ml_model()

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
            mail.logout()
            return []

        email_ids = data[0].split()
        emails = []

        for eid in email_ids[-10:]:  # last 10 unseen emails
            res, msg_data = mail.fetch(eid, "(RFC822)")
            if res != "OK":
                continue
            msg = email.message_from_bytes(msg_data[0][1])

            subject, encoding = decode_header(msg.get("Subject"))[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or "utf-8", errors="ignore")

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

            emails.append({
                "customer": from_,
                "issue": body.strip(),
                "subject": subject,
                "date": date
            })

            mail.store(eid, '+FLAGS', '\\Seen')

        mail.logout()
        return emails
    except Exception as e:
        st.error(f"Error fetching emails: {e}")
        return []

def analyze_issue(issue_text):
    text_lower = issue_text.lower()
    vs = analyzer.polarity_scores(issue_text)
    sentiment = "Positive" if vs["compound"] >= 0 else "Negative"
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    priority = "High" if sentiment == "Negative" and neg_count >= 2 else "Low"
    escalation_flag = 1 if priority == "High" else 0
    # For demo: simple urgency, severity, criticality, category tagging heuristics
    urgency = "High" if "urgent" in text_lower or neg_count >= 3 else "Low"
    severity = "Critical" if "critical" in text_lower or neg_count >= 4 else "Normal"
    criticality = "High" if "fail" in text_lower or "shutdown" in text_lower else "Low"
    category = "Technical" if any(k in text_lower for k in NEGATIVE_KEYWORDS[:12]) else "General"
    return sentiment, priority, escalation_flag, urgency, severity, criticality, category

def save_emails_to_db(emails):
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]
    new_entries = 0
    for e in emails:
        cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (e['customer'], e['issue'][:500]))
        if cursor.fetchone():
            continue
        count += 1
        esc_id = f"SESICE-{count+250000}"
        sentiment, priority, escalation_flag, urgency, severity, criticality, category = analyze_issue(e['issue'])
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
        cursor.execute("""
            INSERT INTO escalations (
                id, customer, issue, sentiment, urgency, severity, criticality, category,
                status, timestamp, action_taken, owner, escalated, priority, escalation_flag,
                action_owner, status_update_date, user_feedback
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (esc_id, e['customer'], e['issue'][:500], sentiment, urgency, severity, criticality, category,
              "Open", e['date'], "", "", escalation_flag, priority, escalation_flag,
              "", now, ""))
        new_entries += 1
        if escalation_flag == 1:
            send_ms_teams_alert(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {e['customer']}\nIssue: {e['issue'][:200]}...")
    conn.commit()
    return new_entries

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

        cursor.execute("SELECT COUNT(*) FROM escalations")
        existing_count = cursor.fetchone()[0]

        count = 0
        for idx, row in df.iterrows():
            customer = str(row[customer_col])
            issue = str(row[issue_col])
            date = str(row[date_col]) if date_col and pd.notna(row[date_col]) else datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (customer, issue[:500]))
            if cursor.fetchone():
                continue
            existing_count += 1
            esc_id = f"SESICE-{existing_count+250000}"
            sentiment, priority, escalation_flag, urgency, severity, criticality, category = analyze_issue(issue)
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("""
                INSERT INTO escalations (
                    id, customer, issue, sentiment, urgency, severity, criticality, category,
                    status, timestamp, action_taken, owner, escalated, priority, escalation_flag,
                    action_owner, status_update_date, user_feedback
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (esc_id, customer, issue[:500], sentiment, urgency, severity, criticality, category,
                  "Open", date, "", "", escalation_flag, priority, escalation_flag,
                  "", now, ""))
            count += 1
            if escalation_flag == 1:
                send_ms_teams_alert(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {customer}\nIssue: {issue[:200]}...")
        conn.commit()
        return count
    except Exception as e:
        st.error(f"Error processing uploaded Excel: {e}")
        return 0

def manual_entry_process(customer, issue):
    if not customer or not issue:
        st.sidebar.error("Please fill customer and issue.")
        return False
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]
    esc_id = f"SESICE-{count+250001}"
    sentiment, priority, escalation_flag, urgency, severity, criticality, category = analyze_issue(issue)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
    cursor.execute("""
        INSERT INTO escalations (
            id, customer, issue, sentiment, urgency, severity, criticality, category,
            status, timestamp, action_taken, owner, escalated, priority, escalation_flag,
            action_owner, status_update_date, user_feedback
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (esc_id, customer, issue[:500], sentiment, urgency, severity, criticality, category,
          "Open", now, "", "", escalation_flag, priority, escalation_flag,
          "", now, ""))
    conn.commit()
    st.sidebar.success(f"Added escalation {esc_id}")
    if escalation_flag == 1:
        send_ms_teams_alert(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {customer}\nIssue: {issue[:200]}...")
    return True

def load_escalations_df():
    df = pd.read_sql_query("""
        SELECT
            id AS escalation_id,
            customer,
            issue,
            sentiment,
            urgency,
            severity,
            criticality,
            category,
            status,
            timestamp AS date,
            action_taken,
            owner,
            escalated,
            priority,
            escalation_flag,
            action_owner,
            status_update_date,
            user_feedback
        FROM escalations
    """, conn)
    return df

def send_ms_teams_alert(message):
    if not MS_TEAMS_WEBHOOK_URL:
        st.warning("MS Teams webhook URL not set; cannot send alerts.")
        return
    headers = {"Content-Type": "application/json"}
    payload = {"text": message}
    try:
        response = requests.post(MS_TEAMS_WEBHOOK_URL, json=payload, headers=headers)
        if response.status_code != 200:
            st.error(f"MS Teams alert failed: {response.status_code} {response.text}")
    except Exception as e:
        st.error(f"Error sending MS Teams alert: {e}")

import smtplib
from email.message import EmailMessage

def send_email_alert(subject, body):
    if not EMAIL_ALERT_ENABLED or not EMAIL_ALERT_RECIPIENT:
        st.warning("Email alerts not configured.")
        return
    try:
        msg = EmailMessage()
        msg["From"] = EMAIL
        msg["To"] = EMAIL_ALERT_RECIPIENT
        msg["Subject"] = subject
        msg.set_content(body)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
    except Exception as e:
        st.error(f"Error sending email alert: {e}")

def check_sla_and_alert():
    df = load_escalations_df()
    now = datetime.datetime.now(datetime.timezone.utc)
    breached = df[(df['priority'] == "High") & (df['status'] == "Open")]

    alerts_sent = 0
    for _, row in breached.iterrows():
        try:
            last_update = datetime.datetime.strptime(row['status_update_date'], "%a, %d %b %Y %H:%M:%S %z")
        except Exception:
            continue
        elapsed = now - last_update
        if elapsed.total_seconds() > 10 * 60:
            message = (
                f"⚠️ SLA breach detected:\n"
                f"ID: {row['escalation_id']}\n"
                f"Customer: {row['customer']}\n"
                f"Open for: {elapsed.seconds // 60} minutes\n"
                f"Issue: {row['issue'][:200]}..."
            )
            send_ms_teams_alert(message)
            send_email_alert(f"SLA Breach: {row['escalation_id']}", message)
            alerts_sent += 1
    return alerts_sent

def display_kanban_card(row):
    esc_id = row['escalation_id']
    sentiment = row['sentiment']
    priority = row['priority']
    status = row['status']
    escalated = row.get('escalated', 0)

    sentiment_colors = {"Positive": "#27ae60", "Negative": "#e74c3c"}
    priority_colors = {"High": "#c0392b", "Low": "#27ae60"}
    status_colors = {"Open": "#f1c40f", "In Progress": "#2980b9", "Resolved": "#2ecc71"}

    border_color = priority_colors.get(priority, "#000000")
    status_color = status_colors.get(status, "#bdc3c7")
    sentiment_color = sentiment_colors.get(sentiment, "#7f8c8d")

    escalated_mark = "🚨" if escalated == 1 else ""

    header_html = f"""
    <div style="
        border-left: 6px solid {border_color};
        padding-left: 10px;
        margin-bottom: 10px;
        font-weight:bold;">
        {esc_id} {escalated_mark} &nbsp; 
        <span style='color:{sentiment_color}; font-weight:bold;'>● {sentiment}</span> / 
        <span style='color:{priority_colors.get(priority, '#000')}; font-weight:bold;'>■ {priority}</span> / 
        <span style='color:{status_color}; font-weight:bold;'>{status}</span>
    </div>
    """

    st.markdown(header_html, unsafe_allow_html=True)

    with st.expander(f"{esc_id} - {row['customer']} [{status}]", expanded=False):
        st.markdown(f"**Issue:** {row['issue']}")
        st.markdown(f"**Date:** {row['date']}")
        st.markdown(f"**Urgency:** {row['urgency']}")
        st.markdown(f"**Severity:** {row['severity']}")
        st.markdown(f"**Criticality:** {row['criticality']}")
        st.markdown(f"**Category:** {row['category']}")
        st.markdown(f"**Owner:** {row['owner'] or ''}")
        st.markdown(f"**Action Taken:** {row['action_taken'] or ''}")

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
        new_owner = st.text_input(
            "Owner",
            value=row['owner'] or "",
            key=f"{esc_id}_owner_main"
        )
        new_feedback = st.text_area(
            "User Feedback",
            value=row['user_feedback'] or "",
            key=f"{esc_id}_feedback"
        )

        if st.button(f"Save changes for {esc_id}", key=f"save_{esc_id}"):
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("""
                UPDATE escalations SET
                status = ?, action_taken = ?, action_owner = ?, owner = ?, user_feedback = ?, status_update_date = ?
                WHERE id = ?
            """, (new_status, new_action_taken, new_action_owner, new_owner, new_feedback, now, esc_id))
            conn.commit()
            st.success(f"Escalation {esc_id} updated.")
            st.experimental_rerun()

def main():
    st.title("EscalateAI - Customer Escalation Management")

    # Sidebar controls
    st.sidebar.title("EscalateAI Controls & Uploads")
    if st.sidebar.button("Fetch new emails from Gmail"):
        new_emails = fetch_gmail_emails()
        count_added = save_emails_to_db(new_emails)
        st.sidebar.success(f"Fetched {len(new_emails)} emails, added {count_added} new escalations.")

    uploaded_file = st.sidebar.file_uploader("Upload Excel file with escalations", type=["xlsx", "xls"])
    if uploaded_file:
        count_added = upload_excel_and_analyze(uploaded_file)
        st.sidebar.success(f"Processed uploaded Excel. Added {count_added} escalations.")

    st.sidebar.markdown("---")
    st.sidebar.header("Manual Entry")
    manual_customer = st.sidebar.text_input("Customer/Email")
    manual_issue = st.sidebar.text_area("Issue")
    if st.sidebar.button("Add Manual Escalation"):
        if manual_entry_process(manual_customer, manual_issue):
            st.sidebar.success("Manual escalation added.")

    st.sidebar.markdown("---")
    if st.sidebar.button("Check SLA breaches and send alerts"):
        count_alerts = check_sla_and_alert()
        st.sidebar.info(f"SLA alerts sent for {count_alerts} escalations.")

    # Filters
    df = load_escalations_df()

    status_filter = st.multiselect(
        "Filter by Status",
        options=["Open", "In Progress", "Resolved"],
        default=["Open", "In Progress", "Resolved"]
    )
    priority_filter = st.multiselect(
        "Filter by Priority",
        options=["High", "Low"],
        default=["High", "Low"]
    )
    escalated_only = st.checkbox("Show only escalated cases")

    filtered_df = df[
        (df['status'].isin(status_filter)) &
        (df['priority'].isin(priority_filter))
    ]
    if escalated_only:
        filtered_df = filtered_df[filtered_df['escalated'] == 1]

    # Show counts
    counts = {status: len(df[df['status'] == status]) for status in ["Open", "In Progress", "Resolved"]}
    escalated_count = len(df[df['escalated'] == 1])
    st.markdown(f"**Counts:** Open: {counts['Open']} | In Progress: {counts['In Progress']} | Resolved: {counts['Resolved']} | Escalated: {escalated_count}")

    # Show Kanban board grouped by status
    for status in status_filter:
        st.header(f"{status} ({len(filtered_df[filtered_df['status'] == status])})")
        for _, row in filtered_df[filtered_df['status'] == status].iterrows():
            display_kanban_card(row)

    # Download filtered escalations
    csv = filtered_df.drop(columns=['escalation_flag']).to_csv(index=False)
    st.download_button(
        label="Download filtered escalations as CSV",
        data=csv,
        file_name="escalations_filtered.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
