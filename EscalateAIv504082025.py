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

DB_FILE = "escalations.db"

# Connect DB & ensure schema
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()

def ensure_schema():
    # Create table if not exists with all required columns
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
            action_owner TEXT,
            status_update_date TEXT,
            user_feedback TEXT
        )
    """)
    conn.commit()

    # Check columns present, add if missing (simple approach)
    cursor.execute("PRAGMA table_info(escalations)")
    columns = [col[1] for col in cursor.fetchall()]
    needed = {
        "escalation_id": "TEXT PRIMARY KEY",
        "customer": "TEXT",
        "issue": "TEXT",
        "date": "TEXT",
        "status": "TEXT",
        "sentiment": "TEXT",
        "priority": "TEXT",
        "escalation_flag": "INTEGER",
        "action_taken": "TEXT",
        "action_owner": "TEXT",
        "status_update_date": "TEXT",
        "user_feedback": "TEXT"
    }
    for col, col_type in needed.items():
        if col not in columns:
            cursor.execute(f"ALTER TABLE escalations ADD COLUMN {col} {col_type}")
    conn.commit()

ensure_schema()

analyzer = SentimentIntensityAnalyzer()

# === Functions ===

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

        for eid in email_ids[-10:]:
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

    # For demonstration, simple static tags for severity, criticality, category
    severity = "Critical" if priority == "High" else "Normal"
    criticality = "High" if priority == "High" else "Low"
    category = "General"

    return sentiment, priority, escalation_flag, severity, criticality, category

def save_emails_to_db(emails):
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]
    new_entries = 0
    for e in emails:
        # Deduplication based on customer + issue snippet
        cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (e['customer'], e['issue'][:500]))
        if cursor.fetchone():
            continue
        count += 1
        esc_id = f"SESICE-{count+250000}"
        sentiment, priority, escalation_flag, severity, criticality, category = analyze_issue(e['issue'])
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
        cursor.execute("""
            INSERT INTO escalations 
            (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, action_taken, action_owner, status_update_date, user_feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (esc_id, e['customer'], e['issue'][:500], e['date'], "Open", sentiment, priority, escalation_flag, "", "", now, ""))
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

        count = 0
        cursor.execute("SELECT COUNT(*) FROM escalations")
        existing_count = cursor.fetchone()[0]

        for idx, row in df.iterrows():
            customer = str(row[customer_col])
            issue = str(row[issue_col])
            date = str(row[date_col]) if date_col and pd.notna(row[date_col]) else datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (customer, issue[:500]))
            if cursor.fetchone():
                continue
            existing_count += 1
            esc_id = f"SESICE-{existing_count+250000}"
            sentiment, priority, escalation_flag, severity, criticality, category = analyze_issue(issue)
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("""
                INSERT INTO escalations 
                (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, action_taken, action_owner, status_update_date, user_feedback)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (esc_id, customer, issue[:500], date, "Open", sentiment, priority, escalation_flag, "", "", now, ""))
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
    sentiment, priority, escalation_flag, severity, criticality, category = analyze_issue(issue)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
    cursor.execute("""
        INSERT INTO escalations 
        (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, action_taken, action_owner, status_update_date, user_feedback)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (esc_id, customer, issue[:500], now, "Open", sentiment, priority, escalation_flag, "", "", now, ""))
    conn.commit()
    st.sidebar.success(f"Added escalation {esc_id}")
    if escalation_flag == 1:
        send_ms_teams_alert(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {customer}\nIssue: {issue[:200]}...")
    return True

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

def check_sla_and_alert():
    df = load_escalations_df()
    now = datetime.datetime.now(datetime.timezone.utc)
    breached = df[
        (df['priority'] == "High") &
        (df['status'] == "Open")
    ]

    alerts_sent = 0
    for _, row in breached.iterrows():
        try:
            last_update = datetime.datetime.strptime(row['status_update_date'], "%a, %d %b %Y %H:%M:%S %z")
        except Exception:
            continue
        elapsed = now - last_update
        if elapsed.total_seconds() > 10 * 60:  # 10 minutes SLA breach
            send_ms_teams_alert(
                f"⚠️ SLA breach detected:\nID: {row['escalation_id']}\nCustomer: {row['customer']}\nOpen for: {elapsed.seconds // 60} minutes\nIssue: {row['issue'][:200]}..."
            )
            alerts_sent += 1
    return alerts_sent

def load_escalations_df():
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    # Debug: print columns
    # st.write("Columns:", df.columns.tolist())
    return df

def save_complaints_excel():
    df = load_escalations_df()
    filename = "complaints_data.xlsx"
    df.to_excel(filename, index=False)
    return filename

def display_kanban_card(row):
    esc_id = row['escalation_id']
    sentiment = row['sentiment']
    priority = row['priority']
    status = row['status']
    escalation_flag = row.get('escalation_flag', 0)  # 1 = escalated

    sentiment_colors = {"Positive": "#27ae60", "Negative": "#e74c3c"}
    priority_colors = {"High": "#c0392b", "Low": "#27ae60"}
    status_colors = {"Open": "#f1c40f", "In Progress": "#2980b9", "Resolved": "#2ecc71"}

    border_color = priority_colors.get(priority, "#000000")
    status_color = status_colors.get(status, "#bdc3c7")
    sentiment_color = sentiment_colors.get(sentiment, "#7f8c8d")

    # Highlight if escalated
    box_shadow = "0 0 10px 3px #c0392b" if escalation_flag == 1 else "none"

    header_html = f"""
    <div style="
        border-left: 6px solid {border_color};
        padding-left: 10px;
        margin-bottom: 10px;
        font-weight:bold;
        box-shadow: {box_shadow};
        ">
        {esc_id} &nbsp; 
        <span style='color:{sentiment_color}; font-weight:bold;'>● {sentiment}</span> / 
        <span style='color:{priority_colors.get(priority, '#000')}; font-weight:bold;'>■ {priority}</span> / 
        <span style='color:{status_color}; font-weight:bold;'>{status}</span>
    </div>
    """

    st.markdown(header_html, unsafe_allow_html=True)

    with st.expander(f"{esc_id} - {row['customer']} [{status}]"):
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
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("""
                UPDATE escalations SET status=?, action_taken=?, action_owner=?, status_update_date=?
                WHERE escalation_id=?
            """, (new_status, new_action_taken, new_action_owner, now, esc_id))
            conn.commit()
            st.success("Updated successfully!")
            st.experimental_rerun()

def sidebar_content():
    st.sidebar.title("EscalateAI Controls & Filters")

    st.sidebar.markdown("### Manual Escalation Entry")
    customer = st.sidebar.text_input("Customer Email or Name", key="manual_customer")
    issue = st.sidebar.text_area("Issue / Complaint", key="manual_issue")
    if st.sidebar.button("Add & Analyze Manual Entry"):
        manual_entry_process(customer, issue)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Upload Complaints Excel File")
    uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx", "xls"], key="upload_excel")
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.sidebar.write("Preview:")
            st.sidebar.dataframe(df.head())
        except Exception as e:
            st.sidebar.error(f"Error reading Excel file: {e}")

        if st.sidebar.button("Analyze Uploaded Excel", key="analyze_upload"):
            count = upload_excel_and_analyze(uploaded_file)
            st.sidebar.success(f"Uploaded and analyzed {count} new complaints.")

    st.sidebar.markdown("---")

    status_filter = st.sidebar.multiselect(
        "Filter by Status",
        options=["Open", "In Progress", "Resolved", "Escalated", "All"],
        default=["Open", "In Progress", "Resolved"],
        key="status_filter"
    )

    # Priority filter for Escalated/All logic
    priority_filter = st.sidebar.multiselect(
        "Filter by Priority",
        options=["High", "Low", "All"],
        default=["High", "Low"],
        key="priority_filter"
    )

    st.sidebar.markdown("---")

    total_escalations = len(load_escalations_df())
    st.sidebar.write(f"Total escalations: {total_escalations}")

    if st.sidebar.button("Fetch New Emails"):
        emails = fetch_gmail_emails()
        if emails:
            new_count = save_emails_to_db(emails)
            st.sidebar.success(f"Fetched and saved {new_count} new emails.")
        else:
            st.sidebar.info("No new emails or error.")

    if st.sidebar.button("Check SLA Alerts"):
        alerts = check_sla_and_alert()
        if alerts > 0:
            st.sidebar.success(f"Sent {alerts} SLA breach alert(s).")
        else:
            st.sidebar.info("No SLA breaches detected.")

    if st.sidebar.button("Download Consolidated Complaints Excel"):
        filename = save_complaints_excel()
        with open(filename, "rb") as f:
            st.sidebar.download_button("Download Excel", f, file_name=filename)

    return status_filter, priority_filter

def main():
    st.title("EscalateAI - Customer Escalation Management")

    ensure_schema()  # Ensure DB schema on every run

    status_filter, priority_filter = sidebar_content()

    df = load_escalations_df()

    # Show columns for debugging - comment out in production
    # st.write("Data columns:", df.columns.tolist())

    # Filter logic: handle 'All' selections
    if "All" in status_filter or not status_filter:
        status_filter = ["Open", "In Progress", "Resolved", "Escalated"]

    if "All" in priority_filter or not priority_filter:
        priority_filter = ["High", "Low"]

    # Add a synthetic 'Escalated' status if escalation_flag==1 and status is Open or In Progress
    df['status'] = df['status'].fillna("Open")
    df['priority'] = df['priority'].fillna("Low")
    df['escalation_flag'] = df['escalation_flag'].fillna(0).astype(int)

    # We add a helper column 'display_status' for filtering and display
    df['display_status'] = df.apply(lambda r: "Escalated" if r['escalation_flag'] == 1 and r['status'] in ["Open", "In Progress"] else r['status'], axis=1)

    filtered_df = df[(df['display_status'].isin(status_filter)) & (df['priority'].isin(priority_filter))]

    # Show counts per bucket on top
    counts = {
        "Open": len(df[df['status'] == "Open"]),
        "In Progress": len(df[df['status'] == "In Progress"]),
        "Resolved": len(df[df['status'] == "Resolved"]),
        "Escalated": len(df[(df['escalation_flag'] == 1) & (df['status'].isin(["Open", "In Progress"]))])
    }

    st.markdown(f"""
        **Counts:**  
        Open: {counts['Open']} | In Progress: {counts['In Progress']} | Resolved: {counts['Resolved']} | Escalated: {counts['Escalated']}
    """)

    # Display Kanban columns
    for status in ["Open", "In Progress", "Resolved", "Escalated"]:
        st.subheader(status)
        cards = filtered_df[filtered_df['display_status'] == status]
        if cards.empty:
            st.info(f"No escalations in {status}")
        else:
            for _, row in cards.iterrows():
                display_kanban_card(row)

if __name__ == "__main__":
    main()
