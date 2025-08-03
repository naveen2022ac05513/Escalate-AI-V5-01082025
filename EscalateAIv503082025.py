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
from streamlit_autorefresh import st_autorefresh
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import smtplib
from email.mime.text import MIMEText

# Load environment variables from .env file
load_dotenv()

EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")

EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", 587))
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")  # change default in .env!

NEGATIVE_KEYWORDS = [
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
    "impact", "loss", "risk", "downtime", "interrupt", "cancel",
    "terminate", "penalty"
]

# Database Setup
conn = sqlite3.connect("escalations.db", check_same_thread=False)
cursor = conn.cursor()

# Create or update table with user_feedback column if missing
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
    user_feedback INTEGER DEFAULT 0
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

def send_email_alert(subject, message):
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        st.warning("Email sender/receiver credentials not set.")
        return

    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER

    try:
        server = smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, [EMAIL_RECEIVER], msg.as_string())
        server.quit()
        st.info("Email alert sent.")
    except Exception as e:
        st.error(f"Failed to send email alert: {e}")

def send_alerts(message):
    send_ms_teams_alert(message)
    send_email_alert("EscalateAI Alert", message)

def analyze_issue(issue_text, use_ml=True):
    text_lower = issue_text.lower()
    vs = analyzer.polarity_scores(issue_text)
    sentiment = "Positive" if vs["compound"] >= 0 else "Negative"

    if use_ml:
        model, vectorizer = load_predictive_model()
        if model and vectorizer:
            priority = predict_priority(issue_text, sentiment, model, vectorizer)
            escalation_flag = 1 if priority == "High" else 0
            return sentiment, priority, escalation_flag

    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    priority = "High" if sentiment == "Negative" and neg_count >= 2 else "Low"
    escalation_flag = 1 if priority == "High" else 0
    return sentiment, priority, escalation_flag

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
        sentiment, priority, escalation_flag = analyze_issue(e['issue'])
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
        cursor.execute("""
            INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, action_taken, action_owner, status_update_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (esc_id, e['customer'], e['issue'][:500], e['date'], "Open", sentiment, priority, escalation_flag, "", "", now))
        new_entries += 1
        if escalation_flag == 1:
            send_alerts(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {e['customer']}\nIssue: {e['issue'][:200]}...")
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
            sentiment, priority, escalation_flag = analyze_issue(issue)
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("""
                INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, action_taken, action_owner, status_update_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (esc_id, customer, issue[:500], date, "Open", sentiment, priority, escalation_flag, "", "", now))
            count += 1
            if escalation_flag == 1:
                send_alerts(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {customer}\nIssue: {issue[:200]}...")
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
    sentiment, priority, escalation_flag = analyze_issue(issue)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
    cursor.execute("""
        INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, action_taken, action_owner, status_update_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (esc_id, customer, issue[:500], now, "Open", sentiment, priority, escalation_flag, "", "", now))
    conn.commit()
    st.sidebar.success(f"Added escalation {esc_id}")
    if escalation_flag == 1:
        send_alerts(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {customer}\nIssue: {issue[:200]}...")
    return True

def display_kanban_card(row):
    esc_id = row['escalation_id']
    sentiment = row['sentiment']
    priority = row['priority']
    status = row['status']

    sentiment_colors = {"Positive": "#27ae60", "Negative": "#e74c3c"}
    priority_colors = {"High": "#c0392b", "Low": "#27ae60"}
    status_colors = {"Open": "#f1c40f", "In Progress": "#2980b9", "Resolved": "#2ecc71"}

    border_color = priority_colors.get(priority, "#000000")
    status_color = status_colors.get(status, "#bdc3c7")
    sentiment_color = sentiment_colors.get(sentiment, "#7f8c8d")

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

    st.markdown(header_html, unsafe_allow_html=True)

    with st.expander("Details", expanded=False):
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
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("""
                UPDATE escalations SET status=?, action_taken=?, action_owner=?, status_update_date=?
                WHERE escalation_id=?
            """, (new_status, new_action_taken, new_action_owner, now, esc_id))
            conn.commit()
            st.success("Updated successfully!")
            st.experimental_rerun()

        # User feedback checkbox for continuous learning
        feedback_key = f"{esc_id}_feedback"
        feedback_correct = st.checkbox("Is the priority classification correct?", key=feedback_key)
        if feedback_correct:
            cursor.execute("UPDATE escalations SET user_feedback=1 WHERE escalation_id=?", (esc_id,))
            conn.commit()
            st.success("Thanks for your feedback!")

def render_kanban():
    st.title("🚀 EscalateAI - Escalations & Complaints Kanban Board")

    search_text = st.text_input("Search by customer or issue text")

    df = load_escalations_df()

    if search_text:
        df = df[df.apply(lambda row: search_text.lower() in str(row['customer']).lower() or search_text.lower() in str(row['issue']).lower(), axis=1)]

    filter_choice = st.radio("Filter Escalations:", ["All", "Escalated Only"])

    if filter_choice == "Escalated Only":
        df = df[df['escalation_flag'] == 1]

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

def save_complaints_excel():
    df = load_escalations_df()
    filename = "complaints_data.xlsx"
    df.to_excel(filename, index=False)
    return filename

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
        if elapsed.total_seconds() > 10 * 60:  # 10 minutes SLA for testing
            send_alerts(
                f"⚠️ SLA breach detected:\nID: {row['escalation_id']}\nCustomer: {row['customer']}\nOpen for: {elapsed.seconds // 60} minutes\nIssue: {row['issue'][:200]}..."
            )
            alerts_sent += 1
    return alerts_sent

# Predictive Model Functions

def train_predictive_model():
    df = load_escalations_df()
    df = df[df['priority'].isin(['High', 'Low'])]

    if len(df) < 20:
        st.warning("Not enough data to train predictive model yet.")
        return None, None

    texts = df['issue'].astype(str).tolist()
    sentiments = [1 if s == "Positive" else 0 for s in df['sentiment']]
    priorities = [1 if p == "High" else 0 for p in df['priority']]

    vectorizer = TfidfVectorizer(max_features=500)
    X_text = vectorizer.fit_transform(texts)

    X = np.hstack([X_text.toarray(), np.array(sentiments).reshape(-1, 1)])

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, priorities)

    joblib.dump((model, vectorizer), "predictive_model.joblib")
    st.success("Predictive model trained and saved.")
    return model, vectorizer

def load_predictive_model():
    if os.path.exists("predictive_model.joblib"):
        model, vectorizer = joblib.load("predictive_model.joblib")
        return model, vectorizer
    return None, None

def predict_priority(issue_text, sentiment, model, vectorizer):
    if model is None or vectorizer is None:
        return "Low"
    X_text = vectorizer.transform([issue_text])
    sentiment_val = 1 if sentiment == "Positive" else 0
    X = np.hstack([X_text.toarray(), np.array([[sentiment_val]])])
    pred = model.predict(X)[0]
    return "High" if pred == 1 else "Low"

# Main Streamlit App
def main():
    st.sidebar.title("EscalateAI Controls")

    # Admin password section for advanced controls
    password = st.sidebar.text_input("Admin Password", type="password")
    admin_mode = password == ADMIN_PASSWORD

    st.sidebar.markdown("---")

    # Manual entry
    st.sidebar.header("Manual Entry")
    manual_customer = st.sidebar.text_input("Customer / Email")
    manual_issue = st.sidebar.text_area("Issue / Complaint")
    if st.sidebar.button("Add Manual Escalation"):
        manual_entry_process(manual_customer, manual_issue)

    # Bulk upload Excel
    st.sidebar.header("Bulk Upload")
    uploaded_file = st.sidebar.file_uploader("Upload Excel file with escalations", type=["xlsx"])
    if uploaded_file:
        count = upload_excel_and_analyze(uploaded_file)
        if count > 0:
            st.sidebar.success(f"Uploaded and added {count} new escalations.")
        else:
            st.sidebar.info("No new escalations found or uploaded.")

    # Export data
    st.sidebar.header("Export Data")
    if st.sidebar.button("Download Escalations Excel"):
        filename = save_complaints_excel()
        with open(filename, "rb") as f:
            st.sidebar.download_button("Download Excel File", f, file_name=filename)

    if st.sidebar.button("Download Escalations CSV"):
        df = load_escalations_df()
        csv_data = df.to_csv(index=False)
        st.sidebar.download_button("Download CSV File", data=csv_data, file_name="escalations.csv", mime="text/csv")

    st.sidebar.markdown("---")

    # Train predictive model - admin only
    if admin_mode:
        st.sidebar.header("Admin Controls")
        if st.sidebar.button("Train Predictive Model"):
            train_predictive_model()

    # Auto-refresh and fetch emails every 60 seconds
    count = st_autorefresh(interval=60000, limit=None, key="email_autorefresh")
    if count > 0:
        emails = fetch_gmail_emails()
        if emails:
            new_count = save_emails_to_db(emails)
            if new_count > 0:
                st.sidebar.success(f"Auto-fetched and saved {new_count} new emails.")

    # Check SLA and alert (run every page load)
    alert_count = check_sla_and_alert()
    if alert_count > 0:
        st.sidebar.warning(f"{alert_count} SLA breaches detected and alerted.")

    # Show Kanban board
    render_kanban()

if __name__ == "__main__":
    main()
