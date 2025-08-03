import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import sqlite3
import os
import pickle
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")

NEGATIVE_KEYWORDS = [
    "fail", "break", "crash", "defect", "fault", "degrade", "damage",
    "trip", "malfunction", "blank", "shutdown", "discharge",
    "dissatisfy", "frustrate", "complain", "reject", "delay", "ignore",
    "escalate", "displease", "noncompliance", "neglect",
    "wait", "pending", "slow", "incomplete", "miss", "omit",
    "unresolved", "shortage", "no response",
    "fire", "burn", "flashover", "arc", "explode", "unsafe",
    "leak", "corrode", "alarm", "incident",
    "impact", "loss", "risk", "downtime", "interrupt", "cancel",
    "terminate", "penalty"
]

# Setup DB connection & table
conn = sqlite3.connect("escalations.db", check_same_thread=False)
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
    action_owner TEXT,
    status_update_date TEXT,
    created_at TEXT,
    resolved_at TEXT,
    added_by TEXT,  -- "auto" or "manual"
    user_feedback INTEGER  -- 1=correct, 0=incorrect, NULL=no feedback
)
""")
conn.commit()

analyzer = SentimentIntensityAnalyzer()

MODEL_FILE = "escalation_predictor.pkl"

# Load or create model pipeline
def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        return model
    else:
        # Dummy model with no training, just predict no escalation
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=500)),
            ('clf', LogisticRegression())
        ])
        return pipe

model = load_model()

def save_model(model):
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

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
    return sentiment, priority, escalation_flag

def predict_escalation(issue_text):
    # Use ML model to predict escalation risk: returns 1 or 0
    pred = model.predict([issue_text])[0]
    return int(pred)

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
        sentiment, priority, keyword_flag = analyze_issue(e['issue'])
        # Also use ML model prediction to flag escalation
        ml_flag = predict_escalation(e['issue'])
        escalation_flag = max(keyword_flag, ml_flag)
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
        cursor.execute("""
            INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, action_taken, action_owner, status_update_date, created_at, resolved_at, added_by, user_feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (esc_id, e['customer'], e['issue'][:500], e['date'], "Open", sentiment, priority, escalation_flag, "", "", now, now, None, "auto", None))
        new_entries += 1
        if escalation_flag == 1:
            send_ms_teams_alert(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {e['customer']}\nIssue: {e['issue'][:200]}...")
    conn.commit()
    return new_entries

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

def load_escalations_df():
    df = pd.read_sql_query("SELECT * FROM escalations ORDER BY status_update_date DESC", conn)
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

        for _, row in df.iterrows():
            customer = str(row[customer_col])
            issue = str(row[issue_col])
            date = str(row[date_col]) if date_col and pd.notna(row[date_col]) else datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (customer, issue[:500]))
            if cursor.fetchone():
                continue
            existing_count += 1
            esc_id = f"SESICE-{existing_count+250000}"
            sentiment, priority, keyword_flag = analyze_issue(issue)
            ml_flag = predict_escalation(issue)
            escalation_flag = max(keyword_flag, ml_flag)
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("""
                INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, action_taken, action_owner, status_update_date, created_at, resolved_at, added_by, user_feedback)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (esc_id, customer, issue[:500], date, "Open", sentiment, priority, escalation_flag, "", "", now, now, None, "auto", None))
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
    sentiment, priority, keyword_flag = analyze_issue(issue)
    ml_flag = predict_escalation(issue)
    escalation_flag = max(keyword_flag, ml_flag)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
    cursor.execute("""
        INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag, action_taken, action_owner, status_update_date, created_at, resolved_at, added_by, user_feedback)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (esc_id, customer, issue[:500], now, "Open", sentiment, priority, escalation_flag, "", "", now, now, None, "manual", None))
    conn.commit()
    st.sidebar.success(f"Added escalation {esc_id}")
    if escalation_flag == 1:
        send_ms_teams_alert(f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {customer}\nIssue: {issue[:200]}...")
    return True

def display_kanban_card(row):
    # Use .get with default values to avoid KeyError
    esc_id = row.get('escalation_id', 'N/A')
    sentiment = row.get('sentiment', 'Unknown')
    priority = row.get('priority', 'Unknown')
    status = row.get('status', 'Unknown')
    created_at = row.get('created_at', 'N/A')
    resolved_at = row.get('resolved_at', 'N/A')
    added_by = row.get('added_by', 'N/A')
    feedback = row.get('user_feedback', "")

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
        st.markdown(f"**Customer:** {row.get('customer', 'N/A')}")
        st.markdown(f"**Issue:** {row.get('issue', 'N/A')}")
        st.markdown(f"**Date:** {row.get('date', 'N/A')}")
        st.markdown(f"**Created At:** {created_at}")
        st.markdown(f"**Resolved At:** {resolved_at}")
        st.markdown(f"**Added By:** {added_by}")
        st.markdown(f"**User Feedback:** {feedback}")

        new_status = st.selectbox(
            "Update Status",
            ["Open", "In Progress", "Resolved"],
            index=["Open", "In Progress", "Resolved"].index(status) if status in ["Open", "In Progress", "Resolved"] else 0,
            key=f"{esc_id}_status"
        )
        new_action_taken = st.text_area(
            "Action Taken",
            value=row.get('action_taken', ""),
            key=f"{esc_id}_action"
        )
        new_action_owner = st.text_input(
            "Action Owner",
            value=row.get('action_owner', ""),
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


        # Feedback on escalation prediction correctness
        feedback_options = {
            None: "No feedback",
            1: "Prediction correct",
            0: "Prediction incorrect"
        }
        current_feedback = feedback if feedback in [0,1] else None
        feedback_val = st.radio(
            "Is the escalation prediction correct?",
            options=[None, 1, 0],
            format_func=lambda x: feedback_options[x],
            index=[None, 1, 0].index(current_feedback) if current_feedback is not None else 0,
            key=f"{esc_id}_feedback"
        )

        if st.button("Save Updates", key=f"save_{esc_id}"):
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            resolved_time = row['resolved_at']
            if new_status == "Resolved" and not resolved_time:
                resolved_time = now
            elif new_status != "Resolved":
                resolved_time = None
            cursor.execute("""
                UPDATE escalations SET status=?, action_taken=?, action_owner=?, status_update_date=?, resolved_at=?, user_feedback=?
                WHERE escalation_id=?
            """, (new_status, new_action_taken, new_action_owner, now, resolved_time, feedback_val, esc_id))
            conn.commit()
            st.success("Updated successfully!")
            st.experimental_rerun()

def save_complaints_excel():
    df = load_escalations_df()
    filename = "complaints_data.xlsx"
    df.to_excel(filename, index=False)
    return filename

def check_sla_and_alert():
    # SLA breach if high priority & Open status for >10 minutes (testing)
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
        if elapsed.total_seconds() > 10 * 60:  # 10 mins for testing
            send_ms_teams_alert(
                f"⚠️ SLA breach detected:\nID: {row['escalation_id']}\nCustomer: {row['customer']}\nOpen for: {elapsed.seconds // 60} minutes\nIssue: {row['issue'][:200]}..."
            )
            alerts_sent += 1
    return alerts_sent

def retrain_model():
    df = load_escalations_df()
    # Use only rows with user feedback for supervised training
    train_df = df[df['user_feedback'].notnull()]
    if train_df.empty:
        st.warning("No labeled data available for training.")
        return None

    X = train_df['issue']
    y = train_df['escalation_flag']  # Use actual escalation_flag as label

    # Train pipeline
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=500)),
        ('clf', LogisticRegression(max_iter=500))
    ])
    pipe.fit(X, y)
    save_model(pipe)
    st.success(f"Model retrained on {len(train_df)} labeled examples.")
    return pipe

def render_kanban():
    st.header("Escalations Kanban Board")

    df = load_escalations_df()
    statuses = ["Open", "In Progress", "Resolved"]

    cols = st.columns(len(statuses))
    for i, status in enumerate(statuses):
        with cols[i]:
            st.subheader(status)
            for idx, row in df[df['status'] == status].iterrows():
                display_kanban_card(row)

def main():
    st.title("EscalateAI - AI-Powered Escalation Management")

    menu = ["Home", "Upload Excel", "Add Escalation Manually", "Retrain Model", "Export Data"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Fetch latest emails and analyze")
        if st.button("Fetch & Process New Emails"):
            emails = fetch_gmail_emails()
            count = save_emails_to_db(emails)
            st.success(f"Processed {count} new escalations from email.")
        if st.button("Check SLA Breaches and Send Alerts"):
            alert_count = check_sla_and_alert()
            st.info(f"SLA alerts sent: {alert_count}")

        render_kanban()

    elif choice == "Upload Excel":
        st.subheader("Upload Excel file with escalations")
        uploaded_file = st.file_uploader("Upload Excel", type=["xls", "xlsx"])
        if uploaded_file:
            count = upload_excel_and_analyze(uploaded_file)
            st.success(f"Added {count} escalations from Excel.")

    elif choice == "Add Escalation Manually":
        st.subheader("Manual Escalation Entry")
        customer = st.text_input("Customer Email/Name")
        issue = st.text_area("Issue Description")
        if st.button("Add Escalation"):
            manual_entry_process(customer, issue)

    elif choice == "Retrain Model":
        st.subheader("Retrain Escalation Prediction Model")
        st.info("Model retrains on escalations with user feedback labels.")
        if st.button("Retrain Now"):
            new_model = retrain_model()
            if new_model:
                global model
                model = new_model

    elif choice == "Export Data":
        st.subheader("Export escalations data")
        filename = save_complaints_excel()
        st.markdown(f"[Download Excel file](./{filename})")

if __name__ == "__main__":
    main()
