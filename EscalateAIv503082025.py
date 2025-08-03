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
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load environment variables from .env file
load_dotenv()

EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")

# Define support team members for assignment
TEAM_MEMBERS = os.getenv("TEAM_MEMBERS", "Alice,Bob,Charlie").split(',')

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

# Setup database connection and create table
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
    assigned_to TEXT,
    resolution_time_minutes INTEGER,
    manual_entry INTEGER,
    user_feedback TEXT
)
""")
conn.commit()

analyzer = SentimentIntensityAnalyzer()

# Load or initialize model and vectorizer
MODEL_PATH = "escalation_model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

def load_model_and_vectorizer():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except Exception:
        return None, None

def save_model_and_vectorizer(model, vectorizer):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

model, vectorizer = load_model_and_vectorizer()

# Round-robin team assignment tracker stored in-memory (can persist to DB/file if needed)
team_assignment_counter = 0

def assign_team_member():
    global team_assignment_counter
    assigned = TEAM_MEMBERS[team_assignment_counter % len(TEAM_MEMBERS)]
    team_assignment_counter += 1
    return assigned

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
    text_lower = issue_text.lower()
    vs = analyzer.polarity_scores(issue_text)
    sentiment = "Positive" if vs["compound"] >= 0 else "Negative"
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    priority = "High" if sentiment == "Negative" and neg_count >= 2 else "Low"
    escalation_flag = 1 if priority == "High" else 0
    return sentiment, priority, escalation_flag

def predict_priority(issue_text):
    global model, vectorizer
    if model is None or vectorizer is None:
        return None
    X = vectorizer.transform([issue_text])
    pred = model.predict(X)[0]
    return pred  # Expected 'High' or 'Low'

def save_emails_to_db(emails, manual_flag=0):
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0]
    new_entries = 0
    for e in emails:
        cursor.execute("SELECT 1 FROM escalations WHERE customer=? AND issue=?", (e['customer'], e['issue'][:500]))
        if cursor.fetchone():
            continue
        count += 1
        esc_id = f"SESICE-{count+250000}"

        # Use prediction model if available, else fallback to analyze_issue
        pred_priority = predict_priority(e['issue'])
        if pred_priority:
            priority = pred_priority
            escalation_flag = 1 if priority == "High" else 0
            sentiment = "Negative" if priority == "High" else "Positive"  # Simplified for demo
        else:
            sentiment, priority, escalation_flag = analyze_issue(e['issue'])

        assigned_to = assign_team_member() if escalation_flag == 1 else ""
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
        cursor.execute("""
            INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag,
            action_taken, action_owner, status_update_date, assigned_to, resolution_time_minutes, manual_entry, user_feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (esc_id, e['customer'], e['issue'][:500], e['date'], "Open", sentiment, priority, escalation_flag,
              "", "", now, assigned_to, None, manual_flag, ""))
        new_entries += 1
        if escalation_flag == 1:
            send_ms_teams_alert(
                f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {e['customer']}\nIssue: {e['issue'][:200]}...\nAssigned to: {assigned_to}")
    conn.commit()
    return new_entries

def send_ms_teams_alert(message):
    if not MS_TEAMS_WEBHOOK_URL:
        st.warning("MS Teams webhook URL not set; cannot send alerts.")
        return
    headers = {"Content-Type": "application/json"}
    payload = {
        "text": message
    }
    try:
        response = requests.post(MS_TEAMS_WEBHOOK_URL, json=payload, headers=headers)
        if response.status_code != 200:
            st.error(f"MS Teams alert failed: {response.status_code} {response.text}")
    except Exception as e:
        st.error(f"Error sending MS Teams alert: {e}")

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

            # Use prediction model if available
            pred_priority = predict_priority(issue)
            if pred_priority:
                priority = pred_priority
                escalation_flag = 1 if priority == "High" else 0
                sentiment = "Negative" if priority == "High" else "Positive"
            else:
                sentiment, priority, escalation_flag = analyze_issue(issue)

            assigned_to = assign_team_member() if escalation_flag == 1 else ""
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("""
                INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag,
                action_taken, action_owner, status_update_date, assigned_to, resolution_time_minutes, manual_entry, user_feedback)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (esc_id, customer, issue[:500], date, "Open", sentiment, priority, escalation_flag,
                  "", "", now, assigned_to, None, 0, ""))
            count += 1
            if escalation_flag == 1:
                send_ms_teams_alert(
                    f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {customer}\nIssue: {issue[:200]}...\nAssigned to: {assigned_to}")
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

    # Use prediction model if available
    pred_priority = predict_priority(issue)
    if pred_priority:
        priority = pred_priority
        escalation_flag = 1 if priority == "High" else 0
        sentiment = "Negative" if priority == "High" else "Positive"
    else:
        sentiment, priority, escalation_flag = analyze_issue(issue)

    assigned_to = assign_team_member() if escalation_flag == 1 else ""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
    cursor.execute("""
        INSERT INTO escalations (escalation_id, customer, issue, date, status, sentiment, priority, escalation_flag,
        action_taken, action_owner, status_update_date, assigned_to, resolution_time_minutes, manual_entry, user_feedback)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (esc_id, customer, issue[:500], now, "Open", sentiment, priority, escalation_flag,
          "", "", now, assigned_to, None, 1, ""))
    conn.commit()
    st.sidebar.success(f"Added escalation {esc_id}")
    if escalation_flag == 1:
        send_ms_teams_alert(
            f"🚨 New HIGH priority escalation detected:\nID: {esc_id}\nCustomer: {customer}\nIssue: {issue[:200]}...\nAssigned to: {assigned_to}")
    return True

def update_resolution_time(escalation_id):
    now = datetime.datetime.now(datetime.timezone.utc)
    cursor.execute("SELECT date FROM escalations WHERE escalation_id=?", (escalation_id,))
    row = cursor.fetchone()
    if not row:
        return
    date_str = row[0]
    try:
        created_time = datetime.datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
    except Exception:
        return
    resolution_minutes = int((now - created_time).total_seconds() // 60)
    cursor.execute("UPDATE escalations SET resolution_time_minutes=? WHERE escalation_id=?", (resolution_minutes, escalation_id))
    conn.commit()

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
        <span style='color:{status_color}; font-weight:bold;'>{status}</span> / 
        <span style='font-size:small; color:#34495e;'>Assigned: {row.get('assigned_to', '')}</span>
    </div>
    """

    st.markdown(header_html, unsafe_allow_html=True)

    with st.expander("Details", expanded=False):
        st.markdown(f"**Customer:** {row['customer']}")
        st.markdown(f"**Issue:** {row['issue']}")
        st.markdown(f"**Date:** {row['date']}")
        st.markdown(f"**User Feedback:** {row.get('user_feedback', '')}")
        st.markdown(f"**Resolution Time (minutes):** {row.get('resolution_time_minutes', 'N/A')}")

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
        new_user_feedback = st.text_input(
            "User Feedback (e.g. correct prediction?)",
            value=row.get('user_feedback', "") or "",
            key=f"{esc_id}_feedback"
        )

        if st.button("Save Updates", key=f"save_{esc_id}"):
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")
            cursor.execute("""
                UPDATE escalations SET status=?, action_taken=?, action_owner=?, status_update_date=?, user_feedback=?
                WHERE escalation_id=?
            """, (new_status, new_action_taken, new_action_owner, now, new_user_feedback, esc_id))
            if new_status == "Resolved":
                update_resolution_time(esc_id)
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
    now = datetime.datetime.now(datetime.timezone.utc)  # timezone aware now
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
        diff = (now - last_update).total_seconds() / 60
        if diff > 10:
            send_ms_teams_alert(f"⚠️ SLA Breach Alert for escalation {row['escalation_id']} - Not resolved for {int(diff)} minutes.")
            alerts_sent += 1
    if alerts_sent > 0:
        st.info(f"{alerts_sent} SLA breach alert(s) sent.")

def train_escalation_model():
    df = load_escalations_df()
    if df.empty:
        st.warning("No data to train model.")
        return

    # We will train on issue text to predict priority (High or Low)
    df = df[df['priority'].isin(['High', 'Low'])]
    X = df['issue']
    y = df['priority']

    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_vect = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("Model trained. Classification report:")
    st.json(report)

    # Save model & vectorizer
    save_model_and_vectorizer(clf, vectorizer)
    global model, vectorizer
    model, vectorizer = clf, vectorizer

def compare_resolution_times_report():
    df = load_escalations_df()
    manual = df[df['manual_entry'] == 1]['resolution_time_minutes'].dropna()
    ai_assisted = df[df['manual_entry'] == 0]['resolution_time_minutes'].dropna()

    avg_manual = manual.mean() if not manual.empty else None
    avg_ai = ai_assisted.mean() if not ai_assisted.empty else None

    st.subheader("Resolution Time Comparison (minutes)")
    st.write(f"Average manual entry resolution time: {avg_manual:.2f}" if avg_manual else "No manual data")
    st.write(f"Average AI-assisted entry resolution time: {avg_ai:.2f}" if avg_ai else "No AI-assisted data")

def main():
    st.title("EscalateAI - AI-Powered Escalation Management")

    # Sidebar
    st.sidebar.header("Actions")

    if st.sidebar.button("Fetch & Analyze Latest Emails"):
        emails = fetch_gmail_emails()
        if emails:
            count = save_emails_to_db(emails, manual_flag=0)
            st.sidebar.success(f"Fetched & saved {count} new escalations from emails.")

    uploaded_file = st.sidebar.file_uploader("Upload Excel for Bulk Upload", type=['xls', 'xlsx'])
    if uploaded_file:
        count = upload_excel_and_analyze(uploaded_file)
        if count:
            st.sidebar.success(f"Imported {count} escalations from Excel.")

    st.sidebar.header("Manual Entry")
    cust = st.sidebar.text_input("Customer Email")
    issue = st.sidebar.text_area("Issue Description")
    if st.sidebar.button("Add Escalation Manually"):
        manual_entry_process(cust, issue)

    if st.sidebar.button("Train Escalation Prediction Model"):
        train_escalation_model()

    if st.sidebar.button("Check SLA Breaches and Send Alerts"):
        check_sla_and_alert()

    st.sidebar.header("Reports")
    if st.sidebar.button("Compare Resolution Times: Manual vs AI"):
        compare_resolution_times_report()

    df = load_escalations_df()

    filter_option = st.radio("Filter Escalations:", ["All", "Escalated Only"])

    if filter_option == "Escalated Only":
        df = df[df['escalation_flag'] == 1]

    status_counts = df['status'].value_counts().to_dict()
    open_count = status_counts.get("Open", 0)
    inprogress_count = status_counts.get("In Progress", 0)
    resolved_count = status_counts.get("Resolved", 0)

    # Kanban board columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### 🟡 Open ({open_count})")
        for _, row in df[df['status'] == "Open"].iterrows():
            display_kanban_card(row)

    with col2:
        st.markdown(f"### 🔵 In Progress ({inprogress_count})")
        for _, row in df[df['status'] == "In Progress"].iterrows():
            display_kanban_card(row)

    with col3:
        st.markdown(f"### 🟢 Resolved ({resolved_count})")
        for _, row in df[df['status'] == "Resolved"].iterrows():
            display_kanban_card(row)

if __name__ == "__main__":
    main()
