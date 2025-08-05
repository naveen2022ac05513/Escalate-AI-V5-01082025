import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import datetime
import time
import os
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client  # WhatsApp notifications
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# --- Constants ---
DB_PATH = "escalate_ai.db"
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_PASS")
MS_TEAMS_WEBHOOK = os.getenv("MS_TEAMS_WEBHOOK")
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")  # For alert emails

# --- Initialize DB connection ---
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# --- Create necessary tables if not exist ---
cursor.execute("""
CREATE TABLE IF NOT EXISTS issues (
    id TEXT PRIMARY KEY,
    timestamp TEXT,
    customer_email TEXT,
    issue_text TEXT,
    severity TEXT,
    criticality TEXT,
    category TEXT,
    sentiment REAL,
    urgency INTEGER,
    escalation_flag INTEGER,
    status TEXT,
    action_taken TEXT,
    action_owner TEXT,
    last_update TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_id TEXT,
    field_changed TEXT,
    old_value TEXT,
    new_value TEXT,
    changed_by TEXT,
    timestamp TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_id TEXT,
    parameter TEXT,
    feedback TEXT,
    timestamp TEXT
)
""")
import sqlite3

def init_db():
    conn = sqlite3.connect('escalate_ai.db')
    c = conn.cursor()
    # Create table if not exists
    c.execute('''
        CREATE TABLE IF NOT EXISTS escalations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            issue_id TEXT UNIQUE,
            customer TEXT,
            issue_desc TEXT,
            status TEXT,
            severity TEXT,
            criticality TEXT,
            category TEXT,
            escalation_flag INTEGER,
            feedback TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn

# --- Sentiment analyzer setup ---
sia = SentimentIntensityAnalyzer()

# --- Negative words list (expanded as per your categories) ---
NEGATIVE_WORDS = {
    # Technical Failures & Product Malfunction
    "fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip", "malfunction", "blank", "shutdown", "discharge",
    # Customer Dissatisfaction & Escalations
    "dissatisfy", "frustrate", "complain", "reject", "delay", "ignore", "escalate", "displease", "noncompliance", "neglect",
    # Support Gaps & Operational Delays
    "wait", "pending", "slow", "incomplete", "miss", "omit", "unresolved", "shortage", "no response",
    # Hazardous Conditions & Safety Risks
    "fire", "burn", "flashover", "arc", "explode", "unsafe", "leak", "corrode", "alarm", "incident",
    # Business Risk & Impact
    "impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate", "penalty"
}

# --- Utility functions ---

def generate_issue_id():
    cursor.execute("SELECT COUNT(*) FROM issues")
    count = cursor.fetchone()[0] + 1
    return f"SESICE-{2500000 + count}"

def extract_features(issue_text):
    # Simple heuristic features: sentiment, negative word count
    sentiment = sia.polarity_scores(issue_text)['compound']
    negative_count = sum(word in issue_text.lower() for word in NEGATIVE_WORDS)
    urgency = 1 if sentiment < -0.3 or negative_count > 2 else 0
    escalation_flag = 1 if urgency and negative_count > 3 else 0
    # Severity, criticality, category can be heuristically assigned or ML predicted
    severity = "High" if escalation_flag else "Low"
    criticality = "Critical" if "safety" in issue_text.lower() or "fire" in issue_text.lower() else "Normal"
    category = "Technical" if any(word in issue_text.lower() for word in NEGATIVE_WORDS) else "General"
    return sentiment, urgency, escalation_flag, severity, criticality, category

def insert_issue(issue):
    cursor.execute("""
    INSERT INTO issues (id, timestamp, customer_email, issue_text, severity, criticality, category,
                        sentiment, urgency, escalation_flag, status, action_taken, action_owner, last_update)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, issue)
    conn.commit()

def update_issue_field(issue_id, field, new_value, changed_by="System"):
    # Fetch old value
    cursor.execute(f"SELECT {field} FROM issues WHERE id = ?", (issue_id,))
    old_value = cursor.fetchone()[0]
    if old_value == new_value:
        return False  # No change
    cursor.execute(f"UPDATE issues SET {field} = ?, last_update = ? WHERE id = ?", (new_value, datetime.datetime.now().isoformat(), issue_id))
    # Log audit
    cursor.execute("""
    INSERT INTO audit_log (issue_id, field_changed, old_value, new_value, changed_by, timestamp)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (issue_id, field, str(old_value), str(new_value), changed_by, datetime.datetime.now().isoformat()))
    conn.commit()
    return True

def fetch_all_issues():
    df = pd.read_sql_query("SELECT * FROM issues", conn)
    return df

def send_ms_teams_alert(issue):
    message = {
        "text": f"🚨 Escalation Alert 🚨\nID: {issue['id']}\nSeverity: {issue['severity']}\nCustomer: {issue['customer_email']}\nIssue: {issue['issue_text'][:100]}..."
    }
    try:
        response = requests.post(MS_TEAMS_WEBHOOK, json=message)
        return response.status_code == 200
    except Exception as e:
        st.error(f"MS Teams alert failed: {e}")
        return False

def send_email(to_email, subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASS)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Email sending failed: {e}")
        return False

def send_whatsapp(to_number, message):
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=message,
            from_=f'whatsapp:{TWILIO_WHATSAPP_FROM}',
            to=f'whatsapp:{to_number}'
        )
        return True
    except Exception as e:
        st.error(f"WhatsApp sending failed: {e}")
        return False

# --- ML Model & Retraining ---

def train_ml_model():
    df = fetch_all_issues()
    if df.empty or 'escalation_flag' not in df.columns:
        return None, None
    X = df[['sentiment', 'urgency']]
    y = df['escalation_flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report = classification_report(y_test, preds)
    accuracy = accuracy_score(y_test, preds)
    return model, (report, accuracy)

def predict_escalation(model, issue_text):
    sentiment = sia.polarity_scores(issue_text)['compound']
    negative_count = sum(word in issue_text.lower() for word in NEGATIVE_WORDS)
    urgency = 1 if sentiment < -0.3 or negative_count > 2 else 0
    if model:
        pred = model.predict(np.array([[sentiment, urgency]]))[0]
    else:
        pred = 0
    return pred

# --- Feedback NLP parsing ---
def parse_feedback_text(feedback_text):
    # Simple sentiment + keyword extraction for demo
    sentiment = sia.polarity_scores(feedback_text)['compound']
    contains_negative = any(word in feedback_text.lower() for word in NEGATIVE_WORDS)
    return {
        "sentiment": sentiment,
        "contains_negative_words": contains_negative
    }

# --- Dashboard summary ---
def draw_dashboard(df):
    st.subheader("EscalateAI Dashboard Summary")
    total_issues = len(df)
    escalated = df[df['escalation_flag'] == 1]
    escalated_count = len(escalated)
    sla_breaches = len(df[(df['status'] != "Resolved") & (pd.to_datetime(df['timestamp']) < (datetime.datetime.now() - datetime.timedelta(minutes=10)))])
    st.metric("Total Issues Logged", total_issues)
    st.metric("Total Escalated", escalated_count)
    st.metric("SLA Breaches", sla_breaches)
    # Severity distribution chart
    severity_counts = df['severity'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(severity_counts.index, severity_counts.values, color='orange')
    ax.set_title("Severity Distribution")
    st.pyplot(fig)

# --- Main Streamlit UI ---

def main():
    st.set_page_config(page_title="EscalateAI vFinal", layout="wide", initial_sidebar_state="expanded")

    st.title("EscalateAI — Customer Escalation & AI Management")

    # --- Sidebar ---
    st.sidebar.header("Controls & Filters")

    # Show count summary for escalated cases in sidebar
    all_issues = fetch_all_issues()
    escalated_issues = all_issues[all_issues['escalation_flag'] == 1] if not all_issues.empty else pd.DataFrame()
    status_counts = escalated_issues['status'].value_counts() if not escalated_issues.empty else pd.Series()
    st.sidebar.markdown("### Escalated Cases Status Summary")
    if not status_counts.empty:
        for status, count in status_counts.items():
            st.sidebar.write(f"- **{status}**: {count}")
    else:
        st.sidebar.write("No escalated cases found.")

    # Search & Filter input for Kanban
    search_term = st.sidebar.text_input("Search issues (text or email):").strip()

    # Status filter for Kanban view
    kanban_status_filter = st.sidebar.selectbox("Filter by Status:", options=["All", "Open", "In Progress", "Resolved", "Escalated"])

    # Buttons for Alerts
    if st.sidebar.button("Send MS Teams Alerts"):
        if escalated_issues.empty:
            st.sidebar.warning("No escalated issues to alert.")
        else:
            successes = 0
            for _, issue in escalated_issues.iterrows():
                if send_ms_teams_alert(issue):
                    successes += 1
            st.sidebar.success(f"MS Teams alerts sent for {successes} issues.")

    if st.sidebar.button("Send Email Alerts"):
        if escalated_issues.empty:
            st.sidebar.warning("No escalated issues to alert.")
        else:
            successes = 0
            for _, issue in escalated_issues.iterrows():
                body = f"Escalated issue detected:\nID: {issue['id']}\nCustomer: {issue['customer_email']}\nIssue: {issue['issue_text'][:100]}"
                if send_email(issue['customer_email'], "Escalation Alert from EscalateAI", body):
                    successes += 1
            st.sidebar.success(f"Email alerts sent for {successes} issues.")

    # Excel Upload (existing functionality retained)
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader("Upload Issues Excel", type=["xlsx"])
    if uploaded_file:
        try:
            df_upload = pd.read_excel(uploaded_file)
            for _, row in df_upload.iterrows():
                issue_text = row.get("issue_text", "")
                customer_email = row.get("customer_email", "")
                sentiment, urgency, escalation_flag, severity, criticality, category = extract_features(issue_text)
                issue_id = generate_issue_id()
                issue = (
                    issue_id,
                    datetime.datetime.now().isoformat(),
                    customer_email,
                    issue_text,
                    severity,
                    criticality,
                    category,
                    sentiment,
                    urgency,
                    escalation_flag,
                    "Open",
                    "",
                    "",
                    datetime.datetime.now().isoformat()
                )
                insert_issue(issue)
            st.sidebar.success("Issues uploaded successfully.")
        except Exception as e:
            st.sidebar.error(f"Failed to upload Excel: {e}")

    # --- Tabs for main content ---
    tabs = st.tabs(["Kanban Board", "Escalated Cases", "Dashboard", "Feedback & Retraining", "Audit Log"])

    # --- Tab 1: Kanban Board ---
    with tabs[0]:
        st.header("Kanban Board")
        df_kanban = all_issues.copy()

        # Apply search & filter
        if search_term:
            df_kanban = df_kanban[
                df_kanban['issue_text'].str.contains(search_term, case=False, na=False) |
                df_kanban['customer_email'].str.contains(search_term, case=False, na=False) |
                df_kanban['id'].str.contains(search_term, case=False, na=False)
            ]
        if kanban_status_filter != "All":
            df_kanban = df_kanban[df_kanban['status'] == kanban_status_filter]

        # Display Kanban columns
        statuses = ["Open", "In Progress", "Resolved", "Escalated"]
        cols = st.columns(len(statuses))
        for idx, status in enumerate(statuses):
            with cols[idx]:
                st.subheader(status)
                status_issues = df_kanban[df_kanban['status'] == status]
                for _, row in status_issues.iterrows():
                    with st.expander(f"{row['id']} | {row['customer_email']}"):
                        st.write(f"**Issue:** {row['issue_text']}")
                        st.write(f"**Severity:** {row['severity']}")
                        st.write(f"**Criticality:** {row['criticality']}")
                        st.write(f"**Category:** {row['category']}")
                        st.write(f"**Sentiment:** {row['sentiment']:.2f}")
                        st.write(f"**Urgency:** {row['urgency']}")
                        st.write(f"**Escalation:** {'Yes' if row['escalation_flag'] else 'No'}")

                        # Editable fields: status, action_taken, action_owner
                        new_status = st.selectbox(f"Change Status (current: {row['status']})", options=statuses, key=f"status_{row['id']}")
                        action_taken = st.text_input("Action Taken", value=row['action_taken'] or "", key=f"action_{row['id']}")
                        action_owner = st.text_input("Action Owner", value=row['action_owner'] or "", key=f"owner_{row['id']}")

                        if st.button("Save Changes", key=f"save_{row['id']}"):
                            changed = False
                            if new_status != row['status']:
                                changed |= update_issue_field(row['id'], 'status', new_status)
                                # Notify customer on status change
                                notify_body = f"Your issue {row['id']} status has been updated to '{new_status}'. If you have questions, reply to this email."
                                send_email(row['customer_email'], f"Issue {row['id']} Status Update", notify_body)
                                # You can add WhatsApp notify here as well, e.g. send_whatsapp(...)
                            if action_taken != row['action_taken']:
                                changed |= update_issue_field(row['id'], 'action_taken', action_taken)
                            if action_owner != row['action_owner']:
                                changed |= update_issue_field(row['id'], 'action_owner', action_owner)
                            if changed:
                                st.success(f"Issue {row['id']} updated.")
                            else:
                                st.info("No changes detected.")

    # --- Tab 2: Escalated Cases Table ---
    with tabs[1]:
        st.header("Escalated Cases")
        escalated_df = all_issues[all_issues['escalation_flag'] == 1]
        if not escalated_df.empty:
            st.dataframe(escalated_df)
        else:
            st.info("No escalated cases to show.")

    # --- Tab 3: Dashboard ---
    with tabs[2]:
        draw_dashboard(all_issues)

    # --- Tab 4: Feedback & Retraining ---
    with tabs[3]:
        st.header("Feedback & Retraining")
        # Select issue for feedback
        #issue_ids = all_issues['id'].tolist() if not all_issues.empty else []
        #$selected_issue_id = st.selectbox("Select Issue ID for Feedback", options=["
        selected_issue_id = st.selectbox("Select Issue ID for Feedback", options=[
            "SESICE-2500001",
            "SESICE-2500002",
            "SESICE-2500003"
        ])

                                                                                  
# ========== CONTINUED: Kanban Board UI and Editing ==========

def update_issue_status(db_conn, issue_id, new_status):
    cursor = db_conn.cursor()
    cursor.execute("UPDATE issues SET status = ? WHERE id = ?", (new_status, issue_id))
    db_conn.commit()

def update_issue_action_taken(db_conn, issue_id, action_taken):
    cursor = db_conn.cursor()
    cursor.execute("UPDATE issues SET action_taken = ? WHERE id = ?", (action_taken, issue_id))
    db_conn.commit()

def update_issue_action_owner(db_conn, issue_id, action_owner):
    cursor = db_conn.cursor()
    cursor.execute("UPDATE issues SET action_owner = ? WHERE id = ?", (action_owner, issue_id))
    db_conn.commit()

def insert_audit_log(db_conn, issue_id, user, action_desc):
    cursor = db_conn.cursor()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO audit_log (issue_id, user, action, timestamp) VALUES (?, ?, ?, ?)",
        (issue_id, user, action_desc, timestamp)
    )
    db_conn.commit()

def send_status_change_notification(issue, new_status):
    # Send notification via Email or WhatsApp based on available contact info
    msg = f"Dear {issue['customer_name']},\n\nYour issue ({issue['issue_id']}) status has been updated to '{new_status}'.\n\nThank you."
    if issue.get('email'):
        send_email(issue['email'], f"Issue {issue['issue_id']} Status Update", msg)
    if issue.get('whatsapp'):
        send_whatsapp_message(issue['whatsapp'], msg)

def kanban_board_ui(db_conn):
    st.header("Issue Kanban Board")

    # Fetch issues from DB
    df = get_issues(db_conn)

    # Filters
    status_filter = st.sidebar.multiselect(
        "Filter by Status",
        options=['Open', 'In Progress', 'Resolved', 'Escalated'],
        default=['Open', 'In Progress', 'Resolved', 'Escalated']
    )
    filtered_df = df[df['status'].isin(status_filter)]

    search_term = st.sidebar.text_input("Search by Customer/Account/Issue")

    if search_term:
        filtered_df = filtered_df[
            filtered_df.apply(lambda row: search_term.lower() in str(row['customer_name']).lower()
                              or search_term.lower() in str(row['account']).lower()
                              or search_term.lower() in str(row['issue_description']).lower(), axis=1)
        ]

    # Group by Status for Kanban columns
    statuses = ['Open', 'In Progress', 'Resolved', 'Escalated']
    columns = st.columns(len(statuses))

    for idx, status in enumerate(statuses):
        with columns[idx]:
            st.subheader(f"{status} ({len(filtered_df[filtered_df['status'] == status])})")
            status_issues = filtered_df[filtered_df['status'] == status]

            for _, issue in status_issues.iterrows():
                with st.expander(f"{issue['issue_id']}: {issue['issue_description'][:40]}..."):
                    st.write(f"**Customer:** {issue['customer_name']}")
                    st.write(f"**Account:** {issue['account']}")
                    st.write(f"**Severity:** {issue['severity']}")
                    st.write(f"**Criticality:** {issue['criticality']}")
                    st.write(f"**Category:** {issue['category']}")
                    st.write(f"**Sentiment Score:** {issue['sentiment_score']:.2f}")
                    st.write(f"**Urgency:** {issue['urgency']}")
                    st.write(f"**Escalation Flag:** {issue['escalation_flag']}")
                    st.write(f"**Action Taken:** {issue['action_taken']}")
                    st.write(f"**Action Owner:** {issue['action_owner']}")
                    st.write(f"**Created At:** {issue['created_at']}")
                    st.write(f"**Last Updated:** {issue['last_updated']}")

                    # Editable fields
                    new_status = st.selectbox(f"Change Status ({issue['issue_id']})", statuses, index=statuses.index(issue['status']))
                    new_action_taken = st.text_area(f"Action Taken ({issue['issue_id']})", issue['action_taken'] or "")
                    new_action_owner = st.text_input(f"Action Owner ({issue['issue_id']})", issue['action_owner'] or "")

                    if st.button(f"Update Issue {issue['issue_id']}", key=f"update_{issue['issue_id']}"):
                        update_issue_status(db_conn, issue['id'], new_status)
                        update_issue_action_taken(db_conn, issue['id'], new_action_taken)
                        update_issue_action_owner(db_conn, issue['id'], new_action_owner)
                        insert_audit_log(db_conn, issue['id'], "User", f"Status changed to {new_status}")
                        send_status_change_notification(issue, new_status)
                        st.success(f"Issue {issue['issue_id']} updated successfully.")
                        st.experimental_rerun()

def feedback_and_retraining_ui(db_conn):
    st.header("Feedback & Model Retraining")

    # Select escalated issues for feedback
    df = get_issues(db_conn)
    escalated_issues = df[df['status'] == 'Escalated']

    issue_ids = escalated_issues['issue_id'].tolist()
    selected_issue_id = st.selectbox("Select Escalated Issue for Feedback", issue_ids)

    if selected_issue_id:
        issue = escalated_issues[escalated_issues['issue_id'] == selected_issue_id].iloc[0]
        st.write(f"**Issue Description:** {issue['issue_description']}")
        st.write(f"**Current Severity:** {issue['severity']}")
        st.write(f"**Current Criticality:** {issue['criticality']}")
        st.write(f"**Current Category:** {issue['category']}")
        st.write(f"**Current Escalation Flag:** {issue['escalation_flag']}")

        st.subheader("Provide Feedback")

        new_severity = st.selectbox("Severity", ["Low", "Medium", "High"], index=["Low", "Medium", "High"].index(issue['severity']))
        new_criticality = st.selectbox("Criticality", ["Low", "Medium", "High"], index=["Low", "Medium", "High"].index(issue['criticality']))
        new_category = st.text_input("Category", issue['category'])
        new_escalation_flag = st.checkbox("Escalation Flag", value=bool(issue['escalation_flag']))

        if st.button("Submit Feedback and Retrain Model"):
            # Update DB
            cursor = db_conn.cursor()
            cursor.execute(
                """UPDATE issues SET severity = ?, criticality = ?, category = ?, escalation_flag = ? WHERE issue_id = ?""",
                (new_severity, new_criticality, new_category, int(new_escalation_flag), selected_issue_id)
            )
            db_conn.commit()

            # Retrain model with updated dataset
            retrain_ml_model(db_conn)
            st.success("Feedback saved and model retrained successfully.")

def retrain_ml_model(db_conn):
    st.info("Retraining ML model...")

    # Fetch data
    df = get_issues(db_conn)

    # Prepare training data
    X = df['issue_description'].values
    y = df['escalation_flag'].astype(int).values

    vectorizer = TfidfVectorizer(max_features=1000)
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    # Save model and vectorizer to disk (or DB) - placeholder
    # For demonstration, just store in session_state
    st.session_state['ml_model'] = model
    st.session_state['vectorizer'] = vectorizer

    st.success("Model retraining complete.")

def ml_prediction_ui(db_conn):
    st.header("Predict Escalation for New Issue")

    issue_text = st.text_area("Enter Issue Description")

    if st.button("Predict Escalation"):
        if not issue_text.strip():
            st.warning("Please enter issue description.")
            return

        if 'ml_model' not in st.session_state or 'vectorizer' not in st.session_state:
            st.warning("ML model not found, retraining first...")
            retrain_ml_model(db_conn)

        model = st.session_state['ml_model']
        vectorizer = st.session_state['vectorizer']

        X_vec = vectorizer.transform([issue_text])
        pred = model.predict(X_vec)[0]
        proba = model.predict_proba(X_vec)[0][1]

        st.write(f"Predicted Escalation Flag: {'Yes' if pred == 1 else 'No'}")
        st.write(f"Probability of Escalation: {proba:.2f}")

def audit_log_ui(db_conn):
    st.header("Audit Log")

    cursor = db_conn.cursor()
    cursor.execute("SELECT id, issue_id, user, action, timestamp FROM audit_log ORDER BY timestamp DESC LIMIT 100")
    rows = cursor.fetchall()

    if rows:
        df = pd.DataFrame(rows, columns=['ID', 'Issue ID', 'User', 'Action', 'Timestamp'])
        st.dataframe(df)
    else:
        st.info("No audit log entries found.")

def sidebar_summary(db_conn):
    st.sidebar.markdown("## Escalation Status Summary")
    df = get_issues(db_conn)
    escalated = df[df['status'] == 'Escalated']

    count_open = len(escalated[escalated['status'] == 'Open'])
    count_in_progress = len(escalated[escalated['status'] == 'In Progress'])
    count_resolved = len(escalated[escalated['status'] == 'Resolved'])
    count_escalated = len(escalated)

    st.sidebar.write(f"**Total Escalated Issues:** {count_escalated}")
    st.sidebar.write(f"Open: {count_open}")
    st.sidebar.write(f"In Progress: {count_in_progress}")
    st.sidebar.write(f"Resolved: {count_resolved}")

def main():
    st.set_page_config(page_title="EscalateAI Final", layout="wide")
    db_conn = init_db()

    sidebar_summary(db_conn)

    st.sidebar.title("EscalateAI Controls")
    page = st.sidebar.radio("Navigate", [
        "Dashboard",
        "Kanban Board",
        "Upload Issues (Excel)",
        "Fetch Emails",
        "Feedback & Retrain",
        "ML Prediction",
        "Audit Log"
    ])

    if page == "Dashboard":
        dashboard_ui(db_conn)
    elif page == "Kanban Board":
        kanban_board_ui(db_conn)
    elif page == "Upload Issues (Excel)":
        excel_upload_ui(db_conn)
    elif page == "Fetch Emails":
        fetch_emails_ui(db_conn)
    elif page == "Feedback & Retrain":
        feedback_and_retraining_ui(db_conn)
    elif page == "ML Prediction":
        ml_prediction_ui(db_conn)
    elif page == "Audit Log":
        audit_log_ui(db_conn)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Alerting")
    if st.sidebar.button("Send MS Teams Alert"):
        send_ms_teams_alert(db_conn)
        st.sidebar.success("MS Teams alert sent.")

    if st.sidebar.button("Send Email Alert"):
        send_email_alert(db_conn)
        st.sidebar.success("Email alert sent.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Search")
    search_text = st.sidebar.text_input("Search Issues")
    if search_text:
        df = get_issues(db_conn)
        filtered = df[df.apply(lambda r: search_text.lower() in str(r['customer_name']).lower() or
                                        search_text.lower() in str(r['issue_description']).lower() or
                                        search_text.lower() in str(r['account']).lower(), axis=1)]
        st.write(f"### Search Results for '{search_text}' ({len(filtered)})")
        st.dataframe(filtered)

import pandas as pd

def get_issues(db_conn):
    query = "SELECT * FROM escalations"
    try:
        df = pd.read_sql_query(query, db_conn)
        return df
    except Exception as e:
        st.error(f"Failed to fetch issues: {e}")
        return pd.DataFrame()  # return empty df on failure
def dashboard_ui(db_conn):
    # Example code for dashboard summary:
    import streamlit as st
    import pandas as pd
    
    df = pd.read_sql_query("SELECT * FROM escalations", db_conn)
    if df.empty:
        st.info("No escalations found.")
        return
    
    st.subheader("Dashboard Summary")
    total = len(df)
    escalated = len(df[df['status'] == 'Escalated'])
    resolved = len(df[df['status'] == 'Resolved'])
    
    st.markdown(f"**Total Issues:** {total}")
    st.markdown(f"**Escalated:** {escalated}")
    st.markdown(f"**Resolved:** {resolved}")
    
    # You can add charts, metrics, filters here

if __name__ == "__main__":
    main()
