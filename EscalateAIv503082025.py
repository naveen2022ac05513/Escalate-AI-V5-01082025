# EscalateAI - Streamlit App

import streamlit as st
import pandas as pd
import sqlite3
import os
import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# DB Setup
DB_FILE = "escalations.db"
os.makedirs("data", exist_ok=True)
conn = sqlite3.connect(os.path.join("data", DB_FILE), check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS escalations (
    id TEXT PRIMARY KEY,
    customer TEXT,
    issue TEXT,
    sentiment TEXT,
    urgency TEXT,
    criticality TEXT,
    status TEXT,
    action_taken TEXT,
    owner TEXT,
    timestamp TEXT,
    last_update TEXT
)''')
conn.commit()

# Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# UI Styles
st.markdown("""
    <style>
    .fixed-title {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 24px;
        font-weight: bold;
        padding: 10px 16px;
        z-index: 999;
        border-bottom: 2px solid #ccc;
        text-align: center;
    }
    .reportview-container .main .block-container{
        padding-top: 100px;
    }
    .custom-button {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 10px 24px;
        text-align: center;
        font-size: 16px;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .custom-button:hover {
        background-color: #45a049;
    }
    </style>
    <div class="fixed-title">EscalateAI - Escalations & Complaints Kanban Board</div>
""", unsafe_allow_html=True)

st.sidebar.title("EscalateAI Menu")

# Utilities

def generate_id():
    result = c.execute("SELECT COUNT(*) FROM escalations").fetchone()[0]
    return f"SESICE-{250001 + result}"

def analyze_issue(issue):
    score = analyzer.polarity_scores(issue)
    sentiment = "Negative" if score['compound'] < -0.2 else "Neutral" if score['compound'] < 0.2 else "Positive"
    urgency = "High" if any(word in issue.lower() for word in ["immediately", "urgent", "asap", "now"]) else "Low"
    criticality = "High" if any(word in issue.lower() for word in ["critical", "escalate", "failure", "loss"]) else "Low"
    return sentiment, urgency, criticality

def fetch_cases():
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    return df

def save_case(data):
    c.execute("""
        INSERT INTO escalations (id, customer, issue, sentiment, urgency, criticality, status, action_taken, owner, timestamp, last_update)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()

def update_case_status(escalation_id, status, action_taken):
    now = datetime.datetime.now().isoformat()
    c.execute("""
        UPDATE escalations
        SET status = ?, action_taken = ?, last_update = ?
        WHERE id = ?
    """, (status, action_taken, now, escalation_id))
    conn.commit()

def check_sla_and_alert():
    df = fetch_cases()
    now = datetime.datetime.now()
    for _, row in df.iterrows():
        if row['status'] != "Resolved" and row['urgency'] == "High":
            last_update = datetime.datetime.fromisoformat(row['last_update'])
            elapsed = now - last_update
            if elapsed.total_seconds() > 3600:
                st.warning(f"⚠️ SLA breach for {row['id']} - {row['customer']}: Last updated {elapsed.seconds // 60} mins ago")

# App Logic

def main():
    check_sla_and_alert()

    menu = ["Add Escalation", "Kanban Board", "Upload Bulk", "Download"]
    choice = st.sidebar.selectbox("Select Action", menu)

    if choice == "Add Escalation":
        st.subheader("Add New Escalation")
        customer = st.text_input("Customer Name")
        issue = st.text_area("Issue Description")
        owner = st.text_input("Owner")
        if st.button("Submit Escalation"):
            sid = generate_id()
            sentiment, urgency, criticality = analyze_issue(issue)
            timestamp = datetime.datetime.now().isoformat()
            data = (sid, customer, issue, sentiment, urgency, criticality, "Open", "", owner, timestamp, timestamp)
            save_case(data)
            st.success(f"Escalation {sid} logged successfully.")

    elif choice == "Kanban Board":
        df = fetch_cases()
        statuses = ["Open", "In Progress", "Resolved"]
        cols = st.columns(len(statuses))
        for i, status in enumerate(statuses):
            with cols[i]:
                st.markdown(f"### {status}")
                for _, row in df[df["status"] == status].iterrows():
                    with st.expander(f"{row['id']} - {row['customer']}"):
                        st.write(f"**Issue:** {row['issue']}")
                        st.write(f"**Sentiment:** {row['sentiment']}, **Urgency:** {row['urgency']}, **Criticality:** {row['criticality']}")
                        action_taken = st.text_input("Action Taken", row['action_taken'], key=row['id']+"action")
                        new_status = st.selectbox("Update Status", statuses, index=statuses.index(row['status']), key=row['id']+"status")
                        if st.button("Update", key=row['id']+"update"):
                            update_case_status(row['id'], new_status, action_taken)
                            st.experimental_rerun()

    elif choice == "Upload Bulk":
        st.subheader("Upload Escalations from Excel")
        uploaded_file = st.file_uploader("Choose Excel File", type=["xlsx"])
        if uploaded_file:
            bulk_df = pd.read_excel(uploaded_file)
            for _, row in bulk_df.iterrows():
                sid = generate_id()
                issue = row['Issue']
                sentiment, urgency, criticality = analyze_issue(issue)
                timestamp = datetime.datetime.now().isoformat()
                data = (sid, row['Customer'], issue, sentiment, urgency, criticality, "Open", "", row['Owner'], timestamp, timestamp)
                save_case(data)
            st.success("Bulk upload completed.")

    elif choice == "Download":
        df = fetch_cases()
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Escalation Data", csv, "escalateai_cases.csv", "text/csv")

if __name__ == '__main__':
    main()
