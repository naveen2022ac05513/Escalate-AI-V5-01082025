# Imports
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Config
DB_PATH = "your_database_path.db"
STATUS_COLORS = {"Open": "red", "In Progress": "orange", "Resolved": "green"}

# Ensure schema exists
def ensure_schema():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS escalations (
            id TEXT PRIMARY KEY,
            customer TEXT,
            issue TEXT,
            sentiment TEXT,
            urgency TEXT,
            severity TEXT,
            criticality TEXT,
            escalated TEXT,
            status TEXT,
            action_taken TEXT,
            owner TEXT,
            feedback_score INTEGER,
            category TEXT
        )
    """)
    conn.commit()
    conn.close()

# Sample seeding (optional)
def seed_data():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    sample = ("ESC001", "Acme Corp", "Cannot log in", "negative", "high", "medium", "critical", "Yes", "Open", "", "", None, "Authentication")
    c.execute("INSERT OR IGNORE INTO escalations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", sample)
    conn.commit()
    conn.close()

# Fetch data
def fetch_escalations():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM escalations", conn)
    conn.close()
    return df

def update_escalation_status(id, status, action_taken, owner, feedback_score=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""UPDATE escalations SET status=?, action_taken=?, owner=?, feedback_score=? WHERE id=?""",
              (status, action_taken, owner, feedback_score, id))
    conn.commit()
    conn.close()

def send_alert(message, via="email"):
    st.info(f"Alert sent via {via}: {message}")

def send_whatsapp_message(phone, message):
    st.info(f"WhatsApp message to {phone}: {message}")

def train_model():
    df = fetch_escalations()
    df = df.dropna(subset=['sentiment', 'urgency', 'severity', 'criticality', 'escalated'])
    if df.shape[0] < 20 or df['escalated'].nunique() < 2:
        st.warning("Not enough data to train model.")
        return None
    X = pd.get_dummies(df[['sentiment', 'urgency', 'severity', 'criticality']])
    y = df['escalated'].apply(lambda x: 1 if x == 'Yes' else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    st.success(f"Model Accuracy: {acc:.2f}")
    return model

def predict_escalation(model, sentiment, urgency, severity, criticality):
    X_pred = pd.DataFrame([{
        f"sentiment_{sentiment}": 1,
        f"urgency_{urgency}": 1,
        f"severity_{severity}": 1,
        f"criticality_{criticality}": 1
    }])
    X_pred = X_pred.reindex(columns=model.feature_names_in_, fill_value=0)
    pred = model.predict(X_pred)
    return "Yes" if pred[0] == 1 else "No"

def parse_feedback(feedback_text):
    text = feedback_text.lower()
    if "wrong" in text or "not needed" in text: return 0
    elif "correct" in text or "appropriate" in text: return 1
    else: return None

# Initialize schema & data
ensure_schema()
seed_data()

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🚨 EscalateAI 2.0 – Customer Escalation Platform")

# Sidebar
st.sidebar.header("⚙️ Controls")
if st.sidebar.button("📣 Send Manual Alert to MS Teams"):
    send_alert("Manual alert triggered from dashboard", via="teams")
if st.sidebar.button("📧 Send Manual Alert via Email"):
    send_alert("Manual alert triggered from dashboard", via="email")

# Main UI Tabs
tabs = st.tabs(["📊 Dashboard", "🗃️ Kanban", "🚩 Escalated", "🔁 Feedback & Retrain", "🧪 Dev Panel"])

# Tab 1: Dashboard
with tabs[0]:
    st.subheader("📊 Summary Dashboard")
    df = fetch_escalations()
    st.metric("Total Escalations", len(df))
    st.metric("Resolved", df[df["status"] == "Resolved"].shape[0])
    st.metric("Negative Sentiment", df[df["sentiment"] == "negative"].shape[0])
    st.bar_chart(df["category"].value_counts())
    st.bar_chart(df["urgency"].value_counts())
    st.bar_chart(df["severity"].value_counts())

# Tab 2: Kanban Board
with tabs[1]:
    st.subheader("🗃️ Escalation Kanban")
    statuses = ["Open", "In Progress", "Resolved"]
    cols = st.columns(len(statuses))
    for i, status in enumerate(statuses):
        with cols[i]:
            st.markdown(f"<h4 style='color:{STATUS_COLORS[status]}'>{status}</h4>", unsafe_allow_html=True)
            bucket = df[df["status"] == status]
            for _, row in bucket.iterrows():
                with st.expander(f"{row['id']} - {row['customer']}"):
                    st.markdown(f"**Issue**: {row['issue']}")
                    st.markdown(f"**Sentiment**: {row['sentiment']}")
                    st.markdown(f"**Urgency**: {row['urgency']}")
                    st.markdown(f"**Severity**: {row['severity']}")
                    st.markdown(f"**Criticality**: {row['criticality']}")
                    st.markdown(f"**Escalated**: {row['escalated']}")
                    new_status = st.selectbox("Update Status", statuses, index=statuses.index(row["status"]), key=f"s_{row['id']}")
                    new_action = st.text_input("Action Taken", row.get("action_taken", ""), key=f"a_{row['id']}")
                    new_owner = st.text_input("Owner", row.get("owner", ""), key=f"o_{row['id']}")
                    if st.button("💾 Save", key=f"save_{row['id']}"):
                        update_escalation_status(row['id'], new_status, new_action, new_owner)
                        if new_status == "Resolved":
                            send_whatsapp_message("whatsapp:+911234567890", f"Your escalation {row['id']} has been resolved.")
                        st.success("Saved successfully.")

# Tab 3: Escalated Cases
with tabs[2]:
    st.subheader("🚩 Escalated Issues")
    st.dataframe(df[df["escalated"] == "Yes"])

# Tab 4: Feedback & Retrain
with tabs[3]:
    st.subheader("🔁 Feedback & Retraining")
    for _, row in df.iterrows():
        feedback = st.text_input(f"Feedback on {row['id']}", key=f"fb_{row['id']}")
        if st.button(f"Submit Feedback for {row['id']}", key=f"fb_btn_{row['id']}"):
            score = parse_feedback(feedback)
            if score is not None:
                update_escalation_status(row['id'], row["status"], row.get("action_taken",""), row.get("owner",""), score)
                st.success("Feedback recorded.")

    if st.button("🔁 Retrain ML Model"):
        model = train_model()
        if model:
            st.success("Model retrained.")
        else:
            st.warning("Model training failed.")

# Tab 5: Dev Panel
with tabs[4]:
    st.subheader("🧪 Developer Utilities")
    st.write("Raw Escalation Data")
    st.dataframe(df)
    if st.button("🧨 Reset Database (Dev Only)"):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS escalations")
        c.execute("DROP TABLE IF EXISTS audit_log")
        conn.commit()
        conn.close()
        st.warning("Database wiped. Refresh app to reinitialize.")
