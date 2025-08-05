import streamlit as st
import pandas as pd
import sqlite3
import threading
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import altair as alt
import uuid
import io
import os

# Constants
DB_PATH = "escalate_ai.db"

# Thread‐safe ID lock
id_lock = threading.Lock()

# Initialize VADER for sentiment analysis
sentiment_analyzer = SentimentIntensityAnalyzer()


def init_db():
    """Create SQLite tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS escalations (
            esc_id TEXT PRIMARY KEY,
            customer TEXT,
            subject TEXT,
            description TEXT,
            predicted_prob REAL,
            predicted_label INTEGER,
            sentiment REAL,
            severity TEXT,
            timestamp TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            esc_id TEXT,
            action TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def generate_escalation_id() -> str:
    """Generate a unique escalation ID in a thread-safe way."""
    with id_lock:
        return uuid.uuid4().hex


def log_audit(esc_id: str, action: str):
    """Record an audit log entry."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute(
        "INSERT INTO audit_logs (esc_id, action, timestamp) VALUES (?, ?, ?)",
        (esc_id, action, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def save_escalation(record: dict):
    """Persist an escalation record to the database and log the action."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        INSERT INTO escalations
        (esc_id, customer, subject, description,
         predicted_prob, predicted_label, sentiment, severity, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record["esc_id"], record["customer"], record["subject"], record["description"],
        record["predicted_prob"], record["predicted_label"],
        record["sentiment"], record["severity"], record["timestamp"]
    ))
    conn.commit()
    conn.close()
    log_audit(record["esc_id"], "CREATED")


def fetch_escalations() -> pd.DataFrame:
    """Retrieve all escalation rows as a DataFrame."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    df = pd.read_sql("SELECT * FROM escalations", conn, parse_dates=["timestamp"])
    conn.close()
    return df


def analyze_sentiment(text: str) -> float:
    """Return the negative sentiment score for the text."""
    vs = sentiment_analyzer.polarity_scores(text)
    return vs["neg"]


def predict_severity(sentiment_score: float) -> (float, int, str):
    """
    Dummy escalation predictor based on negative sentiment:
      - prob = sentiment_score
      - label = 1 if prob > 0.2 else 0
      - severity tiers: Low / Medium / High
    """
    prob = sentiment_score
    label = 1 if prob > 0.2 else 0
    if prob > 0.5:
        sev = "High"
    elif prob > 0.2:
        sev = "Medium"
    else:
        sev = "Low"
    return prob, label, sev


def send_email(recipient: str, subject: str, body: str):
    """Simulate sending an email notification."""
    st.info(f"📧 Email sent to {recipient}: {subject}")


def send_whatsapp_notification(message: str):
    """Simulate sending a WhatsApp alert."""
    st.info(f"📱 WhatsApp notification: {message}")


def render_dashboard():
    """Display metrics, charts, and data download on the dashboard."""
    df = fetch_escalations()
    if df.empty:
        st.warning("No escalations to display yet.")
        return

    st.subheader("Key Metrics")
    total = len(df)
    avg_sentiment = df["sentiment"].mean().round(3)
    high_count = (df["severity"] == "High").sum()
    medium_count = (df["severity"] == "Medium").sum()
    low_count = (df["severity"] == "Low").sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Escalations", total)
    col2.metric("Avg Negative Sentiment", avg_sentiment)
    col3.metric("High Severity", high_count)
    col4.metric("Medium Severity", medium_count)

    st.markdown("---")
    st.subheader("Severity Distribution")
    severity_bar = alt.Chart(df).mark_bar().encode(
        x="severity",
        y="count()",
        color="severity"
    )
    st.altair_chart(severity_bar, use_container_width=True)

    st.subheader("Escalations Over Time")
    df["date"] = df["timestamp"].dt.date
    time_line = alt.Chart(df).mark_line(point=True).encode(
        x="date",
        y="count()",
        color="severity"
    )
    st.altair_chart(time_line, use_container_width=True)

    st.markdown("---")
    st.subheader("Escalations Data")
    st.dataframe(df)

    # Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="escalations_export.csv",
        mime="text/csv"
    )


def upload_csv():
    """Handle CSV upload, record escalations, and notify stakeholders."""
    uploaded = st.file_uploader("Upload escalation CSV", type="csv")
    if not uploaded:
        return

    df = pd.read_csv(uploaded)
    required = {"customer", "subject", "description"}
    if not required.issubset(df.columns):
        st.error(f"CSV must contain columns: {required}")
        return

    results = []
    for _, row in df.iterrows():
        text = f"{row.subject} {row.description}"
        sentiment = analyze_sentiment(text)
        prob, label, severity = predict_severity(sentiment)

        record = {
            "esc_id": generate_escalation_id(),
            "customer": row.customer,
            "subject": row.subject,
            "description": row.description,
            "predicted_prob": prob,
            "predicted_label": label,
            "sentiment": sentiment,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat()
        }
        save_escalation(record)
        send_email(row.customer, f"[Escalation ID {record['esc_id']}]", record["subject"])
        results.append(record)

    st.success(f"Processed {len(results)} escalations.")
    res_df = pd.DataFrame(results)
    st.dataframe(res_df)

    csv = res_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Processed Escalations",
        data=csv, file_name="processed_escalations.csv", mime="text/csv"
    )


def parse_feedback():
    """Parse feedback CSV to identify negative comments and log for retraining."""
    uploaded = st.file_uploader("Upload feedback CSV", type="csv")
    if not uploaded:
        return

    fb = pd.read_csv(uploaded)
    if "feedback" not in fb.columns:
        st.error("CSV must have a 'feedback' column.")
        return

    fb["neg_score"] = fb.feedback.apply(analyze_sentiment)
    neg_fb = fb[fb.neg_score > 0.2]
    count_neg = len(neg_fb)

    # Stub for retraining pipeline
    st.info(f"Identified {count_neg} negative feedback entries for retraining.")
    log_audit("FEEDBACK", f"{count_neg} negative comments queued for model retraining")

    if not neg_fb.empty:
        st.dataframe(neg_fb[["feedback", "neg_score"]])


def main():
    st.set_page_config(page_title="EscalateAI 2.0", layout="wide")
    st.title("🚨 EscalateAI – Smart Escalation Manager")

    init_db()
    menu = st.sidebar.radio("Navigation", ["Dashboard", "Upload", "Feedback Parser"])
    st.sidebar.markdown("---")
    if st.sidebar.button("🔔 Send Test WhatsApp Alert"):
        send_whatsapp_notification("Test escalation resolved")

    if menu == "Dashboard":
        render_dashboard()
    elif menu == "Upload":
        upload_csv()
    elif menu == "Feedback Parser":
        parse_feedback()


if __name__ == "__main__":
    main()
