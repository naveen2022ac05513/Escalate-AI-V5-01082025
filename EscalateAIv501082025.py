# ==============================================================
# EscalateAI – Escalation Management Tool (No Outlook)
# --------------------------------------------------------------
# • Manual or Excel-based escalation ingestion
# • Sentiment detection (HF transformers if available, else rule‑based)
# • Negative issues auto‑tagged as potential escalations
# • SPOC notifications + 24‑hour manager escalation
# • Streamlit Kanban dashboard, CSV/Excel upload, manual entry, export
# --------------------------------------------------------------
# Author: Naveen Gandham • v1.2.2 • July 2025
# ==============================================================

"""Setup
pip install streamlit pandas openpyxl python-dotenv apscheduler scikit-learn joblib xlsxwriter
# Optional for better NLP
pip install transformers torch --index-url https://download.pytorch.org/whl/cpu
"""

# ========== Standard Lib ==========
import os, re, sqlite3, smtplib, time, io, uuid
from email.mime.text import MIMEText
from datetime import datetime
from pathlib import Path
from typing import Tuple

# ========== Third‑party ==========
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Optional sentiment NLP ------------------------------------------------------
try:
    from transformers import pipeline as hf_pipeline  # type: ignore
    import torch  # noqa: F401 (CPU wheel)
    HAS_NLP = True
except Exception:
    HAS_NLP = False

# ========== Paths & ENV ==========
APP_DIR   = Path(__file__).resolve().parent
DATA_DIR  = APP_DIR / "data";  DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR = APP_DIR / "models"; MODEL_DIR.mkdir(exist_ok=True)
DB_PATH   = DATA_DIR / "escalateai.db"

load_dotenv()
SMTP_SERVER   = os.getenv("SMTP_SERVER")
SMTP_PORT     = int(os.getenv("SMTP_PORT", 587))
SMTP_USER     = os.getenv("SMTP_USER")
SMTP_PASS     = os.getenv("SMTP_PASS")

AUTHORIZED_EMAILS = ["seservices.support@gmail.com", "seservices.schneider@gmail.com"]

# ========== Sentiment Model ==========
@st.cache_resource(show_spinner=False)
def load_sentiment():
    if not HAS_NLP:
        return None
    try:
        return hf_pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    except Exception:
        return None

sent_model = load_sentiment()
NEG_WORDS = [r"\b(delay|issue|failure|dissatisfaction|unacceptable|complaint|escalation|critical|risk|faulty)\b"]

def rule_sent(text: str) -> str:
    return "Negative" if any(re.search(p, text, re.I) for p in NEG_WORDS) else "Positive"

def analyze_issue(text: str) -> Tuple[str, str, bool]:
    if sent_model:
        lbl = sent_model(text[:512])[0]["label"].lower()
        sentiment = "Negative" if lbl == "negative" else "Positive"
    else:
        sentiment = rule_sent(text)
    urgency = "High" if any(k in text.lower() for k in ["urgent", "immediate", "critical", "asap"]) else "Low"
    return sentiment, urgency, sentiment == "Negative" and urgency == "High"

# ========== DB Setup ==========
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS escalations (
                id TEXT PRIMARY KEY,
                customer TEXT,
                issue TEXT,
                criticality TEXT,
                impact TEXT,
                sentiment TEXT,
                urgency TEXT,
                escalated INTEGER,
                date_reported TEXT,
                owner TEXT,
                status TEXT,
                action_taken TEXT,
                risk_score REAL,
                spoc_email TEXT,
                spoc_boss_email TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

init_db()

# ========== Risk Prediction ==========
MODEL_PATH = MODEL_DIR / "risk_model.joblib"
@st.cache_resource(show_spinner=False)
def load_risk_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

risk_model = load_risk_model()

def predict_risk(txt: str) -> float:
    return float(risk_model.predict_proba([txt])[0][1]) if risk_model else 0.0

# ========== DB Insertion ==========
def insert_case(data: dict):
    data["id"] = f"SESICE-{str(uuid.uuid4())[:8].upper()}"
    with sqlite3.connect(DB_PATH) as conn:
        cols = ",".join(data.keys())
        vals = tuple(data.values())
        placeholders = ",".join(["?"] * len(data))
        conn.execute(f"INSERT INTO escalations ({cols}) VALUES ({placeholders})", vals)
        conn.commit()

# ========== UI Note ==========
st.sidebar.markdown("✅ Outlook integration disabled. Use Excel or Manual entry.")

# ========== Sidebar Upload ==========
st.sidebar.header("📤 Upload Escalation File")
uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    for _, row in df.iterrows():
        issue_text = str(row.get("Issue", ""))
        sentiment, urgency, escalate = analyze_issue(issue_text)
        insert_case({
            "customer": row.get("Customer", ""),
            "issue": issue_text,
            "criticality": row.get("Criticality", "High" if escalate else "Medium"),
            "impact": row.get("Impact", "High" if escalate else "Medium"),
            "sentiment": sentiment,
            "urgency": urgency,
            "escalated": int(escalate),
            "date_reported": row.get("Date Reported", datetime.utcnow().date().isoformat()),
            "owner": row.get("Owner", "Unassigned"),
            "status": row.get("Status", "Open"),
            "action_taken": row.get("Action Taken", ""),
            "risk_score": predict_risk(issue_text),
            "spoc_email": row.get("SPOC Email", ""),
            "spoc_boss_email": row.get("SPOC Boss Email", "")
        })
    st.success(f"✅ Uploaded and processed {len(df)} records")

# ========== Manual Entry ==========
st.sidebar.header("📝 Manual Escalation Entry")
with st.sidebar.form("manual_entry"):
    cust = st.text_input("Customer")
    issue = st.text_area("Issue")
    owner = st.text_input("Owner", "Unassigned")
    spoc = st.text_input("SPOC Email")
    boss = st.text_input("SPOC Boss Email")
    submitted = st.form_submit_button("Add Escalation")
    if submitted and issue:
        sentiment, urgency, escalate = analyze_issue(issue)
        insert_case({
            "customer": cust,
            "issue": issue,
            "criticality": "High" if escalate else "Medium",
            "impact": "High" if escalate else "Medium",
            "sentiment": sentiment,
            "urgency": urgency,
            "escalated": int(escalate),
            "date_reported": datetime.utcnow().date().isoformat(),
            "owner": owner,
            "status": "Open",
            "action_taken": "",
            "risk_score": predict_risk(issue),
            "spoc_email": spoc,
            "spoc_boss_email": boss
        })
        st.success("✅ Escalation added successfully")

# ========== Main Area: Kanban View ==========
st.title("📌 Escalation Dashboard")
view_df = pd.read_sql("SELECT * FROM escalations ORDER BY datetime(created_at) DESC", sqlite3.connect(DB_PATH))
if view_df.empty:
    st.info("No escalations logged yet.")
else:
    for status in ["Open", "In Progress", "Closed"]:
        st.subheader(f"🗂 {status} Cases")
        subset = view_df[view_df.status == status]
        for _, row in subset.iterrows():
            with st.expander(f"{row['id']} - {row['customer']} ({row['sentiment']}/{row['urgency']})"):
                st.markdown(f"**Issue:** {row['issue']}")
                st.markdown(f"**Risk Score:** {round(row['risk_score'], 2)}")
                st.markdown(f"**Owner:** {row['owner']}")
                st.markdown(f"**Status:** {row['status']}")
                st.markdown(f"**Action Taken:** {row['action_taken']}")
                st.markdown(f"**Reported On:** {row['date_reported']}")
