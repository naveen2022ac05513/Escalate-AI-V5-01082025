# EscalateAI - Full Functional Escalation Management Tool
# ------------------------------------------------------
# Features:
# - Parses Gmail emails & Excel uploads for customer issues
# - NLP analysis: Sentiment (VADER), urgency & category tagging
# - Extensive negative keywords list for escalation detection
# - Unique escalation IDs (SESICE-XXXXX)
# - Kanban board for Open, In Progress, Resolved statuses
# - Inline editable action taken and owner fields
# - SLA breach alert after 10 mins (via MS Teams webhook)
# - Predictive ML model using BERT for escalation priority
# - Continuous feedback & retraining loop
#
# Requirements:
# pip install streamlit pandas vaderSentiment python-dotenv requests transformers torch scikit-learn openpyxl

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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split

# ---------------------------------------
# Load environment variables for email and MS Teams webhook
load_dotenv()
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
IMAP_SERVER = "imap.gmail.com"
MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")
DB_PATH = "escalations.db"

# Setup device for PyTorch (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Expanded negative words list covering categories you provided
NEGATIVE_KEYWORDS = [
    # Technical Failures & Product Malfunction
    "fail", "break", "crash", "defect", "fault", "degrade", "damage",
    "trip", "malfunction", "blank", "shutdown", "discharge",
    # Customer Dissatisfaction & Escalations
    "dissatisfy", "frustrate", "complain", "reject", "delay", "ignore",
    "escalate", "displease", "noncompliance", "neglect",
    # Support Gaps & Operational Delays
    "wait", "pending", "slow", "incomplete", "miss", "omit", "unresolved",
    "shortage", "no response",
    # Hazardous Conditions & Safety Risks
    "fire", "burn", "flashover", "arc", "explode", "unsafe", "leak",
    "corrode", "alarm", "incident",
    # Business Risk & Impact
    "impact", "loss", "risk", "downtime", "interrupt", "cancel",
    "terminate", "penalty"
]

# Urgency phrases for quick detection
URGENCY_PHRASES = ["asap", "urgent", "immediately", "right away", "critical"]

# Category keywords mapping
CATEGORY_KEYWORDS = {
    "Safety": ["fire", "burn", "leak", "unsafe", "flashover", "arc", "explode", "alarm", "incident"],
    "Performance": ["slow", "crash", "malfunction", "degrade", "fault", "defect", "shutdown"],
    "Delay": ["delay", "pending", "wait", "incomplete", "miss", "omit", "unresolved", "shortage"],
    "Compliance": ["noncompliance", "violation", "penalty", "reject"],
    "Service": ["ignore", "unavailable", "no response", "cancel", "terminate"],
    "Quality": ["defect", "fault", "damage", "break", "fail"],
    "Business Risk": ["impact", "loss", "risk", "downtime", "interrupt", "penalty"],
}

# ---------------------------------------
# Database setup
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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
    urgency TEXT,
    category TEXT,
    action_taken TEXT,
    action_owner TEXT,
    status_update_date TEXT,
    feedback INTEGER -- 1=Positive feedback, -1=Negative, 0=No feedback
)
""")
conn.commit()

# ---------------------------------------
# Utility functions

def generate_id():
    """Generate unique escalation ID SESICE-XXXXX starting from 250001"""
    cursor.execute("SELECT COUNT(*) FROM escalations")
    count = cursor.fetchone()[0] + 250001
    return f"SESICE-{count}"

def classify_sentiment(text):
    """Classify sentiment as Positive, Negative, or Neutral using VADER"""
    score = analyzer.polarity_scores(text)['compound']
    if score < -0.05:
        return "Negative"
    elif score > 0.05:
        return "Positive"
    else:
        return "Neutral"

def detect_urgency(text):
    """Detect urgency based on keywords"""
    text_lower = text.lower()
    return "High" if any(p in text_lower for p in URGENCY_PHRASES) else "Normal"

def detect_category(text):
    """Detect issue category based on keyword mapping"""
    text_lower = text.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return cat
    return "General"

def is_escalation(text):
    """Flag issue as escalation if it contains negative keywords"""
    text_lower = text.lower()
    return 1 if any(word in text_lower for word in NEGATIVE_KEYWORDS) else 0

def insert_to_db(data):
    """Insert new escalation record into DB"""
    cursor.execute("""
        INSERT INTO escalations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()

def fetch_escalations():
    """Fetch all escalations from DB as DataFrame"""
    return pd.read_sql_query("SELECT * FROM escalations ORDER BY date DESC", conn)

def update_escalation(escalation_id, status, action_taken, action_owner):
    """Update escalation record with new status/action/owner"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        UPDATE escalations SET status=?, action_taken=?, action_owner=?, status_update_date=? WHERE escalation_id=?
    """, (status, action_taken, action_owner, now, escalation_id))
    conn.commit()

def update_feedback(escalation_id, feedback_value):
    """Update feedback (1 or -1) for feedback loop"""
    cursor.execute("""
        UPDATE escalations SET feedback=? WHERE escalation_id=?
    """, (feedback_value, escalation_id))
    conn.commit()

# ---------------------------------------
# Alerting (MS Teams)

def send_teams_alert(msg):
    """Send alert to MS Teams via webhook if configured"""
    if MS_TEAMS_WEBHOOK_URL:
        try:
            response = requests.post(MS_TEAMS_WEBHOOK_URL, json={"text": msg})
            if response.status_code != 200:
                st.error(f"MS Teams alert failed: {response.status_code} {response.text}")
        except Exception as e:
            st.error(f"Exception sending MS Teams alert: {e}")

def detect_sla_breach():
    """Check if any high-priority Open escalations exceed 10 min SLA"""
    now = datetime.datetime.now()
    df = fetch_escalations()
    breaches = []
    for _, row in df.iterrows():
        if row["priority"] == "High" and row["status"] != "Resolved":
            created_dt = datetime.datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")
            if (now - created_dt).total_seconds() > 600:  # 10 minutes
                breaches.append(row["escalation_id"])
    for eid in breaches:
        send_teams_alert(f"⚠️ SLA Breach: Escalation {eid} has been open for more than 10 minutes!")

# ---------------------------------------
# Email Parsing (Gmail example, only fetches UNSEEN emails)

def parse_email():
    """Parse unread emails from Gmail inbox and process issues"""
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL, PASSWORD)
        mail.select("inbox")
        status, messages = mail.search(None, 'UNSEEN')
        if status != 'OK':
            st.warning("Failed to fetch emails.")
            return

        for num in messages[0].split():
            _, data = mail.fetch(num, '(RFC822)')
            msg = email.message_from_bytes(data[0][1])
            subject = decode_header(msg["Subject"])[0][0]
            from_ = msg.get("From")
            body = ""

            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode(errors='ignore')
                        break
            else:
                body = msg.get_payload(decode=True).decode(errors='ignore')

            if body:
                process_case(from_, body)

        mail.logout()
    except Exception as e:
        st.error(f"Email parsing error: {e}")

# ---------------------------------------
# Excel Upload processing

def process_excel(uploaded_file):
    """Process uploaded Excel file, expected columns: Customer, Issue"""
    try:
        df = pd.read_excel(uploaded_file)
        if 'Customer' not in df.columns or 'Issue' not in df.columns:
            st.error("Excel must contain 'Customer' and 'Issue' columns")
            return
        for _, row in df.iterrows():
            process_case(str(row['Customer']), str(row['Issue']))
    except Exception as e:
        st.error(f"Excel processing error: {e}")

# ---------------------------------------
# ML Predictive Model Using BERT

# Dataset class for PyTorch
class IssueDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# BERT classifier model definition
class BertClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)  # binary classification: High(1), Low(0)
    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
        output = self.dropout(pooled_output)
        return self.out(output)

# Train one epoch
def train_epoch(model, data_loader, loss_fn, optimizer):
    model.train()
    losses = []
    correct = 0
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        losses.append(loss.item())
        _, preds = torch.max(outputs, dim=1)
        correct += torch.sum(preds == labels)
        loss.backward()
        optimizer.step()
    return sum(losses)/len(losses), correct.double()/len(data_loader.dataset)

# Evaluate model on validation set
def eval_model(model, data_loader, loss_fn):
    model.eval()
    losses = []
    correct = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            _, preds = torch.max(outputs, dim=1)
            correct += torch.sum(preds == labels)
    return sum(losses)/len(losses), correct.double()/len(data_loader.dataset)

# Save model and tokenizer
def save_model(model, tokenizer, model_path="bert_model.pt", tokenizer_path="bert_tokenizer"):
    torch.save(model.state_dict(), model_path)
    tokenizer.save_pretrained(tokenizer_path)

# Load model and tokenizer
def load_model(model_path="bert_model.pt", tokenizer_path="bert_tokenizer"):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model = BertClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer

# Train predictive model from feedback-labeled data
def train_predictive_model(epochs=3, batch_size=16):
    df = fetch_escalations()
    # Only use records with feedback labeled as 1 or -1
    df_train = df[df['feedback'].isin([1, -1])]
    if len(df_train) < 20:
        st.warning("Not enough labeled data (min 20) for ML training. Provide feedback on escalations.")
        return None, None
    X = df_train['issue'].values
    # Priority High -> 1, else 0
    y = (df_train['priority'] == "High").astype(int).values

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = IssueDataset(X, y, tokenizer)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BertClassifier()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer)
        st.write(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

    save_model(model, tokenizer)
    st.success("ML model trained and saved.")
    return model, tokenizer

# Predict priority using ML model (returns High or Low)
def predict_priority(issue_text, model, tokenizer):
    model.eval()
    encoding = tokenizer.encode_plus(
        issue_text,
        max_length=128,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return "High" if pred == 1 else "Low"

# ---------------------------------------
# Main processing function for new cases

def process_case(customer, issue, ml_model=None, ml_tokenizer=None):
    """
    Process a single customer issue:
    - Assign escalation ID
    - Get sentiment (VADER)
    - Detect urgency and category
    - Flag escalation based on negative keywords
    - Predict priority with ML model if available; else heuristic
    - Insert record in DB and alert if high priority escalation
    """
    eid = generate_id()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sentiment = classify_sentiment(issue)
    urgency = detect_urgency(issue)
    category = detect_category(issue)
    flag = is_escalation(issue)

    # Use ML prediction if model provided, else heuristic
    if ml_model and ml_tokenizer:
        priority = predict_priority(issue, ml_model, ml_tokenizer)
    else:
        priority = "High" if sentiment == "Negative" else "Normal"

    data = (eid, customer, issue, now, "Open", sentiment, priority, flag, urgency, category, "", "", now, 0)
    insert_to_db(data)

    # Send alert for high priority escalations
    if flag and priority == "High":
        send_teams_alert(f"🚨 New Escalation {eid}\nCustomer: {customer}\nIssue: {issue}")

# ---------------------------------------
# Streamlit UI and app logic

def main():
    st.title("🚨 EscalateAI - Customer Escalation Management")

    # Load or initialize ML model & tokenizer
    model, tokenizer = None, None
    if os.path.exists("bert_model.pt") and os.path.exists("bert_tokenizer"):
        try:
            model, tokenizer = load_model()
            st.info("ML model loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load ML model: {e}")

    # Sidebar: Upload Excel or manual entry
    with st.sidebar:
        st.header("Add New Issues")

        # Email parsing button
        if st.button("Parse Unread Emails (Gmail)"):
            parse_email()
            st.success("Parsed unread emails.")

        uploaded_file = st.file_uploader("Upload Excel (.xlsx) with Customer, Issue columns")
        if uploaded_file:
            process_excel(uploaded_file)
            st.success("Excel uploaded and processed.")

        st.markdown("---")
        st.subheader("Manual Entry")
        cust = st.text_input("Customer Name")
        iss = st.text_area("Issue Description")
        if st.button("Add Manually"):
            if cust.strip() == "" or iss.strip() == "":
                st.warning("Please provide both customer and issue.")
            else:
                process_case(cust, iss, model, tokenizer)
                st.success("Issue added.")

        st.markdown("---")
        st.subheader("ML Model")

        if st.button("Train/ Retrain ML Model"):
            with st.spinner("Training ML model (may take some time)..."):
                model, tokenizer = train_predictive_model()

    # Main Kanban view
    st.header("Escalations Kanban Board")

    df = fetch_escalations()

    # Show SLA breaches alert
    detect_sla_breach()

    statuses = ["Open", "In Progress", "Resolved"]
    cols = st.columns(len(statuses))

    for i, status in enumerate(statuses):
        with cols[i]:
            st.subheader(status)
            filtered = df[df['status'] == status]

            if filtered.empty:
                st.write("_No cases_")
                continue

            for idx, row in filtered.iterrows():
                with st.expander(f"{row['escalation_id']} - {row['customer']}"):
                    st.write(f"**Issue:** {row['issue']}")
                    st.markdown(f"**Sentiment:** {row['sentiment']} | **Urgency:** {row['urgency']} | **Category:** {row['category']}")
                    action_taken = st.text_input("Action Taken", value=row['action_taken'] or "", key=f"action_{row['escalation_id']}")
                    action_owner = st.text_input("Action Owner", value=row['action_owner'] or "", key=f"owner_{row['escalation_id']}")
                    new_status = st.selectbox("Update Status", options=statuses, index=statuses.index(row['status']), key=f"status_{row['escalation_id']}")

                    feedback = st.radio("Feedback on escalation prediction:",
                                        options=["No feedback", "Priority Correct", "Priority Incorrect"],
                                        index=0,
                                        key=f"feedback_{row['escalation_id']}")
                    if st.button("Save", key=f"save_{row['escalation_id']}"):
                        update_escalation(row['escalation_id'], new_status, action_taken, action_owner)
                        # Update feedback: 1 if "Priority Correct", -1 if "Priority Incorrect", 0 no feedback
                        fb_val = 0
                        if feedback == "Priority Correct":
                            fb_val = 1
                        elif feedback == "Priority Incorrect":
                            fb_val = -1
                        update_feedback(row['escalation_id'], fb_val)
                        st.success("Escalation updated.")

# ---------------------------------------
# Entry point
if __name__ == "__main__":
    main()
