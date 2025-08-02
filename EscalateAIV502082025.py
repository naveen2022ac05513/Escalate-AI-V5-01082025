import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import re
import os
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

# Load env vars
EMAIL = os.getenv("EMAIL_USER")
APP_PASSWORD = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("EMAIL_SERVER", "imap.gmail.com")

CSV_FILE = "complaints_escalations.csv"

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Negative word sets as per user
NEGATIVE_WORDS = {
    "technical": [
        "fail", "break", "crash", "defect", "fault", "degrade", "damage", "trip",
        "malfunction", "blank", "shutdown", "discharge"
    ],
    "customer": [
        "dissatisfy", "frustrate", "complain", "reject", "delay", "ignore",
        "escalate", "displease", "noncompliance", "neglect"
    ],
    "support": [
        "wait", "pending", "slow", "incomplete", "miss", "omit", "unresolved",
        "shortage", "no response"
    ],
    "hazardous": [
        "fire", "burn", "flashover", "arc", "explode", "unsafe", "leak", "corrode",
        "alarm", "incident"
    ],
    "business": [
        "impact", "loss", "risk", "downtime", "interrupt", "cancel", "terminate",
        "penalty"
    ],
}
NEGATIVE_WORDS_FLAT = set(word for group in NEGATIVE_WORDS.values() for word in group)

STATUS_OPTIONS = ["Open", "In Progress", "Resolved"]
FILTER_OPTIONS = ["All", "Escalated"]

# Utility functions

def load_complaints():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    else:
        # Create empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            "escalation_id", "customer", "issue", "date", "status",
            "sentiment", "priority", "action_taken", "action_owner"
        ])

def save_complaints(df):
    df.to_csv(CSV_FILE, index=False)

def generate_escalation_id(df):
    if df.empty:
        return "SESICE-250001"
    else:
        last = df["escalation_id"].str.extract(r'(\d+)').astype(int)
        max_id = last.max()[0] if not last.empty else 250000
        return f"SESICE-{max_id + 1}"

def clean_text(text):
    # Basic cleaning for text analysis
    return re.sub(r'\s+', ' ', str(text).lower())

def analyze_sentiment_priority(issue_text):
    text = clean_text(issue_text)
    vs = analyzer.polarity_scores(text)
    sentiment = "Negative" if vs['compound'] <= -0.05 else "Positive"
    # Check if any negative words from sets present
    neg_word_hits = [w for w in NEGATIVE_WORDS_FLAT if w in text]
    priority = "High" if len(neg_word_hits) >= 2 else "Low"
    return sentiment, priority

# Email fetching and parsing

def fetch_emails():
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL, APP_PASSWORD)
        mail.select("inbox")
        # Fetch unseen emails only
        status, messages = mail.search(None, '(UNSEEN)')
        if status != 'OK':
            st.warning("No new emails found.")
            mail.logout()
            return []
        email_ids = messages[0].split()
        fetched = []
        for num in email_ids[-10:]:  # Limit to last 10 new emails
            _, msg_data = mail.fetch(num, '(RFC822)')
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding or 'utf-8', errors='ignore')
                    from_ = msg.get("From")
                    date = msg.get("Date")
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            content_disp = str(part.get("Content-Disposition"))
                            if content_type == "text/plain" and "attachment" not in content_disp:
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
                    fetched.append({
                        "customer": from_,
                        "issue": body.strip()[:1000],
                        "date": date,
                        "status": "Open",
                        "sentiment": "",
                        "priority": "",
                        "action_taken": "",
                        "action_owner": "",
                    })
            mail.store(num, '+FLAGS', '\\Seen')  # Mark as read
        mail.logout()
        return fetched
    except Exception as e:
        st.error(f"Failed to fetch emails: {e}")
        return []

# Append new complaints from fetched emails
def append_complaints(fetched, df):
    added = 0
    for item in fetched:
        # Check duplicate by customer+issue snippet
        existing = df[(df['customer'] == item['customer']) & (df['issue'].str.startswith(item['issue'][:50]))]
        if existing.empty:
            # Analyze sentiment & priority
            sentiment, priority = analyze_sentiment_priority(item['issue'])
            item['sentiment'] = sentiment
            item['priority'] = priority
            item['escalation_id'] = generate_escalation_id(df)
            df = pd.concat([df, pd.DataFrame([item])], ignore_index=True)
            added += 1
    return df, added

# Manual entry in sidebar
def manual_entry(df):
    st.sidebar.header("✍️ Manual Escalation Entry")
    with st.sidebar.form("manual_entry_form", clear_on_submit=True):
        customer = st.text_input("Customer Email / Name")
        issue = st.text_area("Issue Description")
        submitted = st.form_submit_button("Add Escalation")
        if submitted:
            if customer and issue:
                sentiment, priority = analyze_sentiment_priority(issue)
                new_entry = {
                    "escalation_id": generate_escalation_id(df),
                    "customer": customer,
                    "issue": issue,
                    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "Open",
                    "sentiment": sentiment,
                    "priority": priority,
                    "action_taken": "",
                    "action_owner": "",
                }
                df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                save_complaints(df)
                st.sidebar.success("Manual escalation added!")
            else:
                st.sidebar.error("Please enter both Customer and Issue.")
    return df

# Bulk upload from Excel in sidebar
def bulk_upload(df):
    st.sidebar.header("📁 Bulk Upload Complaints (Excel)")
    uploaded_file = st.sidebar.file_uploader("Upload Excel", type=['xlsx', 'xls'])
    if uploaded_file is not None:
        try:
            bulk_df = pd.read_excel(uploaded_file)
            # Expected columns check: 'customer', 'issue', optional others
            for idx, row in bulk_df.iterrows():
                customer = str(row.get("customer", "")).strip()
                issue = str(row.get("issue", "")).strip()
                if customer and issue:
                    # Avoid duplicates
                    existing = df[(df['customer'] == customer) & (df['issue'].str.startswith(issue[:50]))]
                    if existing.empty:
                        sentiment, priority = analyze_sentiment_priority(issue)
                        new_entry = {
                            "escalation_id": generate_escalation_id(df),
                            "customer": customer,
                            "issue": issue,
                            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "status": "Open",
                            "sentiment": sentiment,
                            "priority": priority,
                            "action_taken": "",
                            "action_owner": "",
                        }
                        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            save_complaints(df)
            st.sidebar.success("Bulk complaints uploaded and analyzed!")
        except Exception as e:
            st.sidebar.error(f"Failed to process uploaded file: {e}")
    return df

# Render Kanban Board with filters
def render_kanban(df):
    st.header("📊 Complaints & Escalations Kanban Board")

    filter_opt = st.selectbox("Filter Cases", FILTER_OPTIONS)
    if filter_opt == "Escalated":
        # Show only high priority and/or negative sentiment
        filtered_df = df[(df['priority'] == "High") | (df['sentiment'] == "Negative")]
    else:
        filtered_df = df.copy()

    # Group by status
    status_groups = {status: filtered_df[filtered_df['status'] == status] for status in STATUS_OPTIONS}

    # Color maps for sentiments and priority
    sentiment_colors = {"Positive": "✅", "Negative": "❌"}
    priority_colors = {"High": "🔴", "Low": "🟢"}

    cols = st.columns(len(STATUS_OPTIONS))

    for idx, status in enumerate(STATUS_OPTIONS):
        with cols[idx]:
            st.subheader(f"{status} ({len(status_groups[status])})")
            for _, row in status_groups[status].iterrows():
                header = (
                    f"{row['escalation_id']} - "
                    f"{sentiment_colors.get(row['sentiment'], '')} {row['sentiment']} / "
                    f"{priority_colors.get(row['priority'], '')} {row['priority']}"
                )
                with st.expander(header):
                    st.markdown(f"**Customer:** {row['customer']}")
                    st.markdown(f"**Issue:** {row['issue']}")
                    st.markdown(f"**Date:** {row['date']}")

                    # Editable status
                    new_status = st.selectbox(
                        "Update Status", STATUS_OPTIONS,
                        index=STATUS_OPTIONS.index(row['status']),
                        key=f"status_{row['escalation_id']}"
                    )
                    # Editable action taken and owner
                    new_action_taken = st.text_area(
                        "Action Taken",
                        value=row.get("action_taken", ""),
                        key=f"action_{row['escalation_id']}"
                    )
                    new_action_owner = st.text_input(
                        "Action Owner",
                        value=row.get("action_owner", ""),
                        key=f"owner_{row['escalation_id']}"
                    )

                    # Save changes
                    if st.button("Save Changes", key=f"save_{row['escalation_id']}"):
                        df.loc[df['escalation_id'] == row['escalation_id'], ['status', 'action_taken', 'action_owner']] = [new_status, new_action_taken, new_action_owner]
                        save_complaints(df)
                        st.success("Saved!")

def main():
    st.title("🚀 EscalateAI - Escalations & Complaints Kanban Board")

    df = load_complaints()

    st.sidebar.header("📥 Fetch & Manage Complaints")

    if st.sidebar.button("📡 Fetch Emails from Gmail"):
        fetched = fetch_emails()
        if fetched:
            df, added = append_complaints(fetched, df)
            save_complaints(df)
            st.sidebar.success(f"Added {added} new complaints from email.")
            df = load_complaints()  # Reload after save

    df = manual_entry(df)
    df = bulk_upload(df)

    st.sidebar.markdown("---")
    st.sidebar.header("📤 Download complaints/escalations")
    st.sidebar.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="complaints_escalations.csv",
        mime="text/csv"
    )

    render_kanban(df)

if __name__ == "__main__":
    main()
