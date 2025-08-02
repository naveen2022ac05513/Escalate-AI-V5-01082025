import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
import pandas as pd
import os
from dotenv import load_dotenv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon if not already done
nltk.download('vader_lexicon')

# Load env vars for Gmail
load_dotenv()
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

# File to store complaints/escalations
CSV_FILE = "complaints_escalations.csv"

# Define your negative keywords (⚙️ Technical Failures, 💢 Customer Dissatisfaction, etc.)
NEGATIVE_KEYWORDS = set([
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
])

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

def load_complaints():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        # Ensure all columns present if schema updated
        expected_cols = ['escalation_id', 'customer', 'subject', 'issue', 'date',
                         'status', 'sentiment', 'priority', 'is_escalation',
                         'action_taken', 'action_owner']
        for col in expected_cols:
            if col not in df.columns:
                df[col] = "" if col != 'is_escalation' else False
        return df
    else:
        # Create empty DataFrame with columns
        return pd.DataFrame(columns=['escalation_id', 'customer', 'subject', 'issue', 'date',
                                     'status', 'sentiment', 'priority', 'is_escalation',
                                     'action_taken', 'action_owner'])

def save_complaints(df):
    df.to_csv(CSV_FILE, index=False)

def generate_escalation_id(df):
    if df.empty:
        return "SESICE-250001"
    else:
        last_id = df['escalation_id'].max()
        if pd.isna(last_id):
            return "SESICE-250001"
        try:
            last_num = int(last_id.split('-')[1])
            new_num = last_num + 1
            return f"SESICE-{new_num}"
        except Exception:
            return f"SESICE-{len(df) + 250001}"

def analyze_sentiment(text):
    # Use VADER compound score
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def analyze_priority(text):
    # Count negative keywords occurrences
    count = sum(word in text.lower() for word in NEGATIVE_KEYWORDS)
    if count >= 3:
        return "High"
    elif count >= 1:
        return "Medium"
    else:
        return "Low"

def fetch_emails():
    st.info("Fetching emails from Gmail...")
    fetched = []
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        mail.select("inbox")

        status, data = mail.search(None, '(UNSEEN)')
        if status != "OK":
            st.warning("No new emails found or unable to search mailbox.")
            return []

        email_ids = data[0].split()
        if not email_ids:
            st.info("No new emails.")
            return []

        # Limit fetching to last 10 unseen emails for performance
        email_ids = email_ids[-10:]

        for num in email_ids:
            status, msg_data = mail.fetch(num, '(RFC822)')
            if status != 'OK':
                continue

            msg = email.message_from_bytes(msg_data[0][1])
            subject, encoding = decode_header(msg["Subject"])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or 'utf-8', errors='ignore')
            from_ = msg.get("From")
            date = msg.get("Date")

            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    ctype = part.get_content_type()
                    cdispo = str(part.get("Content-Disposition"))
                    if ctype == "text/plain" and "attachment" not in cdispo:
                        try:
                            body = part.get_payload(decode=True).decode()
                        except Exception:
                            pass
                        break
            else:
                try:
                    body = msg.get_payload(decode=True).decode()
                except Exception:
                    pass

            # Append as complaint record
            fetched.append({
                "customer": from_,
                "subject": subject,
                "issue": body,
                "date": date
            })

            # Mark email seen
            mail.store(num, '+FLAGS', '\\Seen')

        mail.logout()
        st.success(f"Fetched {len(fetched)} new complaints.")
        return fetched

    except imaplib.IMAP4.error as e:
        st.error(f"IMAP error: {str(e)}")
        return []
    except Exception as e:
        st.error(f"Failed to fetch emails: {str(e)}")
        return []

def append_complaints(fetched, df):
    count_added = 0
    for email_data in fetched:
        # Check for duplicates (customer + snippet of issue)
        if not ((df['customer'] == email_data['customer']) & 
                (df['issue'].str.slice(0,50) == email_data['issue'][:50])).any():

            esc_id = generate_escalation_id(df)
            sentiment = analyze_sentiment(email_data['issue'])
            priority = analyze_priority(email_data['issue'])
            is_escalation = priority in ("High", "Medium")

            new_row = {
                "escalation_id": esc_id,
                "customer": email_data['customer'],
                "subject": email_data['subject'],
                "issue": email_data['issue'],
                "date": email_data['date'],
                "status": "Open",
                "sentiment": sentiment,
                "priority": priority,
                "is_escalation": is_escalation,
                "action_taken": "",
                "action_owner": ""
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            count_added += 1
    return df, count_added

def manual_entry(df):
    st.sidebar.header("➕ Manual Complaint Entry")
    customer = st.sidebar.text_input("Customer Email / Name")
    subject = st.sidebar.text_input("Subject")
    issue = st.sidebar.text_area("Issue / Complaint Description")
    date = datetime.datetime.now().strftime("%a, %d %b %Y %H:%M:%S")
    if st.sidebar.button("Add Complaint"):
        if not customer or not issue:
            st.sidebar.warning("Please enter customer and issue details.")
            return df
        esc_id = generate_escalation_id(df)
        sentiment = analyze_sentiment(issue)
        priority = analyze_priority(issue)
        is_escalation = priority in ("High", "Medium")
        new_row = {
            "escalation_id": esc_id,
            "customer": customer,
            "subject": subject,
            "issue": issue,
            "date": date,
            "status": "Open",
            "sentiment": sentiment,
            "priority": priority,
            "is_escalation": is_escalation,
            "action_taken": "",
            "action_owner": ""
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        st.sidebar.success(f"Complaint {esc_id} added.")
    return df

def bulk_upload(df):
    st.sidebar.header("⬆️ Upload Complaints Excel")
    uploaded_file = st.sidebar.file_uploader("Upload complaints file (.xlsx or .csv)", type=["xlsx", "csv"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                upload_df = pd.read_excel(uploaded_file)
            else:
                upload_df = pd.read_csv(uploaded_file)
            required_cols = ['customer', 'subject', 'issue', 'date']
            if not all(col in upload_df.columns for col in required_cols):
                st.sidebar.error(f"Uploaded file must contain columns: {required_cols}")
                return df

            # Analyze each row
            count_added = 0
            for _, row in upload_df.iterrows():
                cust = str(row['customer'])
                subj = str(row['subject'])
                iss = str(row['issue'])
                dt = str(row['date'])
                if not ((df['customer'] == cust) & (df['issue'].str.slice(0,50) == iss[:50])).any():
                    esc_id = generate_escalation_id(df)
                    sentiment = analyze_sentiment(iss)
                    priority = analyze_priority(iss)
                    is_escalation = priority in ("High", "Medium")
                    new_row = {
                        "escalation_id": esc_id,
                        "customer": cust,
                        "subject": subj,
                        "issue": iss,
                        "date": dt,
                        "status": "Open",
                        "sentiment": sentiment,
                        "priority": priority,
                        "is_escalation": is_escalation,
                        "action_taken": "",
                        "action_owner": ""
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    count_added += 1
            st.sidebar.success(f"Bulk uploaded {count_added} complaints.")
        except Exception as e:
            st.sidebar.error(f"Failed to process uploaded file: {str(e)}")
    return df

def render_kanban(df):
    st.header("🚀 EscalateAI - Escalations & Complaints Kanban Board")

    filter_option = st.selectbox("Filter to show", ["All Complaints", "Escalations Only"])
    if filter_option == "Escalations Only":
        df = df[df['is_escalation'] == True]

    # Status columns
    status_columns = ["Open", "In Progress", "Resolved"]
    colors = {
        "Positive": "🟢",
        "Neutral": "🟡",
        "Negative": "🔴",
        "Low": "⬜",
        "Medium": "🟧",
        "High": "🟥"
    }

    # Group by status
    grouped = {status: df[df['status'] == status] for status in status_columns}

    cols = st.columns(len(status_columns))
    for idx, status in enumerate(status_columns):
        with cols[idx]:
            st.subheader(f"{status} ({len(grouped[status])})")
            for i, row in grouped[status].iterrows():
                header = f"{row['escalation_id']} - {colors.get(row['sentiment'],'')} {row['sentiment']} / {colors.get(row['priority'],'')} {row['priority']}"
                with st.expander(header):
                    st.markdown(f"**Customer:** {row['customer']}")
                    st.markdown(f"**Subject:** {row['subject']}")
                    st.markdown(f"**Date:** {row['date']}")
                    st.markdown(f"**Issue:**\n{row['issue']}")
                    new_status = st.selectbox(
                        "Update Status",
                        status_columns,
                        index=status_columns.index(row['status']),
                        key=f"{row['escalation_id']}_status"
                    )
                    new_action_taken = st.text_area(
                        "Action Taken",
                        value=row['action_taken'],
                        key=f"{row['escalation_id']}_action"
                    )
                    new_action_owner = st.text_input(
                        "Action Owner",
                        value=row['action_owner'],
                        key=f"{row['escalation_id']}_owner"
                    )

                    # Update dataframe and CSV if changed
                    if (new_status != row['status'] or 
                        new_action_taken != row['action_taken'] or 
                        new_action_owner != row['action_owner']):
                        df.loc[i, 'status'] = new_status
                        df.loc[i, 'action_taken'] = new_action_taken
                        df.loc[i, 'action_owner'] = new_action_owner
                        save_complaints(df)
                        st.experimental_rerun()

def main():
    st.title("🚀 EscalateAI - Customer Complaint & Escalation Management")

    # Load existing data
    df = load_complaints()

    # Sidebar
    st.sidebar.header("📥 Fetch & Manage Complaints")
    if st.sidebar.button("📡 Fetch Emails from Gmail"):
        fetched = fetch_emails()
        if fetched:
            df, added = append_complaints(fetched, df)
            save_complaints(df)
            st.sidebar.success(f"Added {added} new complaints from email.")

    df = manual_entry(df)
    df = bulk_upload(df)

    # Download CSV button
    st.sidebar.markdown("---")
    st.sidebar.header("📤 Download complaints/escalations")
    st.sidebar.download_button(
        label="Download CSV",
        data=df.to_csv(index=False),
        file_name="complaints_escalations.csv",
        mime="text/csv"
    )

    # Render Kanban Board
    render_kanban(df)

if __name__ == "__main__":
    main()
