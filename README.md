<img width="1418" height="891" alt="image" src="https://github.com/user-attachments/assets/7b495c83-94ea-4832-9367-4be77e1ca820" /># Escalate-AI-V5-01082025
Escalation Management with seservices.support@gmail.com and Email Notifications
1. Automated Escalation Detection
Uses VADER sentiment analysis and keyword matching to assess urgency, severity, and criticality.
Categorizes issues into types like technical, safety, support, etc.
2. Email Integration
Connects to an IMAP server to fetch unseen emails.
Parses subject and body to extract customer issues.
Automatically inserts escalations into a SQLite database.
3. Machine Learning Prediction
Trains a RandomForestClassifier to predict whether an issue should be escalated.
Uses historical data for retraining and feedback loops.
4. Streamlit UI Dashboard
Interactive Kanban board for managing escalations.
Sidebar controls for filtering, uploading Excel files, downloading reports, and sending alerts.
Tabs for viewing all issues, escalated ones, and retraining the model.
5. Multi-Channel Alerting
Sends notifications via Email, Microsoft Teams, and WhatsApp.
SLA breach alerts and manual notifications supported.
6. Feedback & Retraining
Allows users to provide feedback on escalation accuracy.
Supports retraining the ML model based on feedback.
7. Database Management
Ensures schema creation and supports reset for development.
Tracks escalation status, owner, actions, and timestamps.
<img width="1418" height="891" alt="image" src="https://github.com/user-attachments/assets/1c2522a7-feec-4e86-b17d-b7028b78db88" />
