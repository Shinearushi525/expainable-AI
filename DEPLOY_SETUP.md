# Deploy on Streamlit Cloud

## 1. Put the code on GitHub
Upload everything in this folder. The `.gitignore` already keeps out secrets (`*.json`) and model files.
**Never upload the Google JSON key.**

## 2. Create the app
1. Go to https://share.streamlit.io and sign in with GitHub
2. New app → choose your repo → main file: `app.py` → Deploy
3. First launch takes a few minutes because it trains the model.

## 3. Add BigQuery credentials (only for the BigQuery tab)
App → Settings → Secrets, paste:

```toml
GCP_PROJECT_ID = "striking-goal-506009-v4"

[gcp_service_account]
type = "service_account"
project_id = "striking-goal-506009-v4"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "striking-goal-506009-v4@striking-goal-506009-v4.iam.gserviceaccount.com"
client_id = "..."
token_uri = "https://oauth2.googleapis.com/token"
```
Copy each value from the JSON key file. Keep the `\n` characters in `private_key`.

## 4. Before going public
Create a NEW key in GCP (IAM → Service Accounts → Keys), use it in Secrets, and delete the old one.
Also change the role from Editor to **BigQuery Data Viewer + BigQuery Job User**, since the app only reads.
