# Return Risk AI: Deploying on Streamlit Cloud

## 1. Put the code on GitHub
Upload everything in this folder. The `.gitignore` already keeps out secrets (`*.json`) and model files.
**Never upload the Google JSON key.**

## 2. Create the app
1. Go to https://share.streamlit.io and sign in with GitHub
2. New app → choose your repo → main file: `app.py` → Deploy
3. First launch takes a few minutes because it trains the model.

## 3. No credentials needed
The app runs without any Google credentials. The BigQuery tab shows the pipeline overview only, so you don't need to add Secrets.

## 4. Keep your Google key private
The key (`.json`) is only used on your own computer for the BigQuery scripts. Never upload it to GitHub. The `.gitignore` already blocks `*.json`.
