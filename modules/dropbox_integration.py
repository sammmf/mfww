# modules/dropbox_integration.py

import dropbox
import streamlit as st
from dropbox.oauth import DropboxOAuth2FlowNoRedirect
from dropbox import DropboxOAuth2Flow
import requests
import time
import json
import os

def initialize_dropbox():
    """
    Initialize Dropbox client with automatic token refresh.
    """

    # Get App Key and Secret from Streamlit secrets
    APP_KEY = st.secrets["DROPBOX_APP_KEY"]
    APP_SECRET = st.secrets["DROPBOX_APP_SECRET"]

    # Check if tokens are already stored
    if "DROPBOX_REFRESH_TOKEN" in st.secrets and "DROPBOX_ACCESS_TOKEN" in st.secrets:
        refresh_token = st.secrets["DROPBOX_REFRESH_TOKEN"]
        access_token = st.secrets["DROPBOX_ACCESS_TOKEN"]
        expires_at = float(st.secrets["DROPBOX_TOKEN_EXPIRES_AT"])

        # Check if access token is expired
        if time.time() > expires_at:
            # Refresh the access token
            access_token, expires_at = refresh_access_token(APP_KEY, APP_SECRET, refresh_token)
            if not access_token:
                st.error("Failed to refresh access token.")
                return None

            # Update tokens in st.secrets
            update_st_secrets({
                "DROPBOX_ACCESS_TOKEN": access_token,
                "DROPBOX_TOKEN_EXPIRES_AT": str(expires_at)
            })
    else:
        # Start OAuth 2.0 flow
        auth_code = get_auth_code(APP_KEY)
        if not auth_code:
            return None

        # Exchange auth code for tokens
        tokens = exchange_code_for_tokens(APP_KEY, APP_SECRET, auth_code)
        if not tokens:
            st.error("Failed to obtain access token.")
            return None

        access_token = tokens["access_token"]
        refresh_token = tokens["refresh_token"]
        expires_in = tokens["expires_in"]

        # Calculate expiration time
        expires_at = time.time() + expires_in

        # Store tokens in st.secrets
        update_st_secrets({
            "DROPBOX_ACCESS_TOKEN": access_token,
            "DROPBOX_REFRESH_TOKEN": refresh_token,
            "DROPBOX_TOKEN_EXPIRES_AT": str(expires_at)
        })

    # Initialize Dropbox client
    dbx = dropbox.Dropbox(access_token, app_key=APP_KEY, app_secret=APP_SECRET, oauth2_refresh_token=refresh_token)
    return dbx

def get_auth_code(APP_KEY):
    """
    Direct the user to authenticate and obtain an authorization code.
    """
    auth_flow = DropboxOAuth2FlowNoRedirect(APP_KEY, use_pkce=True, token_access_type='offline')
    authorize_url = auth_flow.start()

    st.info("Click the link below to authorize the application with Dropbox:")
    st.write(f"[Authorize with Dropbox]({authorize_url})")

    auth_code = st.text_input("Enter the authorization code here:")
    auth_submit = st.button("Submit Authorization Code")

    if auth_submit and auth_code:
        return auth_code.strip()
    else:
        st.stop()
        return None

def exchange_code_for_tokens(APP_KEY, APP_SECRET, auth_code):
    """
    Exchange the authorization code for access and refresh tokens.
    """
    token_url = "https://api.dropboxapi.com/oauth2/token"
    data = {
        "code": auth_code,
        "grant_type": "authorization_code",
        "client_id": APP_KEY,
        "client_secret": APP_SECRET,
        "redirect_uri": "https://your-streamlit-app-url/redirect"
    }
    response = requests.post(token_url, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error exchanging code: {response.text}")
        return None

def refresh_access_token(APP_KEY, APP_SECRET, refresh_token):
    """
    Refresh the access token using the refresh token.
    """
    token_url = "https://api.dropboxapi.com/oauth2/token"
    data = {
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
        "client_id": APP_KEY,
        "client_secret": APP_SECRET
    }
    response = requests.post(token_url, data=data)
    if response.status_code == 200:
        tokens = response.json()
        access_token = tokens["access_token"]
        expires_in = tokens["expires_in"]
        expires_at = time.time() + expires_in
        return access_token, expires_at
    else:
        st.error(f"Error refreshing access token: {response.text}")
        return None, None

def update_st_secrets(new_secrets):
    """
    Update st.secrets with new tokens.
    """
    # This function depends on how you manage st.secrets.
    # For Streamlit Cloud, st.secrets are read-only and cannot be updated at runtime.
    # So, we'll need to write tokens to a file or another persistent storage.
    # For this example, we'll write to a local JSON file.

    secrets_file = "dropbox_tokens.json"

    # Load existing secrets
    if os.path.exists(secrets_file):
        with open(secrets_file, "r") as f:
            secrets = json.load(f)
    else:
        secrets = {}

    # Update secrets
    secrets.update(new_secrets)

    # Save secrets
    with open(secrets_file, "w") as f:
        json.dump(secrets, f)

    # Optionally, update st.secrets if possible
    for key, value in new_secrets.items():
        st.secrets[key] = value

def load_tokens_from_file():
    """
    Load tokens from the local JSON file.
    """
    secrets_file = "dropbox_tokens.json"
    if os.path.exists(secrets_file):
        with open(secrets_file, "r") as f:
            tokens = json.load(f)
        return tokens
    else:
        return {}

def download_data_file(dbx):
    """
    Download the data file from Dropbox.
    """
    try:
        metadata, res = dbx.files_download("/daily_data.xlsx")
        with open("daily_data.xlsx", "wb") as f:
            f.write(res.content)
        return True
    except dropbox.exceptions.ApiError as err:
        st.error(f"Failed to download file: {err}")
        return False
