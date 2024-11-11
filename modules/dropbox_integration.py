# modules/dropbox_integration.py

import dropbox
import streamlit as st
import time
import requests
from modules import firebase_integration
from dropbox.oauth import DropboxOAuth2FlowNoRedirect

def initialize_dropbox():
    """
    Initialize Dropbox client with automatic token refresh.
    """
    # Get App Key from Streamlit secrets
    try:
        APP_KEY = st.secrets["DROPBOX_APP_KEY"]
        # APP_SECRET is not needed when using PKCE
    except KeyError as e:
        st.error(f"Missing key in st.secrets: {e}")
        st.stop()
        return dbx

    # Retrieve tokens from Firebase
    tokens = firebase_integration.get_tokens_from_firebase()
    if tokens:
        access_token = tokens.get('access_token')
        refresh_token = tokens.get('refresh_token')
        expires_at = tokens.get('expires_at')
    else:
        # Tokens not found, initiate OAuth flow to get initial tokens
        access_token, refresh_token, expires_at = get_initial_tokens(APP_KEY)
        if not access_token:
            st.error("Failed to obtain access token.")
            st.stop()
            return None

        # Save tokens to Firebase
        firebase_integration.save_tokens_to_firebase(access_token, refresh_token, expires_at)

    # Check if access token is expired
    current_time = time.time()
    if current_time > expires_at:
        # Refresh the access token
        access_token, refresh_token, expires_at = refresh_access_token(APP_KEY, refresh_token)
        if not access_token:
            st.error("Failed to refresh access token.")
            st.stop()
            return None

        # Save updated tokens to Firebase
        firebase_integration.save_tokens_to_firebase(access_token, refresh_token, expires_at)

    # Initialize Dropbox client
    dbx = dropbox.Dropbox(
        oauth2_access_token=access_token,
        oauth2_refresh_token=refresh_token,
        app_key=APP_KEY
        # Do not include app_secret
    )
    return dbx

def get_initial_tokens(APP_KEY):
    """
    Initiate OAuth flow to get initial access and refresh tokens.
    """
    if 'auth_flow' not in st.session_state:
        # Use PKCE by setting use_pkce=True
        auth_flow = DropboxOAuth2FlowNoRedirect(
            APP_KEY,
            token_access_type='offline',
            use_pkce=True
        )
        # Store the auth_flow and authorize_url in session_state
        st.session_state['auth_flow'] = auth_flow
        authorize_url = auth_flow.start()
        st.session_state['authorize_url'] = authorize_url
    else:
        # Retrieve auth_flow and authorize_url from session_state
        auth_flow = st.session_state['auth_flow']
        authorize_url = st.session_state['authorize_url']

    st.info("Click the link below to authorize the application with Dropbox:")
    st.write(f"[Authorize with Dropbox]({authorize_url})")

    auth_code = st.text_input("Enter the authorization code here:")
    auth_submit = st.button("Submit Authorization Code")

    if auth_submit and auth_code:
        try:
            oauth_result = auth_flow.finish(auth_code.strip())
            access_token = oauth_result.access_token
            refresh_token = oauth_result.refresh_token
            # Dropbox does not provide expires_in; set a default value
            expires_in = 14400  # 4 hours in seconds
            expires_at = time.time() + expires_in
            # Clear auth_flow from session_state
            del st.session_state['auth_flow']
            del st.session_state['authorize_url']
            return access_token, refresh_token, expires_at
        except Exception as e:
            st.error(f"Error obtaining access token: {e}")
            return None, None, None
    else:
        st.stop()
        return None, None, None

def refresh_access_token(APP_KEY, refresh_token):
    """
    Refresh the access token using the refresh token.

    :return: Tuple of (access_token, refresh_token, expires_at)
    """
    token_url = "https://api.dropboxapi.com/oauth2/token"
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": APP_KEY
    }
    response = requests.post(token_url, data=data)
    if response.status_code == 200:
        tokens = response.json()
        access_token = tokens["access_token"]
        # Dropbox does not return a new refresh token; use the existing one
        expires_in = tokens["expires_in"]
        expires_at = time.time() + expires_in
        return access_token, refresh_token, expires_at
    else:
        st.error(f"Error refreshing access token: {response.text}")
        return None, None, None

def download_data_file(dbx):
    """
    Download the data file from Dropbox.

    :param dbx: Initialized Dropbox client
    :return: True if successful, False otherwise
    """
    try:
        metadata, res = dbx.files_download("/Work/McCall_Farms/McCall_Shared_Data/daily_data.xlsx")
        with open("daily_data.xlsx", "wb") as f:
            f.write(res.content)
        return True
    except dropbox.exceptions.ApiError as err:
        st.error(f"Failed to download file from Dropbox: {err}")
        return False

def upload_file_to_dropbox(dbx, local_file_path, dropbox_destination_path):
    """
    Upload a local file to Dropbox.

    :param dbx: Initialized Dropbox client
    :param local_file_path: Path to the local file
    :param dropbox_destination_path: Destination path in Dropbox
    :return: True if successful, False otherwise
    """
    import dropbox

    try:
        with open(local_file_path, 'rb') as f:
            dbx.files_upload(
                f.read(),
                dropbox_destination_path,
                mode=dropbox.files.WriteMode.overwrite
            )
        return True
    except Exception as e:
        st.error(f"Failed to upload file to Dropbox: {e}")
        return False

