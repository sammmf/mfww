# modules/firebase_integration.py

import firebase_admin
from firebase_admin import credentials, firestore, storage
import streamlit as st

def initialize_firebase():
    """
    Initialize Firebase app using credentials from Streamlit's secrets.
    If the app is already initialized, this function will do nothing.
    """
    if not firebase_admin._apps:
        # Access the credentials under 'firebase' in st.secrets
        firebase_secrets = st.secrets["firebase"]
        
        # Convert the secrets into a dict format required by Firebase
        firebase_creds = {
            "type": firebase_secrets["type"],
            "project_id": firebase_secrets["project_id"],
            "private_key_id": firebase_secrets["private_key_id"],
            "private_key": firebase_secrets["private_key"].replace("\\n", "\n"),
            "client_email": firebase_secrets["client_email"],
            "client_id": firebase_secrets["client_id"],
            "auth_uri": firebase_secrets["auth_uri"],
            "token_uri": firebase_secrets["token_uri"],
            "auth_provider_x509_cert_url": firebase_secrets["auth_provider_x509_cert_url"],
            "client_x509_cert_url": firebase_secrets["client_x509_cert_url"]
            # Include 'universe_domain' if it's required
        }
        
        # Use the credentials to initialize the app
        cred = credentials.Certificate(firebase_creds)
        firebase_admin.initialize_app(cred)
        
def get_firestore_client():
    """
    Get a client for the Firestore database.
    Requires Firebase to be initialized first.
    """
    initialize_firebase()
    return firestore.client()

def save_tokens_to_firebase(access_token, refresh_token, expires_at):
    """
    Save Dropbox tokens to Firebase Firestore.

    :param access_token: The Dropbox access token
    :param refresh_token: The Dropbox refresh token
    :param expires_at: Expiration time of the access token (timestamp)
    """
    db = get_firestore_client()
    tokens_doc = db.collection('tokens').document('dropbox')
    tokens_data = {
        'access_token': access_token,
        'refresh_token': refresh_token,
        'expires_at': expires_at
    }
    tokens_doc.set(tokens_data)

def get_tokens_from_firebase():
    """
    Retrieve Dropbox tokens from Firebase Firestore.

    :return: A dictionary containing 'access_token', 'refresh_token', and 'expires_at'
    """
    db = get_firestore_client()
    tokens_doc = db.collection('tokens').document('dropbox')
    doc = tokens_doc.get()
    if doc.exists:
        return doc.to_dict()
    else:
        return None
