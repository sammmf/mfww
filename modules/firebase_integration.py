# modules/firebase_integration.py

import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st
import json

def initialize_firebase():
    """
    Initialize Firebase app using credentials from Streamlit's secrets.
    If the app is already initialized, this function will do nothing.
    """
    if not firebase_admin._apps:
        # Convert the secrets into a dict format required by Firebase
        firebase_creds = {
            "type": st.secrets["type"],
            "project_id": st.secrets["project_id"],
            "private_key_id": st.secrets["private_key_id"],
            "private_key": st.secrets["private_key"].replace("\\n", "\n"),  # Replace escaped newlines
            "client_email": st.secrets["client_email"],
            "client_id": st.secrets["client_id"],
            "auth_uri": st.secrets["auth_uri"],
            "token_uri": st.secrets["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["client_x509_cert_url"]
        }
        
        # Use the credentials from Streamlit secrets to initialize the app
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
