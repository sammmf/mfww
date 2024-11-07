import firebase_admin
from firebase_admin import credentials, firestore, storage
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
            "private_key": st.secrets["private_key"].replace("\\n", "\n"),  # Make sure to replace escaped newlines
            "client_email": st.secrets["client_email"],
            "client_id": st.secrets["client_id"],
            "auth_uri": st.secrets["auth_uri"],
            "token_uri": st.secrets["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["client_x509_cert_url"]
        }
        
        # Use the credentials from Streamlit secrets to initialize the app
        cred = credentials.Certificate(firebase_creds)
        firebase_admin.initialize_app(cred, {
            'storageBucket': f"{st.secrets['project_id']}.appspot.com"
        })

def get_firestore_client():
    """
    Get a client for the Firestore database.
    Requires Firebase to be initialized first.
    """
    initialize_firebase()
    return firestore.client()

def upload_file_to_firebase(file, filename):
    """
    Uploads a file to Firebase Storage.
    
    :param file: File-like object to upload
    :param filename: Name to use for the file in Firebase Storage
    """
    initialize_firebase()
    bucket = storage.bucket()
    blob = bucket.blob(filename)
    blob.upload_from_file(file)
    blob.make_public()  # Optional: make the file public, remove if not needed
    
    # Return the public URL of the uploaded file
    return blob.public_url

def download_file_from_firebase(filename, local_path):
    """
    Downloads a file from Firebase Storage.
    
    :param filename: Name of the file in Firebase Storage
    :param local_path: Local path where the file should be saved
    """
    initialize_firebase()
    bucket = storage.bucket()
    blob = bucket.blob(filename)
    blob.download_to_filename(local_path)
    st.success(f"File downloaded to {local_path}")

# Usage example in your main app
def main():
    st.title("Firebase Integration Example")
    
    # Example: Upload a file
    uploaded_file = st.file_uploader("Choose a file to upload to Firebase")
    if uploaded_file is not None:
        url = upload_file_to_firebase(uploaded_file, uploaded_file.name)
        st.write("File uploaded to Firebase! Public URL:", url)
    
    # Example: Download a file
    if st.button("Download example file"):
        filename = "example.txt"  # Replace with your file's name in Firebase
        download_file_from_firebase(filename, f"./{filename}")
