import firebase_admin
from firebase_admin import credentials, firestore

# Load the service account key
cred = credentials.Certificate("/serviceAccountKey.json")

# Initialize Firebase app
firebase_admin.initialize_app(cred)

# Initialize Firestore database client
db = firestore.client()

def save_tokens(access_token, refresh_token, expires_at):
    """
    Store tokens in Firestore with an expiration timestamp.
    """
    # Save tokens in a collection called "tokens" and a document called "dropbox_tokens"
    db.collection("tokens").document("dropbox_tokens").set({
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expires_at
    })

def load_tokens():
    """
    Retrieve tokens from Firestore.
    """
    # Get the document "dropbox_tokens" from the "tokens" collection
    doc = db.collection("tokens").document("dropbox_tokens").get()
    if doc.exists:
        data = doc.to_dict()
        return data["access_token"], data["refresh_token"], data["expires_at"]
    else:
        # If the document doesn't exist, return None values
        return None, None, None
