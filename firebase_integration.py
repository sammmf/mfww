import firebase_admin
from firebase_admin import credentials, firestore

# Load the service account key
cred = credentials.Certificate("/serviceAccountKey.json")

# Initialize Firebase app
firebase_admin.initialize_app(cred)

# Initialize Firestore database client
db = firestore.client()
