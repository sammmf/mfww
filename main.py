import streamlit as st
from dashboard import run_dashboard  # Import the full dashboard layout with tabs
from modules import dropbox_integration

def run_app():
    st.title("Welcome to the Wastewater Dashboard")

    # Prompt for 4-digit access code
    code = st.text_input("Please enter your 4-digit access code:", max_chars=4, type='password')

    if st.button("Login"):
        if code in dropbox_integration.facility_codes:
            # Store the access code in session state after successful login
            st.session_state['facility_code'] = code
            st.success("Access granted.")
            # Call the full dashboard once access is granted
            run_dashboard()
        else:
            st.error("Invalid code. Please try again.")

if __name__ == "__main__":
    # If 'facility_code' is stored in session state, user has already logged in, so load the dashboard
    if 'facility_code' in st.session_state:
        run_dashboard()
    else:
        run_app()
