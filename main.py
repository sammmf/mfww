# main.py

from dashboard import run_dashboard
import streamlit as st
from modules import dropbox_integration

def run_app():
    st.title("Welcome to Wastewater Dashboard")

    # Prompt for 4-digit code
    code = st.text_input("Please enter your 4-digit access code:", max_chars=4, type='password')

    if st.button("Login"):
        if code in dropbox_integration.facility_codes:
            st.session_state['facility_code'] = code
            st.success("Access granted.")
            # Proceed to the main dashboard
            run_dashboard()
        else:
            st.error("Invalid code. Please try again.")

def run_dashboard():
    pass

if __name__ == "__main__":
    if 'facility_code' in st.session_state:
        run_dashboard()
    else:
        run_app()
