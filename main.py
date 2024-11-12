# main.py

from dashboard import run_dashboard
import streamlit as st
from modules import dropbox_integration
from modules.machine_learning import run_machine_learning_tab
# Import other modules as needed

def run_app():
    st.title("Welcome to the McCall Farms Dashboard")

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
    # Initialize Dropbox client
    dbx = dropbox_integration.initialize_dropbox()

    # Get the facility code from session state
    facility_code = st.session_state['facility_code']
    dropbox_file_path = dropbox_integration.facility_codes[facility_code]

    # Download the data file for the facility
    if dropbox_integration.download_data_file(dbx, dropbox_file_path):
        st.success("Data file downloaded successfully.")
        # Load the data into your application
        ml_data, configuration = load_data_from_excel("daily_data.xlsx")
        # Run the machine learning tab with the facility-specific data
        run_machine_learning_tab(ml_data, configuration)
        # Run other tabs or modules as needed
    else:
        st.error("Failed to load data file.")

def load_data_from_excel(file_path):
    # Load ml_data and configuration from the Excel file
    ml_data = pd.read_excel(file_path, sheet_name='ml_data')
    configuration = pd.read_excel(file_path, sheet_name='configuration')
    # Additional processing if necessary
    return ml_data, configuration

if __name__ == "__main__":
    if 'facility_code' in st.session_state:
        run_dashboard()
    else:
        run_app()
