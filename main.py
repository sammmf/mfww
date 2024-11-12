import streamlit as st
import pandas as pd
from modules import dropbox_integration
from modules.machine_learning import run_machine_learning_tab
# Import other modules as needed

# Set page configuration at the very top
st.set_page_config(page_title="McCall Farms Dashboard", layout="wide")

def run_app():
    st.title("Welcome to the McCall Farms Dashboard")

    # Prompt for 4-digit code
    if 'code' not in st.session_state:
        st.session_state['code'] = ''

    code = st.text_input("Please enter your 4-digit access code:", max_chars=4, type='password')

    if st.button("Login"):
        if code in dropbox_integration.facility_codes:
            # Set session state variables
            st.session_state['facility_code'] = code
            st.session_state['logged_in'] = True
            # Clear any previous messages
            st.session_state['login_message'] = 'Access granted.'
        else:
            st.session_state['login_message'] = 'Invalid code. Please try again.'

    # Display login message
    if 'login_message' in st.session_state:
        st.info(st.session_state['login_message'])

    # Proceed to dashboard if logged in
    if st.session_state.get('logged_in'):
        run_dashboard()

def run_dashboard():
    st.title("McCall Farms Dashboard")

    # Add a logout button
    if st.sidebar.button("Logout"):
        # Clear session state and reload
        st.session_state.clear()
        st.experimental_set_query_params()  # Reset query parameters
        st.experimental_rerun()  # Rerun the script

    # Initialize Dropbox client
    dbx = dropbox_integration.initialize_dropbox()
    if dbx is None:
        st.error("Failed to initialize Dropbox client.")
        return

    # Get the facility code from session state
    facility_code = st.session_state.get('facility_code')
    dropbox_file_path = dropbox_integration.facility_codes.get(facility_code)

    if not dropbox_file_path:
        st.error("No file path found for the given facility code.")
        return

    # Download the data file for the facility
    if dropbox_integration.download_data_file(dbx, dropbox_file_path):
        # Load the data into your application
        ml_data, configuration = load_data_from_excel("daily_data.xlsx")
        if ml_data is not None and configuration is not None:
            # Store data in session state for access in tabs
            st.session_state['ml_data'] = ml_data
            st.session_state['configuration'] = configuration
            # Display the dashboard with tabs
            display_tabs()
        else:
            st.error("Failed to load data from Excel file.")
    else:
        st.error("Failed to download data file from Dropbox.")

def load_data_from_excel(file_path):
    try:
        # Load ml_data and configuration from the Excel file
        ml_data = pd.read_excel(file_path, sheet_name='ml_data')
        configuration = pd.read_excel(file_path, sheet_name='configuration')
        return ml_data, configuration
    except FileNotFoundError:
        st.error(f"The file '{file_path}' was not found.")
        return None, None
    except ValueError as e:
        st.error(f"Error reading Excel file: {e}")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None, None

def display_tabs():
    # Create tabs
    tabs = st.tabs(["Dashboard", "Data Query", "Machine Learning", "Process Optimizer"])

    # Unpack tabs
    dashboard_tab, data_query_tab, machine_learning_tab, process_optimizer_tab = tabs

    with dashboard_tab:
        st.header("Dashboard")
        st.write("Content for the Dashboard tab.")

    with data_query_tab:
        st.header("Data Query")
        st.write("Content for the Data Query tab.")

    with machine_learning_tab:
        st.header("Machine Learning")
        # Use ml_data and configuration from session state
        ml_data = st.session_state.get('ml_data')
        configuration = st.session_state.get('configuration')
        if ml_data is not None and configuration is not None:
            run_machine_learning_tab(ml_data, configuration)
        else:
            st.error("Data not available. Please log in again.")

    with process_optimizer_tab:
        st.header("Process Optimizer")
        st.write("Content for the Process Optimizer tab.")

if __name__ == "__main__":
    # Initialize 'logged_in' in session state if not present
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if st.session_state['logged_in']:
        run_dashboard()
    else:
        run_app()
