import streamlit as st
import pandas as pd
from modules import dropbox_integration
from modules.machine_learning import run_machine_learning_tab
# Import other modules as needed (e.g., data_query, process_optimizer)

# Set the page configuration at the top of the script
st.set_page_config(page_title="McCall Farms Dashboard", layout="wide")

def run_app():
    st.title("Welcome to the McCall Farms Dashboard")

    # Prompt for 4-digit code
    code = st.text_input("Please enter your 4-digit access code:", max_chars=4, type='password')

    if st.button("Login"):
        if code in dropbox_integration.facility_codes:
            # Store the facility code and set logged_in to True
            st.session_state['facility_code'] = code
            st.session_state['logged_in'] = True
            st.success("Access granted.")
            # Since the script reruns with each interaction, the next run will load the dashboard
        else:
            st.error("Invalid code. Please try again.")

def run_dashboard():
    # Add a logout button
    if st.sidebar.button("Logout"):
        # Clear session state and rerun the app
        st.session_state.clear()
        st.session_state['logged_in'] = False
        st.experimental_rerun()

    # Initialize Dropbox client
    dbx = dropbox_integration.initialize_dropbox()
    if dbx is None:
        st.error("Failed to initialize Dropbox client.")
        return

    # Get the facility code from session state
    facility_code = st.session_state['facility_code']
    dropbox_file_path = dropbox_integration.facility_codes.get(facility_code)

    if not dropbox_file_path:
        st.error("No file path found for the given facility code.")
        return

    # Download the data file for the facility
    if dropbox_integration.download_data_file(dbx, dropbox_file_path):
        st.success("Data file downloaded successfully.")
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
        # Additional processing if necessary
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
        # Add dashboard content here

    with data_query_tab:
        st.header("Data Query")
        # Add data query content here

    with machine_learning_tab:
        st.header("Machine Learning")
        # Use ml_data and configuration from session state
        ml_data = st.session_state['ml_data']
        configuration = st.session_state['configuration']
        run_machine_learning_tab(ml_data, configuration)

    with process_optimizer_tab:
        st.header("Process Optimizer")
        # Add process optimizer content here

# Main application logic
if __name__ == "__main__":
    # Set up session state for 'logged_in' if not present
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if st.session_state['logged_in']:
        run_dashboard()
    else:
        run_app()
