# dashboard.py

import streamlit as st
import pandas as pd
from modules import dropbox_integration
from modules import data_preprocessing
from modules import configuration_handler
from modules import scoring
from modules import visualization
from modules import machine_learning
from modules import optimizer

# Set page configuration at the very top
st.set_page_config(page_title="Wastewater Treatment Plant Dashboard", layout="wide")

# Define the mapping between codes and Dropbox file paths
facility_codes = {
    '3876': '/Work/McCall_Farms/McCall_Shared_Data/daily_data.xlsx',
    '7354': '/sage/daily_data.xlsx',
    '2381': '/scp/daily_data.xlsx',
}

def run_dashboard():
    # Check if the user is logged in
    if 'facility_code' not in st.session_state:
        # Display the login screen
        st.title("Wastewater Treatment Plant Dashboard")
        st.write("Please enter your 4-digit access code to proceed.")

        # Use a form to capture user input and handle submission
        with st.form(key='login_form'):
            code = st.text_input("Access Code:", max_chars=4, type='password')
            submit_button = st.form_submit_button(label='Login')

        if submit_button:
            if code in facility_codes:
                st.session_state['facility_code'] = code
                st.session_state['logged_in'] = True
                st.success("Access granted.")
                # No need to rerun; the app will rerun automatically
            else:
                st.error("Invalid code. Please try again.")

        return  # Stop execution until the user logs in

    # User is logged in; proceed with the dashboard
    facility_code = st.session_state['facility_code']
    dropbox_file_path = facility_codes[facility_code]

    st.title("Wastewater Treatment Plant Dashboard")

    # Initialize Dropbox and download data
    dbx = dropbox_integration.initialize_dropbox()
    if not dbx:
        st.error("Failed to initialize Dropbox.")
        return

    data_downloaded = dropbox_integration.download_data_file(dbx, dropbox_file_path)
    if not data_downloaded:
        st.error("Failed to download data file from Dropbox.")
        return

    # Load data
    ml_data, configuration = data_preprocessing.load_data("daily_data.xlsx")
    if ml_data is None or configuration is None:
        st.error("Failed to load data.")
        return

    # Preprocess data
    ml_data = data_preprocessing.preprocess_ml_data(ml_data)
    if ml_data is None:
        st.error("Failed to preprocess ML data.")
        return

    configuration = configuration_handler.preprocess_configuration(configuration)
    if configuration is None:
        st.error("Failed to preprocess configuration.")
        return

    # Validate data
    valid = data_preprocessing.validate_data(ml_data, configuration)
    if not valid:
        st.error("Data validation failed.")
        return

    # Compute scores
    plant_scores = scoring.compute_plant_scores(ml_data, configuration)
    if plant_scores is None:
        st.error("Failed to compute plant scores.")
        return

    # Extract variables for visualization
    unit_process_scores = plant_scores.get('unit_process_scores', {})
    formatted_unit_process_names = plant_scores.get('formatted_unit_process_names', {})
    data_completeness = plant_scores.get('data_completeness', pd.Series())

    # Calculate scores over time
    scores_over_time = scoring.calculate_scores_over_time(ml_data, configuration)
    if scores_over_time is None:
        st.error("Failed to calculate scores over time.")
        return

    # Create Tabs and unpack them
    tab_dashboard, tab_data_query, tab_machine_learning, tab_process_optimizer = st.tabs([
        "Dashboard", "Data Query", "Machine Learning", "Process Optimizer"
    ])

    with tab_dashboard:
        # Visualize results
        visualization.display_dashboard(
            plant_scores,
            scores_over_time,
            unit_process_scores,
            formatted_unit_process_names,
            data_completeness
        )

    with tab_data_query:
        visualization.display_data_query(ml_data)
        visualization.display_recent_complete_day_summary(ml_data)

    with tab_machine_learning:
        machine_learning.run_machine_learning_tab(ml_data, configuration)

    with tab_process_optimizer:
       optimizer.run_process_optimizer(ml_data, configuration)

if __name__ == "__main__":
    run_dashboard()
