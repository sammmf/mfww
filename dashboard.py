# dashboard.py

import streamlit as st
import pandas as pd
from modules import dropbox_integration
from modules import data_preprocessing
from modules import configuration_handler
from modules import scoring
from modules import visualization
from modules import machine_learning  # New module

def run_dashboard():
    st.set_page_config(page_title="Wastewater Treatment Plant Dashboard", layout="wide")
    st.title("Wastewater Treatment Plant Dashboard")

    # Initialize Dropbox and download data
    dbx = dropbox_integration.initialize_dropbox()
    if not dbx:
        st.error("Failed to initialize Dropbox.")
        return

    data_downloaded = dropbox_integration.download_data_file(dbx)
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
    data_completeness = plant_scores.get('data_completeness', pd.Series(dtype='float64'))
    
    # Calculate scores over time
    scores_over_time = scoring.calculate_scores_over_time(ml_data, configuration)
    if scores_over_time is None:
        st.error("Failed to calculate scores over time.")
        return

    # Create Tabs
    tabs = st.tabs(["Dashboard", "Data Query", "Machine Learning"])

    with tabs[0]:
        # Visualize results
        visualization.display_dashboard(
            plant_scores,
            scores_over_time,
            unit_process_scores,
            formatted_unit_process_names,
            data_completeness
        )

    with tabs[1]:
        visualization.display_data_query(ml_data)
        visualization.display_recent_complete_day_summary(ml_data)

    with tabs[2]:
        # Machine Learning Tab
        machine_learning.run_machine_learning_tab(ml_data, configuration)

if __name__ == "__main__":
    run_dashboard()
