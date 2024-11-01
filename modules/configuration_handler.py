# modules/configuration_handler.py

import pandas as pd
import streamlit as st
from modules.data_preprocessing import clean_feature_names

def preprocess_configuration(configuration):
    """Preprocess the configuration DataFrame."""
    # Clean feature names
    configuration['feature_name'] = configuration['feature_name'].apply(clean_feature_names)

    # Clean and standardize text columns
    text_columns = ['adjustability', 'zero_invalid', 'difficulty_score_parameter', 'preference_direction']
    for col in text_columns:
        configuration[col] = configuration[col].str.strip().str.lower()

    # Ensure every feature has a preference direction
    missing_preference = configuration[configuration['preference_direction'].isna()]['feature_name'].tolist()
    if missing_preference:
        st.error(f"The following features are missing a 'preference_direction': {missing_preference}")
        return None

    return configuration
