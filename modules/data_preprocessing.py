# modules/data_preprocessing.py

import pandas as pd
import numpy as np
import re
from dateutil import parser
import streamlit as st

def clean_feature_names(name):
    """Clean and standardize feature names."""
    name = str(name).strip().lower()
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^a-z0-9_]', '', name)
    return name

def load_data(file_path):
    """Load ml_data and configuration sheets from Excel file."""
    try:
        ml_data = pd.read_excel(file_path, sheet_name='ml_data')
        configuration = pd.read_excel(file_path, sheet_name='configuration')
        return ml_data, configuration
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None, None

def preprocess_ml_data(ml_data):
    """Preprocess ml_data DataFrame."""
    # Clean feature names
    ml_data.columns = [clean_feature_names(col) for col in ml_data.columns]

    # Ensure 'date' column exists
    if 'date' not in ml_data.columns:
        st.error("The 'date' column is missing from the ml_data DataFrame.")
        return None

    # Parse 'date' column
    ml_data['date'] = ml_data['date'].apply(parse_dates)
    ml_data = ml_data.dropna(subset=['date'])

    # Verify 'date' column is datetime
    if not pd.api.types.is_datetime64_any_dtype(ml_data['date']):
        st.error("The 'date' column is not in datetime format.")
        return None

    # Exclude future dates
    today = pd.Timestamp('today').normalize()
    st.write(f"Today's date according to the system is: {today}")
    ml_data = ml_data[ml_data['date'] <= today]

    ml_data.sort_values('date', inplace=True)
    ml_data.reset_index(drop=True, inplace=True)
    return ml_data

def parse_dates(date_str):
    """Parse date strings into datetime objects."""
    try:
        return parser.parse(str(date_str), dayfirst=False, yearfirst=False)
    except (ValueError, TypeError):
        return pd.NaT

def validate_data(ml_data, configuration):
    """Validate that all features from configuration are in ml_data."""
    # Validate feature names
    config_features = configuration['feature_name'].tolist()
    ml_data_columns = ml_data.columns.tolist()

    missing_in_ml_data = list(set(config_features) - set(ml_data_columns))
    if missing_in_ml_data:
        st.warning(f"The following features are missing in ml_data and will be filled with NaN: {missing_in_ml_data}")
        for missing_feature in missing_in_ml_data:
            ml_data[missing_feature] = np.nan

    # Ensure 'date' is not in feature columns
    if 'date' in config_features:
        config_features.remove('date')

    # Replace zeros with NaN where zero is invalid
    zero_invalid_fields = configuration[configuration['zero_invalid'] == 'yes']['feature_name'].tolist()
    zero_invalid_fields = [f for f in zero_invalid_fields if f in ml_data.columns]
    ml_data[zero_invalid_fields] = ml_data[zero_invalid_fields].replace(0, np.nan)

    # Ensure numeric types
    for col in config_features:
        ml_data[col] = pd.to_numeric(ml_data[col], errors='coerce')
        if ml_data[col].dtype not in ['float64', 'int64']:
            st.warning(f"Column '{col}' could not be converted to numeric and will be filled with NaN.")
            ml_data[col] = np.nan

    # Interpolate missing values for variable features
    variable_features = configuration[configuration['adjustability'] == 'variable']['feature_name'].tolist()
    variable_features = [f for f in variable_features if f in ml_data.columns]
    ml_data[variable_features] = ml_data[variable_features].interpolate(method='linear', limit_direction='both', axis=0)

    # Drop rows with less than 70% valid data
    total_features = len(config_features)
    threshold = int(0.7 * total_features)
    ml_data['valid_feature_count'] = ml_data[config_features].notna().sum(axis=1)
    ml_data = ml_data[ml_data['valid_feature_count'] >= threshold]
    ml_data.drop(columns=['valid_feature_count'], inplace=True)

    if ml_data.empty:
        st.error("No data left after dropping rows with insufficient valid data.")
        return False

    ml_data.reset_index(drop=True, inplace=True)
    return True

