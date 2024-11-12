# modules/optimizer.py

import streamlit as st
import pandas as pd
from scipy.optimize import minimize
import numpy as np
import xgboost as xgb
import joblib
import os
from modules import dropbox_integration
import tempfile

def load_model_from_dropbox(dbx, dropbox_model_path):
    """
    Download the trained model from Dropbox and load it.

    Parameters:
    - dbx: Dropbox client instance.
    - dropbox_model_path: Path to the model file in Dropbox.

    Returns:
    - model: The loaded model object, or None if failed.
    """
    try:
        metadata, res = dbx.files_download(dropbox_model_path)
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(res.content)
            tmp_file_path = tmp_file.name

        model = joblib.load(tmp_file_path)
        os.unlink(tmp_file_path)  # Clean up the temporary file
        st.success(f"Model loaded from Dropbox path: {dropbox_model_path}")
        return model
    except dropbox.exceptions.ApiError as err:
        st.error(f"Failed to download model from Dropbox: {err}")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

def run_process_optimizer(ml_data, configuration):
    st.header("Process Optimizer")

    # Check if the user is logged in and get the facility code
    if 'facility_code' not in st.session_state:
        st.error("Facility code not found. Please log in again.")
        return

    facility_code = st.session_state['facility_code']

    # Map facility codes to Dropbox model paths
    facility_model_paths = {
        '3876': '/Work/McCall_Farms/McCall_Shared_Data/trained_model.joblib',
        '7354': '/sage/trained_model.joblib',
        '2381': '/scp/trained_model.joblib',
    }

    if facility_code not in facility_model_paths:
        st.error(f"No model path configured for facility code: {facility_code}")
        return

    dropbox_model_path = facility_model_paths[facility_code]

    # Initialize Dropbox client
    dbx = dropbox_integration.initialize_dropbox()
    if not dbx:
        st.error("Failed to initialize Dropbox.")
        return

    # Load the model if not in session state
    if 'trained_model' not in st.session_state:
        trained_model = load_model_from_dropbox(dbx, dropbox_model_path)
        if trained_model is not None:
            st.session_state['trained_model'] = trained_model
        else:
            st.error("Trained model not found. Please run the machine learning pipeline first.")
            return
    else:
        trained_model = st.session_state['trained_model']

    # Get adjustable features by filtering the configuration DataFrame
    adjustable_features = configuration[configuration['adjustability'] == 'Variable']['feature_name'].tolist()
    if not adjustable_features:
        st.error("No adjustable features found in the configuration.")
        return

    # Verify that all adjustable features are present in ml_data
    missing_features = [feature for feature in adjustable_features if feature not in ml_data.columns]
    if missing_features:
        st.error(f"The following adjustable features are missing from ml_data: {missing_features}")
        return

    # Get min and max bounds for adjustable features
    bounds = get_bounds(configuration, adjustable_features)
    if bounds is None:
        return  # Error messages are handled within get_bounds

    # Get fixed features
    fixed_features = configuration[configuration['adjustability'] != 'variable']['feature_name'].tolist()

    # User selects the target feature to optimize
    target_feature = st.selectbox("Select the target feature to optimize", ml_data.columns)

    # User inputs the desired target value
    desired_target_value = st.number_input(f"Enter the desired value for {target_feature}", value=0.0)

    # Optional: Ask if lower or higher values are preferred
    optimization_goal = st.radio(
        "Optimization Goal",
        ("Minimize", "Maximize", "Reach Desired Value"),
        index=2  # Default to "Reach Desired Value"
    )

    if st.button("Run Optimization"):
        with st.spinner("Running optimization..."):
            result, optimized_target_value, prediction_std = optimize_process(
                trained_model,
                ml_data,
                adjustable_features,
                fixed_features,
                target_feature,
                desired_target_value,
                bounds,
                optimization_goal
            )
            if result is not None:
                display_optimization_results(
                    result,
                    adjustable_features,
                    fixed_features,
                    ml_data,
                    optimized_target_value,
                    prediction_std,
                    target_feature
                )
            else:
                st.error("Optimization failed. Please check your inputs.")

def get_bounds(configuration, adjustable_features):
    bounds = []
    for feature in adjustable_features:
        # Find the row in configuration DataFrame for this feature
        feature_row = configuration[configuration['feature_name'] == feature]
        if feature_row.empty:
            st.error(f"Feature '{feature}' not found in configuration.")
            return None
        min_val = feature_row['min'].values[0]
        max_val = feature_row['max'].values[0]
        if pd.isnull(min_val) or pd.isnull(max_val):
            st.error(f"Bounds not specified for feature '{feature}'.")
            return None
        bounds.append((min_val, max_val))
    return bounds

def optimize_process(
    model,
    ml_data,
    adjustable_features,
    fixed_features,
    target_feature,
    desired_target_value,
    bounds,
    optimization_goal
):
    # Initial guess: mean of adjustable feature values
    initial_guess = ml_data[adjustable_features].mean().values

    # Prepare fixed variables
    fixed_values = ml_data[fixed_features].iloc[-1]

    # Objective function
    def objective_function(x):
        # Create a DataFrame with adjustable and fixed features
        input_data = pd.Series(index=adjustable_features + fixed_features)
        input_data[adjustable_features] = x
        input_data[fixed_features] = fixed_values.values

        # Predict using the model
        prediction = model.predict(pd.DataFrame([input_data]))[0]

        if optimization_goal == "Minimize":
            return prediction
        elif optimization_goal == "Maximize":
            return -prediction  # Negative for maximization
        else:  # Reach Desired Value
            return (prediction - desired_target_value) ** 2  # Squared error

    # Run optimization
    try:
        result = minimize(
            objective_function,
            initial_guess,
            bounds=bounds,
            method='SLSQP',  # Sequential Least Squares Programming
            options={'disp': False}
        )

        # Get the optimized target value and prediction standard deviation
        optimized_input = pd.Series(index=adjustable_features + fixed_features)
        optimized_input[adjustable_features] = result.x
        optimized_input[fixed_features] = fixed_values.values

        # Get prediction and uncertainty
        optimized_target_value, prediction_std = predict_with_uncertainty(model, optimized_input)

        return result, optimized_target_value, prediction_std
    except Exception as e:
        st.error(f"Optimization error: {e}")
        return None, None, None

def predict_with_uncertainty(model, input_data):
    """
    Predict the target value and estimate uncertainty using the model.
    """
    # For XGBoost, we can use DMatrix and get the prediction standard deviation
    try:
        # Convert input data to DMatrix
        dmatrix = xgb.DMatrix(pd.DataFrame([input_data]))
        # Predict using the model
        prediction = model.predict(dmatrix)
        prediction_mean = prediction[0]
        prediction_std = 0  # XGBoost does not provide prediction uncertainty by default
        return prediction_mean, prediction_std
    except Exception as e:
        st.error(f"Error calculating prediction uncertainty: {e}")
        return None, None

def display_optimization_results(
    result,
    adjustable_features,
    fixed_features,
    ml_data,
    optimized_target_value,
    prediction_std,
    target_feature
):
    if result.success:
        st.success("Optimization successful!")

        # Display optimized controllable variables
        st.subheader("Optimized Controllable Variables")
        optimized_values = result.x
        controllable_df = pd.DataFrame({
            'Feature': adjustable_features,
            'Optimized Value': optimized_values
        })
        st.table(controllable_df)

        # Display fixed variables
        st.subheader("Fixed Variables")
        fixed_values = ml_data[fixed_features].iloc[-1].values
        fixed_df = pd.DataFrame({
            'Feature': fixed_features,
            'Value': fixed_values
        })
        st.table(fixed_df)

        # Display optimized target feature value with confidence interval
        st.subheader(f"Optimized {target_feature}")
        confidence_interval = 1.96 * prediction_std  # For approximately 95% confidence
        st.write(f"Predicted {target_feature}: {optimized_target_value:.3f} ± {confidence_interval:.3f}")
        st.write(f"Confidence Interval (95%): [{optimized_target_value - confidence_interval:.3f}, {optimized_target_value + confidence_interval:.3f}]")

        # Additional Metrics
        st.subheader("Optimization Metrics")
        st.write(f"Optimization Success: {result.success}")
        st.write(f"Optimization Message: {result.message}")
        st.write(f"Number of Iterations: {result.nit}")
    else:
        st.error("Optimization did not converge.")
        st.write("Message:", result.message)
