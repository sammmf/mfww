# modules/optimizer.py

import streamlit as st
import pandas as pd
from scipy.optimize import minimize, differential_evolution
import numpy as np
import xgboost as xgb
import joblib
import os
from modules import dropbox_integration
import tempfile
import plotly.express as px

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

    # Get adjustable features by filtering the configuration DataFrame
    adjustable_features = configuration[configuration['adjustability'] == 'variable']['feature_name'].tolist()

    if not adjustable_features:
        st.error("No adjustable features found in the configuration.")
        return

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
    adjustable_features = configuration[configuration['adjustability'] == 'variable']['feature_name'].tolist()
    if not adjustable_features:
        st.error("No adjustable features found in the configuration.")
        return

    # Get adjustable features included in the model
    adjustable_features = st.session_state.get('adjustable_features_in_model', [])
    if not adjustable_features:
        st.error("No adjustable features included in the model. Optimization cannot proceed.")
        return

    # Get fixed features (those that are both fixed and included in the model)
    selected_features = st.session_state.get('selected_features', [])
    fixed_features = [feature for feature in selected_features if feature not in adjustable_features]

     # Get selected features from the model
    selected_features = st.session_state.get('selected_features', [])
    if not selected_features:
        st.error("Selected features are not available.")
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
                optimization_goal,
                selected_features
            )
            if result is not None:
                display_optimization_results(
                    result,
                    adjustable_features,
                    fixed_features,
                    ml_data,
                    optimized_target_value,
                    prediction_std,
                    target_feature,
                    selected_features
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

from scipy.optimize import differential_evolution, minimize

def optimize_process(
    model,
    ml_data,
    adjustable_features,
    fixed_features,
    target_feature,
    desired_target_value,
    bounds,
    optimization_goal,
    selected_features
):
    # Prepare fixed variables
    fixed_values = ml_data[fixed_features].iloc[-1]

    # Get remaining features
    remaining_features = [feature for feature in selected_features if feature not in adjustable_features and feature not in fixed_features]
    remaining_values = ml_data[remaining_features].iloc[-1]

    # Objective function
    def objective_function(x):
        # Create a Series with all selected features
        input_data = pd.Series(index=selected_features, dtype=float)
        input_data[adjustable_features] = x
        input_data[fixed_features] = fixed_values.values
        input_data[remaining_features] = remaining_values.values

        # Ensure input_data has all features the model expects
        input_df = pd.DataFrame([input_data], columns=selected_features)

        # Predict using the model
        prediction = model.predict(input_df)[0]

        if optimization_goal == "Minimize":
            return prediction
        elif optimization_goal == "Maximize":
            return -prediction  # Negative for maximization
        else:  # Reach Desired Value
            return (prediction - desired_target_value) ** 2  # Squared error

    # Run Differential Evolution
    try:
        # Global optimization with Differential Evolution
        de_result = differential_evolution(
            objective_function,
            bounds=bounds,
            strategy='best1bin',
            maxiter=200,
            popsize=5,
            tol=0.05,
            mutation=(0.5, 1),
            recombination=0.7,
            disp=False
        )

        # Local optimization using the result from DE as the initial guess
        local_result = minimize(
            objective_function,
            de_result.x,
            bounds=bounds,
            method='L-BFGS-B',  # You can also try 'SLSQP' or other methods
            options={'disp': False, 'maxiter': 500}
        )

        # Decide which result to use based on the objective function value
        if local_result.fun < de_result.fun:
            final_result = local_result
        else:
            final_result = de_result

        # Get the optimized target value
        optimized_input = pd.Series(index=selected_features, dtype=float)
        optimized_input[adjustable_features] = final_result.x
        optimized_input[fixed_features] = fixed_values.values
        optimized_input[remaining_features] = remaining_values.values

        # Get prediction
        optimized_target_value, prediction_std = predict_with_uncertainty(model, optimized_input)

        return final_result, optimized_target_value, prediction_std

    except Exception as e:
        st.error(f"Optimization error: {e}")
        return None, None, None

def predict_with_uncertainty(model, input_data):
    """
    Predict the target value. Estimating uncertainty is not supported with the current model.
    """
    try:
        # Convert input_data to DataFrame
        input_df = pd.DataFrame([input_data])
        # Predict using the model
        prediction = model.predict(input_df)[0]
        prediction_std = 0  # Uncertainty estimation not available
        return prediction, prediction_std
    except Exception as e:
        st.error(f"Error calculating prediction: {e}")
        return None, None

def display_optimization_results(
    result,
    adjustable_features,
    fixed_features,
    ml_data,
    optimized_target_value,
    prediction_std,
    target_feature,
    selected_features
):
    if result.success:
        st.success("Optimization successful!")

        # Display optimized target feature value
        st.subheader(f"Optimized {target_feature}")
        if optimized_target_value is not None:
            st.write(f"Predicted {target_feature}: {optimized_target_value:.3f}")
            st.write("Note: Prediction uncertainty estimation is not available.")
        else:
            st.error("Unable to compute optimized target value.")

        # Provide visualizations
        display_optimization_visuals(
            result,
            adjustable_features,
            fixed_features,
            ml_data,
            optimized_target_value,
            target_feature,
            selected_features
        )

        # Additional Metrics
        with st.expander("Optimization Metrics"):
            st.write(f"Optimization Success: {result.success}")
            st.write(f"Optimization Message: {result.message}")
            st.write(f"Number of Iterations: {result.nit}")

        # Display fixed variables in a collapsible section
        with st.expander("Fixed Variables"):
            fixed_values = ml_data[fixed_features].iloc[-1].values
            fixed_df = pd.DataFrame({
                'Feature': fixed_features,
                'Value': fixed_values
            })
            st.table(fixed_df)

        # Display optimized controllable variables in a collapsible section
        with st.expander("Optimized Controllable Variables"):
            optimized_values = result.x
            controllable_df = pd.DataFrame({
                'Feature': adjustable_features,
                'Optimized Value': optimized_values
            })
            st.table(controllable_df)

    else:
        st.error("Optimization did not converge.")
        st.write("Message:", result.message)

def display_optimization_visuals(
    result,
    adjustable_features,
    fixed_features,
    ml_data,
    optimized_target_value,
    target_feature,
    selected_features
):
    st.subheader("Optimization Visualization")

    # Prepare data for visualization
    # We'll compare the target feature before and after optimization
    # For simplicity, let's assume we can simulate the original target value using the model

    # Get the latest data point
    latest_data = ml_data.iloc[-1].copy()

    # Prepare input data for original prediction
    original_input = latest_data[selected_features]

    # Predict original target value
    original_prediction = st.session_state['trained_model'].predict(pd.DataFrame([original_input]))[0]

    # Prepare input data for optimized prediction
    optimized_input = pd.Series(index=selected_features, dtype=float)
    optimized_input[adjustable_features] = result.x
    optimized_input[fixed_features] = latest_data[fixed_features].values
    # For any remaining features, use the latest values
    remaining_features = [feature for feature in selected_features if feature not in adjustable_features and feature not in fixed_features]
    optimized_input[remaining_features] = latest_data[remaining_features].values

    # Predict optimized target value (already calculated, but we can recompute for consistency)
    optimized_prediction = optimized_target_value

    # Create a DataFrame for visualization
    comparison_df = pd.DataFrame({
        'Scenario': ['Original', 'Optimized'],
        target_feature: [original_prediction, optimized_prediction]
    })

    # Plot the comparison
    fig = px.bar(
        comparison_df,
        x='Scenario',
        y=target_feature,
        text=target_feature,
        title=f'Comparison of {target_feature} Before and After Optimization'
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='auto')
    st.plotly_chart(fig, use_container_width=True)
