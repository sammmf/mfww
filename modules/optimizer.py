# modules/optimizer.py

import streamlit as st
import pandas as pd
from scipy.optimize import minimize
import numpy as np
import xgboost as xgb
import joblib
import os

def run_process_optimizer(ml_data, configuration, model):
    st.header("Process Optimizer")
 
    # Path to the model file
    model_filename = os.path.join('/Work/McCall_Farms/McCall_Shared_Data', 'trained_model.joblib')

    # Load the model if not in session state
    if 'trained_model' not in st.session_state:
        trained_model = load_model(model_filename)
        if trained_model is not None:
            st.session_state['trained_model'] = trained_model
        else:
            st.error("Trained model not found. Please run the machine learning pipeline first.")
            return
    else:
        trained_model = st.session_state['trained_model']
        
    # Get adjustable features
    adjustable_features = list(configuration.get('adjustable_features', {}).keys())
    if not adjustable_features:
        st.error("No adjustable features found in the configuration.")
        return

    # Get fixed features
    fixed_features = list(configuration.get('fixed_features', {}).keys())

    # Get min and max bounds for adjustable features
    bounds = get_bounds(configuration, adjustable_features)

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
                model,
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
        feature_config = configuration['adjustable_features'][feature]
        min_val = feature_config.get('min', None)
        max_val = feature_config.get('max', None)
        if min_val is None or max_val is None:
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
    # For RandomForestRegressor, we can use the individual estimators
    try:
        # Get predictions from all individual trees
        tree_predictions = np.array([estimator.predict(pd.DataFrame([input_data]))[0] for estimator in model.estimators_])
        prediction_mean = tree_predictions.mean()
        prediction_std = tree_predictions.std()
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
