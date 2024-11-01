# modules/machine_learning.py

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.callback import TrainingCallback
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFECV
import plotly.express as px
import time
import logging

def run_machine_learning_tab(ml_data, configuration):
    st.header("Machine Learning Predictions and Optimization")

    # Initialize session state variables
    if 'feature_importances_calculated' not in st.session_state:
        st.session_state['feature_importances_calculated'] = False
    if 'selected_features' not in st.session_state:
        st.session_state['selected_features'] = []
    if 'learning_rate' not in st.session_state:
        st.session_state['learning_rate'] = 0.1
    if 'max_depth' not in st.session_state:
        st.session_state['max_depth'] = 6
    if 'n_estimators' not in st.session_state:
        st.session_state['n_estimators'] = 100

    # Step 1: Target Metric Selection
    target_options = [col for col in ml_data.columns if col != 'date']
    target_display_names = [col.replace('_', ' ').title() for col in target_options]
    target_name_mapping = dict(zip(target_display_names, target_options))
    selected_target_display = st.selectbox("Select Target Metric to Predict or Optimize", target_display_names)
    selected_target = target_name_mapping[selected_target_display]
    st.session_state['selected_target'] = selected_target

    if st.button("Run Machine Learning Pipeline"):
        with st.spinner('Running machine learning pipeline...'):
            results = run_machine_learning_pipeline(ml_data, configuration, selected_target)
        st.success("Machine learning pipeline completed.")

        # Display Logs
        st.subheader("Feature Elimination Logs")
        if results['logs']:
            for log in results['logs']:
                st.write(log)
        else:
            st.write("No features were eliminated.")

        # Display Selected Features
        st.subheader("Selected Features")
        selected_features_display = [f.replace('_', ' ').title() for f in results['selected_features']]
        st.write(", ".join(selected_features_display))

        # Display Model Performance
        st.subheader("Model Performance")
        rmse = results['model_results']['rmse']
        r2 = results['model_results']['r2']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RMSE", f"{rmse:.2f}", help="Root Mean Square Error: Lower values indicate better fit.")
        with col2:
            st.metric("R² Score", f"{r2:.2f}", help="R-squared: Proportion of variance explained by the model.")

        # Display Feature Importances
        st.subheader("Feature Importances")
        importance_df = results['feature_importances']
        importance_df['Feature'] = importance_df['Feature'].apply(lambda x: x.replace('_', ' ').title())
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
        st.plotly_chart(fig, use_container_width=True)

        # Display Predictions vs Actual
        st.subheader("Predictions vs Actual")
        y_test = results['model_results']['y_test']
        y_pred = results['model_results']['y_pred']
        X_test = results['model_results']['X_test']
        pred_df = pd.DataFrame({
            'Actual': y_test.reset_index(drop=True),
            'Predicted': y_pred,
            'Date': X_test.index
        }).sort_values(by='Date')
        fig = px.line(pred_df, x='Date', y=['Actual', 'Predicted'])
        st.plotly_chart(fig, use_container_width=True)

        # Download Option
        st.subheader("Download Results")
        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv'
        )

        # Store selected features in session state for optimization
        st.session_state['selected_features'] = results['selected_features']

        # Proceed to Optimization Section
        display_optimization_section(ml_data, configuration)

def display_optimization_section(ml_data, configuration):
    st.header("Optimization")
    st.write("Optimize parameters to achieve desired target values.")

    if 'selected_features' not in st.session_state or not st.session_state['selected_features']:
        st.warning("Please run the machine learning pipeline first.")
        return

    selected_features = st.session_state['selected_features']
    selected_target = st.session_state['selected_target']

    # Allow users to input desired target value
    desired_target_value = st.number_input(
        f"Desired {selected_target.replace('_', ' ').title()} Value:",
        value=float(ml_data[selected_target].mean())
    )
    st.session_state['desired_target_value'] = desired_target_value

    # Add a button to trigger optimization
    if st.button("Run Optimization"):
        with st.spinner('Running optimization...'):
            optimization_results = run_optimization(
                ml_data,
                configuration,
                selected_features,
                selected_target,
                st.session_state['desired_target_value']
            )
        st.success("Optimization completed.")

        # Display Optimized Parameters
        st.subheader("Optimized Parameters")
        optimized_df = pd.DataFrame({
            'Feature': [f.replace('_', ' ').title() for f in selected_features],
            'Optimized Value': optimization_results['optimized_values']
        })
        st.table(optimized_df)

        # Display Predicted Target Value
        st.subheader("Optimized Prediction")
        optimized_prediction = optimization_results['optimized_prediction']
        st.write(f"Predicted {selected_target.replace('_', ' ').title()}: {optimized_prediction:.2f}")

        # Download Option for Optimization Results
        st.subheader("Download Optimization Results")
        optimization_df = optimized_df.copy()
        optimization_df['Desired Target'] = st.session_state['desired_target_value']
        optimization_df['Predicted Target'] = optimized_prediction
        csv = optimization_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Optimization Results as CSV",
            data=csv,
            file_name='optimization_results.csv',
            mime='text/csv'
        )

def run_machine_learning_pipeline(ml_data, configuration, selected_target):
    """
    Main function to run the machine learning pipeline, including data preprocessing,
    feature selection with RFE, hyperparameter tuning, and final model training.
    """
    # Initialize logging
    logging.basicConfig(level=logging.INFO)

    # Step 1: Preprocess Data
    X, y = preprocess_data_for_modeling(ml_data, configuration, selected_target)

    # Step 2: Identify and Remove Highly Correlated Features
    correlated_features = get_correlated_features(X, threshold=0.9)
    if correlated_features:
        X.drop(columns=correlated_features, inplace=True)

    # Step 3: Perform Feature Selection using RFE
    selected_features, feature_ranking = perform_feature_selection(X, y)

    # Step 4: Hyperparameter Tuning
    best_params = hyperparameter_tuning(X[selected_features], y)

    # Step 5: Retrain Model with Selected Features and Best Parameters
    model_results = train_and_evaluate_model(X[selected_features], y, best_params)

    # Step 6: Calculate Feature Importances
    feature_importances = calculate_feature_importances(model_results['model'], selected_features)

    # Step 7: Prepare Logs
    logs = log_feature_elimination(correlated_features, feature_ranking)

    # Step 8: Prepare Results
    results = {
        'selected_features': selected_features,
        'feature_ranking': feature_ranking,
        'model_results': model_results,
        'feature_importances': feature_importances,
        'logs': logs
    }
    return results

def preprocess_data_for_modeling(ml_data, configuration, selected_target):
    """
    Preprocess the data for modeling.
    """
    # Clean 'adjustability' values
    configuration['adjustability'] = configuration['adjustability'].astype(str).str.strip().str.lower()

    # Get variable features
    variable_features = configuration[configuration['adjustability'] == 'variable']['feature_name'].tolist()

    # Exclude the target variable from the features
    features = [f for f in variable_features if f != selected_target]

    # Prepare data
    X = ml_data[features]
    y = ml_data[selected_target]

    # Handle missing values
    X = X.interpolate(method='linear', limit_direction='both').fillna(method='ffill').fillna(method='bfill')
    y = y.interpolate(method='linear', limit_direction='both').fillna(method='ffill').fillna(method='bfill')

    return X, y

def get_correlated_features(X, threshold=0.9):
    """
    Identify highly correlated features to consider removing them.
    """
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation greater than threshold
    correlated_features = [
        column for column in upper_tri.columns if any(upper_tri[column] > threshold)
    ]
    return correlated_features

def perform_feature_selection(X, y):
    """
    Perform Recursive Feature Elimination with Cross-Validation (RFECV) to select features.
    """
    # Initialize the model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.1,
        max_depth=6,
        n_estimators=100,
        verbosity=0
    )

    # Use TimeSeriesSplit for cross-validation due to time series data
    tscv = TimeSeriesSplit(n_splits=5)

    # Perform RFECV
    rfecv = RFECV(
        estimator=xgb_model,
        step=1,
        cv=tscv,
        scoring='r2',
        n_jobs=-1,
        min_features_to_select=5  # Ensure we have at least 5 features
    )

    rfecv.fit(X, y)

    # Get selected features and their rankings
    selected_features = X.columns[rfecv.support_].tolist()
    feature_ranking = pd.Series(rfecv.ranking_, index=X.columns).sort_values()

    return selected_features, feature_ranking

def hyperparameter_tuning(X, y):
    """
    Perform hyperparameter tuning using GridSearchCV.
    """
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', verbosity=0)

    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1],
    }

    tscv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='r2',
        n_jobs=-1
    )

    grid_search.fit(X, y)

    return grid_search.best_params_

def train_and_evaluate_model(X, y, params):
    """
    Train the model with the selected features and best hyperparameters, then evaluate its performance.
    """
    # Split data (keeping the time series order)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Initialize the model with best parameters
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        verbosity=0,
        **params
    )

    # Train the model
    xgb_model.fit(X_train, y_train)

    # Make predictions
    y_pred = xgb_model.predict(X_test)

    # Evaluate model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Prepare results
    results = {
        'model': xgb_model,
        'rmse': rmse,
        'r2': r2,
        'y_test': y_test,
        'y_pred': y_pred,
        'X_test': X_test
    }

    return results

def calculate_feature_importances(model, selected_features):
    """
    Calculate feature importances from the trained model.
    """
    importance = model.get_booster().get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'Feature': [f for f in selected_features if f in importance],
        'Importance': [importance.get(f, 0) for f in selected_features if f in importance]
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    return importance_df

def log_feature_elimination(correlated_features, feature_ranking):
    """
    Log the features that were eliminated and why.
    """
    logs = []
    if correlated_features:
        logs.append(f"Removed correlated features (threshold > 0.9): {', '.join(correlated_features)}")

    eliminated_features = feature_ranking[feature_ranking > 1].index.tolist()
    if eliminated_features:
        logs.append(f"Features eliminated by RFE: {', '.join(eliminated_features)}")

    return logs

def run_optimization(ml_data, configuration, selected_features, selected_target, desired_target_value):
    """
    Optimize feature values to achieve the desired target value.
    """
    # Prepare data
    X = ml_data[selected_features]
    y = ml_data[selected_target]

    # Handle missing values
    X = X.fillna(method='ffill').fillna(method='bfill')
    y = y.fillna(method='ffill').fillna(method='bfill')

    # Train the model on all data with best hyperparameters
    best_params = hyperparameter_tuning(X, y)
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        verbosity=0,
        **best_params
    )
    xgb_model.fit(X, y)

    # Run optimization
    bounds = bounds_factory(ml_data, configuration, selected_features)
    initial_guess = initial_guess_factory(ml_data, selected_features)
    objective_function = objective_function_factory(xgb_model, selected_features, desired_target_value)

    result = minimize(
        objective_function,
        x0=initial_guess,
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 100}
    )

    optimized_feature_values = result.x

    # Predicted value with optimized parameters
    feature_dict = {feature: [value] for feature, value in zip(selected_features, optimized_feature_values)}
    input_df = pd.DataFrame(feature_dict)
    optimized_prediction = xgb_model.predict(input_df)[0]

    optimization_results = {
        'optimized_values': optimized_feature_values,
        'optimized_prediction': optimized_prediction
    }

    return optimization_results

def objective_function_factory(model, selected_features, desired_target_value):
    def objective_function(feature_values):
        # Convert feature_values to DataFrame
        feature_dict = {feature: [value] for feature, value in zip(selected_features, feature_values)}
        input_df = pd.DataFrame(feature_dict)

        # Predict using the model
        predicted = model.predict(input_df)[0]

        # Objective is the squared difference between predicted and desired target value
        return (predicted - desired_target_value) ** 2
    return objective_function

def initial_guess_factory(ml_data, selected_features):
    # Initial guess: mean of the feature values
    initial_guess = [ml_data[feature].mean() for feature in selected_features]
    return initial_guess

def bounds_factory(ml_data, configuration, selected_features):
    # Get bounds for features from configuration
    config = configuration.set_index('feature_name')
    bounds = []
    for feature in selected_features:
        if feature in config.index:
            min_val = config.loc[feature, 'min']
            max_val = config.loc[feature, 'max']
            bounds.append((min_val, max_val))
        else:
            # If feature not in configuration, use data min and max
            min_val = ml_data[feature].min()
            max_val = ml_data[feature].max()
            bounds.append((min_val, max_val))
    return bounds
