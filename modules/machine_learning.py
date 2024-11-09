# modules/machine_learning.py

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFECV
import plotly.express as px
from datetime import datetime
import logging
import joblib as joblib
import os as os

# Configure logging
logging.basicConfig(
    filename='ml_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def run_machine_learning_tab(ml_data, configuration):
    """
    This function defines the Machine Learning tab in the Streamlit app.
    It allows users to select a target variable, run the ML pipeline, and view results.
    """
    st.header("Machine Learning Predictions and Optimization")

    # Initialize session state variables
    if 'ml_pipeline_ran' not in st.session_state:
        st.session_state['ml_pipeline_ran'] = False

    # Step 1: Target Metric Selection
    target_options = [col for col in ml_data.columns if col != 'date']
    target_display_names = [col.replace('_', ' ').title() for col in target_options]
    target_name_mapping = dict(zip(target_display_names, target_options))
    selected_target_display = st.selectbox("Select Target Metric to Predict or Optimize", target_display_names)
    selected_target = target_name_mapping[selected_target_display]
    st.session_state['selected_target'] = selected_target

    if st.button("Run Machine Learning Pipeline"):
        # Initialize progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            # Run the pipeline with progress indicators
            results = run_machine_learning_pipeline(
                ml_data, configuration, selected_target, progress_bar, status_text
            )
            st.success("Machine learning pipeline completed.")
            st.session_state['ml_pipeline_ran'] = True
            st.session_state['ml_results'] = results

            # Store the trained model in session state for use in the optimizer
            st.session_state['trained_model'] = results['model_results']['model']

            # Display Logs
            st.subheader("Feature Selection Logs")
            if results['logs']:
                for log in results['logs']:
                    st.write(log)
            else:
                st.write("No features were combined.")

            # Display Selected Features
            st.subheader("Selected Features")
            selected_features_display = [f.replace('_', ' ').title() for f in results['selected_features']]
            st.write(", ".join(selected_features_display))
            st.session_state['selected_features'] = results['selected_features']

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

        except Exception as e:
            st.error(f"An error occurred during the pipeline: {e}")
            st.session_state['ml_pipeline_ran'] = False

    # Proceed to Optimization Section if ML pipeline ran successfully
    if st.session_state.get('ml_pipeline_ran', False):
        # You can add optimization code here if needed
        pass

def run_machine_learning_pipeline(ml_data, configuration, selected_target, progress_bar, status_text):
    import time  # Import time for sleep in example (remove in production)
    total_steps = 7
    current_step = 0

    try:
        # Step 1: Preprocess Data
        status_text.text("Step 1/7: Preprocessing data...")
        X, y = preprocess_data_for_modeling(ml_data, selected_target)
        current_step += 1
        progress_bar.progress(current_step / total_steps)

        # Step 2: Identify Correlated Features
        status_text.text("Step 2/7: Identifying correlated features...")
        correlated_groups = get_correlated_feature_groups(X, threshold=0.8)
        current_step += 1
        progress_bar.progress(current_step / total_steps)

        # Step 3: Select Features from Correlated Groups
        status_text.text("Step 3/7: Selecting features from correlated groups...")
        # Retrieve adjustable features from configuration
        adjustable_features = list(configuration.get('adjustable_features', {}).keys())
        X, dropped_features = select_features_from_correlated_groups(X, correlated_groups, adjustable_features)
        current_step += 1
        progress_bar.progress(current_step / total_steps)

        # Log the dropped features
        logs = log_dropped_features(dropped_features)

        # Step 4: Feature Selection
        status_text.text("Step 4/7: Performing feature selection...")
        selected_features, feature_ranking = perform_feature_selection(X, y)
        current_step += 1
        progress_bar.progress(current_step / total_steps)

        # Step 5: Hyperparameter Tuning
        status_text.text("Step 5/7: Hyperparameter tuning (this may take several minutes)...")
        best_params = hyperparameter_tuning(X[selected_features], y)
        current_step += 1
        progress_bar.progress(current_step / total_steps)

        # Step 6: Train Final Model
        status_text.text("Step 6/7: Training the final model...")
        model_results = train_and_evaluate_model(X[selected_features], y, best_params)
        current_step += 1
        progress_bar.progress(current_step / total_steps)

        # Save the model
        model_filename = os.path.join(dropbox_folder_path, 'trained_model.joblib')

        # Check if model file already exists
        if os.path.exists(model_filename):
            overwrite = st.checkbox("A trained model already exists. Overwrite?", value=False)
            if overwrite:
                save_model(model_results['model'], model_filename)
            else:
                st.info("Model was not saved. Uncheck the box to save the model.")
        else:
            save_model(model_results['model'], model_filename)

        # Step 7: Calculate Feature Importances
        status_text.text("Step 7/7: Calculating feature importances...")
        feature_importances = calculate_feature_importances(model_results['model'], selected_features)
        current_step += 1
        progress_bar.progress(current_step / total_steps)

        # Finalize
        progress_bar.progress(1.0)
        status_text.text("Pipeline completed.")

        # Prepare Results
        results = {
            'selected_features': selected_features,
            'feature_ranking': feature_ranking,
            'model_results': model_results,
            'feature_importances': feature_importances,
            'logs': logs
        }
        return results

    except Exception as e:
        progress_bar.progress(1.0)
        status_text.text("An error occurred during the pipeline.")
        st.error(f"Exception: {e}")
        logging.exception("An exception occured during the machine learning pipeline")
        raise e  # Re-raise the exception to be caught in the calling function

# Rest of your function definitions at the module level (no indentation)

def preprocess_data_for_modeling(ml_data, selected_target):
    """
    Preprocess the data for modeling.
    """
    # Exclude future dates
    today = pd.Timestamp(datetime.today().date())
    ml_data = ml_data[ml_data['date'] <= today]

    # Include all features except 'date' and the target variable
    features = [col for col in ml_data.columns if col not in ['date', selected_target]]

    # Prepare data
    X = ml_data[features]
    y = ml_data[selected_target]

    # Handle missing values
    X = X.interpolate(method='linear', limit_direction='both').fillna(method='ffill').fillna(method='bfill')
    y = y.interpolate(method='linear', limit_direction='both').fillna(method='ffill').fillna(method='bfill')

    return X, y

def get_correlated_feature_groups(X, threshold=0.8):
    """
    Identify groups of highly correlated features without overlapping columns.

    Parameters:
    - X: DataFrame of features.
    - threshold: Correlation threshold to consider features as highly correlated.

    Returns:
    - correlated_groups: A list of lists containing groups of correlated features.
    """
    corr_matrix = X.corr().abs()
    correlated_groups = []
    visited = set()

    for col in corr_matrix.columns:
        if col not in visited:
            # Find features correlated with 'col' beyond the threshold and not yet visited
            high_corr_features = corr_matrix.loc[~corr_matrix.index.isin(visited), col]
            correlated_features = high_corr_features[high_corr_features > threshold].index.tolist()
            correlated_features = [f for f in correlated_features if f != col]
            if correlated_features:
                group = [col] + correlated_features
                correlated_groups.append(group)
                visited.update(group)
            else:
                visited.add(col)
    return correlated_groups

def select_features_from_correlated_groups(X, correlated_groups, adjustable_features):
    """
    Select one feature from each group of highly correlated features to keep.
    
    Parameters:
    - X: DataFrame of features.
    - correlated_groups: List of lists containing groups of correlated features.
    - adjustable_features: List of adjustable feature names.
    
    Returns:
    - X_selected: DataFrame with selected features.
    - dropped_features: List of features that were dropped.
    """
    features_to_keep = set(X.columns)
    dropped_features = []

    for group in correlated_groups:
        # Exclude groups with adjustable features from dropping
        if any(feature in adjustable_features for feature in group):
            continue  # Keep all features in this group
        else:
            # For groups without adjustable features, select one feature to keep
            # For simplicity, we can keep the feature with the highest correlation with the target
            # Alternatively, you can use domain knowledge or feature importance
            feature_to_keep = select_most_useful_feature(group, X)
            # Drop the other features in the group
            features_to_drop = set(group) - {feature_to_keep}
            features_to_keep -= features_to_drop
            dropped_features.extend(features_to_drop)

    X_selected = X[list(features_to_keep)]
    return X_selected, dropped_features

def select_most_useful_feature(group, X):
    """
    Select the most useful feature from a group of correlated features.
    For simplicity, select the feature with the highest variance.
    You can replace this with a more sophisticated method if needed.
    
    Parameters:
    - group: List of feature names in the correlated group.
    - X: DataFrame of features.
    
    Returns:
    - feature_to_keep: Name of the selected feature to keep.
    """
    variances = X[group].var()
    feature_to_keep = variances.idxmax()
    return feature_to_keep

def log_dropped_features(dropped_features):
    """
    Log the features that were dropped during feature selection.

    Parameters:
    - dropped_features: List of feature names that were dropped.

    Returns:
    - logs: List of log messages.
    """
    logs = []
    if dropped_features:
        logs.append("Dropped the following correlated features:")
        for feature in dropped_features:
            logs.append(f" - {feature}")
    else:
        logs.append("No features were dropped.")
    return logs
    
def perform_feature_selection(X, y):
    """
    Perform Recursive Feature Elimination with Cross-Validation (RFECV) to select features.

    Parameters:
    - X: DataFrame of features.
    - y: Series of target variable.

    Returns:
    - selected_features: List of selected feature names.
    - feature_ranking: Series with feature rankings.
    """
    # Initialize the model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        verbosity=0
    )

    # Use TimeSeriesSplit for cross-validation due to time series data
    tscv = TimeSeriesSplit(n_splits=3)

    # Perform RFECV
    rfecv = RFECV(
        estimator=xgb_model,
        step=5,
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

    Parameters:
    - X: DataFrame of features.
    - y: Series of target variable.

    Returns:
    - best_params: Dictionary of best hyperparameters found.
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

    Parameters:
    - X: DataFrame of features.
    - y: Series of target variable.
    - params: Dictionary of hyperparameters.

    Returns:
    - results: Dictionary containing the trained model and evaluation metrics.
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

    Parameters:
    - model: Trained XGBoost model.
    - selected_features: List of feature names used in the model.

    Returns:
    - importance_df: DataFrame containing features and their importance scores.
    """
    importance = model.get_booster().get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'Feature': [f for f in selected_features if f in importance],
        'Importance': [importance.get(f, 0) for f in selected_features if f in importance]
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    return importance_df

def log_feature_combination(correlated_groups):
    """
    Log the groups of features that were combined.

    Parameters:
    - correlated_groups: A list of lists containing groups of correlated features.

    Returns:
    - logs: A list of log messages.
    """
    logs = []
    if correlated_groups:
        logs.append("Combined correlated features (threshold > 0.8):")
        for group in correlated_groups:
            if len(group) > 1:
                logs.append(f" - Combined features {', '.join(group)} into a single feature by averaging.")
    else:
        logs.append("No correlated features found to combine.")
    return logs
