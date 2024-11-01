# modules/machine_learning.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFECV
import logging

def run_machine_learning_pipeline(ml_data, configuration, selected_target):
    """
    Main function to run the machine learning pipeline, including data preprocessing,
    feature selection with RFE, hyperparameter tuning, and final model training.
    """
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    
    # Step 1: Preprocess Data
    logging.info("Preprocessing data...")
    X, y = preprocess_data_for_modeling(ml_data, configuration, selected_target)
    
    # Step 2: Identify and Remove Highly Correlated Features
    logging.info("Identifying highly correlated features...")
    correlated_features = get_correlated_features(X, threshold=0.9)
    if correlated_features:
        logging.info(f"Removing correlated features: {correlated_features}")
        X.drop(columns=correlated_features, inplace=True)
    
    # Step 3: Perform Feature Selection using RFE
    logging.info("Performing feature selection using RFE...")
    selected_features, feature_ranking = perform_feature_selection(X, y)
    logging.info(f"Selected features: {selected_features}")
    
    # Step 4: Hyperparameter Tuning
    logging.info("Performing hyperparameter tuning...")
    best_params = hyperparameter_tuning(X[selected_features], y)
    logging.info(f"Best hyperparameters: {best_params}")
    
    # Step 5: Retrain Model with Selected Features and Best Parameters
    logging.info("Training final model...")
    model_results = train_and_evaluate_model(X[selected_features], y, best_params)
    
    # Step 6: Calculate Feature Importances
    logging.info("Calculating feature importances...")
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
