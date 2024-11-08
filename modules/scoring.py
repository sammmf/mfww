# modules/scoring.py

import numpy as np
import pandas as pd
import streamlit as st

def calculate_feature_score(value, min_val, max_val, optimal_min, optimal_max):
    """Calculate individual feature scores using optimal ranges."""
    try:
        value = float(value)
        min_val = float(min_val)
        max_val = float(max_val)
        optimal_min = float(optimal_min)
        optimal_max = float(optimal_max)
    except (ValueError, TypeError):
        return np.nan

    if any(np.isnan([value, min_val, max_val, optimal_min, optimal_max])):
        return np.nan

    if min_val >= max_val or optimal_min >= optimal_max:
        return np.nan

    optimal_min = max(min_val, optimal_min)
    optimal_max = min(max_val, optimal_max)
    optimal_mid = (optimal_min + optimal_max) / 2

    if value < min_val or value > max_val:
        return 0
    elif optimal_min <= value <= optimal_max:
        if value <= optimal_mid:
            score = 1 - (optimal_mid - value) / (optimal_mid - optimal_min) * 0.1
        else:
            score = 1 - (value - optimal_mid) / (optimal_max - optimal_mid) * 0.1
        return np.clip(score, 0, 1)
    elif min_val <= value < optimal_min:
        score = 0.9 - ((optimal_min - value) / (optimal_min - min_val)) * 0.9
        return np.clip(score, 0, 0.9)
    elif optimal_max < value <= max_val:
        score = 0.9 - ((value - optimal_max) / (max_val - optimal_max)) * 0.9
        return np.clip(score, 0, 0.9)
    else:
        return 0

def calculate_difficulty_score(value, min_val, max_val, optimal_min, optimal_max):
    """Calculate difficulty feature scores (flipped)."""
    score = calculate_feature_score(value, min_val, max_val, optimal_min, optimal_max)
    if not np.isnan(score):
        return np.clip(1 - score, 0, 1)
    else:
        return np.nan

def compute_plant_scores(ml_data, configuration):
    """Compute current plant performance and difficulty scores."""
    # Clean and standardize feature names in configuration
    config = configuration.set_index('feature_name')
    unit_processes = config['unit_process'].unique()

    # Format unit process names for display
    def format_unit_process_name(name):
        name = name.replace('_', ' ').title()
        return name

    formatted_unit_process_names = {process: format_unit_process_name(process) for process in unit_processes}

    # Separate performance features and difficulty features
    difficulty_features = config[config['difficulty_score_parameter'] == 'yes'].index.tolist()
    feature_columns = [col for col in ml_data.columns if col != 'date']
    performance_features = [f for f in feature_columns if f not in difficulty_features]

    # Calculate the number of data points per feature in the recent_data
    recent_data = ml_data.copy()
    data_point_counts = recent_data[feature_columns].count()

    # Calculate the data completeness metric
    expected_data_points = len(recent_data)
    data_completeness = (data_point_counts / expected_data_points).clip(upper=1.0)

    # Compute the average values for each feature
    average_values = recent_data[feature_columns].mean()

    # Initialize dictionaries to hold scores
    unit_process_scores = {}
    difficulty_scores = {}

    # Iterate over each unit process
    for process in unit_processes:
        features = config[(config['unit_process'] == process) & (config['difficulty_score_parameter'] != 'yes')]

        # Skip unit processes that have no performance features
        if features.empty:
            continue

        feature_scores = []

        for feature_name, params in features.iterrows():
            if feature_name not in performance_features:
                continue  # Skip features not in the data
            value = average_values.get(feature_name, np.nan)
            min_val = params['min']
            max_val = params['max']
            optimal_min = params['optimal_min']
            optimal_max = params['optimal_max']
            weight = params['weight']

            # Calculate the feature score
            score = calculate_feature_score(value, min_val, max_val, optimal_min, optimal_max)
            weighted_score = score * weight if not np.isnan(score) else np.nan

            # Get data completeness for the feature
            completeness = data_completeness.get(feature_name, 0)

            feature_scores.append({
                'feature_name': feature_name,
                'score': score,
                'weighted_score': weighted_score,
                'weight': weight,
                'value': value,
                'completeness': completeness
            })

        # Remove features with NaN weighted_scores
        valid_feature_scores = [fs for fs in feature_scores if not np.isnan(fs['weighted_score'])]

        # Aggregate feature scores
        if valid_feature_scores:
            aggregation_method = features['aggregation_method'].iloc[0].strip().lower()
            weights = np.array([fs['weight'] for fs in valid_feature_scores])
            scores = np.array([fs['weighted_score'] for fs in valid_feature_scores])
            if np.nansum(weights) > 0:
                if aggregation_method == 'mean':
                    unit_score = np.nansum(scores) / np.nansum(weights)
                elif aggregation_method == 'median':
                    unit_score = np.nanmedian(scores / weights)
                else:
                    unit_score = np.nan
            else:
                unit_score = np.nan
        else:
            unit_score = np.nan

        unit_process_scores[process] = {
            'unit_score': unit_score,
            'features': feature_scores
        }

    # Calculate overall plant score as weighted average of unit process scores
    total_weight = sum([
        sum(fs['weight'] for fs in process_info['features'] if not np.isnan(fs['weighted_score']))
        for process_info in unit_process_scores.values()
    ])

    if total_weight > 0:
        plant_performance_score = sum([
            process_info['unit_score'] * sum(fs['weight'] for fs in process_info['features'] if not np.isnan(fs['weighted_score'])) / total_weight
            for process_info in unit_process_scores.values() if not np.isnan(process_info['unit_score'])
        ])
    else:
        plant_performance_score = np.nan

    # Difficulty Score
    difficulty_parameters = config[config['difficulty_score_parameter'] == 'yes']
    difficulty_feature_scores = []

    for feature_name, params in difficulty_parameters.iterrows():
        if feature_name not in feature_columns:
            continue
        value = average_values.get(feature_name, np.nan)
        min_val = params['min']
        max_val = params['max']
        optimal_min = params['optimal_min']
        optimal_max = params['optimal_max']
        weight = params['weight']

        # Calculate the difficulty score (flipped)
        score = calculate_difficulty_score(value, min_val, max_val, optimal_min, optimal_max)
        weighted_score = score * weight if not np.isnan(score) else np.nan

        # Get data completeness for the feature
        completeness = data_completeness.get(feature_name, 0)

        difficulty_feature_scores.append({
            'feature_name': feature_name,
            'score': score,
            'weighted_score': weighted_score,
            'weight': weight,
            'value': value,
            'completeness': completeness
        })

    # Remove features with NaN scores
    valid_difficulty_scores = [dfs for dfs in difficulty_feature_scores if not np.isnan(dfs['weighted_score'])]

    if valid_difficulty_scores:
        total_difficulty_weight = sum(dfs['weight'] for dfs in valid_difficulty_scores)
        if total_difficulty_weight > 0:
            difficulty_score = sum(dfs['weighted_score'] for dfs in valid_difficulty_scores) / total_difficulty_weight
        else:
            difficulty_score = np.nan
    else:
        difficulty_score = np.nan

    # Adjust the overall plant performance score based on the difficulty score
    if not np.isnan(plant_performance_score) and not np.isnan(difficulty_score):
        adjustment_factor = 0.1  # Max adjustment of 10%
        adjusted_performance_score = plant_performance_score * (1 + adjustment_factor * difficulty_score)
    else:
        adjusted_performance_score = plant_performance_score

    # Return the computed scores and other data
    plant_scores = {
        'plant_performance_score': plant_performance_score,
        'difficulty_score': difficulty_score,
        'adjusted_performance_score': adjusted_performance_score,
        'unit_process_scores': unit_process_scores,
        'data_completeness': data_completeness,
        'formatted_unit_process_names': formatted_unit_process_names
    }

    return plant_scores
    
def calculate_scores_over_time(ml_data, configuration):
    """Calculate plant performance and difficulty scores over time."""
    # Clean and standardize feature names in configuration
    config = configuration.set_index('feature_name')
    unit_processes = config['unit_process'].unique()

    # Convert 'date' column to datetime
    if 'date' not in ml_data.columns:
        st.error("The 'date' column is missing from ml_data.")
        return None

    ml_data['date'] = pd.to_datetime(ml_data['date'], errors='coerce')
    ml_data.dropna(subset=['date'], inplace=True)

    # **Ensure dates are not in the future**
    today = pd.Timestamp(datetime.today().date())
    ml_data = ml_data[ml_data['date'] <= today]

    # Prepare time ranges: last 12 months in two-week intervals
    max_date_in_data = ml_data['date'].max()
    max_date = min(max_date_in_data, today)
    min_date = max_date - pd.Timedelta(days=365)
    date_ranges = pd.date_range(start=min_date, end=max_date, freq='2W')
    
    # Lists to store time series data
    time_series_dates = []
    overall_scores = []
    difficulty_scores_list = []
    adjusted_scores = []
    unit_process_scores_over_time = {process: [] for process in unit_processes}

    # Loop over each two-week period
    for i in range(len(date_ranges) - 1):
        start_period = date_ranges[i]
        end_period = date_ranges[i + 1]

        period_data = ml_data[(ml_data['date'] >= start_period) & (ml_data['date'] < end_period)]
        if period_data.empty:
            # Append NaN values for this period
            time_series_dates.append(start_period.strftime('%Y-%m-%d'))
            overall_scores.append(np.nan)
            difficulty_scores_list.append(np.nan)
            adjusted_scores.append(np.nan)
            for process in unit_processes:
                unit_process_scores_over_time[process].append(np.nan)
            continue  # Skip to next period

        # Recompute averages for this period
        feature_columns = [col for col in ml_data.columns if col != 'date']
        period_average_values = period_data[feature_columns].mean()
        period_data_point_counts = period_data[feature_columns].count()
        expected_data_points = len(period_data)
        period_data_completeness = (period_data_point_counts / expected_data_points).clip(upper=1.0)

        # Initialize dictionaries to hold scores
        period_unit_process_scores = {}
        total_weight = 0
        total_score = 0

        # Iterate over each unit process
        for process in unit_processes:
            features = config[(config['unit_process'] == process) & (config['difficulty_score_parameter'] != 'yes')]

            if features.empty:
                continue

            feature_scores = []

            for feature_name, params in features.iterrows():
                if feature_name not in feature_columns:
                    continue
                value = period_average_values.get(feature_name, np.nan)
                min_val = params['min']
                max_val = params['max']
                optimal_min = params['optimal_min']
                optimal_max = params['optimal_max']
                weight = params['weight']

                # Calculate the feature score
                score = calculate_feature_score(value, min_val, max_val, optimal_min, optimal_max)
                weighted_score = score * weight if not np.isnan(score) else np.nan

                feature_scores.append({
                    'weighted_score': weighted_score,
                    'weight': weight
                })

            valid_feature_scores = [fs for fs in feature_scores if not np.isnan(fs['weighted_score'])]
            if valid_feature_scores:
                aggregation_method = features['aggregation_method'].iloc[0].strip().lower()
                weights = np.array([fs['weight'] for fs in valid_feature_scores])
                scores = np.array([fs['weighted_score'] for fs in valid_feature_scores])
                if np.nansum(weights) > 0:
                    if aggregation_method == 'mean':
                        unit_score = np.nansum(scores) / np.nansum(weights)
                    elif aggregation_method == 'median':
                        unit_score = np.nanmedian(scores / weights)
                    else:
                        unit_score = np.nan
                else:
                    unit_score = np.nan
            else:
                unit_score = np.nan

            period_unit_process_scores[process] = unit_score

            # Accumulate for overall plant score
            process_weight = sum(fs['weight'] for fs in valid_feature_scores)
            if not np.isnan(unit_score):
                total_weight += process_weight
                total_score += unit_score * process_weight

        # Append unit process scores to the over time dictionary
        for process in unit_processes:
            if process in period_unit_process_scores:
                unit_process_scores_over_time[process].append(period_unit_process_scores[process])
            else:
                unit_process_scores_over_time[process].append(np.nan)

        if total_weight > 0:
            plant_performance_score = total_score / total_weight
        else:
            plant_performance_score = np.nan

        # Difficulty Score
        difficulty_parameters = config[config['difficulty_score_parameter'] == 'yes']
        difficulty_feature_scores = []
        for feature_name, params in difficulty_parameters.iterrows():
            if feature_name not in feature_columns:
                continue  # Skip features not in the data
            value = period_average_values.get(feature_name, np.nan)
            min_val = params['min']
            max_val = params['max']
            optimal_min = params['optimal_min']
            optimal_max = params['optimal_max']
            weight = params['weight']

            # Calculate the difficulty score (flipped)
            score = calculate_difficulty_score(value, min_val, max_val, optimal_min, optimal_max)
            weighted_score = score * weight if not np.isnan(score) else np.nan
            difficulty_feature_scores.append({
                'weighted_score': weighted_score,
                'weight': weight
            })

        valid_difficulty_scores = [dfs for dfs in difficulty_feature_scores if not np.isnan(dfs['weighted_score'])]
        if valid_difficulty_scores:
            total_difficulty_weight = sum(dfs['weight'] for dfs in valid_difficulty_scores)
            if total_difficulty_weight > 0:
                difficulty_score = sum(dfs['weighted_score'] for dfs in valid_difficulty_scores) / total_difficulty_weight
            else:
                difficulty_score = np.nan
        else:
            difficulty_score = np.nan

        # Adjust the overall plant performance score based on the difficulty score
        if not np.isnan(plant_performance_score) and not np.isnan(difficulty_score):
            adjustment_factor = 0.1  # Max adjustment of 10%
            adjusted_performance_score = plant_performance_score * (1 + adjustment_factor * difficulty_score)
        else:
            adjusted_performance_score = plant_performance_score

        # Store results
        time_series_dates.append(start_period.strftime('%Y-%m-%d'))
        overall_scores.append(plant_performance_score)
        difficulty_scores_list.append(difficulty_score)
        adjusted_scores.append(adjusted_performance_score)

    # Prepare the final scores_over_time dictionary
    scores_over_time = {
        'dates': time_series_dates,
        'overall_scores': overall_scores,
        'difficulty_scores': difficulty_scores_list,
        'adjusted_scores': adjusted_scores,
        'unit_process_scores': unit_process_scores_over_time
    }

    return scores_over_time

# ... rest of the code remains unchanged ...
