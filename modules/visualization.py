# modules/visualization.py

import streamlit as st
from streamlit_echarts import st_echarts
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from math import ceil
import pandas as pd

def run_visualization(
    plant_scores,
    scores_over_time,
    unit_process_scores,
    formatted_unit_process_names,
    data_completeness,
    ml_data
):
    tabs = st.tabs(["Dashboard", "Data Query"])

    with tabs[0]:
        display_dashboard(
            plant_scores,
            scores_over_time,
            unit_process_scores,
            formatted_unit_process_names,
            data_completeness
        )

    with tabs[1]:
        display_data_query(ml_data)
        display_recent_complete_day_summary(ml_data)

def display_dashboard(
    plant_scores,
    scores_over_time,
    unit_process_scores,
    formatted_unit_process_names,
    data_completeness
):
    """Display the dashboard with all visualizations."""
    st.header("Plant Performance Overview")

    col1, col2, col3 = st.columns(3)

    # Overall Plant Score
    with col1:
        display_gauge(
            value=plant_scores.get('plant_performance_score'),
            title="Overall Plant Score",
            key="overall_performance_gauge"
        )

    # Difficulty Score
    with col2:
        display_gauge(
            value=plant_scores.get('difficulty_score'),
            title="Difficulty Score",
            key="difficulty_gauge",
            thresholds=[
                [0.4, "#4CAF50"],
                [0.7, "#FFD700"],
                [1, "#FF4D4D"]
            ]
        )

    # Adjusted Performance Score
    with col3:
        display_gauge(
            value=plant_scores.get('adjusted_performance_score'),
            title="Performance Adjusted for Difficulty",
            key="combined_performance_gauge"
        )

    # Time Series Charts for KPIs
    display_time_series(scores_over_time)

    # Visualization of Current Performance vs. Historical Difficulty Levels
    display_scatter_performance_difficulty(
        scores_over_time,
        plant_scores.get('plant_performance_score'),
        plant_scores.get('difficulty_score')
    )

    # Unit Process Scores
    display_unit_process_scores(unit_process_scores, formatted_unit_process_names)

    # Unit Process Performance Trends
    display_unit_process_trends(scores_over_time, formatted_unit_process_names)

    # Data Completeness
    display_data_completeness(data_completeness)

    # Detailed Data View
    display_detailed_data(unit_process_scores, formatted_unit_process_names)

def display_gauge(value, title, key, thresholds=None):
    """Display a gauge chart."""
    st.markdown(f"<h3 style='text-align: center;'>{title}</h3>", unsafe_allow_html=True)
    if value is None or pd.isna(value):
        st.write("N/A (Insufficient data)")
        return

    if thresholds is None:
        thresholds = [
            [0.6, "#FF4D4D"],
            [0.8, "#FFD700"],
            [1, "#4CAF50"]
        ]

    option = {
        "series": [
            {
                "type": "gauge",
                "progress": {"show": True},
                "detail": {"show": False},
                "data": [{"value": round(value * 100)}],
                "axisLine": {
                    "lineStyle": {
                        "width": 10,
                        "color": thresholds
                    }
                },
                "max": 100,
                "min": 0,
                "splitNumber": 2,
                "axisTick": {"show": False},
                "axisLabel": {"fontSize": 10, "formatter": "{value}%"},
                "pointer": {"show": True, "length": "60%"},
                "title": {"show": False}
            }
        ]
    }
    st_echarts(options=option, height="250px", key=key)
    st.markdown(
        f"<p style='text-align: center; font-size: 18px;'>{round(value * 100)}%</p>",
        unsafe_allow_html=True
    )

def display_time_series(scores_over_time):
    st.header("12-Month Performance Trends")

    # Check if scores_over_time is a dictionary
    if not isinstance(scores_over_time, dict):
        st.error("Error: 'scores_over_time' should be a dictionary.")
        st.write(f"Received type: {type(scores_over_time)}")
        return

    # Check if scores_over_time has the required keys
    required_keys = ['dates', 'overall_scores', 'difficulty_scores', 'adjusted_scores']
    if not all(key in scores_over_time for key in required_keys):
        st.error("Incomplete data: Missing required keys in 'scores_over_time'.")
        st.write("Available keys:", list(scores_over_time.keys()))
        return

    # Extract data
    dates = scores_over_time['dates']
    overall_scores = scores_over_time['overall_scores']
    difficulty_scores = scores_over_time['difficulty_scores']
    adjusted_scores = scores_over_time['adjusted_scores']

    # Ensure that the lists are of equal length
    lengths = [len(dates), len(overall_scores), len(difficulty_scores), len(adjusted_scores)]
    if len(set(lengths)) != 1:
        st.error("Data length mismatch: The lists in 'scores_over_time' must have the same length.")
        st.write("Lengths:", dict(zip(required_keys, lengths)))
        return

    # Create DataFrame
    trend_df = pd.DataFrame({
        'Date': dates,
        'Overall Plant Score': overall_scores,
        'Difficulty Score': difficulty_scores,
        'Adjusted Performance Score': adjusted_scores
    })

    # Convert 'Date' column to datetime
    trend_df['Date'] = pd.to_datetime(trend_df['Date'], errors='coerce')
    trend_df.dropna(subset=['Date'], inplace=True)

    # Convert scores to percentages
    score_columns = ['Overall Plant Score', 'Difficulty Score', 'Adjusted Performance Score']
    for col in score_columns:
        trend_df[col] = pd.to_numeric(trend_df[col], errors='coerce') * 100

    # Drop rows where all scores are NaN
    trend_df.dropna(subset=score_columns, how='all', inplace=True)

    # Ensure DataFrame is not empty
    if trend_df.empty:
        st.warning("No valid data available to plot.")
        return

    # Sort DataFrame by 'Date'
    trend_df.sort_values('Date', inplace=True)

    # Debug: Show the DataFrame
    st.write("Trend DataFrame:")
    st.dataframe(trend_df)
    st.write("Data Types:")
    st.write(trend_df.dtypes)

    # Plotting
    try:
        fig = px.line(
            trend_df,
            x='Date',
            y=score_columns,
            labels={'value': 'Score (%)', 'variable': 'KPI', 'Date': 'Date'},
            title='12-Month Performance Trends'
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred while plotting the time series: {e}")
        st.write("Please check the data above for inconsistencies.")

def display_scatter_performance_difficulty(
    scores_over_time,
    plant_performance_score,
    difficulty_score
):
    st.header("Performance Relative to Historical Difficulty Levels")

    # Check if required data is available
    required_keys = ['dates', 'overall_scores', 'difficulty_scores']
    if not all(key in scores_over_time for key in required_keys):
        st.error("Incomplete data: Missing required keys in 'scores_over_time'.")
        return

    # Extract data
    dates = scores_over_time['dates']
    overall_scores = scores_over_time['overall_scores']
    difficulty_scores = scores_over_time['difficulty_scores']

    # Ensure that the lists are of equal length
    lengths = [len(dates), len(overall_scores), len(difficulty_scores)]
    if len(set(lengths)) != 1:
        st.error("Data length mismatch in 'scores_over_time'.")
        st.write("Lengths:", dict(zip(required_keys, lengths)))
        return

    # Create DataFrame
    scatter_df = pd.DataFrame({
        'Date': dates,
        'Overall Plant Score': overall_scores,
        'Difficulty Score': difficulty_scores
    })

    # Convert 'Date' column to datetime
    scatter_df['Date'] = pd.to_datetime(scatter_df['Date'], errors='coerce')
    scatter_df.dropna(subset=['Date'], inplace=True)

    # Convert scores to percentages
    scatter_df['Overall Plant Score'] = pd.to_numeric(scatter_df['Overall Plant Score'], errors='coerce') * 100
    scatter_df['Difficulty Score'] = pd.to_numeric(scatter_df['Difficulty Score'], errors='coerce') * 100

    # Drop rows with NaN values in scores
    scatter_df.dropna(subset=['Overall Plant Score', 'Difficulty Score'], inplace=True)

    # Ensure DataFrame is not empty
    if scatter_df.empty:
        st.warning("No valid data available to plot.")
        return

    # Plotting
    try:
        fig = px.scatter(
            scatter_df,
            x='Difficulty Score',
            y='Overall Plant Score',
            hover_data=['Date'],
            labels={
                'Difficulty Score': 'Difficulty Score (%)',
                'Overall Plant Score': 'Overall Plant Score (%)'
            },
            title='Performance vs. Difficulty'
        )

        # Highlight current performance
        if plant_performance_score is not None and difficulty_score is not None:
            fig.add_trace(
                go.Scatter(
                    x=[difficulty_score * 100],
                    y=[plant_performance_score * 100],
                    mode='markers',
                    marker=dict(color='red', size=12),
                    name='Current Performance'
                )
            )

        st.plotly_chart(fig, use_container_width=True)
        st.write("The red dot represents the current performance relative to historical difficulty levels.")
    except Exception as e:
        st.error(f"An error occurred while plotting the scatter plot: {e}")
        st.write("Please check the data for inconsistencies.")

def display_unit_process_scores(unit_process_scores, formatted_unit_process_names):
    st.header("Unit Process Scores")

    unit_process_items = list(unit_process_scores.items())
    num_processes = len(unit_process_items)
    max_cols_per_row = 4
    num_rows = ceil(num_processes / max_cols_per_row)

    for row in range(num_rows):
        cols = st.columns(max_cols_per_row)
        for idx in range(max_cols_per_row):
            item_idx = row * max_cols_per_row + idx
            if item_idx >= num_processes:
                break
            process, process_info = unit_process_items[item_idx]
            unit_score = process_info.get('unit_score')
            process_name = formatted_unit_process_names.get(process, process)
            with cols[idx]:
                st.markdown(
                    f"<h4 style='text-align: center;'>{process_name}</h4>",
                    unsafe_allow_html=True
                )
                if unit_score is None or pd.isna(unit_score):
                    st.write("N/A")
                else:
                    # Use a smaller gauge chart for visual representation
                    option = {
                        "series": [
                            {
                                "type": "gauge",
                                "radius": "75%",
                                "progress": {"show": True},
                                "detail": {"show": False},
                                "data": [{"value": round(unit_score * 100)}],
                                "axisLine": {
                                    "lineStyle": {
                                        "width": 8,
                                        "color": [
                                            [0.6, "#FF4D4D"],
                                            [0.8, "#FFD700"],
                                            [1, "#4CAF50"]
                                        ]
                                    }
                                },
                                "max": 100,
                                "min": 0,
                                "splitNumber": 2,
                                "axisTick": {
                                    "length": 5,
                                    "lineStyle": {"color": "auto"},
                                    "show": False
                                },
                                "axisLabel": {
                                    "fontSize": 8,
                                    "distance": 10,
                                    "formatter": "{value}%"
                                },
                                "pointer": {
                                    "show": True,
                                    "length": "60%"
                                },
                                "title": {"show": False}
                            }
                        ]
                    }
                    st_echarts(
                        options=option,
                        height="150px",
                        key=f"gauge_{process}"
                    )
                    st.markdown(
                        f"<p style='text-align: center; font-size: 18px;'>{round(unit_score * 100)}%</p>",
                        unsafe_allow_html=True
                    )

def display_unit_process_trends(scores_over_time, formatted_unit_process_names):
    st.header("Unit Process Performance Trends")
    with st.expander("Show/Hide Unit Process Graphs"):
        if 'unit_process_scores' not in scores_over_time or 'dates' not in scores_over_time:
            st.warning("Insufficient data to display unit process trends.")
            return

        dates = scores_over_time['dates']
        unit_process_scores_dict = scores_over_time['unit_process_scores']

        for process, scores in unit_process_scores_dict.items():
            process_name = formatted_unit_process_names.get(process, process)
            if any(s is not None and not pd.isna(s) for s in scores):
                # Ensure that dates and scores have the same length
                if len(dates) != len(scores):
                    st.warning(f"Data length mismatch for {process_name}.")
                    continue

                process_df = pd.DataFrame({
                    'Date': dates,
                    'Score': scores
                })

                # Convert 'Date' column to datetime
                process_df['Date'] = pd.to_datetime(process_df['Date'], errors='coerce')
                process_df.dropna(subset=['Date'], inplace=True)

                # Convert scores to percentages
                process_df['Score'] = pd.to_numeric(process_df['Score'], errors='coerce') * 100
                process_df.dropna(subset=['Score'], inplace=True)

                if not process_df.empty:
                    st.subheader(process_name)
                    fig = px.line(
                        process_df,
                        x='Date',
                        y='Score',
                        labels={'Score': 'Score (%)'},
                        title=f"{process_name} Performance Over Time"
                    )
                    st.plotly_chart(fig, use_container_width=True)

def display_data_completeness(data_completeness):
    st.header("Data Completeness")
    if data_completeness is not None and not pd.isna(data_completeness.mean()):
        overall_completeness = data_completeness.mean() * 100
        st.progress(overall_completeness / 100)
        st.write(f"Overall Data Completeness: {overall_completeness:.0f}% Complete")
    else:
        st.write("Data completeness information is not available.")

def display_detailed_data(unit_process_scores, formatted_unit_process_names):
    st.header("Detailed Data")
    st.write("Select a unit process to view detailed feature scores:")
    process_names = [
        formatted_unit_process_names.get(process, process)
        for process in unit_process_scores.keys()
    ]
    process_name_to_key = {
        formatted_unit_process_names.get(process, process): process
        for process in unit_process_scores.keys()
    }
    selected_process_name = st.selectbox("Unit Processes", process_names)

    if selected_process_name:
        selected_process = process_name_to_key[selected_process_name]
        process_info = unit_process_scores.get(selected_process)
        if process_info:
            feature_scores = process_info.get('features', [])
            feature_data = []
            for fs in feature_scores:
                completeness_percent = fs.get('completeness', 0) * 100
                score = fs.get('score')
                feature_status = "No Data" if score is None or pd.isna(score) else f"{score * 100:.1f}%"
                feature_data.append({
                    'Feature': fs['feature_name'].replace('_', ' ').title(),
                    'Score (%)': score * 100 if score is not None and not pd.isna(score) else np.nan,
                    'Status': feature_status,
                    'Completeness (%)': completeness_percent
                })
            feature_df = pd.DataFrame(feature_data)
            st.table(feature_df)
            # Display charts for feature scores
            st.subheader(f"{selected_process_name} Feature Scores")
            valid_feature_df = feature_df.dropna(subset=['Score (%)'])
            if not valid_feature_df.empty:
                fig = px.bar(
                    valid_feature_df,
                    x='Feature',
                    y='Score (%)',
                    color='Score (%)',
                    range_y=[0, 100],
                    color_continuous_scale=['#FF4D4D', '#FFD700', '#4CAF50'],
                    title=f"{selected_process_name} Feature Scores"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No valid feature scores available to display.")
        else:
            st.write("No information available for the selected unit process.")

def display_data_query(ml_data):
    st.header("Data Query")
    st.write("Select features and time frame to visualize trends.")

    # Searchable Multi-Select Dropdown for Feature Selection
    all_features = ml_data.columns.drop('date').tolist()
    feature_display_names = [f.replace('_', ' ').title() for f in all_features]
    feature_name_mapping = dict(zip(feature_display_names, all_features))
    selected_display_features = st.multiselect("Select Features", feature_display_names)

    if selected_display_features:
        selected_features = [feature_name_mapping[fdn] for fdn in selected_display_features]

        # Dropdown for Time Frame Selection
        timeframe = st.selectbox(
            "Select Time Frame",
            ["Last Week", "Last Month", "Last 3 Months", "Last 6 Months", "Last Year"]
        )

        # Filter data based on selected time frame
        def filter_data_by_time(data, timeframe):
            max_date = data['date'].max()
            start_date = {
                'Last Week': max_date - pd.Timedelta(days=7),
                'Last Month': max_date - pd.Timedelta(days=30),
                'Last 3 Months': max_date - pd.Timedelta(days=90),
                'Last 6 Months': max_date - pd.Timedelta(days=180),
                'Last Year': max_date - pd.Timedelta(days=365),
            }.get(timeframe, max_date - pd.Timedelta(days=365))  # Default: Last Year
            return data[data['date'] >= start_date]

        query_data = filter_data_by_time(ml_data, timeframe)

        if query_data.empty:
            st.warning("No data available for the selected time frame.")
        else:
            # Data Smoothing using Two-Week Rolling Average
            smoothed_data = query_data[['date'] + selected_features].set_index('date')
            smoothed_data = smoothed_data.rolling(window='14D').mean().reset_index()

            # Remove rows with NaN values in selected features
            smoothed_data.dropna(subset=selected_features, how='all', inplace=True)

            if smoothed_data.empty:
                st.warning("No valid data available after smoothing.")
                return

            # Plot the selected features
            try:
                fig = go.Figure()
                for feature in selected_features:
                    fig.add_trace(go.Scatter(
                        x=smoothed_data['date'],
                        y=smoothed_data[feature],
                        name=feature.replace('_', ' ').title()
                    ))
                fig.update_layout(
                    yaxis_title='Value',
                    xaxis_title='Date',
                    title='Feature Trends Over Time',
                    legend_title='Features'
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred while plotting the data: {e}")
                st.write("Please check the selected features for inconsistencies.")

            # Create a CSV download button for the queried data
            csv = smoothed_data.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="Download Queried Data as CSV",
                data=csv,
                file_name='queried_data.csv',
                mime='text/csv'
            )

def display_recent_complete_day_summary(ml_data):
    st.header("Recent Complete Day Summary")
    key_features = ['d2_ph', 'd3_ph', 'mlss', 'sbd', 'eff_flow']

    # Check if key features are in the data
    missing_key_features = [kf for kf in key_features if kf not in ml_data.columns]
    if missing_key_features:
        st.warning(f"The following key features are missing from the data: {missing_key_features}")
    else:
        def get_recent_complete_day(data, features):
            valid_days = data.dropna(subset=features)
            if not valid_days.empty:
                return valid_days.sort_values('date', ascending=False).iloc[0]
            return None  # If no complete days are found

        recent_day = get_recent_complete_day(ml_data, key_features)
        if recent_day is not None:
            st.subheader(f"Data for {recent_day['date'].date()}")
            recent_day_data = recent_day[key_features].rename(lambda x: x.replace('_', ' ').title())
            df_recent_day = pd.DataFrame(recent_day_data).reset_index()
            df_recent_day.columns = ['Feature', 'Value']
            st.table(df_recent_day)
        else:
            st.warning("No recent complete day found with all key features available.")

