import pandas as pd
import numpy as np
from modules import data_preprocessing
from scipy.stats import zscore


## Pass the data frame from data preprocessing 
## Pass column selection from the first page (ex: selected_target from machinelearning.py??)
## 
def identify_outliers(df, method='zscore', column='selected_target'):
    """
    Identify outliers in a DataFrame column using Z-score or IQR method.

    Returns:
    - A list of indices (or boolean array) marking the rows that are outliers.
    """
    if method == 'zscore':
        # Z-score method
        # Calculate Z-scores for the column
        z_scores = zscore(df[column].dropna())
        # Identify rows with absolute Z-scores greater than a threshold (typically 3)
        outliers = np.abs(z_scores) > 3  # You can adjust the threshold as needed
    elif method == 'iqr':
        # IQR method
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        # Calculate IQR
        IQR = Q3 - Q1
        # Define outlier boundaries (1.5 * IQR is a common rule)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Identify outliers
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    else:
        raise ValueError("Method must be either 'zscore' or 'iqr'")

    # Filter out the outliers from the DataFrame
    filtered_df = df[~outliers]
    
    # Print the total number of outliers detected
    total_outliers = outliers.sum()
    print(f"Total number of outliers detected: {total_outliers}")
    
    # Return the new DataFrame with outliers removed
    return filtered_df

# Example usage:
# df = pd.DataFrame({'TESTCOLUMN': [10, 20, 30, 100, 200, 300, 1000]})
# outliers = identify_outliers(df, method='iqr')
# print(outliers)