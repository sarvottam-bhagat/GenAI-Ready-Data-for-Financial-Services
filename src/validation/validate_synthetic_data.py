import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def validate_synthetic_data():
    # Get the absolute path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Construct the full paths to the input CSV files
    original_data_path = os.path.join(project_root, 'data', 'raw', 'financial_data.csv')
    synthetic_data_path = os.path.join(project_root, 'data', 'synthetic', 'synthetic_financial_data.csv')
    
    # Read the CSV files using the absolute paths
    original_data = pd.read_csv(original_data_path)
    synthetic_data = pd.read_csv(synthetic_data_path)
    
    # Ensure both datasets have the same columns
    common_columns = list(set(original_data.columns) & set(synthetic_data.columns))
    original_data = original_data[common_columns]
    synthetic_data = synthetic_data[common_columns]
    
    # Calculate MSE for numerical columns
    mse_results = {}
    for column in common_columns:
        if np.issubdtype(original_data[column].dtype, np.number):
            mse = mean_squared_error(original_data[column], synthetic_data[column])
            mse_results[column] = mse
    
    # Calculate overall MSE
    overall_mse = np.mean(list(mse_results.values()))
    
    return overall_mse, mse_results
