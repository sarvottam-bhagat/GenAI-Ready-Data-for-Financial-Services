import os
import pandas as pd
import numpy as np
from diffprivlib import mechanisms

def apply_differential_privacy():
    # Get the absolute path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Construct the full path to the input CSV file
    input_csv_path = os.path.join(project_root, 'data', 'raw', 'financial_data.csv')
    
    # Read the CSV file using the absolute path
    financial_df = pd.read_csv(input_csv_path)
    
    # Apply differential privacy techniques
    epsilon = 1.0  # Privacy parameter
    privacy_results = financial_df.copy()
    
    for column in financial_df.columns:
        if np.issubdtype(financial_df[column].dtype, np.number):
            # Apply Laplace mechanism for numerical columns
            laplace = mechanisms.Laplace(epsilon=epsilon, sensitivity=1.0)
            privacy_results[column] = financial_df[column].apply(laplace.randomise)
        else:
            # For non-numerical columns, we'll just keep them as is
            # You might want to implement a different strategy for categorical data
            pass
    
    # Save the privacy-preserving results
    output_csv_path = os.path.join(project_root, 'data', 'privacy', 'privacy_results.csv')
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    privacy_results.to_csv(output_csv_path, index=False)
    
    return privacy_results
