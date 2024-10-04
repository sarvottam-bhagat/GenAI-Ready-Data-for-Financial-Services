import os
import pandas as pd
import numpy as np
from diffprivlib import mechanisms

def apply_differential_privacy():
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
   
    input_csv_path = os.path.join(project_root, 'data', 'raw', 'financial_data.csv')
    
   
    financial_df = pd.read_csv(input_csv_path)
    
    
    epsilon = 1.0  # Privacy parameter
    privacy_results = financial_df.copy()
    
    for column in financial_df.columns:
        if np.issubdtype(financial_df[column].dtype, np.number):
            
            laplace = mechanisms.Laplace(epsilon=epsilon, sensitivity=1.0)
            privacy_results[column] = financial_df[column].apply(laplace.randomise)
        else:
            
            pass
    
    # Save the privacy-preserving results
    output_csv_path = os.path.join(project_root, 'data', 'privacy', 'privacy_results.csv')
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    privacy_results.to_csv(output_csv_path, index=False)
    
    return privacy_results
