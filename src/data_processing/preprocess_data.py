import pandas as pd
import numpy as np
import os

def load_and_preprocess_data():
    np.random.seed(42)
    data = {
        'Customer_ID': np.arange(1000, 1100),
        'Transaction_Amount': np.random.normal(500, 100, 100),
        'Balance': np.random.normal(10000, 2000, 100),
        'Age': np.random.randint(20, 70, 100),
        'Credit_Score': np.random.randint(300, 850, 100),
    }
    
    financial_df = pd.DataFrame(data)
    
    # Get the absolute path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Create directory using absolute path
    data_dir = os.path.join(project_root, 'data', 'raw')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save the CSV file using the absolute path
    csv_path = os.path.join(data_dir, 'financial_data.csv')
    financial_df.to_csv(csv_path, index=False)
    
    return financial_df
