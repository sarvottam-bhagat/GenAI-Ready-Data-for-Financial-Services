import os
import pandas as pd
from ctgan import CTGAN

def generate_synthetic_data():
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    
    input_csv_path = os.path.join(project_root, 'data', 'raw', 'financial_data.csv')
    
    
    financial_df = pd.read_csv(input_csv_path)
    
    
    model = CTGAN()
    model.fit(financial_df)
    
    
    synthetic_data = model.sample(len(financial_df))
    
    
    synthetic_data_dir = os.path.join(project_root, 'data', 'synthetic')
    os.makedirs(synthetic_data_dir, exist_ok=True)
    
    
    output_csv_path = os.path.join(synthetic_data_dir, 'synthetic_financial_data.csv')
    
    
    synthetic_data.to_csv(output_csv_path, index=False)
    
    return synthetic_data
