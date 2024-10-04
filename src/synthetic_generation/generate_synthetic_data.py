import os
import pandas as pd
from ctgan import CTGAN

def generate_synthetic_data():
    # Get the absolute path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Construct the full path to the input CSV file
    input_csv_path = os.path.join(project_root, 'data', 'raw', 'financial_data.csv')
    
    # Read the CSV file using the absolute path
    financial_df = pd.read_csv(input_csv_path)
    
    # Initialize and train the CTGAN model
    model = CTGAN()
    model.fit(financial_df)
    
    # Generate synthetic data
    synthetic_data = model.sample(len(financial_df))
    
    # Create the directory for synthetic data if it doesn't exist
    synthetic_data_dir = os.path.join(project_root, 'data', 'synthetic')
    os.makedirs(synthetic_data_dir, exist_ok=True)
    
    # Construct the full path for the output CSV file
    output_csv_path = os.path.join(synthetic_data_dir, 'synthetic_financial_data.csv')
    
    # Save the synthetic data using the absolute path
    synthetic_data.to_csv(output_csv_path, index=False)
    
    return synthetic_data
