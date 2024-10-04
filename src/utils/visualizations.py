import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data_distribution():
    # Get the absolute path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Construct the full paths to the input CSV files
    original_data_path = os.path.join(project_root, 'data', 'raw', 'financial_data.csv')
    synthetic_data_path = os.path.join(project_root, 'data', 'synthetic', 'synthetic_financial_data.csv')
    
    # Read the CSV files using the absolute paths
    original_data = pd.read_csv(original_data_path)
    synthetic_data = pd.read_csv(synthetic_data_path)
    
    # Select numerical columns for visualization
    numerical_columns = original_data.select_dtypes(include=[np.number]).columns
    
    # Create subplots for each numerical column
    fig, axes = plt.subplots(len(numerical_columns), 2, figsize=(15, 5*len(numerical_columns)))
    fig.suptitle('Distribution Comparison: Original vs Synthetic Data')
    
    for i, column in enumerate(numerical_columns):
        # Plot original data
        sns.histplot(original_data[column], ax=axes[i, 0], kde=True)
        axes[i, 0].set_title(f'Original {column}')
        
        # Plot synthetic data
        sns.histplot(synthetic_data[column], ax=axes[i, 1], kde=True)
        axes[i, 1].set_title(f'Synthetic {column}')
    
    plt.tight_layout()
    return fig
