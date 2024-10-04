import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data_distribution():
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    
    original_data_path = os.path.join(project_root, 'data', 'raw', 'financial_data.csv')
    synthetic_data_path = os.path.join(project_root, 'data', 'synthetic', 'synthetic_financial_data.csv')
    
    
    original_data = pd.read_csv(original_data_path)
    synthetic_data = pd.read_csv(synthetic_data_path)
    
    
    numerical_columns = original_data.select_dtypes(include=[np.number]).columns
    
    
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
