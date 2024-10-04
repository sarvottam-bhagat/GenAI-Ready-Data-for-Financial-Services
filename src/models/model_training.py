import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_and_benchmark_model():
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    
    financial_data_path = os.path.join(project_root, 'data', 'raw', 'financial_data.csv')
    synthetic_data_path = os.path.join(project_root, 'data', 'synthetic', 'synthetic_financial_data.csv')
    
    
    financial_df = pd.read_csv(financial_data_path)
    synthetic_df = pd.read_csv(synthetic_data_path)
   
    target_column = financial_df.columns[-1]
    feature_columns = financial_df.columns[:-1]
    
    
    X_original = financial_df[feature_columns]
    y_original = financial_df[target_column]
    X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(X_original, y_original, test_size=0.2, random_state=42)
    
    
    model_original = RandomForestRegressor(n_estimators=100, random_state=42)
    model_original.fit(X_train_original, y_train_original)
    y_pred_original = model_original.predict(X_test_original)
    mae_original = mean_absolute_error(y_test_original, y_pred_original)
    
    
    X_synthetic = synthetic_df[feature_columns]
    y_synthetic = synthetic_df[target_column]
    X_train_synthetic, X_test_synthetic, y_train_synthetic, y_test_synthetic = train_test_split(X_synthetic, y_synthetic, test_size=0.2, random_state=42)
    
    # Train model on synthetic data
    model_synthetic = RandomForestRegressor(n_estimators=100, random_state=42)
    model_synthetic.fit(X_train_synthetic, y_train_synthetic)
    y_pred_synthetic = model_synthetic.predict(X_test_synthetic)
    mae_synthetic = mean_absolute_error(y_test_synthetic, y_pred_synthetic)
    
    return mae_original, mae_synthetic
