import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data_processing.preprocess_data import load_and_preprocess_data
from src.synthetic_generation.generate_synthetic_data import generate_synthetic_data
from src.privacy.differential_privacy import apply_differential_privacy
from src.validation.validate_synthetic_data import validate_synthetic_data
from src.models.model_training import train_and_benchmark_model
from src.utils.visualizations import plot_data_distribution


# Streamlit App
st.title("Synthetic Financial Data Generation and Model Benchmarking")

# Sidebar Navigation
st.sidebar.title("Project Steps")
steps = st.sidebar.radio("Select a Step", ("Load & Preprocess Data", 
                                           "Generate Synthetic Data",
                                           "Apply Differential Privacy", 
                                           "Validate Synthetic Data",
                                           "Train & Benchmark Model"))

# Load and Preprocess Data
if steps == "Load & Preprocess Data":
    st.header("Step 1: Load and Preprocess Data")
    if st.button("Preprocess Data"):
        df = load_and_preprocess_data()
        st.success("Data preprocessed and saved!")
        st.write(df.head())

# Generate Synthetic Data
elif steps == "Generate Synthetic Data":
    st.header("Step 2: Generate Synthetic Data")
    if st.button("Generate Synthetic Data"):
        synthetic_df = generate_synthetic_data()
        st.success("Synthetic data generated!")
        st.write(synthetic_df.head())

# Apply Differential Privacy
elif steps == "Apply Differential Privacy":
    st.header("Step 3: Apply Differential Privacy")
    if st.button("Apply Differential Privacy"):
        privacy_results = apply_differential_privacy()
        st.success("Differential privacy applied!")
        st.write(privacy_results)

# Validate Synthetic Data
elif steps == "Validate Synthetic Data":
    st.header("Step 4: Validate Synthetic Data")
    if st.button("Validate Data"):
        mse = validate_synthetic_data()
        st.success(f"Data validated with Mean Squared Error: {mse}")
        st.write("Comparison of Original and Synthetic Data Distributions:")
        fig = plot_data_distribution()
        st.pyplot(fig)

# Train and Benchmark Model
elif steps == "Train & Benchmark Model":
    st.header("Step 5: Train and Benchmark Model")
    if st.button("Train Model"):
        mae = train_and_benchmark_model()
        st.success(f"Model trained with Mean Absolute Error: {mae}")

# Run the app
if __name__ == '__main__':
    st.sidebar.markdown("Use the buttons to run each step interactively.")
