# Synthetic Financial Data Generator

## Project Overview

This project is a Synthetic Financial Data Generator that uses advanced machine learning techniques to create realistic, privacy-preserving financial datasets. It's designed to help financial institutions, researchers, and data scientists work with high-quality synthetic data that maintains the statistical properties of real financial data without compromising individual privacy.

## Features

- Generate synthetic financial data using state-of-the-art generative models
- Validate synthetic data against original data distributions
- Preprocess and clean financial datasets
- Visualize data distributions and comparisons
- Web-based interface for easy interaction and data generation

## Technologies Used

- Python 3.9
- Streamlit for the web interface
- Pandas and NumPy for data manipulation
- Scikit-learn for machine learning utilities
- CTGAN (Conditional Tabular GAN) for synthetic data generation
- Diffprivlib for differential privacy
- Matplotlib and Seaborn for data visualization
- Docker for containerization

## Getting Started

### Prerequisites

- Python 3.9+
- Docker (optional)

### Local Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/synthetic-financial-data.git
   cd synthetic-financial-data
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

5. Open your web browser and navigate to `http://localhost:8501`

### Running with Docker

1. Build the Docker image:
   ```
   docker build -t synthetic-financial-data .
   ```

2. Run the Docker container:
   ```
   docker run -p 8501:8501 synthetic-financial-data
   ```

3. Open your web browser and navigate to `http://localhost:8501`

## Project Structure

- `app.py`: Main Streamlit application
- `src/`:
  - `data_processing/`: Data preprocessing scripts
  - `synthetic_generation/`: Synthetic data generation models
  - `validation/`: Data validation utilities
  - `models/`: Model training scripts
  - `utils/`: Utility functions and visualizations
- `Dockerfile`: Docker configuration for containerization
- `requirements.txt`: Python dependencies

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [CTGAN](https://github.com/sdv-dev/CTGAN) for the synthetic data generation model
- [Streamlit](https://streamlit.io/) for the web application framework
- [Diffprivlib](https://github.com/IBM/differential-privacy-library) for differential privacy implementation

