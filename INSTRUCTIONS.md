# Usage Instructions

This document provides instructions on how to set up and use the Wet Bulb Temperature Analysis project.

## Setup

### Option 1: Using pip

1. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
   ```
   venv\Scripts\activate
   ```
   - On macOS/Linux:
   ```
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Option 2: Using conda

1. Create and activate a conda environment:
   ```
   conda env create -f environment.yml
   conda activate wet-bulb-temp
   ```

## Verifying Environment Setup

To ensure your environment is properly configured:

```
python scripts/verify_environment.py
```

This will check that all required libraries are installed and accessible.

## Processing Raw Data

To clean and prepare the raw data files for analysis:

```
python scripts/preprocess_data.py
```

This script will:
- Load raw climate and greenhouse gas data
- Clean, transform, and combine the datasets
- Save the processed data for analysis and visualization

## Running the Streamlit Dashboard

The Streamlit dashboard provides an interactive way to explore the data and visualize the relationships between wet-bulb temperature and various climate variables.

1. Run the dashboard using the convenience script:
   ```
   python run_dashboard.py
   ```

   Or run it directly with Streamlit:
   ```
   streamlit run dashboard/app.py
   ```

2. Once the dashboard is running, you can access it in your web browser at http://localhost:8501

3. Use the navigation menu in the sidebar to explore different aspects of the data:
   - Home: Overview with key metrics
   - Data Explorer: Examine variable distributions
   - Time Series Analysis: Trends and patterns over time
   - Correlation Analysis: Relationships between variables
   - Regression Modeling: Build predictive models
   - About: Project information and methodology

## Running Sample Analysis

To run a quick analysis that demonstrates the key features of the project:

```
python scripts/analyze.py
```

This will generate visualizations and regression analysis results in the `data/output` directory.

## Working with Notebooks

The project includes two main notebooks:

1. **Original Analysis Notebook** (`notebooks/data_analysis_of_wet_bulb_temperature.ipynb`):
   - Comprehensive exploratory analysis with detailed explanations
   - Includes background research and in-depth interpretation

2. **Sample Analysis Notebook** (`notebooks/sample_analysis.ipynb`):
   - Streamlined version demonstrating key analysis techniques
   - Can be regenerated using:
     ```
     python scripts/create_sample_notebook.py
     ```

## Using the Python Modules

If you want to use the Python modules directly in your own scripts or notebooks, you can follow these steps:

1. Import the required modules:
   ```python
   from src.data_processing.data_loader import load_data, prepare_data_for_analysis
   from src.visualization.exploratory import plot_time_series, plot_correlation_matrix
   from src.models.regression import build_linear_regression_model, evaluate_regression_model
   ```

2. Load and prepare the data:
   ```python
   data_folder = "path/to/data"
   data = prepare_data_for_analysis(data_folder)
   ```

3. Visualize the data:
   ```python
   import matplotlib.pyplot as plt
   fig = plot_time_series(data, 'avg_wet_bulb', rolling_window=12)
   plt.show()
   ```

4. Build and evaluate a regression model:
   ```python
   from src.models.regression import preprocess_for_regression
   
   target = 'avg_wet_bulb'
   features = ['mean_air_temp', 'mean_relative_humidity', 'average_co2']
   
   X_train, X_test, y_train, y_test, scaler = preprocess_for_regression(
       data, target, features
   )
   
   model = build_linear_regression_model(X_train, y_train)
   results = evaluate_regression_model(model, X_train, X_test, y_train, y_test, features)
   ```

## Project Structure Overview

The project is organized into the following key components:

```
wet-bulb-temperature-analysis/
├── dashboard/           # Streamlit interactive dashboard
├── data/                # Data files (raw, processed, and outputs)
├── notebooks/           # Jupyter notebooks for analysis
├── scripts/             # Utility and analysis scripts
├── src/                 # Source code organized in modules
└── run_dashboard.py     # Convenience script to start the dashboard
```

For a more detailed overview of the project structure and implementation details, please refer to the README.md file.
