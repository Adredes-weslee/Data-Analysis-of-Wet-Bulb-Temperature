"""
Create Sample Notebook
=====================
This script creates a sample Jupyter notebook that demonstrates the usage of our refactored modules.

The generated notebook serves as both documentation and a tutorial, showcasing:
- Data loading and preprocessing
- Exploratory data analysis
- Time series analysis
- Feature engineering
- Regression modeling

The notebook is saved to the 'notebooks' directory in the project root.
"""
import os
import nbformat as nbf
from pathlib import Path

def main():
    """
    Create a sample Jupyter notebook
    
    Generates a Jupyter notebook that demonstrates the usage of the project's modules.
    The notebook includes sections for data loading, exploratory analysis, feature
    engineering, and regression modeling. It serves as both documentation and a
    tutorial for new users of the codebase.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
        Creates and saves a Jupyter notebook file in the notebooks directory
    """
    # Define notebook path
    project_root = Path(__file__).parent
    notebook_path = os.path.join(str(project_root), 'notebooks', 'sample_analysis.ipynb')
    
    # Create notebooks directory if it doesn't exist
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    
    # Create a new notebook
    nb = nbf.v4.new_notebook()
    
    # Add cells to the notebook
    cells = [
        # Title and introduction
        nbf.v4.new_markdown_cell("""# Wet Bulb Temperature Analysis
        
This notebook demonstrates the usage of the refactored Python modules for analyzing wet bulb temperature data in Singapore.

The analysis covers:
1. Data loading and preprocessing
2. Exploratory data analysis
3. Time series analysis
4. Correlation analysis
5. Regression modeling
"""),
        # Setup and imports
        nbf.v4.new_markdown_cell("## Setup and Imports"),
        nbf.v4.new_code_cell("""# Import necessary modules
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path().absolute().parent
sys.path.insert(0, str(project_root))

# Import our modules
from src.data_processing.data_loader import load_data, prepare_data_for_analysis
from src.visualization.exploratory import plot_time_series, plot_correlation_matrix, plot_scatter_with_regression
from src.models.regression import preprocess_for_regression, build_linear_regression_model, evaluate_regression_model
from src.features.feature_engineering import create_temporal_features, create_interaction_features

# Other imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
"""),
        # Data loading
        nbf.v4.new_markdown_cell("## Data Loading and Preprocessing"),
        nbf.v4.new_code_cell("""# Define data path
data_path = os.path.join(str(project_root), 'data')

# Option 1: Load processed data if available
processed_path = os.path.join(data_path, 'processed', 'final_dataset.csv')
if os.path.exists(processed_path):
    print(f"Loading processed data from {processed_path}")
    df = pd.read_csv(processed_path, parse_dates=['month'], index_col='month')
else:
    # Option 2: Prepare data from raw files
    print("Processing data from raw files")
    df = prepare_data_for_analysis(data_path)

# Display basic information
print(f"Dataset shape: {df.shape}")
print(f"Date range: {df.index.min().strftime('%b %Y')} to {df.index.max().strftime('%b %Y')}")
df.head()
"""),
        # Data overview
        nbf.v4.new_markdown_cell("## Data Overview"),
        nbf.v4.new_code_cell("""# Display column information
df.info()

# Display summary statistics
df.describe()
"""),
        # Time series analysis
        nbf.v4.new_markdown_cell("## Time Series Analysis"),
        nbf.v4.new_code_cell("""# Plot wet bulb temperature over time
from src.visualization.exploratory import plot_time_series, plot_monthly_patterns

# Plot time series with trend line
fig = plot_time_series(df, 'avg_wet_bulb', 'Monthly Average Wet Bulb Temperature', 'Temperature (°C)', add_trend=True)
plt.show()

# Plot monthly patterns
fig = plot_monthly_patterns(df, 'avg_wet_bulb')
plt.show()

# Analyze seasonality and trends
from src.utils.statistics import calculate_trends

trends = calculate_trends(df['avg_wet_bulb'])
print(f"Overall trend direction: {trends['trend_direction']}")
print(f"Trend slope: {trends['trend_slope']:.6f}°C per month")
print(f"Is trend statistically significant? {'Yes' if trends['is_significant'] else 'No'} (p-value: {trends['p_value']:.4f})")
"""),
        # Correlation analysis
        nbf.v4.new_markdown_cell("## Correlation Analysis"),
        nbf.v4.new_code_cell("""# Select climate and greenhouse gas variables
climate_vars = [col for col in df.columns if any(x in col.lower() for x in ['temperature', 'humidity', 'rainfall', 'sunshine'])]
ghg_vars = [col for col in df.columns if any(x in col.lower() for x in ['co2', 'ch4', 'n2o', 'sf6'])]

# Create correlation matrix
selected_vars = ['avg_wet_bulb'] + climate_vars + ghg_vars
correlation_df = df[selected_vars].dropna()

# Plot correlation matrix
fig = plot_correlation_matrix(correlation_df.corr(), title='Correlation Matrix')
plt.show()

# Scatter plot with regression line for specific variables
fig = plot_scatter_with_regression(df, 'co2', 'avg_wet_bulb')
plt.show()
"""),
        # Feature engineering
        nbf.v4.new_markdown_cell("## Feature Engineering"),
        nbf.v4.new_code_cell("""# Create temporal features
df_features = create_temporal_features(df.copy())
print("Temporal features created:", [col for col in df_features.columns if col not in df.columns])

# Create interaction features
interaction_columns = ['mean_temperature', 'mean_rainfall']
if all(col in df.columns for col in interaction_columns):
    df_features = create_interaction_features(df_features, columns=interaction_columns)
    print("Interaction features created:", [col for col in df_features.columns if col not in df.columns and 'interaction' in col])

# Show the new features
df_features[[col for col in df_features.columns if col not in df.columns]].head()
"""),
        # Regression modeling
        nbf.v4.new_markdown_cell("## Regression Modeling"),
        nbf.v4.new_code_cell("""# Prepare data for regression modeling
target_var = 'avg_wet_bulb'
feature_vars = [col for col in df.select_dtypes(include=['number']).columns 
                if col != target_var and 'wet_bulb' not in col][:5]  # Using a few features for simplicity

print(f"Target variable: {target_var}")
print(f"Feature variables: {feature_vars}")

# Preprocess data for regression
X_train, X_test, y_train, y_test, scaler = preprocess_for_regression(
    df, target_var, feature_vars, test_size=0.2, random_state=42
)

# Build and evaluate the model
model = build_linear_regression_model(X_train, y_train)
results = evaluate_regression_model(model, X_train, X_test, y_train, y_test, feature_vars)

# Print model results
print("\\nModel Results:")
print(f"Training R²: {results['train_r2']:.4f}")
print(f"Testing R²: {results['test_r2']:.4f}")
print(f"RMSE: {results['test_rmse']:.4f}")

# Plot feature importance
from src.models.regression import plot_feature_importance
fig = plot_feature_importance(model, feature_vars)
plt.show()

# Plot actual vs predicted
from src.models.regression import plot_actual_vs_predicted, plot_residuals
y_pred = model.predict(X_test)

fig = plot_actual_vs_predicted(y_test, y_pred)
plt.show()

fig = plot_residuals(y_test, y_pred)
plt.show()
"""),
        # Conclusion
        nbf.v4.new_markdown_cell("""## Conclusion

In this notebook, we've demonstrated how to use the refactored Python modules for wet bulb temperature analysis:

1. **Data Processing**: We loaded and preprocessed the data using the `data_processing` module.
2. **Visualization**: We created time series plots, correlation matrices, and scatter plots using the `visualization` module.
3. **Feature Engineering**: We added temporal and interaction features using the `features` module.
4. **Statistical Analysis**: We analyzed trends and patterns using the `utils` module.
5. **Regression Modeling**: We built and evaluated a linear regression model using the `models` module.

This demonstrates how our modular code structure makes it easy to perform complex analyses in a clean, organized manner.
"""),
    ]
    
    # Add cells to notebook
    nb['cells'] = cells
    
    # Write notebook to file
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)
    
    print(f"Sample notebook created at {notebook_path}")

if __name__ == "__main__":
    main()
