"""
Analysis script to demonstrate how to use the project modules

This script provides a standalone example of how to use the project's various modules
to load data, create visualizations, and build regression models for wet bulb temperature
analysis. It serves as both documentation and a functional example for users of the
codebase.

The script performs several key steps:
1. Loads and preprocesses climate and atmospheric data
2. Generates exploratory visualizations
3. Builds and evaluates a regression model for wet bulb temperature

Outputs are saved to an 'output' directory within the project root.
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Add the project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our modules
from src.data_processing.data_loader import load_data, prepare_data_for_analysis
from src.visualization.exploratory import (
    set_plot_style, plot_time_series, plot_correlation_matrix,
    plot_scatter_with_regression
)
from src.models.regression import (
    preprocess_for_regression, build_linear_regression_model,
    evaluate_regression_model, plot_feature_importance
)

def main():
    """
    Main analysis function
    
    Executes the complete analysis workflow:
    - Data loading and preprocessing
    - Creating and saving visualizations
    - Building and evaluating a regression model for wet bulb temperature
    
    This function demonstrates the standard workflow for analysis using
    the project's modules and serves as an example for users.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
        Results are printed to console and saved as files
    """
    print("Starting wet bulb temperature analysis...")
    
    # Define data folder path
    data_folder = Path(__file__).parent / "data"
    
    # Set plot style
    set_plot_style()
    
    # Load and prepare data
    print("Loading and preparing data...")
    try:
        data = prepare_data_for_analysis(data_folder)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"Data loaded. Shape: {data.shape}")
    print("\nFirst 5 rows:")
    print(data.head())
    
    # Create output directory for plots
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Generate some basic visualizations
    print("\nGenerating visualizations...")
    
    # Time series plot of wet bulb temperature
    fig1 = plot_time_series(
        data, 'avg_wet_bulb', 
        title='Average Monthly Wet Bulb Temperature',
        ylabel='Temperature (°C)',
        rolling_window=12
    )
    fig1.savefig(output_dir / "wet_bulb_time_series.png")
    
    # Correlation matrix
    fig2 = plot_correlation_matrix(data)
    fig2.savefig(output_dir / "correlation_matrix.png")
    
    # Scatter plot with regression line
    if 'mean_air_temp' in data.columns and 'avg_wet_bulb' in data.columns:
        fig3 = plot_scatter_with_regression(
            data, 'mean_air_temp', 'avg_wet_bulb',
            title='Wet Bulb Temperature vs Air Temperature'
        )
        fig3.savefig(output_dir / "temp_scatter.png")
    
    # Build a simple regression model
    print("\nBuilding regression model...")
    target = 'avg_wet_bulb'
    features = ['mean_air_temp', 'mean_relative_humidity', 'average_co2']
    
    # Make sure all features are available
    available_features = [f for f in features if f in data.columns]
    if len(available_features) < 2:
        print(f"Error: Not enough features available. Found: {available_features}")
        return
    
    # Preprocess data
    X_train, X_test, y_train, y_test, _ = preprocess_for_regression(
        data, target, available_features
    )
    
    # Build model
    model = build_linear_regression_model(X_train, y_train)
    
    # Evaluate model
    results = evaluate_regression_model(
        model, X_train, X_test, y_train, y_test, available_features
    )
    
    # Print results
    print("\nRegression Model Results:")
    print(f"Training R²: {results['train_r2']:.4f}")
    print(f"Testing R²: {results['test_r2']:.4f}")
    print(f"Training RMSE: {results['train_rmse']:.4f}")
    print(f"Testing RMSE: {results['test_rmse']:.4f}")
    
    # Feature importance plot
    fig4 = plot_feature_importance(model, available_features)
    fig4.savefig(output_dir / "feature_importance.png")
    
    print("\nAnalysis completed. Results saved to output directory.")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
