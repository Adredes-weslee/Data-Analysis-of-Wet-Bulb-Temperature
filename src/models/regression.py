"""
Regression Models Module
=======================
This module provides functions for building and evaluating regression models.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

def preprocess_for_regression(df, target_col, feature_cols, test_size=0.2, random_state=42):
    """
    Preprocess data for regression modeling
    
    Prepares data for regression modeling by extracting features and target variables,
    splitting the data into training and testing sets, and scaling the features. This 
    standardization helps improve model performance and convergence by ensuring all 
    features are on similar scales.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing both features and target variable
    target_col : str
        Name of the target variable column
    feature_cols : list
        List of feature column names to be used in the model
    test_size : float, optional
        Proportion of the dataset to include in the test split (0.0-1.0), defaults to 0.2 (20%)
    random_state : int, optional
        Random state for reproducibility of the train/test split, defaults to 42
        
    Returns
    -------
    tuple
        X_train_scaled : array-like
            Scaled training feature data
        X_test_scaled : array-like
            Scaled testing feature data
        y_train : array-like
            Training target values
        y_test : array-like
            Testing target values
        scaler : sklearn.preprocessing.StandardScaler
            Fitted scaler object for future transformations
    """
    # Extract features and target
    X = df[feature_cols]
    y = df[target_col]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def build_linear_regression_model(X_train, y_train):
    """
    Build a linear regression model
    
    Creates and fits a standard linear regression model using scikit-learn's
    LinearRegression implementation. The model finds the best-fit line that
    minimizes the sum of squared errors between predicted and actual values.
    
    Parameters
    ----------
    X_train : array-like
        Training features of shape (n_samples, n_features)
    y_train : array-like
        Training target values of shape (n_samples,)
        
    Returns
    -------
    sklearn.linear_model.LinearRegression
        Fitted linear regression model ready for predictions
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_regression_model(model, X_train, X_test, y_train, y_test, feature_names=None):
    """
    Evaluate regression model performance
    
    Calculates various performance metrics for a regression model on both training
    and testing datasets. The metrics include Root Mean Squared Error (RMSE), 
    R-squared (R²), and Mean Absolute Error (MAE). For linear models, also extracts
    the coefficients and intercept.
    
    Parameters
    ----------
    model : sklearn model
        Trained regression model
    X_train : array-like
        Training features
    X_test : array-like
        Testing features
    y_train : array-like
        Training target
    y_test : array-like
        Testing target
    feature_names : list, optional
        List of feature names for coefficient mapping
        
    Returns
    -------
    dict
        Dictionary of evaluation metrics and model details including:
        - train_rmse: Root Mean Squared Error on training data
        - test_rmse: Root Mean Squared Error on test data
        - train_r2: R-squared on training data
        - test_r2: R-squared on test data
        - train_mae: Mean Absolute Error on training data
        - test_mae: Mean Absolute Error on test data
        - coefficients: Dictionary mapping feature names to coefficients (if available)
        - intercept: Model intercept (if available)
    """
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Extract coefficients if it's a linear model
    coefficients = None
    if feature_names is not None and hasattr(model, 'coef_'):
        coefficients = {feature: coef for feature, coef in zip(feature_names, model.coef_)}
    
    # Create results dictionary
    results = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'coefficients': coefficients
    }
    
    if hasattr(model, 'intercept_'):
        results['intercept'] = model.intercept_
    
    return results


def plot_actual_vs_predicted(y_actual, y_predicted, title='Actual vs Predicted'):
    """
    Create a scatter plot of actual vs predicted values
    
    Generates a scatter plot comparing actual target values with model predictions.
    A diagonal red dashed line represents perfect predictions (where actual equals 
    predicted). This visualization helps assess model performance and identify 
    regions where the model may systematically over- or under-predict.
    
    Parameters
    ----------
    y_actual : array-like
        Actual target values
    y_predicted : array-like
        Predicted target values
    title : str, optional
        Plot title, defaults to 'Actual vs Predicted'
        
    Returns
    -------
    matplotlib.figure.Figure
        The Figure object containing the plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the scatter plot
    ax.scatter(y_actual, y_predicted, alpha=0.5)
    
    # Add the perfect prediction line
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    # Add title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Actual', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_residuals(y_actual, y_predicted, title='Residual Plot'):
    """
    Create a residual plot
    
    Generates a scatter plot of model residuals (actual - predicted values) against 
    predicted values. This visualization helps identify patterns in the errors, 
    which can indicate model misspecification, heteroscedasticity, or other issues.
    A horizontal red dashed line at y=0 represents perfect predictions.
    
    Parameters
    ----------
    y_actual : array-like
        Actual target values
    y_predicted : array-like
        Predicted target values
    title : str, optional
        Plot title, defaults to 'Residual Plot'
        
    Returns
    -------
    matplotlib.figure.Figure
        The Figure object containing the plot
    """
    # Calculate residuals
    residuals = y_actual - y_predicted
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot residuals
    ax.scatter(y_predicted, residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--')
    
    # Add title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_feature_importance(model, feature_names, title='Feature Importance'):
    """
    Plot feature importance for linear models
    
    Creates a horizontal bar chart showing the coefficient values for each feature
    in a linear regression model. Positive coefficients are shown in green and 
    negative coefficients in red. Features are sorted by the absolute magnitude
    of their coefficients to highlight the most influential variables.
    
    Parameters
    ----------
    model : sklearn.linear_model
        Trained linear model with a coef_ attribute
    feature_names : list
        List of feature names corresponding to the model coefficients
    title : str, optional
        Plot title, defaults to 'Feature Importance'
        
    Returns
    -------
    matplotlib.figure.Figure
        The Figure object containing the plot
        
    Raises
    ------
    ValueError
        If the model does not have a coef_ attribute
    """
    if not hasattr(model, 'coef_'):
        raise ValueError("Model does not have coefficients")
    
    # Get coefficients
    coefficients = model.coef_
    
    # Create a DataFrame for easier plotting
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    
    # Sort by absolute coefficient value
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot horizontal bar chart
    colors = ['green' if c > 0 else 'red' for c in coef_df['Coefficient']]
    ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
    
    # Add title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Coefficient Value', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return fig
