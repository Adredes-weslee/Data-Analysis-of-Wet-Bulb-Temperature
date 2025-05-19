"""
Exploratory Data Analysis and Visualization Module
================================================
This module provides functions for exploratory data analysis and visualization.
It contains various utilities for creating time series plots, correlation matrices,
scatter plots, and other visualizations useful for analyzing climate and
atmospheric data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_plot_style():
    """
    Set the default plotting style
    
    Configures matplotlib and seaborn with consistent styling for all plots
    in the application, ensuring visual coherence across different charts
    and visualizations.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
        Modifies global matplotlib and seaborn settings
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    sns.set_palette('viridis')


def plot_time_series(df, column_name, title=None, ylabel=None, rolling_window=None):
    """
    Create a time series plot for a specific column
    
    Generates a line plot showing the time evolution of a variable, with
    optional rolling mean to highlight trends by smoothing out short-term
    fluctuations.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a datetime index
    column_name : str
        Name of the column to plot
    title : str, optional
        Plot title, defaults to 'Time Series of {column_name}' if not provided
    ylabel : str, optional
        Y-axis label, defaults to column_name if not provided
    rolling_window : int, optional
        Window size for rolling mean calculation, no rolling mean shown if None
        
    Returns
    -------
    matplotlib.figure.Figure
        The Figure object containing the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the raw data
    ax.plot(df.index, df[column_name], 'o-', alpha=0.6, label=column_name)
    
    # Add rolling mean if specified
    if rolling_window is not None:
        rolling_mean = df[column_name].rolling(window=rolling_window).mean()
        ax.plot(df.index, rolling_mean, 'r-', linewidth=2, 
                label=f'{rolling_window}-period Rolling Mean')
        
    # Add title and labels
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Time Series of {column_name}', fontsize=14)
        
    ax.set_xlabel('Date', fontsize=12)
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    else:
        ax.set_ylabel(column_name, fontsize=12)
        
    # Add legend and grid
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    return fig


def plot_correlation_matrix(df, title='Correlation Matrix'):
    """
    Create a correlation matrix heatmap
    
    Generates a triangular heatmap showing Pearson correlation coefficients
    between all pairs of variables in the dataframe. Uses color intensity
    to indicate the strength and direction (positive/negative) of the
    correlations.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing variables to correlate
    title : str, optional
        Plot title, defaults to 'Correlation Matrix'
        
    Returns
    -------
    matplotlib.figure.Figure
        The Figure object containing the plot
        
    Notes
    -----
    This function uses pandas.DataFrame.corr() which computes the pairwise correlation
    of columns, excluding NA/null values. Only numeric columns are included in the
    correlation calculation.
    
    The upper triangle of the correlation matrix is masked to avoid redundant display
    of the symmetric matrix.
    
    Raises
    ------
    ValueError
        If the DataFrame has no numeric columns for correlation calculation
    """    
    
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['number'])
    
    # Calculate the correlation matrix on numeric columns only
    corr_matrix = numeric_df.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw the heatmap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt='.2f', cbar_kws={'shrink': .5})
    
    # Add title
    plt.title(title, fontsize=16)
    
    return fig


def plot_scatter_with_regression(df, x_col, y_col, title=None):
    """
    Create a scatter plot with regression line
    
    Generates a scatter plot of two variables with an overlaid regression
    line to visualize the relationship between them. Useful for identifying
    linear trends and outliers.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the variables to plot
    x_col : str
        Name of the column for x-axis
    y_col : str
        Name of the column for y-axis
    title : str, optional
        Plot title, defaults to '{y_col} vs {x_col}' if not provided
        
    Returns
    -------
    matplotlib.figure.Figure
        The Figure object containing the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create the scatter plot
    sns.regplot(x=x_col, y=y_col, data=df, scatter_kws={'alpha': 0.5}, ax=ax)
    
    # Add title and labels
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'{y_col} vs {x_col}', fontsize=14)
        
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_boxplots(df, columns, title='Distribution of Variables'):
    """
    Create boxplots for multiple columns
    
    Generates boxplots for the specified columns, providing a visual summary
    of the distributions, including medians, quartiles, and potential outliers.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the variables to plot
    columns : list
        List of column names to plot
    title : str, optional
        Plot title, defaults to 'Distribution of Variables'
        
    Returns
    -------
    matplotlib.figure.Figure
        The Figure object containing the plot
    """
    # Create a copy of the data without the index to avoid conflicts
    df_copy = df[columns].copy().reset_index(drop=True)
    
    # Melt the dataframe for easier plotting, without using the index
    df_melt = pd.melt(df_copy, value_vars=columns, 
                      var_name='Variable', value_name='Value')
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create the boxplot
    sns.boxplot(x='Variable', y='Value', data=df_melt, ax=ax)
    
    # Add title and labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Variable', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    
    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def plot_monthly_patterns(df, column_name, title=None):
    """
    Create box plots showing monthly patterns for a specific variable
    
    Generates boxplots for each month, summarizing the distribution of the
    specified variable. Useful for identifying seasonal trends and variations.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with datetime index
    column_name : str
        Name of the column to analyze
    title : str, optional
        Plot title, defaults to 'Monthly Distribution of {column_name}' if not provided
        
    Returns
    -------
    matplotlib.figure.Figure
        The Figure object containing the plot
        
    Notes
    -----
    This function creates a copy of the input DataFrame and adds a 'month_num' column
    derived from the DataFrame's index. The original DataFrame is not modified.
    
    Raises
    ------
    KeyError
        If the specified column_name does not exist in the DataFrame
    TypeError
        If the DataFrame's index is not a datetime index
    """
    # Extract month from index and create a new column
    monthly_data = df.copy()
    monthly_data['month_num'] = monthly_data.index.month
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create the boxplot with order to ensure months appear in correct order
    sns.boxplot(x='month_num', y=column_name, data=monthly_data, ax=ax, 
                order=list(range(1, 13)))  # Explicitly specify month order 1-12
    
    # Add title and labels
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Monthly Distribution of {column_name}', fontsize=14)
        
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel(column_name, fontsize=12)
    
    # Set x-axis labels to month names with proper tick positions
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Make sure ticks match the actual data values (1-12)
    ax.set_xticks(list(range(1, 13)))
    ax.set_xticklabels(month_names)
    
    return fig
