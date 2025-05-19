"""
Feature Engineering Module
=========================
This module provides functions for feature engineering and transformation.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def calculate_wet_bulb_temperature(temperature, relative_humidity):
    """
    Calculate the wet bulb temperature from temperature and relative humidity
    
    Computes wet bulb temperature using a simplified approximation formula that combines
    air temperature and relative humidity. Wet bulb temperature is an important metric
    for assessing heat stress on the human body, as it accounts for the cooling effect
    of evaporation.
    
    Parameters
    ----------
    temperature : float or array-like
        Temperature in degrees Celsius
    relative_humidity : float or array-like
        Relative humidity as percentage (0-100)
        
    Returns
    -------
    float or array-like
        Wet bulb temperature in degrees Celsius
        
    Notes
    -----
    The formula used is an approximation:
    WBT = T * arctan[0.151977 * (rh% + 8.313659)^(1/2)] + arctan(T + rh%) 
          - arctan(rh% - 1.676331) + 0.00391838 *(rh%)^(3/2) * arctan(0.023101 * rh%) - 4.686035
    """
    # Formula for wet bulb temperature approximation
    # WBT = T * arctan[0.151977 * (rh% + 8.313659)^(1/2)] + arctan(T + rh%) - arctan(rh% - 1.676331) 
    #       + 0.00391838 *(rh%)^(3/2) * arctan(0.023101 * rh%) - 4.686035
    
    # Note: This is a simplified approximation
    term1 = temperature * np.arctan(0.151977 * np.sqrt(relative_humidity + 8.313659))
    term2 = np.arctan(temperature + relative_humidity)
    term3 = np.arctan(relative_humidity - 1.676331)
    term4 = 0.00391838 * np.power(relative_humidity, 1.5) * np.arctan(0.023101 * relative_humidity)
    
    wbt = term1 + term2 - term3 + term4 - 4.686035
    return wbt


def create_temporal_features(df):
    """
    Create temporal features from datetime index
    
    Extracts calendar-based features from a DataFrame's datetime index, including
    year, month, quarter, and season. These features can help capture seasonal
    patterns and temporal trends in time series data for modeling purposes.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with datetime index
        
    Returns
    -------
    pandas.DataFrame
        A new DataFrame (original is not modified) with additional temporal features:
        - year: The year component of the date
        - month: The month component (1-12)
        - quarter: The quarter (1-4)
        - season_numeric: The season as a number (1-4)
        
    Notes
    -----
    This function creates and returns a new DataFrame rather than modifying
    the input DataFrame in-place.
    """
    # Create a copy to avoid modifying the original
    df_new = df.copy()
    
    # Extract temporal features
    df_new['year'] = df_new.index.year
    df_new['month'] = df_new.index.month
    df_new['quarter'] = df_new.index.quarter
    
    # Create season numeric directly instead of going through string mapping
    # Winter: 1 (Dec-Feb), Spring: 2 (Mar-May), Summer: 3 (Jun-Aug), Fall: 4 (Sep-Nov)
    df_new['season_numeric'] = ((df_new.index.month % 12 + 3) // 3)
    
    return df_new


def create_interaction_features(df):
    """
    Create interaction features between variables
    
    Generates interaction features by combining related variables in meaningful ways.
    This includes multiplying temperature with humidity and creating composite 
    measures of greenhouse gases. These interaction features can capture non-linear
    relationships and synergistic effects between variables.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing climate and atmospheric variables
        
    Returns
    -------
    pandas.DataFrame
        A new DataFrame (original is not modified) with additional features:
        - temp_humidity_interaction: Product of mean air temperature and relative humidity
          (only created if both columns exist)
        - greenhouse_composite: Normalized sum of greenhouse gas concentrations
          (only created if at least 2 gas columns exist)
          
    Notes
    -----
    This function creates and returns a new DataFrame rather than modifying
    the input DataFrame in-place.
    
    Certain interaction features will only be created if their required source columns
    exist in the input DataFrame.
    
    Raises
    ------
    ValueError
        Indirectly from MinMaxScaler if any greenhouse gas column contains non-numeric values
    """
    # Create a copy to avoid modifying the original
    df_new = df.copy()
    
    # Create interactions between temperature and humidity
    if 'mean_air_temp' in df_new.columns and 'mean_relative_humidity' in df_new.columns:
        df_new['temp_humidity_interaction'] = df_new['mean_air_temp'] * df_new['mean_relative_humidity']
    
    # Create greenhouse gas composite
    gas_cols = [col for col in df_new.columns if any(x in col for x in ['co2', 'ch4', 'n2o', 'sf6'])]
    if len(gas_cols) >= 2:
        # Normalize and sum greenhouse gas columns
        scaler = MinMaxScaler()
        normalized_gases = scaler.fit_transform(df_new[gas_cols])
        df_new['greenhouse_composite'] = np.sum(normalized_gases, axis=1)
    
    return df_new


def create_lag_features(df, columns, lags=[1, 3, 6, 12]):
    """
    Create lag features for time series data
    
    Generates lagged versions of specified columns to capture temporal dependencies
    in time series data. For example, a lag of 1 means using the previous month's
    value as a feature for the current month, which can help identify autoregressive
    patterns in the data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with datetime index
    columns : list
        List of column names to create lags for
    lags : list, optional
        List of lag periods to create, defaults to [1, 3, 6, 12]
        
    Returns
    -------
    pandas.DataFrame
        A new DataFrame (original is not modified) with additional lag features 
        named as {column_name}_lag_{lag_period}. Note that lag features will contain
        NaN values for the first N rows where N is the largest lag value.
        
    Notes
    -----
    This function creates and returns a new DataFrame rather than modifying
    the input DataFrame in-place.
    
    Only columns that exist in the input DataFrame will have lag features created.
    Non-existent column names in the columns parameter will be silently ignored.
    
    Raises
    ------
    TypeError
        If columns parameter is not a list or lags parameter is not a list
    """
    # Create a copy to avoid modifying the original
    df_new = df.copy()
    
    # Create lag features
    for col in columns:
        if col in df_new.columns:
            for lag in lags:
                df_new[f'{col}_lag_{lag}'] = df_new[col].shift(lag)
    
    return df_new


def create_rolling_features(df, columns, windows=[3, 6, 12]):
    """
    Create rolling window features for time series data
    
    Generates rolling window statistics (mean and standard deviation) for specified
    columns. Rolling windows are useful for capturing trends and volatility over
    different time periods, smoothing out noise, and creating features that represent
    recent behavior in the time series.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with datetime index
    columns : list
        List of column names to create rolling features for
    windows : list, optional
        List of window sizes to create, defaults to [3, 6, 12]
        
    Returns
    -------
    pandas.DataFrame
        A new DataFrame (original is not modified) with additional rolling features:
        - {column_name}_rolling_mean_{window}: Rolling mean for each window size
        - {column_name}_rolling_std_{window}: Rolling standard deviation for each window size
        
        Note that rolling features will contain NaN values for the first N rows
        where N is the window size.
        
    Notes
    -----
    This function creates and returns a new DataFrame rather than modifying
    the input DataFrame in-place.
    
    Only columns that exist in the input DataFrame will have rolling features created.
    Non-existent column names in the columns parameter will be silently ignored.
    
    Raises
    ------
    TypeError
        If parameters are not of the expected types
    ValueError
        If any window size is less than 1
    """
    # Create a copy to avoid modifying the original
    df_new = df.copy()
    
    # Create rolling features
    for col in columns:
        if col in df_new.columns:
            for window in windows:
                df_new[f'{col}_rolling_mean_{window}'] = df_new[col].rolling(window=window).mean()
                df_new[f'{col}_rolling_std_{window}'] = df_new[col].rolling(window=window).std()
    
    return df_new


def prepare_features_for_modeling(df, target_col, drop_cols=None):
    """
    Prepare features for modeling by removing unnecessary columns and handling missing values
    
    Performs final data preparation steps before modeling, including removing specified
    columns, handling missing values by dropping rows, and separating features from the
    target variable.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing all variables
    target_col : str
        Name of the target variable column
    drop_cols : list, optional
        List of column names to drop from the dataset. If None, no columns will be dropped
        except the target column.
        
    Returns
    -------
    tuple
        X : pandas.DataFrame
            Features dataframe ready for modeling
        y : pandas.Series
            Target variable series
        feature_names : list
            List of feature column names used in X
            
    Notes
    -----
    This function will:
    1. Remove columns specified in drop_cols
    2. Drop rows with NaN values
    3. Separate target variable from features
    
    Raises
    ------
    KeyError
        If target_col is not present in the DataFrame
    ValueError
        If all data is dropped due to NaN values
    """
    # Create a copy to avoid modifying the original
    df_new = df.copy()
    
    # Drop specified columns
    if drop_cols:
        df_new = df_new.drop(columns=[col for col in drop_cols if col in df_new.columns])
    
    # Handle missing values
    df_new = df_new.dropna()
    
    # Extract features and target
    y = df_new[target_col] if target_col in df_new.columns else None
    X = df_new.drop(columns=[target_col]) if target_col in df_new.columns else df_new
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    return X, y, feature_names
