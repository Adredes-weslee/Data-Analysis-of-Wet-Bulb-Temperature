"""
Custom Statistical Functions Module
==================================
This module provides custom statistical functions for data analysis.
"""
import numpy as np
import math

def custom_mean(data_list):
    """
    Calculate the mean of a list of numbers
    
    Parameters
    ----------
    data_list : list or array-like
        List of numbers
        
    Returns
    -------
    float
        Mean value
    """
    total = 0
    for element in data_list:
        total += element
    return total / len(data_list)


def custom_mean_nan(data_list):
    """
    Calculate the mean of a list of numbers, ignoring NaN values
    
    Parameters
    ----------
    data_list : list or array-like
        List of numbers which may contain NaN values
        
    Returns
    -------
    float
        Mean value (excluding NaNs)
    """
    total = 0
    no_nan_length = 0
    for element in data_list:
        if not math.isnan(element): 
            total += element
            no_nan_length += 1
    return total / no_nan_length if no_nan_length > 0 else float('nan')


def custom_std(data_list):
    """
    Calculate the standard deviation of a list of numbers
    
    Parameters
    ----------
    data_list : list or array-like
        List of numbers
        
    Returns
    -------
    float
        Standard deviation
    """
    # Calculate the mean
    mean = custom_mean(data_list)
    
    # Sum of squared differences
    sum_squared_diff = 0
    for element in data_list:
        sum_squared_diff += (element - mean) ** 2
    
    # Return the square root of the average squared difference
    return math.sqrt(sum_squared_diff / len(data_list))


def custom_std_nan(data_list):
    """
    Calculate the standard deviation of a list of numbers, ignoring NaN values
    
    Computes the population standard deviation (square root of variance) after
    filtering out NaN values from the input list. If all values are NaN, returns NaN.
    
    Parameters
    ----------
    data_list : list or array-like
        List of numbers which may contain NaN values
        
    Returns
    -------
    float
        Standard deviation (excluding NaNs), or NaN if all input values are NaN
        
    Notes
    -----
    The standard deviation is calculated using the formula:
    σ = sqrt(Σ(x_i - μ)² / N)
    where μ is the mean of the non-NaN values and N is the count of non-NaN values.
    """
    # Get list without NaN values
    clean_list = [x for x in data_list if not math.isnan(x)]
    
    if len(clean_list) == 0:
        return float('nan')
    
    # Calculate the mean
    mean = custom_mean(clean_list)
    
    # Sum of squared differences
    sum_squared_diff = 0
    for element in clean_list:
        sum_squared_diff += (element - mean) ** 2
    
    # Return the square root of the average squared difference
    return math.sqrt(sum_squared_diff / len(clean_list))


def custom_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient between two variables
    
    Computes the Pearson correlation coefficient, which measures the linear 
    relationship between two variables. The coefficient ranges from -1 (perfect 
    negative correlation) through 0 (no correlation) to 1 (perfect positive correlation).
    
    Parameters
    ----------
    x : list or array-like
        First variable
    y : list or array-like
        Second variable
        
    Returns
    -------
    float
        Pearson correlation coefficient
        
    Raises
    ------
    ValueError
        If the input lists have different lengths
    
    Notes
    -----
    The Pearson correlation is calculated as:
    r = Σ[(x_i - x̄)(y_i - ȳ)] / √[Σ(x_i - x̄)² * Σ(y_i - ȳ)²]
    where x̄ and ȳ are the means of x and y respectively.
    """
    # Check if the lengths match
    if len(x) != len(y):
        raise ValueError("The lengths of the input lists must be equal")
    
    # Calculate the means
    mean_x = custom_mean(x)
    mean_y = custom_mean(y)
    
    # Calculate the numerator (covariance)
    numerator = 0
    for i in range(len(x)):
        numerator += (x[i] - mean_x) * (y[i] - mean_y)
    
    # Calculate the denominators (standard deviations)
    sum_x_squared_diff = 0
    sum_y_squared_diff = 0
    for i in range(len(x)):
        sum_x_squared_diff += (x[i] - mean_x) ** 2
        sum_y_squared_diff += (y[i] - mean_y) ** 2
    
    # Calculate the correlation coefficient
    return numerator / (math.sqrt(sum_x_squared_diff) * math.sqrt(sum_y_squared_diff))
