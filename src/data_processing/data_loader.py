"""
Data Loading and Preprocessing Module
====================================
This module provides functions to load and preprocess the data for analysis.
It handles reading various climate and atmospheric gas datasets, preprocessing
them into a consistent format, and merging them for analysis.
"""
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_folder):
    """
    Load all data files from the data folder
    
    Reads multiple datasets including wet bulb temperature, surface air temperature,
    climate variables (rainfall, sunshine, humidity), and greenhouse gases (CO2, CH4, 
    N2O, SF6) from CSV files in the specified data folder.
    
    Parameters
    ----------
    data_folder : str
        Path to the data folder containing raw data files
        
    Returns
    -------
    dict
        Dictionary of dataframes with keys:
        - 'wet_bulb': Wet bulb temperature data
        - 'air_temp': Surface air temperature data
        - 'climate_vars': Rainfall, sunshine, humidity data
        - 'co2': Carbon dioxide concentration data
        - 'ch4': Methane concentration data
        - 'n2o': Nitrous oxide concentration data
        - 'sf6': Sulfur hexafluoride concentration data
    """
    data = {}
    
    # Define raw data path
    raw_data_path = os.path.join(data_folder, 'raw')
    if not os.path.exists(raw_data_path):
        raw_data_path = data_folder  # Fallback if raw subfolder doesn't exist
        logger.warning(f"Raw data directory not found at {raw_data_path}, using {data_folder} instead")
    
    logger.info(f"Loading data from {raw_data_path}")
    
    # Wet bulb temperature data
    try:
        wet_bulb_path = os.path.join(raw_data_path, 'wet-bulb-temperature-hourly.csv')
        data['wet_bulb'] = pd.read_csv(wet_bulb_path)
        logger.info(f"Loaded wet bulb temperature data: {data['wet_bulb'].shape[0]} records")
    except FileNotFoundError:
        logger.error(f"Wet bulb temperature data file not found at {wet_bulb_path}")
    
    # Surface air temperature data
    try:
        air_temp_path = os.path.join(raw_data_path, 'surface-air-temperature-monthly-mean.csv')
        data['air_temp'] = pd.read_csv(air_temp_path)
        logger.info(f"Loaded surface air temperature data: {data['air_temp'].shape[0]} records")
    except FileNotFoundError:
        logger.error(f"Surface air temperature data file not found at {air_temp_path}")
        
    # Rainfall, sunshine, humidity data
    try:
        climate_vars_path = os.path.join(raw_data_path, 'M890081.csv')
        data['climate_vars'] = pd.read_csv(climate_vars_path)
        logger.info(f"Loaded climate variables data: {data['climate_vars'].shape[0]} records")
    except FileNotFoundError:
        logger.error(f"Climate variables data file not found at {climate_vars_path}")    
    # Carbon dioxide data
    try:
        co2_path = os.path.join(raw_data_path, 'co2_mm_mlo.csv')
        # Using comment='#' to skip header comments
        data['co2'] = pd.read_csv(co2_path, comment='#')
        logger.info(f"Loaded CO2 data: {data['co2'].shape[0]} records")
    except FileNotFoundError:
        logger.error(f"CO2 data file not found at {co2_path}")
    except Exception as e:
        logger.error(f"Error loading CO2 data: {e}")    
    # Methane data
    try:
        ch4_path = os.path.join(raw_data_path, 'ch4_mm_gl.csv')
        data['ch4'] = pd.read_csv(ch4_path, comment='#')
        logger.info(f"Loaded CH4 data: {data['ch4'].shape[0]} records")
    except FileNotFoundError:
        logger.error(f"CH4 data file not found at {ch4_path}")
    except Exception as e:
        logger.error(f"Error loading CH4 data: {e}")
    # Nitrous oxide data
    try:
        n2o_path = os.path.join(raw_data_path, 'n2o_mm_gl.csv')
        data['n2o'] = pd.read_csv(n2o_path, comment='#')
        logger.info(f"Loaded N2O data: {data['n2o'].shape[0]} records")
    except FileNotFoundError:
        logger.error(f"N2O data file not found at {n2o_path}")
    except Exception as e:
        logger.error(f"Error loading N2O data: {e}")
    # Sulfur hexafluoride data
    try:
        sf6_path = os.path.join(raw_data_path, 'sf6_mm_gl.csv')
        data['sf6'] = pd.read_csv(sf6_path, comment='#')
        logger.info(f"Loaded SF6 data: {data['sf6'].shape[0]} records")
    except FileNotFoundError:
        logger.error(f"SF6 data file not found at {sf6_path}")
    except Exception as e:
        logger.error(f"Error loading SF6 data: {e}")
    
    # Check if any data was loaded
    if not data:
        logger.error("No data files were loaded successfully")
    else:
        logger.info(f"Successfully loaded {len(data)} datasets")
        
    return data


def clean_wet_bulb_data(df):
    """
    Clean the wet bulb temperature data
    
    Processes the wet bulb temperature data by converting datetime strings,
    aggregating monthly statistics, and preparing the data for analysis.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Wet bulb temperature dataframe
        
    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe with columns:
        - 'month': Datetime object representing the month
        - 'avg_wet_bulb': Average wet bulb temperature for the month
        - 'max_wet_bulb': Maximum wet bulb temperature for the month
        - 'min_wet_bulb': Minimum wet bulb temperature for the month
        - 'std_wet_bulb': Standard deviation of wet bulb temperature for the month
    """
    # Copy the dataframe to avoid modifying the original
    df_cleaned = df.copy()
    
    # Check if we have wbt_date and wbt_time columns (new format)
    if 'wbt_date' in df_cleaned.columns and 'wbt_time' in df_cleaned.columns:
        # Convert date and time columns to a single datetime
        df_cleaned['date'] = pd.to_datetime(df_cleaned['wbt_date'])
    else:
        # Fallback for the old format (Date Time column)
        df_cleaned['date'] = pd.to_datetime(df_cleaned['Date Time'])
    
    # Create a year-month column for aggregation
    df_cleaned['year_month'] = df_cleaned['date'].dt.to_period('M')
      # Get the wet bulb temperature column name
    if 'wet_bulb_temperature' in df_cleaned.columns:
        temp_col = 'wet_bulb_temperature'
    elif 'Wet Bulb Temperature (°C)' in df_cleaned.columns:
        temp_col = 'Wet Bulb Temperature (°C)'
    else:
        raise ValueError("Cannot find wet bulb temperature column in the dataset")
    
    # Group by year-month and calculate monthly statistics
    monthly_stats = df_cleaned.groupby('year_month').agg(
        avg_wet_bulb=(temp_col, 'mean'),
        max_wet_bulb=(temp_col, 'max'),
        min_wet_bulb=(temp_col, 'min'),
        std_wet_bulb=(temp_col, 'std')
    ).reset_index()
    # Convert period to datetime for merging
    monthly_stats['month'] = monthly_stats['year_month'].dt.to_timestamp()
    
    # Return the dataframe without the year_month column to avoid Period datatype issues
    return monthly_stats[['month', 'avg_wet_bulb', 'max_wet_bulb', 'min_wet_bulb', 'std_wet_bulb']]


def clean_greenhouse_gas_data(df, gas_name):
    """
    Clean greenhouse gas data (CO2, CH4, N2O, SF6)
    
    Processes greenhouse gas data by creating datetime columns, renaming columns,
    and preparing the data for analysis.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Greenhouse gas dataframe
    gas_name : str
        Name of the greenhouse gas
        
    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe with columns:
        - 'month': Datetime object representing the month
        - 'average_<gas_name>': Average concentration of the greenhouse gas
    """
    # Copy the dataframe to avoid modifying the original
    df_cleaned = df.copy()
    
    # Identify the average column based on the headers in the file
    avg_col = None
    for col in df_cleaned.columns:
        if 'average' in str(col).lower():
            avg_col = col
            break
            
    # If no average column found, use the 4th column (index 3) which typically contains the concentration
    if not avg_col and len(df_cleaned.columns) > 3:
        avg_col = df_cleaned.columns[3]
    
    # Make sure we have year and month columns
    year_col = 'year' if 'year' in df_cleaned.columns else df_cleaned.columns[0]
    month_col = 'month' if 'month' in df_cleaned.columns else df_cleaned.columns[1]
    
    # Create new date column using month and year
    df_cleaned['date'] = df_cleaned[year_col].astype(str) + "-" + df_cleaned[month_col].astype(str)
    
    # Convert to datetime
    df_cleaned['month'] = pd.to_datetime(df_cleaned['date'], format='%Y-%m')
    
    # Rename and select columns
    df_cleaned = df_cleaned.rename(columns={avg_col: f'average_{gas_name}'})
    df_cleaned = df_cleaned[['month', f'average_{gas_name}']]
    
    return df_cleaned


def clean_climate_vars_data(df):
    """
    Clean climate variables data (rainfall, sunshine, humidity)
    
    Processes climate variables data by converting datetime columns, renaming columns,
    and preparing the data for analysis.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Climate variables dataframe
        
    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe with columns:
        - 'month': Datetime object representing the month
        - 'total_rainfall': Total rainfall for the month
        - 'daily_mean_sunshine': Daily mean sunshine hours for the month
        - 'mean_relative_humidity': Mean relative humidity for the month
    """
    # Copy the dataframe to avoid modifying the original
    df_cleaned = df.copy()
      # Find the row with 'Data Series' in the first column
    data_row_idx = None
    for i, row in df_cleaned.iterrows():
        if isinstance(row.iloc[0], str) and 'Data Series' in row.iloc[0]:
            data_row_idx = i
            break
    
    if data_row_idx is not None:
        # Get column names from this row
        header_row = df_cleaned.iloc[data_row_idx]
        # Get data below this row
        df_cleaned = df_cleaned.iloc[data_row_idx+1:].reset_index(drop=True)
        # Set column names (avoid None or empty)
        columns = []
        for col in header_row:
            if pd.isna(col) or col == '':
                columns.append(f'col_{len(columns)}')
            else:
                columns.append(col)
        df_cleaned.columns = columns
      # Filter out rows with non-date entries like "Footnotes:"
    date_col = df_cleaned.columns[0]
    valid_rows = []
    for i, value in enumerate(df_cleaned[date_col]):
        if isinstance(value, str) and len(value.strip()) > 0:
            # Keep only rows that look like they have a year at the beginning (e.g., "2023 May")
            if value.strip()[0:4].isdigit():
                valid_rows.append(i)
    
    # Filter dataframe to keep only valid rows
    df_cleaned = df_cleaned.iloc[valid_rows].reset_index(drop=True)
    
    # Extract date from first column (Data Series), which contains values like '2023 May'
    df_cleaned['month'] = pd.to_datetime(df_cleaned[date_col].str.strip(), format='%Y %b', errors='coerce')
    
    # Drop rows where date conversion failed
    df_cleaned = df_cleaned.dropna(subset=['month'])
      # Identify and rename columns based on column names
    rainfall_col = None
    sunshine_col = None
    humidity_col = None
    
    # Look for columns by keyword
    for col in df_cleaned.columns:
        col_str = str(col).lower()
        if 'rainfall' in col_str or 'rain' in col_str:
            rainfall_col = col
        elif 'sunshine' in col_str or 'sun' in col_str:
            sunshine_col = col
        elif 'humidity' in col_str or 'humid' in col_str:
            humidity_col = col
    
    # If not found, use positions
    if rainfall_col is None and len(df_cleaned.columns) > 1:
        rainfall_col = df_cleaned.columns[1]  
    if sunshine_col is None and len(df_cleaned.columns) > 2:
        sunshine_col = df_cleaned.columns[2]  
    if humidity_col is None and len(df_cleaned.columns) > 3:
        humidity_col = df_cleaned.columns[3]  
    
    # Create a new dataframe with standardized columns
    result = pd.DataFrame()
    result['month'] = df_cleaned['month']
    
    # Add data columns if they exist
    if rainfall_col:
        result['total_rainfall'] = pd.to_numeric(df_cleaned[rainfall_col], errors='coerce')
    else:
        result['total_rainfall'] = np.nan
        
    if sunshine_col:
        result['daily_mean_sunshine'] = pd.to_numeric(df_cleaned[sunshine_col], errors='coerce')
    else:
        result['daily_mean_sunshine'] = np.nan
        
    if humidity_col:
        result['mean_relative_humidity'] = pd.to_numeric(df_cleaned[humidity_col], errors='coerce')
    else:
        result['mean_relative_humidity'] = np.nan
    
    return result


def clean_air_temp_data(df):
    """
    Clean surface air temperature data
    
    Processes surface air temperature data by converting datetime columns, renaming columns,
    and preparing the data for analysis.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Surface air temperature dataframe
        
    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe with columns:
        - 'month': Datetime object representing the month
        - 'mean_air_temp': Mean air temperature for the month
    """
    # Copy the dataframe to avoid modifying the original
    df_cleaned = df.copy()
    
    # Convert month column to datetime
    df_cleaned['month'] = pd.to_datetime(df_cleaned['month'], format='%Y-%m')
      # Rename columns for clarity
    df_cleaned = df_cleaned.rename(columns={
        'surface_air_temperature': 'mean_air_temp',
        'mean_temp': 'mean_air_temp'  
    })
    
    return df_cleaned[['month', 'mean_air_temp']]


def merge_all_datasets(data_dict):
    """
    Merge all datasets into a single dataframe
    
    Combines cleaned datasets into a single dataframe for analysis by merging
    on the 'month' column.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary of dataframes
        
    Returns
    -------
    pandas.DataFrame
        Merged dataframe with all datasets combined
    """
    # Clean individual datasets
    wet_bulb_monthly = clean_wet_bulb_data(data_dict['wet_bulb'])
    co2_clean = clean_greenhouse_gas_data(data_dict['co2'], 'co2')
    ch4_clean = clean_greenhouse_gas_data(data_dict['ch4'], 'ch4')
    n2o_clean = clean_greenhouse_gas_data(data_dict['n2o'], 'n2o')
    sf6_clean = clean_greenhouse_gas_data(data_dict['sf6'], 'sf6')
    climate_vars_clean = clean_climate_vars_data(data_dict['climate_vars'])
    air_temp_clean = clean_air_temp_data(data_dict['air_temp'])
    
    # Merge all datasets on month
    merged_df = wet_bulb_monthly.merge(co2_clean, on='month', how='left')
    merged_df = merged_df.merge(ch4_clean, on='month', how='left')
    merged_df = merged_df.merge(n2o_clean, on='month', how='left')
    merged_df = merged_df.merge(sf6_clean, on='month', how='left')
    merged_df = merged_df.merge(climate_vars_clean, on='month', how='left')
    merged_df = merged_df.merge(air_temp_clean, on='month', how='left')
    
    # Set month as index for time series analysis
    merged_df.set_index('month', inplace=True)
    
    return merged_df


def prepare_data_for_analysis(data_folder, save_output=True, output_filename="final_dataset.csv"):
    """
    Prepare data for analysis by loading, cleaning and merging all datasets
    
    Loads raw datasets, cleans and preprocesses them, merges them into a single
    dataframe, and optionally saves the processed data to a CSV file.
    
    Parameters
    ----------
    data_folder : str
        Path to the data folder
    save_output : bool, optional
        Whether to save the processed data to a CSV file
    output_filename : str, optional
        Name of the output file if saving
        
    Returns
    -------
    pandas.DataFrame
        Fully prepared dataframe for analysis
    """
    # Load all datasets
    logger.info("Loading raw datasets...")
    data_dict = load_data(data_folder)
    
    if not data_dict:
        logger.error("No data loaded, cannot continue with preparation")
        return None
    
    # Check if all required datasets are available
    required_keys = ['wet_bulb', 'air_temp', 'climate_vars']
    missing_keys = [key for key in required_keys if key not in data_dict]
    if missing_keys:
        logger.warning(f"Missing required datasets: {missing_keys}")
    
    # Merge all datasets
    logger.info("Merging datasets...")
    merged_df = merge_all_datasets(data_dict)
    
    # Handle missing values
    initial_rows = merged_df.shape[0]
    merged_df = merged_df.dropna()
    final_rows = merged_df.shape[0]
    
    if initial_rows > final_rows:
        logger.warning(f"Removed {initial_rows - final_rows} rows with missing values")
    
    # Save the processed data if requested
    if save_output:
        save_processed_data(merged_df, data_folder, output_filename)
    
    logger.info(f"Data preparation complete. Final dataset shape: {merged_df.shape}")
    return merged_df


def save_processed_data(df, data_folder, filename="processed_data.csv"):
    """
    Save processed dataframe to the processed data folder
    
    Saves the processed dataframe to a CSV file in the 'processed' subfolder
    of the specified data folder. Creates the subfolder if it does not exist.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to save
    data_folder : str
        Path to the data folder
    filename : str, optional
        Name of the output file
        
    Returns
    -------
    str
        Path to the saved file
    """
    # Define processed data path
    processed_data_path = os.path.join(data_folder, 'processed')
    
    # Create processed data directory if it doesn't exist
    if not os.path.exists(processed_data_path):
        try:
            os.makedirs(processed_data_path)
            logger.info(f"Created processed data directory at {processed_data_path}")
        except Exception as e:
            logger.error(f"Error creating processed data directory: {e}")
            processed_data_path = data_folder
            logger.warning(f"Falling back to {data_folder} for saving processed data")
    
    # Full path to output file
    output_path = os.path.join(processed_data_path, filename)
    
    # Save dataframe
    try:
        df.to_csv(output_path)
        logger.info(f"Saved processed data to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        return None
