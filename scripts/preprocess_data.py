"""
Data Preprocessing Script
========================
This script loads the raw data, processes it, and saves the processed data for the dashboard.

The preprocessing workflow consists of:
1. Loading raw wet bulb temperature, climate variables, and greenhouse gas data
2. Cleaning and transforming each dataset (handling missing values, date formatting, etc.)
3. Merging the datasets into a unified timeline with consistent formatting
4. Saving the processed dataset for use by the analysis tools and dashboard

Usage:
    python preprocess_data.py

The script outputs a processed CSV file in the data/processed directory and logs
its activities to preprocessing.log.
"""
import os
import logging
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent

# Ensure logs directory exists
logs_dir = project_root / 'logs'
logs_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / 'preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
import sys
sys.path.insert(0, str(project_root))

# Import our data processing functions
from src.data_processing.data_loader import (
    load_data, clean_wet_bulb_data, 
    clean_greenhouse_gas_data, clean_climate_vars_data,
    clean_air_temp_data
)

def main():
    """
    Main function to preprocess and save the data
    
    Orchestrates the complete data preprocessing workflow:
    1. Loading raw data files from the data/raw directory
    2. Preprocessing each individual dataset:
       - Wet bulb temperature data
       - Climate variables (temperature, humidity, rainfall, etc.)
       - Greenhouse gas data (CO2, CH4, N2O, SF6)
    3. Merging all datasets into a unified dataframe
    4. Saving the processed data to the data/processed directory
    
    All processing steps are logged to both the console and a preprocessing.log file
    to track the data transformation process.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
        The processed data is saved to disk as CSV file
        
    Side Effects
    -----------
    - Creates preprocessing.log file in the current directory
    - Creates data/processed directory if it doesn't exist
    - Writes processed data to data/processed/final_dataset.csv
    
    Raises
    ------
    Exception
        If any required dataset is missing or if processing fails
    """
    logger.info("Starting data preprocessing")
    
    # Define data paths
    raw_data_path = os.path.join(str(project_root), 'data', 'raw')
    processed_data_path = os.path.join(str(project_root), 'data', 'processed')
    
    # Create processed directory if it doesn't exist
    os.makedirs(processed_data_path, exist_ok=True)
    
    try:
        # Load raw data
        logger.info("Loading raw data files")
        data_dict = load_data(raw_data_path)
        
        # Check if all necessary data is loaded
        required_datasets = ['wet_bulb', 'air_temp', 'climate_vars', 'co2']
        missing_data = [dataset for dataset in required_datasets if dataset not in data_dict or data_dict[dataset] is None]
        if missing_data:
            logger.error(f"Missing required datasets: {missing_data}")
            logger.error("Cannot proceed with preprocessing")
            return
            
        # Process and merge all datasets
        logger.info("Processing and merging datasets")
        # Use the functions from data_loader.py
        wet_bulb_monthly = clean_wet_bulb_data(data_dict['wet_bulb'])
        climate_vars_clean = clean_climate_vars_data(data_dict['climate_vars'])
        air_temp_clean = clean_air_temp_data(data_dict['air_temp'])
        
        # Process greenhouse gas data if available
        co2_clean = None
        ch4_clean = None
        n2o_clean = None
        sf6_clean = None
        
        if 'co2' in data_dict and data_dict['co2'] is not None:
            co2_clean = clean_greenhouse_gas_data(data_dict['co2'], 'co2')
        if 'ch4' in data_dict and data_dict['ch4'] is not None:
            ch4_clean = clean_greenhouse_gas_data(data_dict['ch4'], 'ch4')
        if 'n2o' in data_dict and data_dict['n2o'] is not None:
            n2o_clean = clean_greenhouse_gas_data(data_dict['n2o'], 'n2o')
        if 'sf6' in data_dict and data_dict['sf6'] is not None:
            sf6_clean = clean_greenhouse_gas_data(data_dict['sf6'], 'sf6')
            
        # Merge datasets manually since we're processing them individually
        logger.info("Merging processed datasets")
        final_df = wet_bulb_monthly.merge(air_temp_clean, on='month', how='left')
        final_df = final_df.merge(climate_vars_clean, on='month', how='left')
        
        if co2_clean is not None:
            final_df = final_df.merge(co2_clean, on='month', how='left')
        if ch4_clean is not None:
            final_df = final_df.merge(ch4_clean, on='month', how='left')
        if n2o_clean is not None:
            final_df = final_df.merge(n2o_clean, on='month', how='left')
        if sf6_clean is not None:
            final_df = final_df.merge(sf6_clean, on='month', how='left')
            
        # Set month as index for time series analysis
        final_df.set_index('month', inplace=True)
        
        # Save processed data
        output_path = os.path.join(processed_data_path, 'final_dataset.csv')
        final_df.to_csv(output_path)
        logger.info(f"Processed data saved to {output_path}")
        
        # Save a data description file
        logger.info("Generating data description")
        data_description = f"""
# Wet Bulb Temperature Dataset Description

## Overview
- Total records: {final_df.shape[0]}
- Date range: {final_df.index.min().strftime('%b %Y')} to {final_df.index.max().strftime('%b %Y')}
- Number of variables: {final_df.shape[1]}

## Variables
{pd.DataFrame({'Column': final_df.columns, 'Non-Null Count': final_df.count().values}).to_string()}

## Summary Statistics
{final_df.describe().to_string()}
"""
        
        with open(os.path.join(processed_data_path, 'dataset_description.md'), 'w') as f:
            f.write(data_description)
        logger.info("Data description saved")
        
        logger.info("Data preprocessing completed successfully")
        return final_df
        
    except Exception as e:
        logger.exception(f"Error during data preprocessing: {e}")
        return None

if __name__ == "__main__":
    main()
