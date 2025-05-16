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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

# Import our data processing functions
from src.data_processing.data_loader import (
    load_data, preprocess_wet_bulb_data, 
    preprocess_ghg_data, preprocess_climate_vars,
    merge_datasets, save_processed_data
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
        
        # Process wet bulb temperature data
        logger.info("Processing wet bulb temperature data")
        wet_bulb_monthly = preprocess_wet_bulb_data(data_dict['wet_bulb'])
        
        # Process climate variables data
        logger.info("Processing climate variables data")
        climate_vars = preprocess_climate_vars(data_dict)
        
        # Process greenhouse gas data
        logger.info("Processing greenhouse gas data")
        ghg_data = preprocess_ghg_data(data_dict)
        
        # Merge all datasets
        logger.info("Merging datasets")
        final_df = merge_datasets(wet_bulb_monthly, climate_vars, ghg_data)
        
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
