"""
Main Streamlit application for wet bulb temperature analysis

This module serves as the entry point for the Streamlit web application that
provides an interactive interface for exploring and analyzing wet bulb temperature
data and its relationships with climate variables and greenhouse gases.

The application includes multiple pages for different types of analysis:
- Home: Overview and key insights
- Data Explorer: Interactive data examination and filtering
- Time Series Analysis: Temporal trends and patterns
- Correlation Analysis: Relationships between variables
- Regression Modeling: Predictive modeling of wet bulb temperature
- About: Project information and methodology
"""
import os
import sys
import streamlit as st
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set page configuration
st.set_page_config(
    page_title="Wet Bulb Temperature Analysis",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_processed_data(data_path):
    """
    Load processed data with caching
    
    Loads the final processed dataset for analysis, with Streamlit's caching mechanism
    to improve performance. If the processed dataset is not found, it attempts to
    prepare it from raw data files.
    
    Parameters
    ----------
    data_path : str
        Path to the data folder containing processed and raw subdirectories
        
    Returns
    -------
    pandas.DataFrame
        Processed data with datetime index and all analysis variables
        
    Raises
    ------
    Exception
        If data loading fails, the exception is logged and propagated
        
    Side Effects
    -----------
    If processed data doesn't exist at data_path/processed/final_dataset.csv, 
    this function will generate it using the raw data and the data preparation pipeline.
    The processed data is not explicitly saved to disk here but is cached in memory
    by Streamlit.
    
    Notes
    -----
    This function will first look for a processed dataset at data_path/processed/final_dataset.csv
    If not found, it will fall back to processing raw data in data_path/raw/
    """
    try:
        # Check for processed data first
        processed_path = os.path.join(data_path, 'processed', 'final_dataset.csv')
        if os.path.exists(processed_path):
            logger.info(f"Loading processed data from {processed_path}")
            return pd.read_csv(processed_path, parse_dates=['month'], index_col='month')
        
        # If not available, prepare from raw data
        logger.info("Processed data not found, preparing from raw data")
        from src.data_processing.data_loader import prepare_data_for_analysis
        df = prepare_data_for_analysis(data_path)
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Error loading data: {e}")
        return None

def main():
    """
    Main dashboard application
    
    This function initializes the Streamlit application, sets up the sidebar
    navigation, loads the data, and displays the selected page based on user input.
    It handles the dynamic rendering of different analysis pages and manages the
    application flow.
    
    The dashboard includes multiple pages:
    - Home: Overview with key statistics and visualizations
    - Data Explorer: Tools for exploring and filtering the dataset
    - Time Series Analysis: Temporal trends and patterns visualization
    - Correlation Analysis: Correlation matrices and scatter plots
    - Regression Modeling: Building and evaluating predictive models
    - About: Project information and methodology
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
        This function directly renders the Streamlit interface
        
    Side Effects
    -----------
    - Loads data using the cached load_processed_data function
    - Renders a sidebar with navigation options
    - Renders the selected page content in the main area
    - May display error messages if data loading fails
    
    Raises
    ------
    Exception
        Handled internally - displays error message to user if data loading fails
    """
    # Sidebar header
    st.sidebar.title("Wet Bulb Temperature Analysis")
    st.sidebar.image("https://i.imgur.com/1ZcRyrc.png", width=100)
    
    # Data path
    data_path = os.path.join(str(Path(__file__).parent.parent), 'data')
    # Load data
    st.sidebar.text("Loading data...")
    df = load_processed_data(data_path)
    
    if df is None:
        st.error("Failed to load data. Please check the data directory and logs.")
        return
    
    st.sidebar.success(f"Data loaded: {df.shape[0]} records")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        options=["Home", "Data Explorer", "Time Series Analysis", 
                 "Correlation Analysis", "Regression Modeling", "About"]
    )
    
    # Import and display selected page
    try:
        if page == "Home":
            from src.app_pages.home import show
            show(df)
        elif page == "Data Explorer":
            from src.app_pages.data_explorer import show
            show(df)
        elif page == "Time Series Analysis":
            from src.app_pages.time_series import show
            show(df)
        elif page == "Correlation Analysis":
            from src.app_pages.correlation import show
            show(df)
        elif page == "Regression Modeling":
            from src.app_pages.regression import show
            show(df)
        elif page == "About":
            from src.app_pages.about import show
            show(df)
    except Exception as e:
        st.error(f"Error displaying page {page}: {e}")
        logger.error(f"Error displaying page {page}: {e}")
    
    # Add footer with project info
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        Created with ❤️ using Streamlit
        
        Data sources: data.gov.sg, NOAA
        """
    )


if __name__ == "__main__":
    main()