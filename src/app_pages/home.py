"""
Home page for the Streamlit dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.visualization.exploratory import plot_time_series

def show(df):
    """
    Display the home page with overview information
    
    Creates a landing page with summary statistics and visualizations to
    give users an overview of the dataset and key insights.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the analysis data
        
    Returns
    -------
    None
        This function directly renders content to the Streamlit app
    """
    st.title("Wet Bulb Temperature Analysis Dashboard")
    
    st.write("""
    This dashboard allows you to explore the relationship between wet bulb temperature in Singapore
    and various climate variables and greenhouse gases. Use the navigation menu on the left to
    explore different aspects of the data.
    """)
    
    # Display key statistics and trends
    col1, col2 = st.columns(2)
    
    with col1:
        if 'avg_wet_bulb' in df.columns:
            st.subheader("Wet Bulb Temperature")
            fig = plot_time_series(df, 'avg_wet_bulb', rolling_window=12)
            st.pyplot(fig)
        elif 'mean_wet_bulb_temperature' in df.columns:
            st.subheader("Wet Bulb Temperature")
            fig = plot_time_series(df, 'mean_wet_bulb_temperature', rolling_window=12)
            st.pyplot(fig)
    
    with col2:
        if 'average_co2_ppm' in df.columns:
            st.subheader("CO2 Concentration")
            fig = plot_time_series(df, 'average_co2_ppm', rolling_window=12)
            st.pyplot(fig)
    
    # Show summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())
    
    # Display additional insights
    st.subheader("Key Insights")
    
    # Check if the necessary columns exist before creating insights
    wet_bulb_col = 'avg_wet_bulb' if 'avg_wet_bulb' in df.columns else 'mean_wet_bulb_temperature' if 'mean_wet_bulb_temperature' in df.columns else None
    air_temp_col = 'mean_air_temp' if 'mean_air_temp' in df.columns else 'mean_surface_airtemp' if 'mean_surface_airtemp' in df.columns else None
    
    has_required_columns = wet_bulb_col is not None and air_temp_col is not None and 'mean_relative_humidity' in df.columns
    
    if has_required_columns:
        # Calculate average wet bulb temperature by month
        monthly_avg = df.groupby(df.index.month)[wet_bulb_col].mean().reset_index()
        monthly_avg['month_name'] = pd.to_datetime(monthly_avg['month'], format='%m').dt.strftime('%b')
        
        # Create a bar chart of monthly averages
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(monthly_avg['month_name'], monthly_avg[wet_bulb_col], color='skyblue')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Average Wet Bulb Temperature (°C)', fontsize=12)
        ax.set_title('Average Wet Bulb Temperature by Month', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Calculate correlation between key variables
        corr_cols = [wet_bulb_col, air_temp_col, 'mean_relative_humidity']
        if 'average_co2_ppm' in df.columns:
            corr_cols.append('average_co2_ppm')
            
        corr_matrix = df[corr_cols].corr()
        
        st.write("**Correlation Between Key Variables:**")
        st.dataframe(corr_matrix.style.format("{:.4f}").background_gradient(cmap='coolwarm'))
    else:
        st.info("Some required columns are missing to generate insights. Please check your data.")
