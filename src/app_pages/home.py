"""
Home Page for Streamlit Dashboard

This module provides the main landing page for the wet bulb temperature analysis
dashboard. It displays key insights, summary visualizations, and overview statistics
to give users a quick understanding of the dataset and main findings.

Features include:
- Project overview and introduction
- Key statistics and metrics display
- Summary time series visualizations
- Monthly temperature patterns
- Correlation analysis overview
- Dataset summary and statistics

The page serves as the entry point for the application and provides navigation
context for users to explore more detailed analysis in other sections.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.visualization.exploratory import plot_time_series


def pick_first_column(df, *candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


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

    st.markdown(
        """
        Track how wet bulb temperature in Singapore moves with air temperature, humidity,
        rainfall, and greenhouse-gas proxies. Start here for the main trend, then move into
        **Correlation Analysis** or **Regression Modeling** when you want the stronger
        analytical breakdown.
        """
    )

    wet_bulb_col = pick_first_column(df, 'avg_wet_bulb', 'mean_wet_bulb_temperature')
    air_temp_col = pick_first_column(df, 'mean_air_temp', 'mean_surface_airtemp')
    humidity_col = pick_first_column(df, 'mean_relative_humidity', 'avg_relative_humidity')
    co2_col = pick_first_column(df, 'average_co2', 'average_co2_ppm')

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Records", f"{len(df):,}")
    with metric_col2:
        if hasattr(df.index, "min") and len(df.index) > 0:
            st.metric("Date range", f"{df.index.min().year}-{df.index.max().year}")
        else:
            st.metric("Date range", "N/A")
    with metric_col3:
        if wet_bulb_col is not None:
            st.metric("Mean wet bulb", f"{df[wet_bulb_col].mean():.2f} °C")
        else:
            st.metric("Mean wet bulb", "N/A")
    with metric_col4:
        if wet_bulb_col is not None and len(df.index) > 0:
            monthly_avg = df.groupby(df.index.month)[wet_bulb_col].mean()
            hottest_month = pd.to_datetime(monthly_avg.idxmax(), format='%m').strftime('%b')
            st.metric("Warmest month", hottest_month)
        else:
            st.metric("Warmest month", "N/A")
    
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
        co2_col = pick_first_column(df, 'average_co2', 'average_co2_ppm')
        if co2_col:
            st.subheader("CO2 Concentration")
            fig = plot_time_series(df, co2_col, rolling_window=12)
            st.pyplot(fig)
    
    # Show summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())
    
    # Display additional insights
    st.subheader("Key Insights")

    has_required_columns = wet_bulb_col is not None and air_temp_col is not None and humidity_col is not None
    
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
        corr_cols = [wet_bulb_col, air_temp_col, humidity_col]
        if co2_col is not None:
            corr_cols.append(co2_col)
            
        corr_matrix = df[corr_cols].corr()
        
        st.write("**Correlation Between Key Variables:**")
        st.dataframe(corr_matrix.style.format("{:.4f}").background_gradient(cmap='coolwarm'))
    else:
        st.info("Some required columns are missing to generate insights. Please check your data.")
