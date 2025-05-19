"""
Time series analysis page for the Streamlit dashboard
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.visualization.exploratory import plot_time_series, plot_monthly_patterns

def show(df):
    """
    Display the time series analysis page
    
    Creates an interactive interface for analyzing time series data with options
    for date range selection, rolling averages, monthly patterns, and year-over-year
    comparisons. Provides visualizations to help users understand temporal trends 
    and seasonal patterns in the selected variables.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the analysis data with a datetime index
        
    Returns
    -------
    None
        This function directly renders content to the Streamlit app
    """
    st.title("Time Series Analysis")
    
    # Variable selection
    var_to_plot = st.selectbox(
        "Select variable to analyze", 
        options=df.columns.tolist(),
        index=0
    )
    
    # Time range selector
    date_range = st.slider(
        "Select date range",
        min_value=df.index.min().to_pydatetime(),
        max_value=df.index.max().to_pydatetime(),
        value=(df.index.min().to_pydatetime(), df.index.max().to_pydatetime()),
        format="YYYY-MM"
    )
    
    # Filter data based on selection
    filtered_df = df.loc[date_range[0]:date_range[1]]
    
    # Time series plot options
    col1, col2 = st.columns(2)
    with col1:
        show_rolling = st.checkbox("Show rolling average", value=True)
    with col2:
        rolling_window = st.slider("Rolling window size", min_value=1, max_value=24, value=12) if show_rolling else None
    
    # Create and display plot
    fig = plot_time_series(
        filtered_df, var_to_plot, 
        title=f'Time Series of {var_to_plot}',
        ylabel=var_to_plot,
        rolling_window=rolling_window
    )
    st.pyplot(fig)
    
    # Monthly patterns
    if st.checkbox("Show Monthly Patterns"):
        fig = plot_monthly_patterns(filtered_df, var_to_plot)
        st.pyplot(fig)
          # Year-over-year comparison
    if st.checkbox("Show Year-over-Year Comparison"):
        """
        Display year-over-year comparison of the selected variable
        
        Groups the filtered data by year and month, and creates a pivot table
        to calculate the mean values for each month across different years.
        Plots the year-over-year comparison to visualize trends and changes
        in the selected variable over time.
        
        Returns
        -------
        None
        """
        # Create a new dataframe without the datetime index to avoid the ambiguity
        yearly_data = pd.DataFrame()
        
        # Extract year and month from the index explicitly
        yearly_data['year'] = filtered_df.index.year
        yearly_data['month_num'] = filtered_df.index.month
        yearly_data[var_to_plot] = filtered_df[var_to_plot].values
        
        # Pivot table for year-over-year comparison
        pivot_data = yearly_data.pivot_table(
            index='month_num', 
            columns='year', 
            values=var_to_plot, 
            aggfunc='mean'
        )
        
        # Plot year-over-year comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for year in pivot_data.columns:
            ax.plot(pivot_data.index, pivot_data[year], marker='o', label=str(year))
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel(var_to_plot, fontsize=12)
        ax.set_title(f'Year-over-Year Comparison of {var_to_plot}', fontsize=14)
        
        # Set proper tick positions and labels to avoid categorical warnings
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        # Make sure the ticks are at the actual month number positions (1-12)
        ax.set_xticks(list(range(1, 13)))
        ax.set_xticklabels(month_labels)
        
        ax.legend(title='Year')
        ax.grid(alpha=0.3)
        
        st.pyplot(fig)
    
    # Statistical decomposition
    if st.checkbox("Show Time Series Decomposition"):
        """
        Perform statistical decomposition of the time series
        
        Decomposes the selected variable into its trend, seasonal, and residual
        components using the seasonal_decompose function from statsmodels. 
        Requires at least 24 data points for monthly data decomposition.
        
        Returns
        -------
        None
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Ensure we have enough data for decomposition
            if len(filtered_df) < 24:  # Need at least 2 years of monthly data
                st.warning("Not enough data for decomposition. Select a wider date range.")
                return
                
            # Decompose the time series
            decomposition = seasonal_decompose(
                filtered_df[var_to_plot], 
                model='additive', 
                period=12  # For monthly data
            )
            
            # Create figure with subplots
            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            
            # Original data
            axes[0].plot(decomposition.observed)
            axes[0].set_title('Original Time Series')
            axes[0].grid(alpha=0.3)
            
            # Trend component
            axes[1].plot(decomposition.trend)
            axes[1].set_title('Trend Component')
            axes[1].grid(alpha=0.3)
            
            # Seasonal component
            axes[2].plot(decomposition.seasonal)
            axes[2].set_title('Seasonal Component')
            axes[2].grid(alpha=0.3)
            
            # Residual component
            axes[3].plot(decomposition.resid)
            axes[3].set_title('Residual Component')
            axes[3].grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error performing time series decomposition: {e}")
            st.info("Try installing statsmodels with: pip install statsmodels")
