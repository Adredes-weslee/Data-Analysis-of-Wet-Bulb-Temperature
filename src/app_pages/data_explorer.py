"""
Data explorer page for the Streamlit dashboard
"""
import streamlit as st
import pandas as pd

def show(df):
    """
    Display the data explorer page
    
    Creates an interactive data exploration interface allowing users to examine
    dataset properties, filter data, and analyze specific columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the analysis data to be explored
        
    Returns
    -------
    None
        This function directly renders content to the Streamlit app
    """
    st.title("Data Explorer")
    
    st.subheader("Dataset Explorer")
    
    # Show dataset shape and column information
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    with col2:
        st.write(f"**Date Range:** {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    
    # Column selector for exploration
    columns_to_display = st.multiselect(
        "Select columns to display", 
        options=df.columns.tolist(),
        default=df.columns[:5].tolist()
    )
    
    if columns_to_display:
        st.dataframe(df[columns_to_display])
    
    # Show descriptive statistics
    if st.checkbox("Show Descriptive Statistics"):
        st.dataframe(df[columns_to_display].describe())
        
    # Show data types
    if st.checkbox("Show Data Types"):
        st.dataframe(pd.DataFrame(df[columns_to_display].dtypes, columns=['Data Type']))
    
    # Missing values analysis
    if st.checkbox("Show Missing Values Analysis"):
        missing_data = df[columns_to_display].isna().sum().to_frame('Missing Count')
        missing_data['Missing Percentage'] = (df[columns_to_display].isna().sum() / len(df) * 100).round(2)
        st.dataframe(missing_data)
    
    # Data filtering options
    st.subheader("Data Filtering")
    
    # Time range filter
    date_range = st.slider(
        "Select date range",
        min_value=df.index.min().to_pydatetime(),
        max_value=df.index.max().to_pydatetime(),
        value=(df.index.min().to_pydatetime(), df.index.max().to_pydatetime()),
        format="YYYY-MM"
    )
    
    # Filter data based on time range
    filtered_df = df.loc[date_range[0]:date_range[1]]
    
    # Column value filter
    if columns_to_display:
        column_to_filter = st.selectbox(
            "Select a column to filter by value",
            options=columns_to_display
        )
        
        if column_to_filter in df.select_dtypes(include=['number']).columns:
            min_val = float(filtered_df[column_to_filter].min())
            max_val = float(filtered_df[column_to_filter].max())
            
            value_range = st.slider(
                f"Filter by {column_to_filter} value",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val)
            )
            
            # Apply filter
            filtered_df = filtered_df[(filtered_df[column_to_filter] >= value_range[0]) & 
                                      (filtered_df[column_to_filter] <= value_range[1])]
    
    # Show filtered data
    st.subheader("Filtered Data")
    st.write(f"Showing {len(filtered_df)} records after filtering")
    st.dataframe(filtered_df[columns_to_display])
