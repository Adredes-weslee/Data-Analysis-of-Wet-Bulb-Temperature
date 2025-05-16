"""
Correlation analysis page for the Streamlit dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.visualization.exploratory import plot_correlation_matrix, plot_scatter_with_regression

def show(df):
    """
    Display the correlation analysis page
    
    Creates an interactive interface for analyzing correlations between variables
    in the dataset. Provides correlation matrix visualization, tabular correlation data,
    scatter plots with regression lines, and optional partial correlation analysis.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the analysis data
        
    Returns
    -------
    None
        This function directly renders content to the Streamlit app
    """
    st.title("Correlation Analysis")
    
    # Variable selection for correlation
    cols_to_correlate = st.multiselect(
        "Select variables for correlation analysis",
        options=df.columns.tolist(),
        default=df.select_dtypes(include=['number']).columns[:5].tolist()
    )
    
    if cols_to_correlate and len(cols_to_correlate) > 1:
        # Correlation matrix
        fig = plot_correlation_matrix(df[cols_to_correlate])
        st.pyplot(fig)
        
        # Display correlation table with formatting
        corr_df = df[cols_to_correlate].corr()
        st.write("Correlation Table:")
        st.dataframe(corr_df.style.format("{:.4f}").background_gradient(cmap='coolwarm'))
        
        # Scatter plot with regression
        st.subheader("Scatter Plot with Regression Line")
        
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Select X variable", options=cols_to_correlate, index=0)
        with col2:
            y_var = st.selectbox("Select Y variable", options=cols_to_correlate, index=min(1, len(cols_to_correlate)-1))
        
        fig = plot_scatter_with_regression(
            df, x_var, y_var,
            title=f'{y_var} vs {x_var}'
        )
        st.pyplot(fig)
        
        # Add partial correlation analysis
        if st.checkbox("Show Partial Correlation Analysis"):
            st.subheader("Partial Correlation Analysis")
            st.write("""
            Partial correlation measures the relationship between two variables while controlling 
            for the effect of one or more additional variables.
            """)
            
            if len(cols_to_correlate) >= 3:
                control_vars = st.multiselect(
                    "Select variables to control for",
                    options=[col for col in cols_to_correlate if col not in [x_var, y_var]],
                    default=[]
                )
                
                if control_vars:
                    try:
                        from scipy.stats import pearsonr
                        
                        # Calculate residuals after controlling for the selected variables
                        X = df[control_vars]
                        
                        # Add constant for regression
                        from statsmodels.api import add_constant
                        X = add_constant(X)
                        
                        # Calculate residuals for x_var
                        from statsmodels.regression.linear_model import OLS
                        model_x = OLS(df[x_var], X).fit()
                        residuals_x = model_x.resid
                        
                        # Calculate residuals for y_var
                        model_y = OLS(df[y_var], X).fit()
                        residuals_y = model_y.resid
                        
                        # Calculate correlation between residuals
                        partial_corr, p_value = pearsonr(residuals_x, residuals_y)
                        
                        st.write(f"**Partial Correlation:** {partial_corr:.4f}")
                        st.write(f"**P-value:** {p_value:.4f}")
                        
                        # Plot residuals
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(residuals_x, residuals_y, alpha=0.6)
                        ax.set_xlabel(f'Residuals of {x_var}', fontsize=12)
                        ax.set_ylabel(f'Residuals of {y_var}', fontsize=12)
                        ax.set_title(f'Partial Correlation: {x_var} vs {y_var} (controlling for {", ".join(control_vars)})', 
                                    fontsize=14)
                        ax.grid(alpha=0.3)
                        
                        # Add regression line for residuals
                        from scipy.stats import linregress
                        slope, intercept, _, _, _ = linregress(residuals_x, residuals_y)
                        x_vals = np.array([min(residuals_x), max(residuals_x)])
                        y_vals = intercept + slope * x_vals
                        ax.plot(x_vals, y_vals, 'r-', linewidth=2)
                        
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Error performing partial correlation analysis: {e}")
                        st.info("Try installing scipy and statsmodels with: pip install scipy statsmodels")
            else:
                st.info("You need at least 3 variables to perform partial correlation analysis.")
    else:
        st.info("Please select at least 2 variables for correlation analysis.")
