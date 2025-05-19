"""
Regression modeling page for the Streamlit dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime

from src.models.regression import (
    preprocess_for_regression, build_linear_regression_model,
    evaluate_regression_model, plot_actual_vs_predicted,
    plot_residuals, plot_feature_importance
)
from src.features.feature_engineering import (
    create_temporal_features, create_interaction_features,
    create_lag_features, create_rolling_features
)

def show(df):
    """
    Display the regression modeling page
    
    Creates an interactive interface for building, training, and evaluating 
    regression models for wet bulb temperature prediction. Allows users to 
    select target variables, features, feature engineering options, and 
    view detailed model evaluation metrics and visualizations.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the analysis data
        
    Returns
    -------
    None
        This function directly renders content to the Streamlit app
    """
    st.title("Regression Modeling")
    
    st.write("""
    Build and evaluate regression models to predict wet bulb temperature based on different variables.
    Select a target variable, features, and various modeling options to understand the relationships.
    """)
    
    # Target variable selection
    target_var = st.selectbox(
        "Select target variable",
        options=df.select_dtypes(include=['number']).columns.tolist(),
        index=0 if 'avg_wet_bulb' in df.columns else 0
    )
    
    # Feature selection
    feature_vars = st.multiselect(
        "Select features for the model",
        options=[col for col in df.select_dtypes(include=['number']).columns if col != target_var],
        default=[col for col in df.select_dtypes(include=['number']).columns[:3] if col != target_var]
    )
    
    if feature_vars and len(feature_vars) > 0:
        # Model configuration options
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Train/test split ratio
            test_size = st.slider("Test set size (%)", min_value=10, max_value=50, value=20) / 100
        
        with col2:
            # Random state for reproducibility
            random_state = st.number_input("Random seed", min_value=0, max_value=1000, value=42)
        
        # Feature engineering options
        with st.expander("Feature Engineering Options"):
            use_temporal = st.checkbox("Add temporal features", value=False)
            use_interactions = st.checkbox("Add interaction features", value=False)
            use_lags = st.checkbox("Add lag features", value=False)
            use_rolling = st.checkbox("Add rolling window features", value=False)
            
            if any([use_temporal, use_interactions, use_lags, use_rolling]):
                # Apply selected feature engineering
                model_df = df.copy()
                
                if use_temporal:
                    model_df = create_temporal_features(model_df)
                
                if use_interactions:
                    model_df = create_interaction_features(model_df)
                
                if use_lags:
                    lag_columns = st.multiselect(
                        "Select columns for lag features",
                        options=feature_vars,
                        default=[]
                    )
                    if lag_columns:
                        lag_periods = st.multiselect(
                            "Select lag periods (months)",
                            options=[1, 2, 3, 6, 12],
                            default=[1, 3]
                        )
                        model_df = create_lag_features(model_df, lag_columns, lags=lag_periods)
                
                if use_rolling:
                    rolling_columns = st.multiselect(
                        "Select columns for rolling window features",
                        options=feature_vars,
                        default=[]
                    )
                    if rolling_columns:
                        window_sizes = st.multiselect(
                            "Select rolling window sizes (months)",
                            options=[2, 3, 6, 12],
                            default=[3, 6]
                        )
                        model_df = create_rolling_features(model_df, rolling_columns, windows=window_sizes)
                
                # Update feature list with new features
                numeric_cols = model_df.select_dtypes(include=['number']).columns
                new_features = [col for col in numeric_cols if col not in df.columns and col != target_var]
                
                if new_features:
                    st.write("**New features created:**")
                    st.write(", ".join(new_features))
                    
                    # Add new features to selection
                    additional_features = st.multiselect(
                        "Select additional features to include in the model",
                        options=new_features,
                        default=[]
                    )
                    selected_features = feature_vars + additional_features
                else:
                    selected_features = feature_vars
            else:
                model_df = df
                selected_features = feature_vars
          # Advanced options
        with st.expander("Advanced Options"):
            handle_missing = st.radio(
                "How to handle missing values",
                options=["Drop rows with missing values", "Fill with mean", "Fill with median", "Fill with zero"],
                index=0
            )
        
        # Train model
        if st.button("Train Regression Model"):
            if len(selected_features) == 0:
                st.error("Please select at least one feature for the model.")
                return
                
            with st.spinner("Training model..."):
                try:
                    # Make a copy to avoid modifying original data
                    model_df_clean = model_df.copy()
                    
                    # First convert any non-numeric data to numeric
                    for col in selected_features + [target_var]:
                        if col in model_df_clean.columns and model_df_clean[col].dtype == 'object':
                            model_df_clean[col] = pd.to_numeric(model_df_clean[col], errors='coerce')
                    
                    # Handle missing values based on user selection
                    if handle_missing == "Fill with mean":
                        model_df_clean = model_df_clean.fillna(model_df_clean.mean())
                    elif handle_missing == "Fill with median":
                        model_df_clean = model_df_clean.fillna(model_df_clean.median()) 
                    elif handle_missing == "Fill with zero":
                        model_df_clean = model_df_clean.fillna(0)
                    # else: Default is to drop rows with missing values in preprocess_for_regression
                    
                    # Check that we have enough valid data
                    relevant_cols = selected_features + [target_var]
                    valid_data_count = model_df_clean[relevant_cols].dropna().shape[0]
                    
                    if valid_data_count < 10:  # Minimum sample size
                        st.error(f"Not enough valid data points ({valid_data_count}) after handling missing values. Try a different method.")
                        return
                        
                    st.info(f"Using {valid_data_count} valid data points for model training")
                    model_df = model_df_clean
                    
                except Exception as e:
                    st.error(f"Error preprocessing data: {e}")
                    return
                  # Preprocess data
                try:
                    st.info("Preprocessing data and training model...")
                    X_train, X_test, y_train, y_test, scaler = preprocess_for_regression(
                        model_df, target_var, selected_features, test_size=test_size, random_state=int(random_state)
                    )
                    
                    # Build model
                    model = build_linear_regression_model(X_train, y_train)
                    
                    # Evaluate model
                    results = evaluate_regression_model(
                        model, X_train, X_test, y_train, y_test, selected_features
                    )
                    
                    # Display results
                    st.subheader("Model Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Training Results:**")
                        st.write(f"R²: {results['train_r2']:.4f}")
                        st.write(f"RMSE: {results['train_rmse']:.4f}")
                        st.write(f"MAE: {results['train_mae']:.4f}")
                        
                    with col2:
                        st.write("**Testing Results:**")
                        st.write(f"R²: {results['test_r2']:.4f}")
                        st.write(f"RMSE: {results['test_rmse']:.4f}")
                        st.write(f"MAE: {results['test_mae']:.4f}")
                    
                    # Predictions and residuals
                    y_test_pred = model.predict(X_test)
                    
                    # Create visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Actual vs Predicted Values:**")
                        fig = plot_actual_vs_predicted(y_test, y_test_pred)
                        st.pyplot(fig)
                    
                    with col2:
                        st.write("**Residual Plot:**")
                        fig = plot_residuals(y_test, y_test_pred)
                        st.pyplot(fig)
                    
                    # Feature importance
                    st.write("**Feature Importance:**")
                    fig = plot_feature_importance(model, selected_features)
                    st.pyplot(fig)
                    # Model equation
                    st.subheader("Model Equation")
                    
                    equation = f"{target_var} = "
                    for i, feature in enumerate(selected_features):
                        coef = results['coefficients'][feature]
                        sign = "+" if coef >= 0 else "-"
                        if i == 0:
                            equation += f"{abs(coef):.4f} × {feature} "
                        else:
                            equation += f"{sign} {abs(coef):.4f} × {feature} "                    
                    if 'intercept' in results:
                        sign = "+" if results['intercept'] >= 0 else "-"
                        equation += f"{sign} {abs(results['intercept']):.4f}"
                    
                    st.write(equation)
                      # Model predictions section
                    st.subheader("Download Results")
                    
                    # Create predictions dataframe with proper index
                    predictions_df = pd.DataFrame({
                        'actual': y_test,
                        'predicted': y_test_pred,
                        'residual': y_test - y_test_pred
                    })
                    
                    # Download section with centered button
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        # Encode CSV data for download
                        csv_data = predictions_df.to_csv().encode('utf-8')
                        
                        # Create a descriptive filename
                        safe_target_name = ''.join(c if c.isalnum() else '_' for c in target_var)
                        current_date = datetime.now().strftime("%Y%m%d")
                        filename = f"{safe_target_name}_predictions_{current_date}.csv"
                        
                        # Show the download button - made more prominent
                        st.download_button(
                            label="📥 Download CSV to Computer",
                            data=csv_data,
                            file_name=filename,
                            mime="text/csv",
                            key="download_predictions",
                            help="Download the predictions directly to your computer",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Error in model training: {e}")
    else:
        st.info("Please select at least one feature for the model")
