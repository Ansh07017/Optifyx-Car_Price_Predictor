import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df, show_details=True):
    """
    Preprocess the car dataset for model training
    
    Args:
        df (pandas.DataFrame): The raw car dataset
        show_details (bool): Whether to display preprocessing details in the UI
        
    Returns:
        tuple: Preprocessed features (X), target (y), feature names, categorical and numerical features
    """
    # Make a copy of the dataframe to avoid modifying the original
    data = df.copy()
    
    # Display preprocessing steps if show_details is True
    if show_details:
        st.write("## Data Preprocessing")
    
    # Remove duplicates if any
    initial_rows = data.shape[0]
    data.drop_duplicates(inplace=True)
    removed_dups = initial_rows - data.shape[0]
    
    if show_details:
        st.write(f"Removed {removed_dups} duplicate rows")
    
    # Handle missing values if any
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        if show_details:
            st.write("Handling missing values:")
            st.write(missing_values[missing_values > 0])
        
        # For numerical columns, fill with median
        num_cols = data.select_dtypes(include=['number']).columns
        for col in num_cols:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        cat_cols = data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].mode()[0], inplace=True)
    
    # Create a new feature for car age (current year - car year)
    current_year = 2025  # Current year
    if 'Year' in data.columns:
        # Create the Car_Age column if it doesn't exist
        if 'Car_Age' not in data.columns:
            data['Car_Age'] = current_year - data['Year']
            if show_details:
                st.write("Created new feature: Car_Age = Current Year - Year")
    
    # Ensure column names are consistently formatted
    # Based on the dataset image, columns might have different formatting
    if 'Selling_Price' in data.columns:
        target_col = 'Selling_Price'
    elif 'Selling_Pri' in data.columns:  # From the image you shared
        target_col = 'Selling_Pri'
        # Rename for consistency
        data = data.rename(columns={'Selling_Pri': 'Selling_Price'})
    else:
        # Default target column if none of the expected columns are found
        st.error("Could not find the target column 'Selling_Price' or 'Selling_Pri' in the dataset.")
        target_col = 'Selling_Price'  # Default value to avoid errors
    
    # Separate features and target
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['number']).columns.tolist()
    
    # Make sure Car_Age is in numerical_features if it exists
    if 'Car_Age' in X.columns and 'Car_Age' not in numerical_features:
        numerical_features.append('Car_Age')
    
    # Display feature information only if show_details is True
    if show_details:
        st.write(f"Target variable: {target_col}")
        st.write(f"Categorical features: {categorical_features}")
        st.write(f"Numerical features: {numerical_features}")
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )
    
    # Transform the data
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after transformation
    cat_encoder = preprocessor.named_transformers_['cat']
    if len(categorical_features) > 0:
        encoded_cat_features = list(cat_encoder.get_feature_names_out(categorical_features))
    else:
        encoded_cat_features = []
        
    feature_names = numerical_features + encoded_cat_features
    
    if show_details:
        st.write(f"After preprocessing, we have {X_processed.shape[1]} features")
    
    return X_processed, y, feature_names, categorical_features, numerical_features
