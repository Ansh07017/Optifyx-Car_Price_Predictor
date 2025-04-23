import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    """
    Preprocess the car dataset for model training
    
    Args:
        df (pandas.DataFrame): The raw car dataset
        
    Returns:
        tuple: Preprocessed features (X), target (y), feature names, categorical and numerical features
    """
    # Make a copy of the dataframe to avoid modifying the original
    data = df.copy()
    
    # Display preprocessing steps
    st.write("## Data Preprocessing")
    
    # Remove duplicates if any
    initial_rows = data.shape[0]
    data.drop_duplicates(inplace=True)
    removed_dups = initial_rows - data.shape[0]
    st.write(f"Removed {removed_dups} duplicate rows")
    
    # Handle missing values if any
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
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
    current_year = 2023  # You can update this or get dynamically
    if 'Year' in data.columns:
        data['Car_Age'] = current_year - data['Year']
        st.write("Created new feature: Car_Age = Current Year - Year")
    
    # Identify target variable
    target_col = 'Selling_Price'
    
    # Separate features and target
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['number']).columns.tolist()
    
    # Display feature information
    st.write(f"Target variable: {target_col}")
    st.write(f"Categorical features: {categorical_features}")
    st.write(f"Numerical features: {numerical_features}")
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'), categorical_features)
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
    
    st.write(f"After preprocessing, we have {X_processed.shape[1]} features")
    
    return X_processed, y, feature_names, categorical_features, numerical_features
