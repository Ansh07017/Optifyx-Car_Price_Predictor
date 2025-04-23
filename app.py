import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
import os

# Import custom modules
from modules.data_loader import load_data, show_column_descriptions
from modules.data_preprocessing import preprocess_data
from modules.eda import perform_eda
from modules.model_training import train_models
from modules.model_evaluation import evaluate_models
from modules.prediction import predict_price
from modules.database import create_tables

# Set page configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set the title of the app
st.title("🚗 Car Price Prediction Application")
st.subheader("Predict car prices using machine learning models")

# Add custom CSS for the sidebar
st.markdown("""
<style>
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #0068c9;
    }
    .menu-item {
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Create sidebar for navigation with icons
st.sidebar.markdown('<div class="sidebar-title">🚗 Car Price Predictor</div>', unsafe_allow_html=True)

# Create menu options with icons
menu_options = {
    "Price Prediction": "💰",
    "Data Exploration": "📊",
    "Model Training": "⚙️",
    "Data Overview": "📋"
}

# Use radio buttons for page selection
page = st.sidebar.radio(
    "Navigation",
    list(menu_options.keys()),
    format_func=lambda x: f"{menu_options[x]} {x}"
)

# Load data
try:
    df = load_data()
    
    if page == "Price Prediction":
        st.write("# 🚗 Car Price Prediction")
        st.write("""
        Instantly estimate the selling price of your car based on its specifications.
        Simply adjust the parameters below and get an accurate price prediction.
        """)
        
        # Create a cleaner layout using columns
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.image("https://img.icons8.com/fluency/96/000000/car.png", width=80)
            
            # Show column descriptions in an expander
            with st.expander("ℹ️ **What Do These Features Mean?**", expanded=False):
                show_column_descriptions()
                
            st.write("#### Why Trust This Model?")
            st.write("""
            - Trained on real market data
            - Uses multiple advanced ML algorithms
            - Considers all important car factors
            """)
        
        with col1:
            # Preprocess data
            X, y, feature_names, categorical_features, numerical_features = preprocess_data(df)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train models
            models = train_models(X_train, y_train, feature_names)
            
            # Predict price
            predict_price(models, df, categorical_features, numerical_features, feature_names)
    
    elif page == "Data Exploration":
        perform_eda(df)
        
    elif page == "Model Training":
        # Preprocess data
        X, y, feature_names, categorical_features, numerical_features = preprocess_data(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train models
        models = train_models(X_train, y_train, feature_names)
        
        # Evaluate models
        evaluate_models(models, X_test, y_test, X_train, y_train, feature_names)
        
    elif page == "Data Overview":
        st.write("## Welcome to the Car Price Prediction App")
        st.write("""
        This application allows you to explore car data, train machine learning models, 
        and predict the selling price of cars based on various features.
        
        ### Dataset Overview
        """)
        
        # Display basic dataset information
        st.write("#### Data sample:")
        st.dataframe(df.head())
        
        # Dataset statistics and info
        st.write("#### Dataset Information:")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Number of records: {df.shape[0]}")
            st.write(f"Number of features: {df.shape[1]}")
        
        with col2:
            st.write(f"Numeric features: {len(df.select_dtypes(include=['number']).columns)}")
            st.write(f"Categorical features: {len(df.select_dtypes(include=['object']).columns)}")
        
        # Display missing values information
        st.write("#### Missing Values:")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            st.write(missing_data[missing_data > 0])
        else:
            st.write("No missing values found in the dataset.")
        
        # Display basic statistics for numerical columns
        st.write("#### Numerical Features Statistics:")
        st.dataframe(df.describe())
        
        # Distribution of price
        st.write("#### Distribution of Car Selling Prices:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Selling_Price'], kde=True, ax=ax)
        plt.xlabel('Selling Price (in lakhs)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Car Selling Prices')
        st.pyplot(fig)
        
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.error("Please check if the dataset file exists and is correctly formatted.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This app was built using Streamlit to predict car prices using various machine learning models."
)
