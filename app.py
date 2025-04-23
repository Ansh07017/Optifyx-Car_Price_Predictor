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

# Add custom CSS for better UI with car background
with open('static/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Custom card styling function
def card(title, content):
    return f"""
    <div class="card">
        <h3>{title}</h3>
        <div>{content}</div>
    </div>
    """

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
        
        # Create a hero section with a stylish intro
        st.markdown(card(
            "💰 Instant Car Valuation",
            """
            <p style="font-size: 16px;">
            Get an accurate estimate of your car's market value in seconds.
            Our AI-powered models analyze multiple factors to deliver precise price predictions.
            </p>
            <div style="display: flex; margin-top: 15px;">
                <div style="background-color: #f8f9fa; border-radius: 5px; padding: 8px 15px; margin-right: 10px;">
                    <span style="font-weight: bold;">✓</span> Trained on real market data
                </div>
                <div style="background-color: #f8f9fa; border-radius: 5px; padding: 8px 15px; margin-right: 10px;">
                    <span style="font-weight: bold;">✓</span> Multiple ML algorithms
                </div>
                <div style="background-color: #f8f9fa; border-radius: 5px; padding: 8px 15px;">
                    <span style="font-weight: bold;">✓</span> Comprehensive analysis
                </div>
            </div>
            """
        ), unsafe_allow_html=True)
        
        # Create a cleaner layout using columns
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown(card(
                "ℹ️ How It Works",
                """
                <div style="text-align: center; margin-bottom: 15px;">
                    <img src="https://img.icons8.com/fluency/96/000000/car.png" width="80" />
                </div>
                <p>Simply adjust the car parameters on the left and click "Predict Price" to get an instant valuation of your vehicle.</p>
                """
            ), unsafe_allow_html=True)
            
            # Show column descriptions in an expander
            with st.expander("📝 **What Do These Features Mean?**", expanded=False):
                show_column_descriptions()
        
        with col1:
            # Get the preprocessed data and models once at the beginning
            # This is done in the background without showing preprocessing details
            with st.spinner("Loading prediction models..."):
                # Preprocess data silently
                X, y, feature_names, categorical_features, numerical_features = preprocess_data(df, show_details=False)
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Train models silently
                models = train_models(X_train, y_train, feature_names, show_details=False)
            
            # Show only the prediction interface to the user
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
        st.write("# Welcome to the Car Price Prediction App")
        
        # Introduction card
        st.markdown(card(
            "🚗 Application Overview",
            """
            This application allows you to explore car data, train machine learning models, 
            and predict the selling price of cars based on various features. You can use the sidebar navigation to:
            <ul>
                <li>💰 <b>Price Prediction</b>: Get instant price estimates for your car</li>
                <li>📊 <b>Data Exploration</b>: Visualize trends and patterns</li>
                <li>⚙️ <b>Model Training</b>: Train and compare machine learning models</li>
                <li>📋 <b>Data Overview</b>: Get familiar with the dataset</li>
            </ul>
            """
        ), unsafe_allow_html=True)
        
        # Display basic dataset information
        st.markdown(card(
            "📋 Dataset Sample",
            ""
        ), unsafe_allow_html=True)
        st.dataframe(df.head())
        
        # Dataset statistics and info in a styled card
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(card(
                "📊 Dataset Statistics",
                f"""
                <ul>
                    <li><b>Number of records:</b> {df.shape[0]}</li>
                    <li><b>Number of features:</b> {df.shape[1]}</li>
                    <li><b>Numeric features:</b> {len(df.select_dtypes(include=['number']).columns)}</li>
                    <li><b>Categorical features:</b> {len(df.select_dtypes(include=['object']).columns)}</li>
                </ul>
                """
            ), unsafe_allow_html=True)
            
            # Missing values information
            missing_data = df.isnull().sum()
            missing_info = ""
            if missing_data.sum() > 0:
                missing_info = "<ul>"
                for col, count in missing_data[missing_data > 0].items():
                    missing_info += f"<li>{col}: {count} missing values</li>"
                missing_info += "</ul>"
            else:
                missing_info = "No missing values found in the dataset."
                
            st.markdown(card(
                "❓ Missing Values",
                missing_info
            ), unsafe_allow_html=True)
        
        with col2:
            # Car image display
            st.markdown(card(
                "🚘 The Car Price Predictor",
                """
                <div style="text-align: center;">
                    <img src="https://img.icons8.com/color/240/000000/car-sale--v1.png" width="160" />
                </div>
                <p style="text-align: center; margin-top: 15px;">
                Predict car prices based on their specifications using trained machine learning models.
                </p>
                """
            ), unsafe_allow_html=True)
        
        # Display basic statistics for numerical columns
        st.markdown(card(
            "🔢 Numerical Features Statistics",
            ""
        ), unsafe_allow_html=True)
        st.dataframe(df.describe())
        
        # Distribution of price
        st.markdown(card(
            "📈 Distribution of Car Selling Prices",
            ""
        ), unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Selling_Price'], kde=True, ax=ax, color='#0068c9')
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
