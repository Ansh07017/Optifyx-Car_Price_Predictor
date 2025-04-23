import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def predict_price(models, df, categorical_features, numerical_features, feature_names):
    """
    Create an interactive price prediction interface
    
    Args:
        models (dict): Dictionary of trained models
        df (pandas.DataFrame): The original dataset
        categorical_features (list): List of categorical feature names
        numerical_features (list): List of numerical feature names
        feature_names (list): List of all feature names after preprocessing
    """
    st.write("## Car Price Prediction")
    
    if not models:
        st.warning("No models available for prediction. Please train at least one model.")
        return
    
    # Provide information about the prediction tool
    st.write("""
    This tool allows you to predict the selling price of a car based on various features.
    Adjust the values below to get a price prediction.
    """)
    
    # Select the model to use for prediction
    model_names = list(models.keys())
    selected_model_name = st.selectbox("Select model for prediction:", model_names)
    selected_model = models[selected_model_name]['model']
    
    # Create the input form
    st.write("### Enter Car Details")
    
    col1, col2 = st.columns(2)
    
    # Create input widgets for categorical features
    cat_input_values = {}
    for feature in categorical_features:
        unique_values = df[feature].unique().tolist()
        
        # Use appropriate widgets based on the number of unique values
        with col1:
            if len(unique_values) <= 5:
                cat_input_values[feature] = st.radio(
                    f"Select {feature}:",
                    unique_values,
                    index=0
                )
            else:
                cat_input_values[feature] = st.selectbox(
                    f"Select {feature}:",
                    unique_values,
                    index=0
                )
    
    # Create input widgets for numerical features
    num_input_values = {}
    for feature in numerical_features:
        if feature in ['Year', 'Selling_Price', 'Car_Age']:  # Skip target and derived variables
            continue
            
        feature_min = float(df[feature].min())
        feature_max = float(df[feature].max())
        feature_mean = float(df[feature].mean())
        
        with col2:
            if feature_min.is_integer() and feature_max.is_integer():
                num_input_values[feature] = st.slider(
                    f"Select {feature}:",
                    int(feature_min),
                    int(feature_max),
                    int(feature_mean)
                )
            else:
                step = (feature_max - feature_min) / 100
                num_input_values[feature] = st.slider(
                    f"Select {feature}:",
                    float(feature_min),
                    float(feature_max),
                    float(feature_mean),
                    step=step
                )
    
    # Handle 'Year' separately for better UX
    with col1:
        if 'Year' in numerical_features:
            min_year = int(df['Year'].min())
            max_year = int(df['Year'].max())
            num_input_values['Year'] = st.slider(
                "Select Manufacturing Year:",
                min_year,
                max_year,
                max_year - 3
            )
            
            # Calculate Car_Age automatically
            current_year = 2025
            if 'Car_Age' in numerical_features:
                num_input_values['Car_Age'] = current_year - num_input_values['Year']
                st.write(f"Car Age: {num_input_values['Car_Age']} years")
    
    # Create a prediction button
    predict_button = st.button("Predict Price")
    
    if predict_button:
        # Create a DataFrame with the input values
        input_data = pd.DataFrame({**cat_input_values, **num_input_values}, index=[0])
        
        # Preprocess the input data
        # Standardize numerical features
        scaler = StandardScaler()
        scaler.fit(df[numerical_features])
        scaled_numerical = scaler.transform(input_data[numerical_features])
        
        # One-hot encode categorical features
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        if len(categorical_features) > 0:  # Only encode if we have categorical features
            encoder.fit(df[categorical_features])
            encoded_categorical = encoder.transform(input_data[categorical_features])
            
            # Combine numerical and categorical features
            input_features = np.hstack([scaled_numerical, encoded_categorical])
        else:
            input_features = scaled_numerical
        
        # Make prediction
        prediction = selected_model.predict(input_features)[0]
        
        # Display the prediction
        st.write("## Predicted Car Price")
        st.write(f"### ₹ {prediction:.2f} Lakhs")
        
        # Provide interpretation guidelines
        st.write("#### Interpretation")
        st.write("""
        - The predicted price is in lakhs of Indian Rupees (1 lakh = 100,000).
        - This prediction is based on the patterns learned from historical data.
        - Actual prices may vary due to factors not captured in the model.
        """)
        
        # Add feature impact if using a tree-based model
        if selected_model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
            st.write("#### Feature Impact on Price")
            st.write("For tree-based models, we can see which features influenced the prediction the most:")
            
            # Get feature importance
            importances = selected_model.feature_importances_
            
            # Create dataframe for feature importance
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            
            # Sort by importance
            feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).head(10)
            
            # Create bar chart for feature importance
            fig = px.bar(
                feature_importance_df,
                x='Feature',
                y='Importance',
                title='Top 10 Features Influencing Price',
                color='Importance',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig)
