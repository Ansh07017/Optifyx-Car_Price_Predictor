import streamlit as st
import pandas as pd
import os

@st.cache_data
def load_data():
    """
    Load the car dataset from CSV file
    
    Returns:
        pandas.DataFrame: The loaded car dataset
    """
    try:
        # Check if the file exists in attached_assets directory
        file_path = "attached_assets/car data.csv"
        
        if os.path.exists(file_path):
            # Load the dataset
            df = pd.read_csv(file_path)
            
            # Map column names to be consistent based on the image shared
            column_mapping = {
                'Car_Name': 'Car_Name',
                'Year': 'Year',
                'Selling_Price': 'Selling_Price',
                'Selling_Pri': 'Selling_Price',
                'Present_Price': 'Present_Price',
                'Present_Pri': 'Present_Price',
                'Driven_kms': 'Driven_kms',
                'Driven_km': 'Driven_kms',
                'Fuel_Type': 'Fuel_Type',
                'Selling_type': 'Selling_type',
                'Transmission': 'Transmission',
                'Transmissi': 'Transmission',
                'Owner': 'Owner'
            }
            
            # Rename columns that exist in the dataframe
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            st.success("Data loaded from CSV file successfully")
            return df
        
        # If not in attached_assets, try current directory
        file_path = "car data.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # Apply the same column mapping for consistency
            column_mapping = {
                'Car_Name': 'Car_Name',
                'Year': 'Year',
                'Selling_Price': 'Selling_Price',
                'Selling_Pri': 'Selling_Price',
                'Present_Price': 'Present_Price',
                'Present_Pri': 'Present_Price',
                'Driven_kms': 'Driven_kms',
                'Driven_km': 'Driven_kms',
                'Fuel_Type': 'Fuel_Type',
                'Selling_type': 'Selling_type',
                'Transmission': 'Transmission',
                'Transmissi': 'Transmission',
                'Owner': 'Owner'
            }
            
            # Rename columns that exist in the dataframe
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
                    
            st.success("Data loaded from CSV file successfully")
            return df
        
        # If we get here, no data source was found
        raise FileNotFoundError("Car dataset file not found in attached_assets or current directory")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        raise e

def show_column_descriptions():
    """
    Display descriptions of each column in the dataset
    """
    # Column descriptions directly defined here
    descriptions = {
        'Car_Name': 'Name/model of the car',
        'Year': 'Year of manufacture',
        'Selling_Price': 'Price at which the car is being sold (in lakhs INR)',
        'Present_Price': 'Current market price of the car when new (in lakhs INR)',
        'Driven_kms': 'Total kilometers the car has been driven',
        'Fuel_Type': 'Type of fuel the car uses (e.g., Petrol, Diesel, CNG)',
        'Selling_type': 'Who is selling the car — Dealer or Individual',
        'Transmission': 'Type of transmission — Manual or Automatic',
        'Owner': 'Number of previous owners before the current seller (0 = first-hand)',
        'Car_Age': 'Age of the car calculated as (Current Year - Year of Manufacture)'
    }
    
    st.subheader("Dataset Column Descriptions")
    
    # Create a table for the descriptions
    col_desc_df = pd.DataFrame({
        'Column Name': list(descriptions.keys()),
        'Description': list(descriptions.values())
    })
    
    st.table(col_desc_df)
