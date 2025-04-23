import streamlit as st
import pandas as pd
import os

@st.cache_data
def load_data():
    """
    Load the car dataset
    
    Returns:
        pandas.DataFrame: The loaded car dataset
    """
    try:
        # Attempt to read from the attached assets directory
        file_path = "attached_assets/car data.csv"
        
        # Check if the file exists
        if os.path.exists(file_path):
            # Load the dataset
            df = pd.read_csv(file_path)
            return df
        else:
            # If file doesn't exist in attached_assets, try the current directory
            file_path = "car data.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                return df
            else:
                raise FileNotFoundError(f"Car dataset file not found in attached_assets or current directory")
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        raise e
