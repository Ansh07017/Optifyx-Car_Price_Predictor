import streamlit as st
import pandas as pd
import os
from modules.database import create_tables, insert_car_data, get_car_data, get_column_descriptions

@st.cache_data
def load_data():
    """
    Load the car dataset from file and store in database
    
    Returns:
        pandas.DataFrame: The loaded car dataset
    """
    try:
        # First, try to get data from database
        db_data = get_car_data()
        
        if db_data is not None and not db_data.empty:
            st.success("Data loaded from database successfully")
            return db_data
        
        # If database is empty, load from CSV and store in database
        # Attempt to read from the attached assets directory
        file_path = "attached_assets/car data.csv"
        
        # Check if the file exists
        if os.path.exists(file_path):
            # Load the dataset
            df = pd.read_csv(file_path)
            
            # Create database tables if they don't exist
            create_tables()
            
            # Store data in database
            insert_car_data(df)
            
            return df
        else:
            # If file doesn't exist in attached_assets, try the current directory
            file_path = "car data.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                
                # Create database tables if they don't exist
                create_tables()
                
                # Store data in database
                insert_car_data(df)
                
                return df
            else:
                raise FileNotFoundError(f"Car dataset file not found in attached_assets or current directory")
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        raise e

def show_column_descriptions():
    """
    Display descriptions of each column in the dataset
    """
    descriptions = get_column_descriptions()
    
    st.subheader("Dataset Column Descriptions")
    
    # Create a table for the descriptions
    col_desc_df = pd.DataFrame({
        'Column Name': list(descriptions.keys()),
        'Description': list(descriptions.values())
    })
    
    st.table(col_desc_df)
