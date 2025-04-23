import os
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import streamlit as st

def get_connection():
    """
    Get a connection to the PostgreSQL database
    
    Returns:
        connection: PostgreSQL database connection
    """
    try:
        # Get database connection parameters from environment variables
        conn = psycopg2.connect(
            host=os.environ.get('PGHOST'),
            database=os.environ.get('PGDATABASE'),
            user=os.environ.get('PGUSER'),
            password=os.environ.get('PGPASSWORD'),
            port=os.environ.get('PGPORT')
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        return conn
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

def create_tables():
    """
    Create the necessary tables in the database if they don't exist
    """
    conn = get_connection()
    if conn is None:
        return
    
    try:
        cursor = conn.cursor()
        
        # Create car_data table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS car_data (
            id SERIAL PRIMARY KEY,
            car_name TEXT,
            year INTEGER,
            selling_price FLOAT,
            present_price FLOAT,
            driven_kms INTEGER,
            fuel_type TEXT,
            selling_type TEXT,
            transmission TEXT,
            owner INTEGER,
            car_age INTEGER
        )
        """)
        
        cursor.close()
        st.success("Database tables created successfully!")
    except Exception as e:
        st.error(f"Error creating tables: {e}")
    finally:
        conn.close()

def insert_car_data(df):
    """
    Insert the car data into the database
    
    Args:
        df (pandas.DataFrame): The car dataset
    """
    conn = get_connection()
    if conn is None:
        return
    
    try:
        cursor = conn.cursor()
        
        # First, check if there's already data in the table
        cursor.execute("SELECT COUNT(*) FROM car_data")
        count = cursor.fetchone()[0]
        
        if count > 0:
            st.info("Data already exists in the database. Skipping insertion.")
            cursor.close()
            conn.close()
            return
        
        # Calculate car_age if it doesn't exist
        if 'Car_Age' not in df.columns and 'Year' in df.columns:
            current_year = 2025
            df['Car_Age'] = current_year - df['Year']
        
        # Insert data into car_data table
        for _, row in df.iterrows():
            cursor.execute(
                """
                INSERT INTO car_data (
                    car_name, year, selling_price, present_price, 
                    driven_kms, fuel_type, selling_type, transmission, owner, car_age
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    row.get('Car_Name', ''),
                    row.get('Year', 0),
                    row.get('Selling_Price', 0.0),
                    row.get('Present_Price', 0.0),
                    row.get('Driven_kms', 0),
                    row.get('Fuel_Type', ''),
                    row.get('Selling_type', ''),
                    row.get('Transmission', ''),
                    row.get('Owner', 0),
                    row.get('Car_Age', 0)
                )
            )
        
        cursor.close()
        st.success(f"Successfully inserted {len(df)} car records into the database!")
    except Exception as e:
        st.error(f"Error inserting data: {e}")
    finally:
        conn.close()

def get_car_data():
    """
    Get the car data from the database
    
    Returns:
        pandas.DataFrame: The car dataset
    """
    conn = get_connection()
    if conn is None:
        return None
    
    try:
        # Query the database
        df = pd.read_sql_query("SELECT * FROM car_data", conn)
        
        # Remove id column
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        
        # Rename columns to match the original dataset (capitalize column names)
        df.columns = [col.capitalize() for col in df.columns]
        
        # Ensure specific column names match exactly
        rename_map = {
            'Car_name': 'Car_Name',
            'Selling_price': 'Selling_Price',
            'Present_price': 'Present_Price',
            'Driven_kms': 'Driven_kms',
            'Fuel_type': 'Fuel_Type',
            'Selling_type': 'Selling_type'
        }
        
        df = df.rename(columns=rename_map)
        
        return df
    except Exception as e:
        st.error(f"Error fetching data from database: {e}")
        return None
    finally:
        conn.close()
        
def get_column_descriptions():
    """
    Returns a dictionary with column descriptions
    
    Returns:
        dict: Dictionary mapping column names to their descriptions
    """
    return {
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