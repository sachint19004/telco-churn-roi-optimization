import pandas as pd

def load_and_clean_data(file_path):
    """Loads the CSV and cleans the data types and missing values."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print("Cleaning data...")
    # Fix the hidden spaces in TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    print("Data cleaning complete. Returning clean dataframe.")
    return df