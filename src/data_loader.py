import pandas as pd
import os

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the heart disease dataset from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
        
    Raises:
        FileNotFoundError: If the file is not found.
    """
    if not os.path.exists(filepath):
        # Fallback check for filename variations if needed, or just error out
        raise FileNotFoundError(f"The dataset file was not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Dataset loaded successfully from {filepath}")
    print(f"Shape: {df.shape}")
    return df
