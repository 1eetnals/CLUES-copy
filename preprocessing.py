import pandas as pd
import re
import os
from utils import IMAGING_TYPES, ensure_output_dir

def load_data(filepath):
    """
    Load data file.
    
    Args:
        filepath (str): Path to the file to load
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    return pd.read_excel(filepath)

def remove_special_characters(text):
    """
    Remove special characters from text.
    
    Args:
        text (str): Text to remove special characters from
        
    Returns:
        str: Text with special characters removed
    """
    return re.sub(r'[^a-zA-Z0-9가-힣\s]', '', str(text))

def preprocess_data(df):
    """
    Remove special characters from text columns in the dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe to preprocess
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Apply special character removal function to each report column
    for col, _ in IMAGING_TYPES:
        df[col] = df[col].apply(remove_special_characters)
    
    return df

def prepare_rag_datasets(low_entropy_df):
    """
    Prepare RAG datasets from Stage 1 low entropy group.
    
    Args:
        low_entropy_df (pd.DataFrame): Stage 1 low entropy dataframe
        
    Returns:
        None
    """
    output_dir = ensure_output_dir()
    
    for imaging_type, base_column_name in IMAGING_TYPES:
        # Extract data for this imaging type
        data = low_entropy_df[[imaging_type, f'Stage1_{base_column_name}_prob']]
        data = data.dropna()
        
        # Skip if no data available
        if len(data) == 0:
            print(f"Warning: No data available for {base_column_name} RAG dataset")
            continue
        
        # Rename columns
        data.columns = ['Input', 'Output']
        
        # Save to CSV file
        data.to_csv(os.path.join(output_dir, f'{base_column_name}.csv'), index=False) 