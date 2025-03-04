import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import (
    chat_completion, 
    detect_prob, 
    calculate_entropy, 
    uc_grouping, 
    IMAGING_TYPES, 
    ZEROSHOT_PROMPT,
    DISEASE_OF_INTEREST,
    ensure_output_dir
)
import os

class MockResponse:
    """Mock response class for handling NaN values"""
    def __init__(self):
        self.choices = [type('obj', (object,), {
            'message': type('obj', (object,), {
                'content': 'nan'
            })
        })]

async def zeroshot_run_async(main_prompt, information, imaging_report, doi):
    """
    Process multiple cases asynchronously using zero-shot prompt.
    
    Args:
        main_prompt (str): Main prompt template
        information (list): List of cases to process
        imaging_report (str): Type of imaging report
        doi (str): Disease of interest
        
    Returns:
        list: List of responses
    """
    tasks = []
    for i in information:
        # Convert to string and clean
        str_val = str(i).strip() if i is not None else ''
        
        # Check for invalid values
        if (pd.isna(i) or 
            not isinstance(i, str) or 
            not str_val or 
            str_val.lower() in ['nan', 'none', 'null']):
            tasks.append(MockResponse())
            continue
            
        new_prompt = main_prompt.format(imaging_report=imaging_report, disease=doi, input_text=str_val)
        tasks.append(chat_completion(new_prompt))
    
    # Handle responses
    responses = []
    for t in tasks:
        if isinstance(t, MockResponse):
            responses.append(t)
        else:
            try:
                response = await t
                responses.append(response)
            except Exception:
                responses.append(MockResponse())
    
    return responses

async def process_stage1(df, openai_api_key=None, batch_size=15):
    """
    Process Stage 1 of the classification system.
    
    Args:
        df (pd.DataFrame): Input dataframe
        openai_api_key (str, optional): OpenAI API key
        batch_size (int): Batch processing size
        
    Returns:
        pd.DataFrame: Dataframe with Stage 1 results added
    """
    output_dir = ensure_output_dir()
    
    # Initialize Stage 1 columns with NaN
    for _, base_column_name in IMAGING_TYPES:
        for i in range(1, 4):
            df[f'Stage1_{base_column_name}_predict_{i}'] = np.nan
        df[f'Stage1_{base_column_name}_prob'] = np.nan
    
    # Process each imaging type with progress bar
    imaging_pbar = tqdm(IMAGING_TYPES, desc="Processing imaging types", unit="type")
    for imaging_type, base_column_name in imaging_pbar:
        imaging_pbar.set_postfix_str(f"Current: {base_column_name}")
        
        if imaging_type not in df.columns:
            continue
        
        # Convert empty strings and whitespace-only strings to NaN
        df[imaging_type] = df[imaging_type].replace(r'^\s*$', np.nan, regex=True)
        
        # Skip if all values are NaN
        if df[imaging_type].isna().all():
            continue
        
        # Create nan mask and handle nan values first
        nan_mask = (
            df[imaging_type].isna() |  # NaN values
            (df[imaging_type].astype(str).str.strip() == '') |  # Empty strings
            (df[imaging_type].astype(str).str.lower() == 'nan') |  # 'nan' strings
            (df[imaging_type].astype(str).str.lower() == 'none') |  # 'none' strings
            ~df[imaging_type].astype(str).str.strip().astype(bool)  # Falsy values
        )
        
        # Set NaN values
        nan_indices = df[nan_mask].index.tolist()
        for idx in nan_indices:
            for i in range(1, 4):
                df.at[idx, f'Stage1_{base_column_name}_predict_{i}'] = np.nan
            df.at[idx, f'Stage1_{base_column_name}_prob'] = np.nan
        
        # Process only valid non-NaN values
        non_nan_mask = ~nan_mask
        non_nan_count = non_nan_mask.sum()
        
        if non_nan_count == 0:
            continue
        
        # Get non-NaN values and indices
        non_nan_indices = df[non_nan_mask].index.tolist()
        non_nan_values = df.loc[non_nan_mask, imaging_type].values
        
        # Additional validation of non-NaN values
        valid_values = []
        valid_indices = []
        
        for idx, val in zip(non_nan_indices, non_nan_values):
            # Convert to string and clean
            str_val = str(val).strip()
            
            # Skip if empty or invalid
            if not str_val or str_val.lower() in ['nan', 'none', 'null', '']:
                for i in range(1, 4):
                    df.at[idx, f'Stage1_{base_column_name}_predict_{i}'] = np.nan
                df.at[idx, f'Stage1_{base_column_name}_prob'] = np.nan
                continue
                
            valid_values.append(str_val)
            valid_indices.append(idx)
        
        if not valid_values:
            continue
        
        # Split valid data into batches
        batches = [valid_values[j:j + batch_size] for j in range(0, len(valid_values), batch_size)]
        batch_indices = [valid_indices[j:j + batch_size] for j in range(0, len(valid_indices), batch_size)]
        
        # Process each report three times with progress bar
        iter_pbar = tqdm(range(1, 4), desc=f"Processing {base_column_name} iterations", unit="iter", leave=False)
        for i in iter_pbar:
            outputs = []
            processed_indices = []
            
            # Process batches with progress bar
            batch_pbar = tqdm(zip(batches, batch_indices), 
                            desc=f"Processing batches", 
                            total=len(batches), 
                            unit="batch",
                            leave=False)
            
            for batch, batch_idx in batch_pbar:
                output = await zeroshot_run_async(ZEROSHOT_PROMPT, batch, imaging_type, DISEASE_OF_INTEREST)
                outputs.extend(output)
                processed_indices.extend(batch_idx)
            
            # Save results
            for idx, output in zip(processed_indices, outputs):
                result = output.choices[0].message.content
                df.at[idx, f'Stage1_{base_column_name}_predict_{i}'] = result
        
        # Process probabilities
        prob_pbar = tqdm(valid_indices, desc="Calculating probabilities", unit="row", leave=False)
        for idx in prob_pbar:
            probs = []
            for i in range(1, 4):
                predict_column = f'Stage1_{base_column_name}_predict_{i}'
                value = df.at[idx, predict_column]
                if pd.notna(value) and value != 'nan':
                    prob = detect_prob(value)
                    if pd.notna(prob):
                        probs.append(prob)
            
            if probs:
                df.at[idx, f'Stage1_{base_column_name}_prob'] = np.mean(probs)
            else:
                df.at[idx, f'Stage1_{base_column_name}_prob'] = np.nan
    
    # Calculate maximum probability across imaging types
    prob_columns = [f'Stage1_{t[1]}_prob' for t in IMAGING_TYPES]
    df['Stage1_prob'] = df[prob_columns].max(axis=1, skipna=True)
    
    # Calculate entropy only for valid probabilities
    probs = df['Stage1_prob'].values
    mask = ~np.isnan(probs)
    entropies = np.full_like(probs, np.nan)
    entropies[mask] = calculate_entropy(probs[mask])
    df['Stage1_entropy'] = entropies
    
    # Save results
    df.to_excel(os.path.join(output_dir, 'Stage1_result.xlsx'), index=False)
    
    return df

def split_stage1_by_entropy(df):
    """
    Split Stage 1 results based on entropy values.
    
    Args:
        df (pd.DataFrame): Dataframe containing Stage 1 results
        
    Returns:
        tuple: (high_entropy_df, low_entropy_df) Two dataframes split by entropy values
    """
    # Check output directory
    output_dir = ensure_output_dir()
    
    print("\n=== Stage 1 Split Information ===")
    print(f"Total cases before split: {len(df)}")
    
    # Group based on entropy values
    Stage1_entropy_list = df['Stage1_entropy'].values
    cutoff_value = df['Stage1_entropy'].median()
    print(f"Entropy cutoff value: {cutoff_value}")
    
    high, low = uc_grouping(Stage1_entropy_list, cutoff=cutoff_value)
    
    Stage1_high = df.loc[high].reset_index(drop=True)
    Stage1_low = df.loc[low].reset_index(drop=True)
    
    print(f"High entropy group size: {len(Stage1_high)}")
    print(f"Low entropy group size: {len(Stage1_low)}")
    print(f"Total after split: {len(Stage1_high) + len(Stage1_low)}")
    
    if len(df) != len(Stage1_high) + len(Stage1_low):
        print("WARNING: Total cases mismatch after split!")
        
    # Check for duplicates between groups
    high_ids = set(Stage1_high['연구번호']) if '연구번호' in Stage1_high.columns else set()
    low_ids = set(Stage1_low['연구번호']) if '연구번호' in Stage1_low.columns else set()
    duplicates = high_ids.intersection(low_ids)
    
    if duplicates:
        print(f"WARNING: Found {len(duplicates)} duplicate cases between groups!")
        print(f"Duplicate IDs: {duplicates}")
    
    # Save results
    Stage1_high.to_excel(os.path.join(output_dir, 'Stage1_high_entropygroup.xlsx'), index=False)
    Stage1_low.to_excel(os.path.join(output_dir, 'Stage1_low_entropygroup.xlsx'), index=False)
    df.to_excel(os.path.join(output_dir, 'Stage1_result.xlsx'), index=False)
    
    return Stage1_high, Stage1_low 