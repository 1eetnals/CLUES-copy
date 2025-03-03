import asyncio
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
    OUTPUT_DIR,
    ensure_output_dir
)
import os
import openai

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
        new_prompt = main_prompt.format(imaging_report=imaging_report, disease=doi, input_text=i)
        tasks.append(chat_completion(new_prompt))
    responses = await asyncio.gather(*tasks)
    return responses

async def process_stage1(df, batch_size=15):
    """
    Perform Stage 1 processing.
    
    Args:
        df (pd.DataFrame): Dataframe to process
        batch_size (int): Batch processing size
        
    Returns:
        pd.DataFrame: Dataframe with Stage 1 results added
    """
    output_dir = ensure_output_dir()
    
    # Initialize columns for each imaging type
    for _, base_column_name in IMAGING_TYPES:
        for i in range(1, 4):  # Process each report three times
            df[f'Stage1_{base_column_name}_predict_{i}'] = np.nan
    
    # Process each imaging type and iteration
    for imaging_type, base_column_name in IMAGING_TYPES:
        for i in range(1, 4):   # 3 iterations
            # Get non-NaN indices and values
            non_nan_indices = df[imaging_type].dropna().index
            non_nan_values = df[imaging_type].dropna().values
                    
            # Split data into batches
            batches = [non_nan_values[j:j + batch_size].tolist() for j in range(0, len(non_nan_values), batch_size)]
            
            outputs = []
            for batch in tqdm(batches, desc=f"Processing {imaging_type}, Round {i}"):
                output = await zeroshot_run_async(ZEROSHOT_PROMPT, batch, imaging_type, DISEASE_OF_INTEREST)
                outputs.extend(output)
            
            # Extract results from outputs
            results = [output.choices[0].message.content for output in outputs]
            
            # Save results to corresponding column
            result_column = f'Stage1_{base_column_name}_predict_{i}'
            for idx, result in zip(non_nan_indices, results):
                df.at[idx, result_column] = result
    
    # Create temporary dictionary for probability calculations
    temp_results = {f'Stage1_{t[1]}_prob': [] for t in IMAGING_TYPES}
    
    # Process each imaging type and iteration
    for imaging_type, base_column_name in IMAGING_TYPES:
        for i in range(1, 4):
            predict_column = f'Stage1_{base_column_name}_predict_{i}'
    
            # Get non-NaN values and corresponding indices
            non_nan_indices = df[predict_column].dropna().index
            non_nan_values = df[predict_column].dropna().values
    
            results = []
            for value in non_nan_values:
                results.append(detect_prob(value))
    
            # Add results to corresponding list in temporary dictionary
            temp_results[f'Stage1_{base_column_name}_prob'].append(pd.Series(results, index=non_nan_indices))
    
    # Calculate mean probabilities for each imaging type
    for key, value in temp_results.items():
        if value:
            df[key + '_mean'] = pd.concat(value, axis=1).mean(axis=1, skipna=True)
    
    # Calculate maximum probability across all imaging types
    df['Stage1_prob'] = df[[f'Stage1_{t[1]}_prob_mean' for t in IMAGING_TYPES]].max(axis=1, skipna=True)
    df['Stage1_prob'].fillna(0, inplace=True)
    
    # Calculate entropy
    probs = df['Stage1_prob'].values
    entropies = calculate_entropy(probs)
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
    
    # Group based on entropy values
    Stage1_entropy_list = df['Stage1_entropy'].values
    cutoff_value = df['Stage1_entropy'].median()
    high, low = uc_grouping(Stage1_entropy_list, cutoff=cutoff_value)
    
    Stage1_high = df.loc[high].reset_index(drop=True)
    Stage1_low = df.loc[low].reset_index(drop=True)
    
    # Save results
    Stage1_high.to_excel(os.path.join(output_dir, 'Stage1_high_entropygroup.xlsx'), index=False)
    Stage1_low.to_excel(os.path.join(output_dir, 'Stage1_low_entropygroup.xlsx'), index=False)
    df.to_excel(os.path.join(output_dir, 'Stage1_result.xlsx'), index=False)
    
    return Stage1_high, Stage1_low 