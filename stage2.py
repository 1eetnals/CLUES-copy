import asyncio
import numpy as np
import pandas as pd
from tqdm import tqdm
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from utils import (
    chat_completion, 
    detect_prob, 
    calculate_entropy, 
    uc_grouping, 
    IMAGING_TYPES, 
    FEWSHOT_PROMPT,
    DISEASE_OF_INTEREST,
    ensure_output_dir
)
from dotenv import load_dotenv
import os

def setup_retrieval_system(imaging_type, openai_api_key=None):
    """
    Set up the retrieval system.
    
    Args:
        imaging_type (str): Type of imaging
        openai_api_key (str, optional): OpenAI API key. If None, loads from environment variables.
        
    Returns:
        tuple: (ensemble_retriever, k) Retriever and number of documents to retrieve
    """
    # Load .env file
    load_dotenv()
    
    # Check output directory
    output_dir = ensure_output_dir()
    
    ##### 1) Load documents
    csv_path = os.path.join(output_dir, f'{imaging_type}.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"RAG dataset file not found: {csv_path}")
        
    # Load and check data
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError(f"RAG dataset is empty for {imaging_type}")
        
    loader = CSVLoader(csv_path, source_column='Output')   
    docs = loader.load()
    if not docs:
        raise ValueError(f"No documents loaded from RAG dataset for {imaging_type}")
        
    doc_list = [doc.page_content for doc in docs]      # Create list for lexical search

    ##### 2) Embeddings
    model_name = 'text-embedding-ada-002' 
    embeddings = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=openai_api_key if openai_api_key else os.getenv('OPENAI_API_KEY')
    )

    ##### 3) Create vector store
    vectorstore = FAISS.from_documents(docs, embeddings)

    ##### 4) Search
    k = min(6, len(docs))  # Number of similar documents to retrieve (not more than available docs)

    # Set up lexical search with BM25
    bm25_retriever = BM25Retriever.from_texts(doc_list)
    bm25_retriever.k = k

    # Set up semantic search with FAISS (cosine similarity)
    faiss_retriever = vectorstore.as_retriever(
        search_type="similarity",  
        search_kwargs={'k': k}
    )

    # Set up hybrid search using weighted combination of BM25 and FAISS
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.6, 0.4]
    )

    return ensemble_retriever, k

async def retrieve_shots(query, retriever, k):
    """
    Retrieve shots related to the query.
    
    Args:
        query (str): Search query
        retriever: Retriever object
        k (int): Number of documents to retrieve
        
    Returns:
        str: Formatted search results
    """
    # Retrieve documents related to query using get_relevant_documents
    shots_tot = retriever.get_relevant_documents(query)
    # Extract and format content from retrieved documents
    shots = [doc.page_content for doc in shots_tot[:k]]
    formatted_shots = '\n'.join(f'{shot.split("Output: ")[0].strip()}\nOutput: {shot.split("Output: ")[1].strip()}' for shot in shots)
    return formatted_shots

async def fewshot_run_async(main_prompt, information, imaging_report, doi, retriever, k):
    """
    Process multiple cases asynchronously using few-shot prompt.
    
    Args:
        main_prompt (str): Main prompt template
        information (list): List of cases to process
        imaging_report (str): Type of imaging report
        doi (str): Disease of interest
        retriever: Retriever object
        k (int): Number of documents to retrieve
        
    Returns:
        list: List of responses
    """
    tasks = []
    for i in information:
        shots = await retrieve_shots(i, retriever, k)
        new_prompt = main_prompt.format(imaging_report=imaging_report, disease=doi, fewshot_retrieval=shots, input_text=i)
        tasks.append(chat_completion(new_prompt))
    responses = await asyncio.gather(*tasks)
    return responses

async def process_stage2(stage1_high_df, openai_api_key=None, batch_size=15):
    """
    Perform Stage 2 processing.
    
    Args:
        stage1_high_df (pd.DataFrame): Stage 1 high entropy group dataframe
        openai_api_key (str, optional): OpenAI API key
        batch_size (int): Batch processing size
        
    Returns:
        pd.DataFrame: Dataframe with Stage 2 results added
    """
    output_dir = ensure_output_dir()
    
    print("\n=== Stage 2 Processing Information ===")
    print(f"Total cases to process: {len(stage1_high_df)}")
    
    # Initialize Stage 2 columns
    for _, base_column_name in IMAGING_TYPES:
        stage1_high_df[f'Stage2_{base_column_name}_predict'] = np.nan
        stage1_high_df[f'Stage2_{base_column_name}_prob_mean'] = np.nan
    
    # Process each imaging type with progress bar
    imaging_pbar = tqdm(IMAGING_TYPES, desc="Processing imaging types", unit="type")
    for imaging_type, base_column_name in imaging_pbar:
        imaging_pbar.set_postfix_str(f"Current: {base_column_name}")
        
        print(f"\nProcessing {base_column_name}:")
        
        if imaging_type not in stage1_high_df.columns:
            print(f"Column {imaging_type} not found in dataframe")
            continue
            
        # Convert empty strings and whitespace-only strings to NaN
        stage1_high_df[imaging_type] = stage1_high_df[imaging_type].replace(r'^\s*$', np.nan, regex=True)
        
        # Skip if all values are NaN
        if stage1_high_df[imaging_type].isna().all():
            print(f"All values are NaN for {base_column_name}")
            continue
        
        # Create nan mask and handle nan values first
        nan_mask = (
            stage1_high_df[imaging_type].isna() |  # NaN values
            (stage1_high_df[imaging_type].astype(str).str.strip() == '') |  # Empty strings
            (stage1_high_df[imaging_type].astype(str).str.lower() == 'nan') |  # 'nan' strings
            (stage1_high_df[imaging_type].astype(str).str.lower() == 'none') |  # 'none' strings
            ~stage1_high_df[imaging_type].astype(str).str.strip().astype(bool)  # Falsy values
        )
        
        print(f"Found {nan_mask.sum()} NaN/invalid values")
        
        # Set NaN values
        nan_indices = stage1_high_df[nan_mask].index
        for idx in nan_indices:
            stage1_high_df.at[idx, f'Stage2_{base_column_name}_predict'] = np.nan
            stage1_high_df.at[idx, f'Stage2_{base_column_name}_prob_mean'] = np.nan
        
        # Process only valid non-NaN values
        non_nan_mask = ~nan_mask
        non_nan_count = non_nan_mask.sum()
        
        print(f"Processing {non_nan_count} valid values")
        
        if non_nan_count == 0:
            continue
        
        # Get non-NaN values and indices
        non_nan_indices = stage1_high_df[non_nan_mask].index
        non_nan_values = stage1_high_df.loc[non_nan_mask, imaging_type].values
        
        # Additional validation of non-NaN values
        valid_values = []
        valid_indices = []
        
        for idx, val in zip(non_nan_indices, non_nan_values):
            # Convert to string and clean
            str_val = str(val).strip()
            
            # Skip if empty or invalid
            if not str_val or str_val.lower() in ['nan', 'none', 'null', '']:
                stage1_high_df.at[idx, f'Stage2_{base_column_name}_predict'] = np.nan
                stage1_high_df.at[idx, f'Stage2_{base_column_name}_prob_mean'] = np.nan
                continue
                
            valid_values.append(str_val)
            valid_indices.append(idx)
        
        print(f"Found {len(valid_values)} values after validation")
        
        if not valid_values:
            continue
            
        try:
            # Load RAG dataset
            rag_file = os.path.join(output_dir, f'{base_column_name}.csv')
            if not os.path.exists(rag_file):
                print(f"RAG dataset not found for {base_column_name}")
                continue
                
            # Set up retrieval system
            retriever, k = setup_retrieval_system(base_column_name, openai_api_key)
            
            # Split data into batches
            batches = [valid_values[j:j + batch_size] for j in range(0, len(valid_values), batch_size)]
            batch_indices = [valid_indices[j:j + batch_size] for j in range(0, len(valid_indices), batch_size)]
            
            print(f"Processing {len(batches)} batches")
            
            # Process batches
            outputs = []
            processed_indices = []
            
            batch_pbar = tqdm(zip(batches, batch_indices), 
                            desc=f"Processing {base_column_name} batches", 
                            total=len(batches), 
                            unit="batch",
                            leave=False)
            
            for batch, batch_idx in batch_pbar:
                output = await fewshot_run_async(FEWSHOT_PROMPT, batch, imaging_type, DISEASE_OF_INTEREST, retriever, k)
                outputs.extend(output)
                processed_indices.extend(batch_idx)
            
            print(f"Successfully processed {len(outputs)} outputs")
            
            # Save results
            for idx, output in zip(processed_indices, outputs):
                result = output.choices[0].message.content
                stage1_high_df.at[idx, f'Stage2_{base_column_name}_predict'] = result
            
            # Process probabilities
            prob_count = 0
            prob_pbar = tqdm(valid_indices, desc="Calculating probabilities", unit="row", leave=False)
            for idx in prob_pbar:
                predict_column = f'Stage2_{base_column_name}_predict'
                value = stage1_high_df.at[idx, predict_column]
                
                if pd.notna(value) and value != 'nan':
                    prob = detect_prob(value)
                    if pd.notna(prob):
                        stage1_high_df.at[idx, f'Stage2_{base_column_name}_prob_mean'] = prob
                        prob_count += 1
                    else:
                        stage1_high_df.at[idx, f'Stage2_{base_column_name}_prob_mean'] = np.nan
                else:
                    stage1_high_df.at[idx, f'Stage2_{base_column_name}_prob_mean'] = np.nan
            
            print(f"Calculated {prob_count} valid probabilities")
            
        except Exception as e:
            print(f"Error processing {base_column_name}: {str(e)}")
            continue
    
    # Calculate maximum probability across all imaging types
    stage1_high_df['Stage2_prob'] = stage1_high_df[[f'Stage2_{t[1]}_prob_mean' for t in IMAGING_TYPES]].max(axis=1, skipna=True)
    
    # Calculate entropy only for non-NaN probabilities
    probs = stage1_high_df['Stage2_prob'].values
    mask = ~np.isnan(probs)
    entropies = np.full_like(probs, np.nan)
    entropies[mask] = calculate_entropy(probs[mask])
    stage1_high_df['Stage2_entropy'] = entropies
    
    print("\nFinal Statistics:")
    print(f"Total valid probabilities: {mask.sum()}")
    print(f"Total NaN probabilities: {(~mask).sum()}")
    
    # Save results
    stage1_high_df.to_excel(os.path.join(output_dir, 'Stage2_result.xlsx'), index=False)
    
    return stage1_high_df

def split_stage2_by_entropy(stage1_high_df):
    """
    Split Stage 2 results based on entropy values.
    
    Args:
        stage1_high_df (pd.DataFrame): Dataframe containing Stage 2 results
        
    Returns:
        tuple: (stage2_high_df, stage2_low_df) Two dataframes split by entropy values
    """
    # Check output directory
    output_dir = ensure_output_dir()
    
    print("\n=== Stage 2 Split Information ===")
    print(f"Total Stage 1 high entropy cases: {len(stage1_high_df)}")
    
    # Group based on entropy values
    Stage2_entropy_list = stage1_high_df['Stage2_entropy'].values
    cutoff_value = stage1_high_df['Stage2_entropy'].median()
    print(f"Entropy cutoff value: {cutoff_value}")
    
    high, low = uc_grouping(Stage2_entropy_list, cutoff=cutoff_value)
    
    Stage2_high = stage1_high_df.loc[high].reset_index(drop=True)
    Stage2_low = stage1_high_df.loc[low].reset_index(drop=True)
    
    print(f"Stage 2 high entropy group size: {len(Stage2_high)}")
    print(f"Stage 2 low entropy group size: {len(Stage2_low)}")
    print(f"Total after split: {len(Stage2_high) + len(Stage2_low)}")
    
    if len(stage1_high_df) != len(Stage2_high) + len(Stage2_low):
        print("WARNING: Total cases mismatch after Stage 2 split!")
        
    # Check for duplicates between groups
    high_ids = set(Stage2_high['연구번호']) if '연구번호' in Stage2_high.columns else set()
    low_ids = set(Stage2_low['연구번호']) if '연구번호' in Stage2_low.columns else set()
    duplicates = high_ids.intersection(low_ids)
    
    if duplicates:
        print(f"WARNING: Found {len(duplicates)} duplicate cases between Stage 2 groups!")
        print(f"Duplicate IDs: {duplicates}")
    
    # Save results
    Stage2_high.to_excel(os.path.join(output_dir, 'Stage2_high_entropygroup.xlsx'), index=False)
    Stage2_low.to_excel(os.path.join(output_dir, 'Stage2_low_entropygroup.xlsx'), index=False)
    
    return Stage2_high, Stage2_low

def combine_results(stage1_low_df, stage1_high_df, stage2_high_df, stage2_low_df):
    """
    Combine results from all stages.
    
    Args:
        stage1_low_df (pd.DataFrame): Stage 1 low entropy group
        stage1_high_df (pd.DataFrame): Stage 1 high entropy group
        stage2_high_df (pd.DataFrame): Stage 2 high entropy group
        stage2_low_df (pd.DataFrame): Stage 2 low entropy group
        
    Returns:
        tuple: (stage2_result_df, stage3_result_df) Combined results from Stage 2 and Stage 3
    """
    # Check output directory
    output_dir = ensure_output_dir()
    
    print("\n=== Results Combination Information ===")
    print(f"Stage 1 low entropy group size: {len(stage1_low_df)}")
    print(f"Stage 1 high entropy group size: {len(stage1_high_df)}")
    print(f"Stage 2 high entropy group size: {len(stage2_high_df)}")
    print(f"Stage 2 low entropy group size: {len(stage2_low_df)}")
    
    # Using Stage 1 results only
    stage1_low_df['Stage2_prob'] = stage1_low_df['Stage1_prob']
    combined_df_1 = pd.concat([stage1_low_df, stage1_high_df], ignore_index=True)
    print(f"\nStage 2 combined result size: {len(combined_df_1)}")
    
    # Check for duplicates in Stage 2 results
    if '연구번호' in combined_df_1.columns:
        duplicates = combined_df_1['연구번호'].duplicated().sum()
        if duplicates > 0:
            print(f"WARNING: Found {duplicates} duplicate cases in Stage 2 results!")
            print("Duplicate IDs:", combined_df_1[combined_df_1['연구번호'].duplicated(keep=False)]['연구번호'].unique())
    
    # Using Stage 1, 2, 3 results
    stage1_low_df['Stage3_prob'] = stage1_low_df['Stage1_prob']
    stage2_low_df['Stage3_prob'] = stage2_low_df['Stage2_prob']
    stage2_high_df['Stage3_prob'] = stage2_high_df['Label']  # Use final label
    combined_df_2 = pd.concat([stage1_low_df, stage2_low_df, stage2_high_df], ignore_index=True)
    print(f"\nStage 3 combined result size: {len(combined_df_2)}")
    
    # Check for duplicates in Stage 3 results
    if '연구번호' in combined_df_2.columns:
        duplicates = combined_df_2['연구번호'].duplicated().sum()
        if duplicates > 0:
            print(f"WARNING: Found {duplicates} duplicate cases in Stage 3 results!")
            print("Duplicate IDs:", combined_df_2[combined_df_2['연구번호'].duplicated(keep=False)]['연구번호'].unique())
    
    # Save results
    combined_df_1.to_excel(os.path.join(output_dir, 'Stage2_result.xlsx'), index=False)
    combined_df_2.to_excel(os.path.join(output_dir, 'Stage3_result.xlsx'), index=False)
    
    return combined_df_1, combined_df_2 