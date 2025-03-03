import asyncio
import numpy as np
import pandas as pd
from tqdm import tqdm
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
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
    OUTPUT_DIR,
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
        
    loader = CSVLoader(csv_path, source_column='Output')   
    docs = loader.load()
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
    k = 6  # Number of similar documents to retrieve

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

def retrieve_shots(query, retriever, k):
    """
    Retrieve shots related to the query.
    
    Args:
        query (str): Search query
        retriever: Retriever object
        k (int): Number of documents to retrieve
        
    Returns:
        str: Formatted search results
    """
    # Retrieve documents related to query
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
        shots = retrieve_shots(i, retriever, k)
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
    
    for imaging_type, base_column_name in IMAGING_TYPES:
        if imaging_type not in stage1_high_df.columns:
            continue
            
        print(f"\nProcessing {base_column_name} for Stage 2...")
        
        # Load RAG dataset
        rag_file = os.path.join(output_dir, f'{base_column_name}.csv')
        if not os.path.exists(rag_file):
            print(f"Warning: RAG dataset not found for {base_column_name}")
            continue
            
        # Set up retrieval system
        retriever, k = setup_retrieval_system(base_column_name, openai_api_key)
            
        # Get non-NaN indices and values
        non_nan_indices = stage1_high_df[imaging_type].dropna().index
        non_nan_values = stage1_high_df[imaging_type].dropna().values
                    
        # Split data into batches
        batches = [non_nan_values[j:j + batch_size].tolist() for j in range(0, len(non_nan_values), batch_size)]
            
        outputs = []
        for batch in tqdm(batches, desc=f"Processing {imaging_type} for Stage 2"):
            output = await fewshot_run_async(FEWSHOT_PROMPT, batch, imaging_type, DISEASE_OF_INTEREST, retriever, k)
            outputs.extend(output)
            
        # Extract results from outputs
        results = [output.choices[0].message.content for output in outputs]
            
        # Save results to corresponding column
        result_column = f'Stage2_{base_column_name}_predict'
        for idx, result in zip(non_nan_indices, results):
            stage1_high_df.at[idx, result_column] = result
    
    # Create temporary dictionary for probability calculations
    temp_results = {f'Stage2_{t[1]}_prob': [] for t in IMAGING_TYPES}
    
    # Process each imaging type
    for imaging_type, base_column_name in IMAGING_TYPES:
        predict_column = f'Stage2_{base_column_name}_predict'
    
        # Get non-NaN values and corresponding indices
        non_nan_indices = stage1_high_df[predict_column].dropna().index
        non_nan_values = stage1_high_df[predict_column].dropna().values
    
        results = []
        for value in non_nan_values:
            results.append(detect_prob(value))
    
        # Add results to corresponding list in temporary dictionary
        temp_results[f'Stage2_{base_column_name}_prob'].append(pd.Series(results, index=non_nan_indices))
    
    # Calculate mean probabilities for each imaging type
    for key, value in temp_results.items():
        if value:
            stage1_high_df[key + '_mean'] = pd.concat(value, axis=1).mean(axis=1, skipna=True)
    
    # Calculate maximum probability across all imaging types
    stage1_high_df['Stage2_prob'] = stage1_high_df[[f'Stage2_{t[1]}_prob_mean' for t in IMAGING_TYPES]].max(axis=1, skipna=True)
    stage1_high_df['Stage2_prob'].fillna(0, inplace=True)
    
    # Calculate entropy
    probs = stage1_high_df['Stage2_prob'].values
    entropies = calculate_entropy(probs)
    stage1_high_df['Stage2_entropy'] = entropies
    
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
    
    # Group based on entropy values
    Stage2_entropy_list = stage1_high_df['Stage2_entropy'].values
    cutoff_value = stage1_high_df['Stage2_entropy'].median()
    high, low = uc_grouping(Stage2_entropy_list, cutoff=cutoff_value)
    
    Stage2_high = stage1_high_df.loc[high].reset_index(drop=True)
    Stage2_low = stage1_high_df.loc[low].reset_index(drop=True)
    
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
    
    # Using Stage 1 results only
    stage1_low_df['Stage2_prob'] = stage1_low_df['Stage1_prob']
    combined_df_1 = pd.concat([stage1_low_df, stage1_high_df], ignore_index=True)
    combined_df_1.to_excel(os.path.join(output_dir, 'Stage2_result.xlsx'), index=False)
    
    # Using Stage 1, 2, 3 results
    stage1_low_df['Stage3_prob'] = stage1_low_df['Stage1_prob']
    stage2_low_df['Stage3_prob'] = stage2_low_df['Stage2_prob']
    stage2_high_df['Stage3_prob'] = stage2_high_df['Label']  # Use final label
    combined_df_2 = pd.concat([stage1_low_df, stage2_low_df, stage2_high_df], ignore_index=True)
    combined_df_2.to_excel(os.path.join(output_dir, 'Stage3_result.xlsx'), index=False)
    
    return combined_df_1, combined_df_2 