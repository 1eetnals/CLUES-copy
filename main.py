import asyncio
import argparse
import pandas as pd
import numpy as np
import os
from utils import (
    set_openai_api_key,
    DATA_FILE,
    DISEASE_OF_INTEREST,
    IMAGING_TYPES,
    setup_device,
    ensure_output_dir
)
from preprocessing import load_data, preprocess_data, prepare_rag_datasets
from stage1 import process_stage1, split_stage1_by_entropy
from stage2 import process_stage2, split_stage2_by_entropy, combine_results
from evaluation import evaluate_all_datasets, print_evaluation_results

async def main():
    """
    Main execution function.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Acute Stroke Patient Classification System')
    parser.add_argument('--api_key', type=str, help='OpenAI API key')
    parser.add_argument('--data_file', type=str, default=DATA_FILE, help='Data file path')
    parser.add_argument('--batch_size', type=int, default=15, help='Batch processing size')
    parser.add_argument('--skip_stage1', action='store_true', help='Skip Stage 1 processing')
    parser.add_argument('--skip_stage2', action='store_true', help='Skip Stage 2 processing')
    parser.add_argument('--evaluate_only', action='store_true', help='Perform evaluation only')
    parser.add_argument('--sample_size', type=int, help='Sample size to process (e.g., 30)')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')
    args = parser.parse_args()
    
    # Set OpenAI API key
    openai_api_key = set_openai_api_key(args.api_key)
    
    # GPU setup (if requested)
    if args.use_gpu:
        device = setup_device()
    else:
        device = "cpu"
        print("Using CPU. Add --use_gpu option to use GPU.")
    
    # Create output directory
    output_dir = ensure_output_dir()
    
    if not args.evaluate_only:
        # Load and preprocess data
        print(f"Loading data file '{args.data_file}'...")
        df = load_data(args.data_file)
        
        # Sampling (if requested)
        if args.sample_size:
            if args.sample_size < len(df):
                df = df.head(args.sample_size)
                print(f"Using first {args.sample_size} entries only.")
            else:
                print(f"Requested sample size ({args.sample_size}) is larger than total data size ({len(df)}). Using all data.")
        
        df = preprocess_data(df)
        print("Data loading and preprocessing complete!")
        print(f"Data size to process: {len(df)} entries")
        
        if not args.skip_stage1:
            # Stage 1 processing
            print("\n===== Stage 1: Zero-Shot + Ensemble =====")
            df = await process_stage1(df, args.batch_size)
            stage1_high, stage1_low = split_stage1_by_entropy(df)
            print("Stage 1 processing complete!")
            
            # Prepare RAG datasets
            print("\nPreparing RAG datasets...")
            prepare_rag_datasets(stage1_low)
            print("RAG dataset preparation complete!")
        else:
            # Load existing results
            print("\nLoading existing Stage 1 results...")
            df = pd.read_excel('Stage1_result.xlsx')
            stage1_high = pd.read_excel('Stage1_high_entropygroup.xlsx')
            stage1_low = pd.read_excel('Stage1_low_entropygroup.xlsx')
            print("Stage 1 results loaded!")
        
        if not args.skip_stage2:
            # Stage 2 processing
            print("\n===== Stage 2: Few-Shot + RAG =====")
            stage1_high = await process_stage2(stage1_high, openai_api_key, args.batch_size)
            stage2_high, stage2_low = split_stage2_by_entropy(stage1_high)
            
            # Combine results
            print("\nCombining results...")
            stage2_result, stage3_result = combine_results(stage1_low, stage1_high, stage2_high, stage2_low)
            print("Stage 2 and result combination complete!")
        else:
            # Load existing results
            print("\nLoading existing Stage 2 results...")
            stage2_high = pd.read_excel('Stage2_high_entropygroup.xlsx')
            stage2_low = pd.read_excel('Stage2_low_entropygroup.xlsx')
            print("Stage 2 results loaded!")
    
    # Perform evaluation
    print("\n===== Performing Evaluation =====")
    evaluation_datasets = [
        ('Stage1_low_entropygroup.xlsx', 'Stage1_prob'),
        ('Stage1_high_entropygroup.xlsx', 'Stage1_prob'),
        ('Stage2_low_entropygroup.xlsx', 'Stage2_prob'),
        ('Stage2_high_entropygroup.xlsx', 'Stage2_prob'),
        ('Stage1_result.xlsx', 'Stage1_prob'),
        ('Stage2_result.xlsx', 'Stage2_prob'),
        ('Stage3_result.xlsx', 'Stage3_prob'),
    ]
    
    binary_results, multiclass_results = evaluate_all_datasets(evaluation_datasets)
    print_evaluation_results(binary_results, multiclass_results)
    print("\nEvaluation complete!")

if __name__ == "__main__":
    asyncio.run(main()) 