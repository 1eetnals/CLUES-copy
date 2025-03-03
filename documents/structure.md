# CLUES - File Structure

## Project Structure

```
.
├── main.py                                 # Main execution script
├── utils.py                                # Utility functions and constants
├── preprocessing.py                        # Data preprocessing functions
├── stage1.py                               # Stage 1 (Zero-Shot + Ensemble) implementation
├── stage2.py                               # Stage 2 (Few-Shot + RAG) implementation
├── evaluation.py                           # Evaluation functions
├── requirements.txt                        # Package requirements
├── README.md                               # Project documentation
├── Data.xlsx                               # Input data file
├── documents/                              # Documentation folder
│   ├── structure.md                        # File structure documentation
│   └── architecture.md                     # Architecture documentation
├── notebooks/                              # Notebooks
│   └── code.ipynb                          # total codes in ipynb file
└── output/                                 # Output results folder (auto-generated)
    ├── Stage1_result.xlsx                  # Stage 1 results file
    ├── Stage1_high_entropygroup.xlsx       # Stage 1 high entropy group
    ├── Stage1_low_entropygroup.xlsx        # Stage 1 low entropy group
    ├── MRI.csv                             # MRI RAG dataset
    ├── CT.csv                              # CT RAG dataset
    ├── ANG.csv                             # Angiography RAG dataset
    ├── Stage2_result.xlsx                  # Stage 2 results file
    ├── Stage2_high_entropygroup.xlsx       # Stage 2 high entropy group
    ├── Stage2_low_entropygroup.xlsx        # Stage 2 low entropy group
    └── Stage3_result.xlsx                  # Stage 3 results file
```

## Key Modules and Functions

### utils.py
Defines utility functions and global constants.

- `set_openai_api_key(api_key=None)`: Set OpenAI API key
- `chat_completion(input_prompt, model='gpt-4-turbo-preview')`: Asynchronous ChatGPT API call
- `detect_prob(a_value)`: Extract probability value from text
- `calculate_entropy(p)`: Calculate entropy for probability
- `uc_grouping(entropy_list, cutoff)`: Entropy-based grouping

### preprocessing.py
Provides data loading and preprocessing functions.

- `load_data(filepath)`: Load data file
- `remove_special_characters(text)`: Remove special characters
- `preprocess_data(df)`: Preprocess data
- `prepare_rag_datasets(low_entropy_df)`: Prepare RAG training data

### stage1.py
Implements Zero-Shot learning and ensemble processing.

- `zeroshot_run_async(main_prompt, information, imaging_report, doi)`: Zero-shot asynchronous execution
- `process_stage1(df, batch_size=15)`: Perform Stage 1 processing
- `split_stage1_by_entropy(df)`: Split results based on entropy

### stage2.py
Implements Few-Shot learning and RAG (Retrieval-Augmented Generation) processing.

- `setup_retrieval_system(imaging_type, openai_api_key)`: Set up retrieval system
- `retrieve_shots(query, retriever, k)`: Retrieve relevant shots
- `fewshot_run_async(main_prompt, information, imaging_report, doi, retriever, k)`: Few-shot asynchronous execution
- `process_stage2(stage1_high_df, openai_api_key, batch_size=15)`: Perform Stage 2 processing
- `split_stage2_by_entropy(stage1_high_df)`: Split results based on entropy
- `combine_results(stage1_low_df, stage1_high_df, stage2_high_df, stage2_low_df)`: Combine results

### evaluation.py
Provides model evaluation functions.

- `evaluate_binary_classification(df, prob_column)`: Binary classification evaluation
- `evaluate_multiclass_classification(df, prob_column, lower_bound=0.45, upper_bound=0.55)`: Multi-class classification evaluation
- `evaluate_all_datasets(evaluation_datasets)`: Evaluate all datasets
- `print_evaluation_results(binary_results, multiclass_results)`: Print evaluation results

### main.py
Entry point of the program that coordinates the overall processing flow.

- `main()`: Main execution function
