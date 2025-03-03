# CLUES - Architecture

## System Overview

The Acute Stroke Patient Classification System is a modular system that analyzes medical imaging reports to classify patients. This system integrates Zero-Shot learning, Few-Shot learning, RAG (Retrieval-Augmented Generation), and ensemble techniques to achieve accurate patient classification.

## Main Module Flow

```mermaid
graph TD
    A[main.py] --> B[Data Loading - preprocessing.py]
    B --> C[Stage 1 Processing - stage1.py]
    C --> D{Entropy-based Classification}
    D -->|Low Entropy| E[Confirmed Results]
    D -->|High Entropy| F[RAG Preparation - preprocessing.py]
    F --> G[Stage 2 Processing - stage2.py]
    G --> H{Entropy-based Classification}
    H -->|Low Entropy| I[Confirmed Results]
    H -->|High Entropy| J[Needs Further Review]
    E --> K[Result Combination - stage2.py]
    I --> K
    J --> K
    K --> L[Evaluation - evaluation.py]
```

## Detailed Module Architecture

### 1. Data Processing Pipeline

```mermaid
graph LR
    A[Data.xlsx] --> B[load_data]
    B --> C[preprocess_data]
    C --> D[Report Data]
    
    subgraph preprocessing.py
        B
        C
    end
```

### 2. Stage 1: Zero-Shot + Ensemble Processing

```mermaid
graph TD
    A[Report Data] --> B[zeroshot_run_async]
    B --> C[process_stage1]
    C --> D[Calculate Probabilities and Entropy]
    D --> E[split_stage1_by_entropy]
    E --> F[Stage1_high_entropygroup.xlsx]
    E --> G[Stage1_low_entropygroup.xlsx]
    
    subgraph stage1.py
        B
        C
        D
        E
    end
    
    subgraph utils.py
        H[chat_completion] --> B
        I[detect_prob] --> D
        J[calculate_entropy] --> D
        K[uc_grouping] --> E
    end
```

### 3. RAG Dataset Preparation

```mermaid
graph LR
    A[Stage1_low_entropygroup.xlsx] --> B[prepare_rag_datasets]
    B --> C[MRI.csv]
    B --> D[CT.csv]
    B --> E[ANG.csv]
    
    subgraph preprocessing.py
        B
    end
```

### 4. Stage 2: Few-Shot + RAG Processing

```mermaid
graph TD
    A[Stage1_high_entropygroup.xlsx] --> B[setup_retrieval_system]
    B --> C[retrieve_shots]
    C --> D[fewshot_run_async]
    D --> E[process_stage2]
    E --> F[Calculate Probabilities and Entropy]
    F --> G[split_stage2_by_entropy]
    G --> H[Stage2_high_entropygroup.xlsx]
    G --> I[Stage2_low_entropygroup.xlsx]
    
    subgraph stage2.py
        B
        C
        D
        E
        F
        G
    end
    
    subgraph utils.py
        J[chat_completion] --> D
        K[detect_prob] --> F
        L[calculate_entropy] --> F
        M[uc_grouping] --> G
    end
```

### 5. Result Combination

```mermaid
graph TD
    A[Stage1_low_entropygroup.xlsx] --> D[combine_results]
    B[Stage2_low_entropygroup.xlsx] --> D
    C[Stage2_high_entropygroup.xlsx] --> D
    D --> E[Stage2_result.xlsx]
    D --> F[Stage3_result.xlsx]
    
    subgraph stage2.py
        D
    end
```

### 6. Evaluation System

```mermaid
graph TD
    A[Result Files] --> B[evaluate_binary_classification]
    A --> C[evaluate_multiclass_classification]
    B --> D[evaluate_all_datasets]
    C --> D
    D --> E[print_evaluation_results]
    
    subgraph evaluation.py
        B
        C
        D
        E
    end
```

### 7. Overall System Operation

```mermaid
sequenceDiagram
    participant Main as main.py
    participant Preproc as preprocessing.py
    participant Stage1 as stage1.py
    participant Stage2 as stage2.py
    participant Utils as utils.py
    participant Eval as evaluation.py
    
    Main->>Preproc: Load and preprocess data
    Preproc->>Main: Preprocessed data
    Main->>Stage1: Request Stage 1 processing
    Stage1->>Utils: Call ChatGPT API
    Utils->>Stage1: API response
    Stage1->>Main: Stage 1 results (entropy-based classification)
    Main->>Preproc: Request RAG dataset preparation
    Preproc->>Main: RAG dataset preparation complete
    Main->>Stage2: Request Stage 2 processing
    Stage2->>Stage2: Set up retrieval system
    Stage2->>Utils: Call ChatGPT API
    Utils->>Stage2: API response
    Stage2->>Main: Stage 2 results (entropy-based classification)
    Main->>Stage2: Request result combination
    Stage2->>Main: Combined results
    Main->>Eval: Request evaluation
    Eval->>Main: Evaluation results
```

## Technical Implementation Details

### OpenAI API Call Implementation

```mermaid
graph LR
    A[Input Prompt] --> B[chat_completion]
    B --> C[AsyncOpenAI Client]
    C --> D[ChatGPT API]
    D --> E[API Response]
    
    subgraph utils.py
        B
    end
```

### Entropy-based Uncertainty Classification

```mermaid
graph TD
    A[Probability Values] --> B[calculate_entropy]
    B --> C[Entropy Values]
    C --> D[uc_grouping]
    D --> E[High Entropy Group]
    D --> F[Low Entropy Group]
    
    subgraph utils.py
        B
        D
    end
```

### RAG System Implementation

```mermaid
graph TD
    A[CSV Data] --> B[CSVLoader]
    B --> C[Vector Embedding]
    C --> D[FAISS Vector Store]
    D --> E[Semantic Search]
    F[Text Data] --> G[BM25 Lexical Search]
    E --> H[EnsembleRetriever]
    G --> H
    H --> I[Search Results]
    
    subgraph stage2.py
        B
        C
        D
        E
        G
        H
    end
```
