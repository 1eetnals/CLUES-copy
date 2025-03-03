import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, cohen_kappa_score
import os
from utils import ensure_output_dir

def evaluate_binary_classification(df, prob_column):
    """
    Evaluate binary classification performance.
    
    Args:
        df (pd.DataFrame): Dataframe to evaluate
        prob_column (str): Column name containing probability values
        
    Returns:
        dict: Evaluation results
    """
    if len(df) == 0:
        return None
        
    y_true = df['Label'].values
    y_pred = (df[prob_column] >= 0.5).astype(int)
    
    # Set average for multi-class data handling
    average = 'macro' if len(np.unique(y_true)) > 2 else 'binary'
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    try:
        kappa = cohen_kappa_score(y_true, y_pred)
    except ValueError:
        kappa = 0  # For single class cases
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'kappa': kappa
    }

def evaluate_multiclass_classification(df, prob_column, lower_bound=0.45, upper_bound=0.55):
    """
    Evaluate multi-class classification performance.
    
    Args:
        df (pd.DataFrame): Dataframe to evaluate
        prob_column (str): Column name containing probability values
        lower_bound (float): Lower probability threshold
        upper_bound (float): Upper probability threshold
        
    Returns:
        str: Classification report
    """
    if len(df) == 0:
        return None
        
    # True labels
    y_true = df['Label'].values
    
    # Generate prediction labels (0: Negative, 1: Equivocal, 2: Positive)
    y_pred = np.zeros_like(y_true)
    probs = df[prob_column].values
    
    # Assign predictions
    y_pred[(probs >= lower_bound) & (probs <= upper_bound)] = 1  # Equivocal
    y_pred[probs > upper_bound] = 2  # Positive
    
    # Generate report only if labels exist
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    if len(unique_labels) > 0:
        report = classification_report(
            y_true, 
            y_pred, 
            labels=[0, 1, 2],
            target_names=['Negative', 'Equivocal', 'Positive'],
            output_dict=False,
            zero_division=0
        )
    else:
        report = "No valid labels found for evaluation"
    
    return report

def evaluate_all_datasets(evaluation_datasets):
    """
    Evaluate all datasets.
    
    Args:
        evaluation_datasets (list): List of (filename, probability column) tuples
        
    Returns:
        tuple: (binary_results, multiclass_results) Binary and multi-class classification results
    """
    output_dir = ensure_output_dir()
    binary_results = {}
    multiclass_results = {}
    
    for filename, prob_column in evaluation_datasets:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filename} not found, skipping evaluation.")
            continue
            
        df = pd.read_excel(filepath)
        
        # Binary classification evaluation
        binary_metrics = evaluate_binary_classification(df, prob_column)
        if binary_metrics:
            binary_results[filename] = binary_metrics
            
        # Multi-class classification evaluation
        multiclass_metrics = evaluate_multiclass_classification(df, prob_column)
        if multiclass_metrics:
            multiclass_results[filename] = multiclass_metrics
    
    return binary_results, multiclass_results

def print_evaluation_results(binary_results, multiclass_results):
    """
    Print evaluation results.
    
    Args:
        binary_results (dict): Binary classification evaluation results
        multiclass_results (dict): Multi-class classification evaluation results
    """
    print("\n=== Binary Classification Results ===")
    for filename, results in binary_results.items():
        if results is not None:
            print(f"\n{filename}:")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall: {results['recall']:.4f}")
            print(f"F1-score: {results['f1']:.4f}")
            print(f"Kappa: {results['kappa']:.4f}")
            print("\nConfusion Matrix:")
            print(results['confusion_matrix'])
    
    print("\n=== Multi-class Classification Results ===")
    for filename, report in multiclass_results.items():
        if report is not None:
            print(f"\n{filename}:")
            print(report) 