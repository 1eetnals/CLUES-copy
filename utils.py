import pandas as pd
import os
import numpy as np
import re 
import json
import openai
import asyncio
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv

from tqdm import tqdm
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from rank_bm25 import BM25Okapi   # lexical search; sparse vector filter
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, cohen_kappa_score 

warnings.filterwarnings("ignore")

# OpenAI API key setup function
def set_openai_api_key(api_key=None):
    # Load .env file
    load_dotenv()
    
    if api_key is not None:
        os.environ['OPENAI_API_KEY'] = api_key
    elif 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("OpenAI API key not found. Please provide it in .env file or as a command line argument.")
    
    return os.environ.get('OPENAI_API_KEY')

# Prompt templates
ZEROSHOT_PROMPT = """
다음과 같은 {imaging_report}이/가 Input으로 주어졌을 때, 
차트 리뷰를 하여 환자가 {disease} 환자인지 판단합니다.

<판단> 에서 {disease} 환자일 확률을 0과 1사이의 값으로 나타냅니다.
논리적으로 생각하고 <결정 근거>에서 자신의 결정에 대한 추론을 제공합니다.

Output에는 다음 형식을 사용합니다.
<판단> : (0과 1사이의 값만 해당) 
<결정근거> : 

Input : {input_text}
Output :
"""

FEWSHOT_PROMPT = """
다음과 같은 {imaging_report}이/가 Input으로 주어졌을 때, 
예시들을 참고해 차트 리뷰를 하여 환자가 {disease} 환자인지 판단합니다.
예시들은 {imaging_report}과/와 판단입니다. 

예시 : {fewshot_retrieval}

<판단> 에서 {disease} 환자일 확률을 0과 1사이의 값으로 나타냅니다.
논리적으로 생각하고 <결정 근거>에서 자신의 결정에 대한 추론을 제공합니다.

Output에는 다음 형식을 사용합니다.
<판단> : (0과 1사이의 값만 해당) 
<결정근거> : 

Input : {input_text}
Output :
"""

# ChatGPT API call function
async def chat_completion(input_prompt, model='gpt-4-turbo-preview'):
    client = AsyncOpenAI()
    
    SYSTEM_PROMPT = "당신은 의료 AI 언어 모델입니다."
    USER_PROMPT_1 = """당신의 역할에 대해 명확합니까?"""
    ASSISTANT_PROMPT_1 = """네. 저는 환자를 직접 치료하지 않습니다. 하지만 의료 전문가를 도와드릴 준비가 되어있습니다. 시작하는 데 필요한 정보를 알려주세요."""

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_1},
            {"role": "assistant", "content": ASSISTANT_PROMPT_1},
            {"role": "user", "content": input_prompt}
        ],
        temperature=0
    )
    return response

# Probability extraction function
def detect_prob(a_value):
    pattern = r'\d+(\.\d+)?'  
    match = re.search(pattern, str(a_value))

    if match:
        return float(match.group())
    elif 'Output: ' in a_value:
        try:
            return float(a_value.split('Output: ')[1])
        except ValueError:
            return np.nan
    else:
        try:
            return float(a_value)
        except ValueError:
            return np.nan
        
# Entropy calculation function
def calculate_entropy(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)  # prevent log(0) 
    return - (p * np.log(p) + (1 - p) * np.log(1 - p))   #binary cross entropy  

# Uncertainty grouping function
def uc_grouping(entropy_list, cutoff):
    high_idx = []
    low_idx = []
    for idx, entropy in enumerate(entropy_list):
        if entropy >= cutoff:
            high_idx.append(idx)
        else:
            low_idx.append(idx)
            
    return high_idx, low_idx

# Data file path
DATA_FILE = 'Data.xlsx'

# Imaging types and corresponding column names - modified to match actual file columns
IMAGING_TYPES = [
    ('MRI', 'MRI'),
    ('CT', 'CT'),
    ('ANG', 'ANG')
]

# 관심 질병
DISEASE_OF_INTEREST = '급성 뇌졸중' 

# Output directory setting
OUTPUT_DIR = 'output'

def ensure_output_dir():
    """
    Create output directory if it doesn't exist.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    return OUTPUT_DIR

def check_device():
    """
    Check and return the best available device.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def setup_device():
    """
    Set up the best available device in the system.
    """
    device = check_device()
    if device == "cuda":
        print("Using CUDA GPU.")
        # CUDA settings
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    elif device == "mps":
        print("Using Apple Silicon GPU (MPS).")
    else:
        print("Using CPU.")
    
    return device 