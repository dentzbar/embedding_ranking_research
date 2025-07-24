#!/usr/bin/env python3
"""
Utility functions for text embedding compression research.
"""

import pickle
import json
import numpy as np
from typing import Dict, List, Union, Optional
import os
from transformers import BertTokenizer
import tiktoken
import re


def load_ranking_dict(filepath: str) -> Dict[Union[str, int], int]:
    """
    Load a saved ranking dictionary from pickle or json file.
    
    Args:
        filepath: Path to the saved ranking file
        
    Returns:
        Dictionary with vocabulary items as keys and ranks as values
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Ranking file not found: {filepath}")
    
    file_ext = os.path.splitext(filepath)[1].lower()
    
    if file_ext == '.pkl':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif file_ext == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Try to convert string keys back to appropriate types
        converted_data = {}
        for k, v in data.items():
            # Try to convert to int if possible (for token IDs)
            try:
                key = int(k)
            except ValueError:
                key = k  # Keep as string
            converted_data[key] = v
        return converted_data
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .pkl or .json")


def tokenize_text_bert(text: str, model_name: str = 'bert-base-uncased') -> List[str]:
    """
    Tokenize text using BERT tokenizer.
    
    Args:
        text: Input text to tokenize
        model_name: BERT model name
        
    Returns:
        List of tokens
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(text)
    return tokens


def tokenize_text_openai(text: str, encoding_name: str = 'cl100k_base') -> List[str]:
    """
    Tokenize text using OpenAI tokenizer.
    
    Args:
        text: Input text to tokenize
        encoding_name: Encoding name for tiktoken
        
    Returns:
        List of token strings
    """
    encoding = tiktoken.get_encoding(encoding_name)
    token_ids = encoding.encode(text)
    
    tokens = []
    for token_id in token_ids:
        try:
            token_bytes = encoding.decode_single_token_bytes(token_id)
            token_str = token_bytes.decode('utf-8', errors='ignore')
            tokens.append(token_str)
        except:
            tokens.append(f"<token_{token_id}>")
    
    return tokens


def get_text_ranks(ranking_dict_path: str, input_text: str, 
                   model_type: str = 'bert', model_name: Optional[str] = None) -> List[int]:
    """
    Get ranks for tokens in input text based on saved ranking dictionary.
    
    Args:
        ranking_dict_path: Path to saved ranking dictionary
        input_text: Text to analyze
        model_type: Type of model used ('bert' or 'openai')
        model_name: Specific model name (optional)
        
    Returns:
        List of ranks corresponding to tokens in the text
    """
    # Load the ranking dictionary
    ranking_dict = load_ranking_dict(ranking_dict_path)
    
    # Tokenize the input text based on model type
    if model_type.lower() == 'bert':
        if model_name is None:
            model_name = 'bert-base-uncased'
        tokens = tokenize_text_bert(input_text, model_name)
    elif model_type.lower() == 'openai':
        if model_name is None:
            encoding_name = 'cl100k_base'
        else:
            # Map model names to encodings
            if 'gpt-3.5' in model_name or 'gpt-4' in model_name:
                encoding = tiktoken.encoding_for_model(model_name)
                encoding_name = encoding.name
            else:
                encoding_name = 'cl100k_base'
        tokens = tokenize_text_openai(input_text, encoding_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Get ranks for each token
    ranks = []
    missing_tokens = []
    
    for token in tokens:
        if token in ranking_dict:
            ranks.append(ranking_dict[token])
        else:
            # Token not found in ranking dictionary
            ranks.append(None)  # or some default value
            missing_tokens.append(token)
    
    if missing_tokens:
        print(f"Warning: {len(missing_tokens)} tokens not found in ranking dictionary:")
        for token in missing_tokens[:10]:  # Show first 10 missing tokens
            print(f"  '{token}'")
        if len(missing_tokens) > 10:
            print(f"  ... and {len(missing_tokens) - 10} more")
    
    return ranks


def analyze_text_compression(ranking_dict_path: str, input_text: str, 
                           model_type: str = 'bert', model_name: Optional[str] = None,
                           show_details: bool = False) -> Dict:
    """
    Analyze text compression potential based on token ranks.
    
    Args:
        ranking_dict_path: Path to saved ranking dictionary
        input_text: Text to analyze
        model_type: Type of model used ('bert' or 'openai')
        model_name: Specific model name (optional)
        show_details: Whether to show detailed token analysis
        
    Returns:
        Dictionary with analysis results
    """
    # Get token ranks
    if model_type.lower() == 'bert':
        if model_name is None:
            model_name = 'bert-base-uncased'
        tokens = tokenize_text_bert(input_text, model_name)
    elif model_type.lower() == 'openai':
        if model_name is None:
            encoding_name = 'cl100k_base'
        else:
            if 'gpt-3.5' in model_name or 'gpt-4' in model_name:
                encoding = tiktoken.encoding_for_model(model_name)
                encoding_name = encoding.name
            else:
                encoding_name = 'cl100k_base'
        tokens = tokenize_text_openai(input_text, encoding_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    ranks = get_text_ranks(ranking_dict_path, input_text, model_type, model_name)
    
    # Filter out None values for analysis
    valid_ranks = [r for r in ranks if r is not None]
    
    if not valid_ranks:
        return {
            'error': 'No valid ranks found',
            'total_tokens': len(tokens),
            'missing_tokens': len(tokens)
        }
    
    # Compute statistics
    analysis = {
        'text_length_chars': len(input_text),
        'total_tokens': len(tokens),
        'valid_ranks': len(valid_ranks),
        'missing_tokens': len(tokens) - len(valid_ranks),
        'mean_rank': np.mean(valid_ranks),
        'median_rank': np.median(valid_ranks),
        'min_rank': min(valid_ranks),
        'max_rank': max(valid_ranks),
        'std_rank': np.std(valid_ranks),
        'tokens': tokens if show_details else None,
        'ranks': ranks if show_details else None
    }
    
    # Compression potential analysis
    # Lower ranks indicate more common/important tokens
    low_rank_threshold = np.percentile(valid_ranks, 25)  # Bottom 25%
    high_rank_threshold = np.percentile(valid_ranks, 75)  # Top 25%
    
    low_rank_count = sum(1 for r in valid_ranks if r <= low_rank_threshold)
    high_rank_count = sum(1 for r in valid_ranks if r >= high_rank_threshold)
    
    analysis.update({
        'low_rank_tokens': low_rank_count,
        'high_rank_tokens': high_rank_count,
        'compression_potential': high_rank_count / len(valid_ranks) if valid_ranks else 0,
        'low_rank_threshold': low_rank_threshold,
        'high_rank_threshold': high_rank_threshold
    })
    
    return analysis


def compare_rankings(ranking_dict_paths: List[str], input_text: str, 
                    model_types: List[str], model_names: Optional[List[str]] = None) -> Dict:
    """
    Compare text analysis across multiple ranking dictionaries.
    
    Args:
        ranking_dict_paths: List of paths to ranking dictionaries
        input_text: Text to analyze
        model_types: List of model types corresponding to each ranking dict
        model_names: List of model names (optional)
        
    Returns:
        Dictionary with comparison results
    """
    if model_names is None:
        model_names = [None] * len(ranking_dict_paths)
    
    if len(ranking_dict_paths) != len(model_types) or len(ranking_dict_paths) != len(model_names):
        raise ValueError("All input lists must have the same length")
    
    results = {}
    
    for i, (dict_path, model_type, model_name) in enumerate(zip(ranking_dict_paths, model_types, model_names)):
        try:
            analysis = analyze_text_compression(dict_path, input_text, model_type, model_name)
            results[f"ranking_{i}_{model_type}"] = analysis
        except Exception as e:
            results[f"ranking_{i}_{model_type}"] = {'error': str(e)}
    
    return results


def find_ranking_files(directory: str = '.') -> List[str]:
    """
    Find all ranking files in a directory.
    
    Args:
        directory: Directory to search in
        
    Returns:
        List of ranking file paths
    """
    ranking_files = []
    
    for filename in os.listdir(directory):
        if ('_ranking.' in filename and 
            (filename.endswith('.pkl') or filename.endswith('.json'))):
            ranking_files.append(os.path.join(directory, filename))
    
    return sorted(ranking_files)


def extract_reference_word_from_filename(filepath: str) -> Optional[str]:
    """
    Extract the reference word from a ranking file name.
    
    Args:
        filepath: Path to ranking file
        
    Returns:
        Reference word or None if not found
    """
    filename = os.path.basename(filepath)
    
    # Pattern: {model}_{word}_ranking.{ext}
    match = re.search(r'([^_]+)_(.+)_ranking\.(pkl|json)$', filename)
    if match:
        return match.group(2)
    
    return None


def batch_analyze_texts(ranking_dict_path: str, texts: List[str], 
                       model_type: str = 'bert', model_name: Optional[str] = None) -> List[Dict]:
    """
    Analyze multiple texts in batch.
    
    Args:
        ranking_dict_path: Path to ranking dictionary
        texts: List of texts to analyze
        model_type: Model type
        model_name: Model name (optional)
        
    Returns:
        List of analysis results
    """
    results = []
    
    for i, text in enumerate(texts):
        try:
            analysis = analyze_text_compression(ranking_dict_path, text, model_type, model_name)
            analysis['text_id'] = i
            analysis['text_preview'] = text[:100] + ('...' if len(text) > 100 else '')
            results.append(analysis)
        except Exception as e:
            results.append({
                'text_id': i,
                'text_preview': text[:100] + ('...' if len(text) > 100 else ''),
                'error': str(e)
            })
    
    return results


# Example usage functions
def demo_text_analysis():
    """Demonstration of text analysis functionality"""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing is a fascinating field of study.",
        "Machine learning models can compress text efficiently.",
        "Supercalifragilisticexpialidocious is a very long word."
    ]
    
    print("=== Text Analysis Demo ===")
    
    # Find available ranking files
    ranking_files = find_ranking_files()
    
    if not ranking_files:
        print("No ranking files found. Please run the main script first to generate rankings.")
        return
    
    print(f"Found {len(ranking_files)} ranking files:")
    for i, filepath in enumerate(ranking_files):
        ref_word = extract_reference_word_from_filename(filepath)
        print(f"  {i}: {filepath} (reference: {ref_word})")
    
    # Use the first ranking file for demo
    ranking_file = ranking_files[0]
    ref_word = extract_reference_word_from_filename(ranking_file)
    
    print(f"\nUsing ranking file: {ranking_file}")
    print(f"Reference word: {ref_word}")
    
    # Determine model type from filename
    model_type = 'bert' if 'bert' in os.path.basename(ranking_file).lower() else 'openai'
    
    print(f"\nAnalyzing {len(sample_texts)} sample texts:")
    
    for i, text in enumerate(sample_texts):
        print(f"\n--- Text {i+1}: {text[:50]}... ---")
        try:
            analysis = analyze_text_compression(ranking_file, text, model_type)
            print(f"Tokens: {analysis['total_tokens']}")
            print(f"Mean rank: {analysis['mean_rank']:.1f}")
            print(f"Compression potential: {analysis['compression_potential']:.2%}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    demo_text_analysis() 