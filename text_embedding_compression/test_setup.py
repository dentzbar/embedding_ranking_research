#!/usr/bin/env python3
"""
Test script to validate the text embedding compression setup.
Run this to check if all components are working correctly.
"""

import sys
import os
import traceback


def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError as e:
        print(f"  ❌ numpy: {e}")
        return False
        
    try:
        import pandas as pd
        print("  ✓ pandas")
    except ImportError as e:
        print(f"  ❌ pandas: {e}")
        return False
        
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        print("  ✓ scikit-learn")
    except ImportError as e:
        print(f"  ❌ scikit-learn: {e}")
        return False
        
    try:
        import torch
        print("  ✓ torch")
    except ImportError as e:
        print(f"  ❌ torch: {e}")
        return False
        
    try:
        from transformers import BertTokenizer, BertModel
        print("  ✓ transformers")
    except ImportError as e:
        print(f"  ❌ transformers: {e}")
        return False
        
    try:
        import tiktoken
        print("  ✓ tiktoken")
    except ImportError as e:
        print(f"  ❌ tiktoken: {e}")
        return False
        
    try:
        from embedding_ranker import EmbeddingRanker, BertEmbeddingRanker, OpenAIEmbeddingRanker
        print("  ✓ embedding_ranker")
    except ImportError as e:
        print(f"  ❌ embedding_ranker: {e}")
        return False
        
    try:
        from utils import load_ranking_dict, get_text_ranks, analyze_text_compression
        print("  ✓ utils")
    except ImportError as e:
        print(f"  ❌ utils: {e}")
        return False
        
    return True


def test_bert_basic():
    """Test basic BERT functionality"""
    print("\n🤖 Testing BERT functionality...")
    
    try:
        from embedding_ranker import BertEmbeddingRanker
        
        # Initialize BERT ranker (this will download the model if not cached)
        print("  Initializing BERT ranker...")
        ranker = BertEmbeddingRanker()
        
        print(f"  ✓ BERT model loaded: {ranker.model_name}")
        print(f"  ✓ Vocabulary size: {ranker.vocab_size}")
        
        # Test word embedding
        print("  Testing word embedding...")
        test_word = "test"
        embedding = ranker.get_word_embedding(test_word)
        print(f"  ✓ Word '{test_word}' embedding shape: {embedding.shape}")
        
        # Test vocabulary embeddings (just check shape, don't compute all)
        print("  Testing vocabulary embeddings...")
        vocab_embeddings = ranker.get_vocab_embeddings()
        print(f"  ✓ Vocabulary embeddings shape: {vocab_embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ BERT test failed: {e}")
        traceback.print_exc()
        return False


def test_openai_basic():
    """Test basic OpenAI functionality (without API calls)"""
    print("\n🌐 Testing OpenAI functionality...")
    
    try:
        from embedding_ranker import OpenAIEmbeddingRanker
        
        # Initialize OpenAI ranker (without making API calls)
        print("  Initializing OpenAI ranker...")
        ranker = OpenAIEmbeddingRanker()
        
        print(f"  ✓ OpenAI model: {ranker.model_name}")
        print(f"  ✓ Vocabulary size: {ranker.vocab_size}")
        
        # Test tokenization
        print("  Testing tokenization...")
        test_text = "Hello world"
        tokens = ranker.encoding.encode(test_text)
        print(f"  ✓ Text '{test_text}' tokenized to {len(tokens)} tokens: {tokens}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ OpenAI test failed: {e}")
        traceback.print_exc()
        return False


def test_utils():
    """Test utility functions"""
    print("\n🛠️ Testing utility functions...")
    
    try:
        from utils import tokenize_text_bert, tokenize_text_openai, find_ranking_files
        
        # Test BERT tokenization
        test_text = "Hello world"
        bert_tokens = tokenize_text_bert(test_text)
        print(f"  ✓ BERT tokenization: '{test_text}' -> {bert_tokens}")
        
        # Test OpenAI tokenization
        openai_tokens = tokenize_text_openai(test_text)
        print(f"  ✓ OpenAI tokenization: '{test_text}' -> {openai_tokens[:5]}{'...' if len(openai_tokens) > 5 else ''}")
        
        # Test finding ranking files
        ranking_files = find_ranking_files('.')
        print(f"  ✓ Found {len(ranking_files)} ranking files")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Utils test failed: {e}")
        traceback.print_exc()
        return False


def test_file_structure():
    """Test if all required files exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        'embedding_ranker.py',
        'run.py', 
        'utils.py',
        'analysis.ipynb',
        'requirements.txt',
        'README.md'
    ]
    
    all_exist = True
    for filename in required_files:
        if os.path.exists(filename):
            print(f"  ✓ {filename}")
        else:
            print(f"  ❌ {filename} - Missing!")
            all_exist = False
            
    return all_exist


def main():
    """Run all tests"""
    print("=" * 50)
    print("🧪 Text Embedding Compression Setup Test")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("BERT Basic", test_bert_basic),
        ("OpenAI Basic", test_openai_basic),
        ("Utils", test_utils)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("  1. Run: python run.py --word 'machine' --model bert")
        print("  2. Open: jupyter notebook analysis.ipynb")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Set OpenAI API key: export OPENAI_API_KEY='your-key'")
        print("  3. Check internet connection for model downloads")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 