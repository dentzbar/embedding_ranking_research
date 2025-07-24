#!/usr/bin/env python3
"""
Main script for running embedding-based vocabulary ranking.
Example usage:
    rank_vocabulary("dog", model="bert", output_format="json")
    rank_vocabulary("cat", model="openai", output_format="pickle")
"""

import os
from embedding_ranker import BertEmbeddingRanker, OpenAIEmbeddingRanker


def rank_vocabulary(word, model="bert", output_format="pickle", bert_model="bert-base-uncased", openai_model="text-embedding-3-small"):
    """
    Rank vocabulary by similarity to a reference word.
    
    Args:
        word (str): Reference word to compare against
        model (str): Either "bert" or "openai"
        output_format (str): Either "pickle" or "json"
        bert_model (str): Name of BERT model to use
        openai_model (str): Name of OpenAI model to use
    
    Returns:
        str: Path to saved ranking file
    """
    print(f"Starting vocabulary ranking for word: '{word}'")
    print(f"Using model: {model}")
    print(f"Output format: {output_format}")
    
    try:
        # Initialize the appropriate ranker
        if model == 'bert':
            print(f"Initializing BERT ranker with model: {bert_model}")
            ranker = BertEmbeddingRanker(model_name=bert_model)
        elif model == 'openai':
            print(f"Initializing OpenAI ranker with model: {openai_model}")
            if not os.getenv('OPENAI_API_KEY'):
                print("Warning: OPENAI_API_KEY environment variable not set!")
                print("You may encounter API errors.")
            ranker = OpenAIEmbeddingRanker(model_name=openai_model)
        else:
            raise ValueError(f"Invalid model: {model}. Must be 'bert' or 'openai'")
        
        # Rank vocabulary
        print("\nStarting vocabulary ranking process...")
        ranking_dict = ranker.rank_vocabulary(word)
        
        # Save results
        print(f"\nSaving results in {output_format} format...")
        filename = ranker.save_ranking(ranking_dict, word, format=output_format)
        
        print(f"\n✓ Successfully completed!")
        print(f"✓ Vocabulary size: {ranker.vocab_size}")
        print(f"✓ Results saved to: {filename}")
        print(f"✓ Dictionary size: {len(ranking_dict)} items")
        
        # Show top 10 results
        print(f"\nTop 10 most similar tokens to '{word}':")
        sorted_items = sorted(ranking_dict.items(), key=lambda x: x[1])
        for i, (token, rank) in enumerate(sorted_items[:10]):
            print(f"  {rank:4d}. {token}")
            
        return filename
            
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Example usage
    rank_vocabulary("dog", model="bert", output_format="json")
    rank_vocabulary("cat", model="openai", output_format="pickle")