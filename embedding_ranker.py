import numpy as np
import pickle
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Tuple
import os
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel
import tiktoken
import openai
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingRanker(ABC):
    """
    Abstract base class for embedding-based vocabulary ranking.
    Given a reference word, ranks the full vocabulary by cosine similarity.
    """
    
    def __init__(self):
        self.vocab = None
        self.embeddings = None
        self.vocab_size = 0
        
    @abstractmethod
    def load_model_and_vocab(self):
        """Load the embedding model and vocabulary"""
        pass
    
    @abstractmethod
    def get_word_embedding(self, word: str) -> np.ndarray:
        """Get embedding for a single word"""
        pass
    
    @abstractmethod
    def get_vocab_embeddings(self) -> np.ndarray:
        """Get embeddings for the entire vocabulary"""
        pass
    
    def compute_cosine_similarities(self, reference_embedding: np.ndarray, vocab_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between reference and vocabulary embeddings"""
        # Reshape reference embedding for sklearn
        reference_embedding = reference_embedding.reshape(1, -1)
        
        # Compute cosine similarity
        similarities = cosine_similarity(reference_embedding, vocab_embeddings)[0]
        return similarities
    
    def rank_vocabulary(self, reference_word: str) -> Dict[Union[str, int], int]:
        """
        Rank vocabulary by cosine similarity to reference word.
        Returns dictionary with vocab items as keys and ranks as values.
        """
        print(f"Computing embeddings for reference word: {reference_word}")
        
        # Get reference word embedding
        reference_embedding = self.get_word_embedding(reference_word)
        
        print("Computing vocabulary embeddings...")
        # Get all vocabulary embeddings
        vocab_embeddings = self.get_vocab_embeddings()
        
        print("Computing cosine similarities...")
        # Compute similarities
        similarities = self.compute_cosine_similarities(reference_embedding, vocab_embeddings)
        
        print("Ranking vocabulary...")
        # Get sorted indices (descending order - highest similarity first)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Create ranking dictionary
        ranking_dict = {}
        for rank, vocab_idx in enumerate(sorted_indices):
            vocab_key = self.get_vocab_key(vocab_idx)
            ranking_dict[vocab_key] = rank + 1  # 1-indexed ranking
            
        return ranking_dict
    
    @abstractmethod
    def get_vocab_key(self, vocab_idx: int) -> Union[str, int]:
        """Get the key for vocabulary item (word or token ID)"""
        pass
    
    def save_ranking(self, ranking_dict: Dict, reference_word: str, format: str = 'pickle'):
        """Save ranking dictionary to file"""
        filename = f"{self.__class__.__name__.lower()}_{reference_word}_ranking"
        
        if format == 'pickle':
            filename += '.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(ranking_dict, f)
        elif format == 'json':
            filename += '.json'
            # Convert keys to strings for JSON compatibility
            json_dict = {str(k): v for k, v in ranking_dict.items()}
            with open(filename, 'w') as f:
                json.dump(json_dict, f, indent=2)
        else:
            raise ValueError("Format must be 'pickle' or 'json'")
            
        print(f"Ranking saved to {filename}")
        return filename


class BertEmbeddingRanker(EmbeddingRanker):
    """BERT-based embedding ranker"""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.load_model_and_vocab()
        
    def load_model_and_vocab(self):
        """Load BERT model and tokenizer"""
        print(f"Loading BERT model: {self.model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        self.model.eval()
        
        # Get vocabulary
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)
        print(f"Vocabulary size: {self.vocab_size}")
        
        # Create reverse mapping for token IDs to words
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
    def get_word_embedding(self, word: str) -> np.ndarray:
        """Get BERT embedding for a word"""
        # Tokenize the word
        tokens = self.tokenizer(word, return_tensors='pt', add_special_tokens=True)
        
        with torch.no_grad():
            outputs = self.model(**tokens)
            # Use the [CLS] token embedding or mean of all token embeddings
            # Here we use the mean of non-special tokens
            embeddings = outputs.last_hidden_state[0]  # Remove batch dimension
            
            # Find non-special tokens (exclude [CLS] and [SEP])
            special_tokens = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id}
            non_special_mask = ~torch.isin(tokens['input_ids'][0], torch.tensor(list(special_tokens)))
            
            if non_special_mask.sum() > 0:
                word_embedding = embeddings[non_special_mask].mean(dim=0)
            else:
                word_embedding = embeddings[0]  # Use [CLS] if no other tokens
                
        return word_embedding.numpy()
    
    def get_vocab_embeddings(self) -> np.ndarray:
        """Get embeddings for all vocabulary tokens"""
        # Use the token embeddings from the model
        token_embeddings = self.model.embeddings.word_embeddings.weight.data
        return token_embeddings.detach().numpy()
    
    def get_vocab_key(self, vocab_idx: int) -> str:
        """Return the token string as key"""
        return self.id_to_token[vocab_idx]


class OpenAIEmbeddingRanker(EmbeddingRanker):
    """OpenAI-based embedding ranker"""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        super().__init__()
        self.model_name = model_name
        self.encoding = None
        self.load_model_and_vocab()
        
    def load_model_and_vocab(self):
        """Load OpenAI tokenizer and setup"""
        print(f"Loading OpenAI model: {self.model_name}")
        
        # Get the appropriate encoding for the model
        if "gpt-3.5" in self.model_name or "gpt-4" in self.model_name:
            self.encoding = tiktoken.encoding_for_model(self.model_name)
        else:
            # For embedding models, use cl100k_base encoding
            self.encoding = tiktoken.get_encoding("cl100k_base")
            
        # Get vocabulary size
        self.vocab_size = self.encoding.n_vocab
        print(f"Vocabulary size: {self.vocab_size}")
        
        # Create vocabulary mapping
        self.vocab = {}
        self.id_to_token = {}
        
        # Note: tiktoken doesn't provide direct vocab access, so we'll work with token IDs
        for i in range(self.vocab_size):
            try:
                token_bytes = self.encoding.decode_single_token_bytes(i)
                token_str = token_bytes.decode('utf-8', errors='ignore')
                self.vocab[token_str] = i
                self.id_to_token[i] = token_str
            except:
                # Some tokens might not be decodable
                self.id_to_token[i] = f"<token_{i}>"
                
    def get_word_embedding(self, word: str) -> np.ndarray:
        """Get OpenAI embedding for a word"""
        try:
            import openai
            response = openai.embeddings.create(
                model=self.model_name,
                input=word
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Error getting OpenAI embedding: {e}")
            print("Note: You need to set your OpenAI API key and have openai package installed")
            # Return a dummy embedding for demonstration
            return np.random.normal(0, 1, 1536)  # text-embedding-3-small has 1536 dimensions
    
    def get_vocab_embeddings(self) -> np.ndarray:
        """
        Get embeddings for all vocabulary tokens.
        Note: This is computationally expensive for OpenAI API.
        In practice, you might want to cache this or use a subset.
        """
        print("Warning: Getting embeddings for entire vocabulary is expensive with OpenAI API")
        print("Consider using a cached version or subset for development")
        
        embeddings = []
        batch_size = 100  # Process in batches to avoid API limits
        
        tokens_to_embed = []
        valid_indices = []
        
        # Collect decodable tokens
        for i in range(min(1000, self.vocab_size)):  # Limit to first 1000 for demo
            if i in self.id_to_token:
                token = self.id_to_token[i]
                if len(token.strip()) > 0 and not token.startswith('<token_'):
                    tokens_to_embed.append(token)
                    valid_indices.append(i)
        
        print(f"Processing {len(tokens_to_embed)} valid tokens...")
        
        # Process in batches
        for i in tqdm(range(0, len(tokens_to_embed), batch_size)):
            batch_tokens = tokens_to_embed[i:i+batch_size]
            try:
                response = openai.embeddings.create(
                    model=self.model_name,
                    input=batch_tokens
                )
                batch_embeddings = [np.array(item.embedding) for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error in batch {i//batch_size}: {e}")
                # Add dummy embeddings for failed batch
                dummy_embeddings = [np.random.normal(0, 1, 1536) for _ in batch_tokens]
                embeddings.extend(dummy_embeddings)
        
        return np.array(embeddings)
    
    def get_vocab_key(self, vocab_idx: int) -> int:
        """Return token ID as key for faster lookup"""
        return vocab_idx 