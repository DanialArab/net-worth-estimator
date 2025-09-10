"""
Similarity computation and net worth estimation service.
Computes similarity between user embeddings and wealthy individuals database.
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class WealthyIndividual:
    """Data class for wealthy individual records."""
    
    def __init__(self, id: int, name: str, net_worth: float, industry: str, embedding: List[float]):
        self.id = id
        self.name = name
        self.net_worth = net_worth
        self.industry = industry
        self.embedding = np.array(embedding, dtype=np.float32)

class SimilarityService:
    """Service for computing similarity and estimating net worth."""
    
    def __init__(self, data_path: str = "data/wealthy_individuals.json"):
        """Initialize similarity service.
        
        Args:
            data_path: Path to wealthy individuals dataset
        """
        self.data_path = Path(data_path)
        self.wealthy_individuals = self._load_wealthy_individuals()
        self._prepare_embeddings_matrix()
        logger.info(f"Loaded {len(self.wealthy_individuals)} wealthy individuals")
    
    def _load_wealthy_individuals(self) -> List[WealthyIndividual]:
        """Load wealthy individuals dataset.
        
        Returns:
            List of WealthyIndividual objects
        """
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            
            individuals = []
            for item in data:
                individual = WealthyIndividual(
                    id=item['id'],
                    name=item['name'],
                    net_worth=item['net_worth'],
                    industry=item['industry'],
                    embedding=item['embedding']
                )
                individuals.append(individual)
            
            return individuals
            
        except Exception as e:
            logger.error(f"Error loading wealthy individuals data: {str(e)}")
            raise
    
    def _prepare_embeddings_matrix(self):
        """Prepare embeddings matrix for efficient similarity computation."""
        embeddings = [individual.embedding for individual in self.wealthy_individuals]
        self.embeddings_matrix = np.vstack(embeddings)
        logger.info(f"Prepared embeddings matrix with shape: {self.embeddings_matrix.shape}")
    
    def compute_similarities(self, user_embedding: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between user embedding and all wealthy individuals.
        
        Args:
            user_embedding: User's facial embedding vector
            
        Returns:
            Array of similarity scores
        """
        try:
            # Ensure user embedding is 2D for sklearn
            if user_embedding.ndim == 1:
                user_embedding = user_embedding.reshape(1, -1)
            
            # Compute cosine similarities
            similarities = cosine_similarity(user_embedding, self.embeddings_matrix)
            return similarities.flatten()
            
        except Exception as e:
            logger.error(f"Error computing similarities: {str(e)}")
            raise
    
    def get_top_matches(self, user_embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        """Get top K most similar wealthy individuals.
        
        Args:
            user_embedding: User's facial embedding vector
            top_k: Number of top matches to return
            
        Returns:
            List of dictionaries with match information
        """
        similarities = self.compute_similarities(user_embedding)
        
        # Get indices of top K similarities
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        matches = []
        for i, idx in enumerate(top_indices):
            individual = self.wealthy_individuals[idx]
            match = {
                "name": individual.name,
                "similarity_score": float(similarities[idx]),
                "industry": individual.industry,
                "net_worth": individual.net_worth,
                "rank": i + 1
            }
            matches.append(match)
        
        return matches
    
    def estimate_net_worth(self, user_embedding: np.ndarray, method: str = "weighted_average") -> float:
        """Estimate user's net worth based on similarity to wealthy individuals.
        
        Args:
            user_embedding: User's facial embedding vector
            method: Estimation method ('weighted_average', 'top_match', 'top_3_average')
            
        Returns:
            Estimated net worth in USD
        """
        similarities = self.compute_similarities(user_embedding)
        
        if method == "weighted_average":
            return self._weighted_average_estimation(similarities)
        elif method == "top_match":
            return self._top_match_estimation(similarities)
        elif method == "top_3_average":
            return self._top_k_average_estimation(similarities, k=3)
        else:
            raise ValueError(f"Unknown estimation method: {method}")
    
    def _weighted_average_estimation(self, similarities: np.ndarray) -> float:
        """Estimate using weighted average of all individuals.
        
        Args:
            similarities: Array of similarity scores
            
        Returns:
            Estimated net worth
        """
        # Use softmax to convert similarities to weights
        weights = self._softmax(similarities * 10)  # Scale for more pronounced differences
        
        net_worths = np.array([individual.net_worth for individual in self.wealthy_individuals])
        estimated_worth = np.sum(weights * net_worths)
        
        # Return deterministic result for consistency and reproducibility
        return float(estimated_worth)
    
    def _top_match_estimation(self, similarities: np.ndarray) -> float:
        """Estimate based on most similar individual.
        
        Args:
            similarities: Array of similarity scores
            
        Returns:
            Estimated net worth
        """
        top_idx = np.argmax(similarities)
        base_worth = self.wealthy_individuals[top_idx].net_worth
        
        # Scale based on similarity score
        similarity_factor = similarities[top_idx]
        estimated_worth = base_worth * similarity_factor
        
        # Return deterministic result for consistency and reproducibility
        return float(estimated_worth)
    
    def _top_k_average_estimation(self, similarities: np.ndarray, k: int = 3) -> float:
        """Estimate using average of top K matches.
        
        Args:
            similarities: Array of similarity scores
            k: Number of top matches to consider
            
        Returns:
            Estimated net worth
        """
        top_indices = np.argsort(similarities)[::-1][:k]
        top_similarities = similarities[top_indices]
        
        # Weighted average of top K
        weights = self._softmax(top_similarities * 5)
        net_worths = np.array([self.wealthy_individuals[i].net_worth for i in top_indices])
        
        estimated_worth = np.sum(weights * net_worths)
        
        # Return deterministic result for consistency and reproducibility
        return float(estimated_worth)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax of array.
        
        Args:
            x: Input array
            
        Returns:
            Softmax normalized array
        """
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)
    
    def get_prediction(self, user_embedding: np.ndarray) -> Dict:
        """Get complete prediction including net worth and top matches.
        
        Args:
            user_embedding: User's facial embedding vector
            
        Returns:
            Dictionary with estimated net worth and top 3 matches
        """
        try:
            # Estimate net worth
            estimated_net_worth = self.estimate_net_worth(user_embedding, method="top_3_average")
            
            # Get top 3 matches
            top_matches = self.get_top_matches(user_embedding, top_k=3)
            
            return {
                "estimated_net_worth": round(estimated_net_worth, 2),
                "currency": "USD",
                "top_matches": top_matches,
                "confidence_score": float(np.mean([match["similarity_score"] for match in top_matches]))
            }
            
        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}")
            raise
