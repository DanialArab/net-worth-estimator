"""
Embedding extraction service for facial features.
Uses FaceNet pre-trained model to extract embeddings from selfie images.
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from typing import Optional, Tuple
import logging
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class EmbeddingExtractor:
    """Extracts facial embeddings from images using FaceNet."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the embedding extractor.
        
        Args:
            device: Device to run inference on ('cpu' or 'cuda'). Auto-detected if None.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device
        )
        
        # Initialize FaceNet model for embeddings
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.model.to(self.device)
        
        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Initialize PCA for dimensionality reduction (512 -> 32)
        self.pca = PCA(n_components=32)
        self._init_pca()
    
    def _init_pca(self):
        """Initialize PCA with some dummy data to fit the transform.
        In a real application, this should be fitted on a representative dataset.
        """
        # Generate dummy 512-dimensional data for PCA fitting
        np.random.seed(42)  # For reproducibility
        dummy_data = np.random.randn(100, 512)
        self.pca.fit(dummy_data)
        logger.info("PCA initialized for dimensionality reduction (512 -> 32)")
    
    def extract_face(self, image: Image.Image) -> Optional[torch.Tensor]:
        """Extract and align face from image.
        
        Args:
            image: PIL Image containing a face
            
        Returns:
            Tensor of aligned face or None if no face detected
        """
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Detect and extract face
            face_tensor = self.mtcnn(image)
            
            if face_tensor is None:
                logger.warning("No face detected in image")
                return None
                
            return face_tensor.unsqueeze(0).to(self.device)
            
        except Exception as e:
            logger.error(f"Error extracting face: {str(e)}")
            return None
    
    def get_embedding(self, face_tensor: torch.Tensor) -> np.ndarray:
        """Get embedding vector from face tensor.
        
        Args:
            face_tensor: Tensor of aligned face
            
        Returns:
            Numpy array of face embedding (reduced to 32 dimensions)
        """
        try:
            with torch.no_grad():
                embedding = self.model(face_tensor)
                # Normalize the embedding
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                full_embedding = embedding.cpu().numpy().flatten()
                
                # Reduce dimensionality from 512 to 32 using PCA
                reduced_embedding = self.pca.transform(full_embedding.reshape(1, -1))
                return reduced_embedding.flatten()
                
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise
    
    def extract_embedding_from_image(self, image: Image.Image) -> Optional[np.ndarray]:
        """Extract embedding directly from image.
        
        Args:
            image: PIL Image containing a face
            
        Returns:
            Numpy array of face embedding or None if extraction failed
        """
        face_tensor = self.extract_face(image)
        if face_tensor is None:
            return None
            
        return self.get_embedding(face_tensor)
    
    def preprocess_image_fallback(self, image: Image.Image) -> torch.Tensor:
        """Fallback preprocessing if face detection fails.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed image tensor
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transform and add batch dimension
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor
    
    def extract_embedding_with_fallback(self, image: Image.Image) -> np.ndarray:
        """Extract embedding with fallback if face detection fails.
        
        Args:
            image: PIL Image
            
        Returns:
            Numpy array of embedding (either from detected face or full image)
        """
        # Try face detection first
        embedding = self.extract_embedding_from_image(image)
        
        if embedding is not None:
            logger.info("Successfully extracted embedding from detected face")
            return embedding
        
        # Fallback: use whole image
        logger.warning("Face detection failed, using whole image for embedding")
        try:
            preprocessed = self.preprocess_image_fallback(image)
            return self.get_embedding(preprocessed)
        except Exception as e:
            logger.error(f"Fallback embedding extraction failed: {str(e)}")
            # Return zero embedding as last resort (32 dimensions to match PCA output)
            return np.zeros(32, dtype=np.float32)
