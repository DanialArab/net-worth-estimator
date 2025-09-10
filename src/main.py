"""
FastAPI application for net worth estimation from selfie images.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

# Force Docker build environment
os.environ['DOCKER_BUILD_ONLY'] = 'true'
os.environ['NIXPACKS_DISABLE'] = 'true'

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

from .embedding_service import EmbeddingExtractor
from .similarity_service import SimilarityService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Net Worth Estimator API",
    description="AI-powered net worth estimation from selfie images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances (initialized on startup)
embedding_extractor: Optional[EmbeddingExtractor] = None
similarity_service: Optional[SimilarityService] = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    global embedding_extractor, similarity_service
    
    try:
        logger.info("Initializing services...")
        
        # Initialize embedding extractor
        embedding_extractor = EmbeddingExtractor()
        logger.info("Embedding extractor initialized successfully")
        
        # Initialize similarity service
        similarity_service = SimilarityService()
        logger.info("Similarity service initialized successfully")
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Net Worth Estimator API",
        "version": "1.0.0",
        "description": "Upload a selfie to get an estimated net worth and similar wealthy individuals",
        "endpoints": {
            "predict": "/predict",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if services are initialized
        if embedding_extractor is None or similarity_service is None:
            raise HTTPException(status_code=503, detail="Services not initialized")
        
        return {
            "status": "healthy",
            "services": {
                "embedding_extractor": "ready",
                "similarity_service": "ready"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/predict")
async def predict_net_worth(file: UploadFile = File(...)) -> Dict:
    """
    Predict net worth from uploaded selfie image.
    
    Args:
        file: Uploaded image file (multipart/form-data)
        
    Returns:
        JSON response with estimated net worth and top 3 similar wealthy individuals
    """
    try:
        # Validate file
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Please upload an image file."
            )
        
        # Check file size (limit to 10MB)
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 10MB."
            )
        
        logger.info(f"Processing uploaded file: {file.filename}, size: {len(content)} bytes")
        
        # Load image
        try:
            image = Image.open(io.BytesIO(content))
            logger.info(f"Image loaded successfully: {image.size}, mode: {image.mode}")
        except Exception as e:
            logger.error(f"Failed to load image: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail="Invalid image file. Could not process the uploaded image."
            )
        
        # Extract embedding
        try:
            user_embedding = embedding_extractor.extract_embedding_with_fallback(image)
            if user_embedding is None:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract features from the uploaded image. Please ensure the image contains a clear face."
                )
            logger.info(f"Embedding extracted successfully, shape: {user_embedding.shape}")
        except Exception as e:
            logger.error(f"Failed to extract embedding: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to process image features. Please try again."
            )
        
        # Get prediction
        try:
            prediction = similarity_service.get_prediction(user_embedding)
            logger.info("Prediction generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate prediction: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate prediction. Please try again."
            )
        
        # Format response
        response = {
            "success": True,
            "estimated_net_worth": prediction["estimated_net_worth"],
            "currency": prediction["currency"],
            "top_matches": [
                {
                    "name": match["name"],
                    "similarity_score": round(match["similarity_score"], 4),
                    "industry": match["industry"],
                    "rank": match["rank"]
                }
                for match in prediction["top_matches"]
            ],
            "confidence_score": round(prediction["confidence_score"], 4),
            "metadata": {
                "filename": file.filename,
                "processing_method": "face_detection" if "detected face" in str(embedding_extractor) else "full_image"
            }
        }
        
        logger.info(f"Returning prediction: estimated worth ${prediction['estimated_net_worth']:,.2f}")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again."
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again."
        }
    )

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
