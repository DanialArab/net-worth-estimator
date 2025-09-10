# Wealth Potential Estimator API

An AI-powered service that estimates a user's potential net worth based on facial features extracted from selfie images. The service compares user features against a database of wealthy individuals to provide net worth estimates and similarity matches.

## 🚀 Live Demo

**Try the API:** https://net-worth-estimator.onrender.com/docs

*Interactive Swagger UI documentation with live API testing capabilities*

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/DanialArab/net-worth-estimator.git
cd net-worth-estimator

# Build and run the service
./run.sh build
./run.sh run

# The API will be available at http://localhost:8000
# Interactive documentation at http://localhost:8000/docs
```

### Manual Docker Build

```bash
# Build the Docker image
docker build -t net-worth-estimator .

# Run the container
docker run -p 8000:8000 net-worth-estimator
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the development server
cd src
python main.py
```

## 📡 API Usage

### Endpoint: `POST /predict`

Upload a selfie image to get net worth estimation and similar wealthy individuals.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_selfie.jpg"
```

**Response:**
```json
{
  "success": true,
  "estimated_net_worth": 85000000.50,
  "currency": "USD",
  "top_matches": [
    {
      "name": "Tech Mogul Alpha",
      "similarity_score": 0.8542,
      "industry": "Technology",
      "rank": 1
    },
    {
      "name": "Finance King Gamma",
      "similarity_score": 0.7891,
      "industry": "Finance",
      "rank": 2
    },
    {
      "name": "Real Estate Queen Delta",
      "similarity_score": 0.7234,
      "industry": "Real Estate",
      "rank": 3
    }
  ],
  "confidence_score": 0.7889,
  "metadata": {
    "filename": "selfie.jpg",
    "processing_method": "face_detection"
  }
}
```

**Response Fields:**
- **`confidence_score`**: Average similarity score of top 3 matches (0.0-1.0)
  - **0.9-1.0**: Very high confidence - strong similarity to wealthy individuals
  - **0.7-0.9**: High confidence - good matches found  
  - **0.5-0.7**: Medium confidence - moderate similarity
  - **< 0.5**: Lower confidence - weaker matches

### Other Endpoints

- `GET /` - API information and available endpoints
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API documentation (Swagger UI)

## 🏗️ Architecture

### System Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client App    │───▶│   FastAPI API    │───▶│  ML Pipeline    │
│   (Frontend)    │    │   (main.py)      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   File Upload    │    │ Embedding       │
                       │   Validation     │    │ Extraction      │
                       └──────────────────┘    │ (FaceNet)       │
                                               └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │ Similarity      │
                                               │ Computation     │
                                               │ & Net Worth     │
                                               │ Estimation      │
                                               └─────────────────┘
```

### Core Components

#### 1. Embedding Extraction Service (`embedding_service.py`)

**Architecture Decision:** Uses FaceNet (InceptionResnetV1) pre-trained on VGGFace2
- **Rationale:** Proven model with excellent facial recognition performance
- **Face Detection:** MTCNN for robust face detection and alignment
- **Fallback Strategy:** Uses full image if face detection fails
- **Output:** 512-dimensional L2-normalized embeddings, reduced to 32 dimensions via PCA

**Key Features:**
- Automatic face detection and alignment
- Graceful fallback to whole-image processing
- GPU support with automatic CPU fallback
- Robust error handling

#### 2. Similarity Computation Service (`similarity_service.py`)

**Architecture Decision:** Cosine similarity with weighted averaging
- **Similarity Metric:** Cosine similarity for normalized embeddings
- **Net Worth Estimation:** Multiple methods available:
  - `weighted_average`: Uses all individuals with softmax weighting
  - `top_match`: Based on most similar individual
  - `top_3_average`: Weighted average of top 3 matches (default)

**Key Features:**
- Efficient matrix operations using NumPy/scikit-learn
- Multiple estimation algorithms for different use cases
- Consistent results - same image always produces same estimate
- Confidence scoring based on similarity scores

#### 3. API Layer (`main.py`)

**Architecture Decision:** FastAPI for high-performance async API
- **Rationale:** Excellent performance, automatic OpenAPI docs, type hints
- **File Handling:** Multipart form-data with size/type validation
- **Error Handling:** Comprehensive exception handling with user-friendly messages
- **Logging:** Structured logging for debugging and monitoring

### Data Pipeline

1. **Image Upload** → Validation (type, size, format)
2. **Preprocessing** → Face detection & alignment using MTCNN
3. **Feature Extraction** → 512-dim embeddings via FaceNet
4. **Dimensionality Reduction** → PCA reduces 512 dims to 32 dims
5. **Similarity Computation** → Cosine similarity against wealthy individuals DB
6. **Net Worth Estimation** → Weighted combination of similar profiles
7. **Response Formatting** → JSON with estimates and top matches

## 🗂️ Project Structure

```
net-worth-estimator/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # FastAPI application & endpoints
│   ├── embedding_service.py     # FaceNet embedding extraction
│   └── similarity_service.py    # Similarity & net worth estimation
├── data/
│   └── wealthy_individuals.json # Mock wealthy individuals database
├── tests/                       # Test files (placeholder)
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container configuration
├── docker-compose.yml           # Multi-container orchestration
├── run.sh                       # Deployment helper script
├── .gitignore                   # Git ignore patterns
└── README.md                    # This file
```

## 🧠 Machine Learning Approach

### Model Selection

**Primary Model: FaceNet (InceptionResnetV1)**
- **Training Data:** VGGFace2 dataset (3.3M images, 9K identities)
- **Architecture:** Inception-ResNet with triplet loss
- **Embedding Dimension:** 512 features
- **Normalization:** L2 normalization for consistent similarity computation

**Face Detection: MTCNN**
- **Multi-stage:** P-Net → R-Net → O-Net cascade
- **Features:** Face detection, landmark detection, alignment
- **Robustness:** Handles various face orientations and lighting

### Dimensionality Reduction: PCA

**Architecture Decision:** Reduce 512-dimensional FaceNet embeddings to 32 dimensions
- **Rationale:** Improves computational efficiency and storage requirements
- **Method:** Principal Component Analysis (PCA) preserves most important variance
- **Benefits:**
  - **94% size reduction** (512 → 32 dimensions)
  - **Faster similarity computation** with smaller vectors
  - **Reduced memory usage** for embedding storage
  - **Maintained accuracy** while improving performance

### Similarity Computation

**Metric:** Cosine Similarity
- **Rationale:** Effective for normalized high-dimensional embeddings
- **Range:** [-1, 1] where 1 = identical, 0 = orthogonal, -1 = opposite
- **Implementation:** Efficient vectorized computation using scikit-learn

### Net Worth Estimation Algorithm

The default algorithm uses **weighted top-3 averaging**:

1. **Compute Similarities:** Cosine similarity with all wealthy individuals
2. **Select Top-K:** Identify 3 most similar profiles
3. **Weight Calculation:** Apply softmax to similarity scores for weighting
4. **Weighted Average:** Combine net worths using computed weights
5. **Deterministic Output:** Return consistent results for reproducibility

**Mathematical Formula:**
```
weights_i = softmax(similarity_i * temperature)
estimated_worth = Σ(weights_i * net_worth_i)
```

**Temperature Parameter:**
- **Purpose**: Controls the "sharpness" of the probability distribution
- **High temperature (10)**: More uniform distribution (used in `weighted_average`)
- **Low temperature (5)**: More focused on top matches (used in `top_3_average`)
- **Effect**: Higher temperature = smoother weights, Lower temperature = sharper weights

## 🚀 Deployment

### Production Deployment on Render.com

This application is deployed on **Render.com**, a modern cloud platform that provides seamless Docker-based deployments.

**Live API:** https://net-worth-estimator.onrender.com/docs

#### Why Render.com?
- **Docker-first approach:** Native support for Dockerfile deployments
- **Automatic deployments:** Deploys automatically on Git push
- **Built-in SSL:** HTTPS enabled by default
- **Health monitoring:** Automatic health checks and restarts
- **Scalable:** Easy horizontal scaling as needed

#### Deployment Process:
1. **Connect Repository:** Link your GitHub repository to Render
2. **Auto-detect Dockerfile:** Render automatically detects and uses the Dockerfile
3. **Configure Resources:** Set memory allocation (1GB+ recommended)
4. **Deploy:** Automatic build and deployment on every push

### Local Development with Docker Compose

```bash
docker-compose up -d
```

**Features:**
- Easy local development setup
- Automatic restarts and health checks
- Volume mounting for development
- Consistent with production environment

## ⚙️ Configuration & Environment

### Environment Variables

- `PYTHONPATH=/app` - Python module path
- `PYTHONUNBUFFERED=1` - Unbuffered Python output
- `LOG_LEVEL=INFO` - Logging level (DEBUG, INFO, WARNING, ERROR)

### Resource Requirements

**Minimum:**
- RAM: 1GB (tested ~480MB usage, with startup buffer)
- CPU: 2 cores
- Disk: 1GB (models + dependencies)

**Recommended (Production):**
- RAM: 2-4GB (for traffic spikes and multiple requests)
- CPU: 4 cores
- GPU: Optional (significant speedup)
- Disk: 2GB + logs

## 🔧 Development

### Adding New Features

1. **New Embedding Models:** Extend `EmbeddingExtractor` class
2. **Different Similarity Metrics:** Add methods to `SimilarityService`
3. **Additional Endpoints:** Add routes to `main.py`

### Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Local Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server with auto-reload
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## 📊 Synthetic Data & Training Dataset

### Wealthy Individuals Database

**Dataset:** `data/wealthy_individuals.json` contains **10 synthetic wealthy individuals**

**Data Structure:**
```json
{
  "id": 1,
  "name": "Tech Mogul Alpha",
  "net_worth": 180000000000,
  "industry": "Technology",
  "embedding": [0.12, -0.34, 0.78, ...] // 32-dimensional synthetic embedding
}
```

### ⚠️ **Important: All Data is Synthetic**

- ✅ **Names**: Fictional individuals ("Tech Mogul Alpha", "Energy Tycoon Beta", etc.)
- ✅ **Net Worth Values**: Realistic but synthetic wealth figures
- ✅ **Embeddings**: Randomly generated 32-dimensional vectors (not real facial features)
- ✅ **Industries**: Representative industry categories

### Why Synthetic Data?

1. **Privacy Protection**: No real individuals' biometric data used
2. **Ethical Compliance**: Avoids potential discrimination or privacy violations
3. **Demonstration Purpose**: Shows technical implementation without real-world risks
4. **Legal Safety**: No copyright, privacy, or consent issues
5. **Reproducible Results**: Consistent testing and development

### Data Generation Process

- **Embeddings**: Generated using `np.random.randn()` with controlled variance
- **Net Worth**: Realistic billionaire wealth ranges ($54B - $180B)
- **Names**: Generic placeholder names with industry themes
- **Industries**: 10 major business sectors represented

ℹ️ **Note:** In a production system, this would be replaced with properly obtained, consented, and legally compliant real data.

## 🎯 Assumptions & Limitations

### Assumptions Made

1. **Face Visibility:** Input images contain visible human faces
2. **Image Quality:** Reasonable resolution and lighting conditions
3. **Single Face:** Images primarily contain one person's face
4. **Wealth Correlation:** Facial features correlate with socioeconomic status (for demonstration purposes)

### Known Limitations

1. **Dataset Size:** Mock dataset contains only 10 wealthy individuals
2. **Embedding Quality:** Dependent on face detection success
3. **Bias Concerns:** May exhibit demographic biases from training data
4. **Accuracy:** Demonstration project - not production-accurate
5. **Real-time Performance:** Model loading adds startup latency

### Ethical Considerations

⚠️ **Important:** This is a **demonstration project** for technical skills assessment. 

**Real-world concerns:**
- Facial analysis for wealth estimation raises ethical issues
- Potential for discrimination and bias
- Privacy implications of facial feature extraction
- Should not be used for actual financial decisions

## 🔧 Troubleshooting

### Common Issues

**Issue:** "No face detected in image"
```
Solution: Ensure image contains a clear, visible face
Alternative: System falls back to whole-image processing
```

**Issue:** "CUDA out of memory"
```
Solution: System automatically falls back to CPU processing
Or: Reduce batch size in embedding extraction
```

**Issue:** "Service unhealthy"
```
Check: curl http://localhost:8000/health
Logs: docker-compose logs -f
```

**Issue:** Slow inference
```
Solution: Enable GPU support or reduce image resolution
Check: GPU availability with torch.cuda.is_available()
```

### Performance Tuning

1. **GPU Acceleration:** Install CUDA-compatible PyTorch
2. **Batch Processing:** Process multiple images simultaneously
3. **Model Quantization:** Use int8 quantization for faster inference
4. **Caching:** Cache embeddings for repeated requests

## 📊 Monitoring & Metrics

### Health Checks

- **Endpoint:** `GET /health`
- **Docker:** Built-in health check every 30s
- **Render.com:** Automatic health monitoring and restarts

### Logging

**Log Levels:**
- `INFO`: Request processing, predictions
- `WARNING`: Fallback processing, face detection failures
- `ERROR`: Service failures, invalid inputs

**Log Format:**
```
2024-01-15 10:30:45 - embedding_service - INFO - Successfully extracted embedding from detected face
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

**⚠️ Disclaimer:** This is a demonstration project for technical assessment. It should not be used for actual financial decisions or real-world wealth estimation.
