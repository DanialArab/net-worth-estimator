# Deployment Guide

## Sevalla Deployment

### Prerequisites
- Docker working locally ✅
- Sevalla configuration ready ✅ (`sevalla.json`)
- Health checks implemented ✅

### Quick Deploy to Sevalla

1. **Push to Git repository** (if not already done):
   ```bash
   git add .
   git commit -m "Ready for production deployment"
   git push origin main
   ```

2. **Connect repository to Sevalla**:
   - Link your GitHub/GitLab repository
   - Sevalla will detect the `sevalla.json` configuration
   - This forces Docker build instead of Nixpacks (avoids build conflicts)

3. **Set Environment Variables** (optional):
   ```
   PYTHONPATH=/app
   PYTHONUNBUFFERED=1
   ```

### Expected Resource Usage
- **Memory**: 4-6GB minimum (due to PyTorch + ML models)
- **Storage**: ~9GB for container image
- **Startup Time**: 2-5 minutes (model loading)
- **Port**: Uses `$PORT` environment variable (Sevalla compatible)

### Health Monitoring
- Health check endpoint: `/health`
- Expected startup time: 60+ seconds
- Auto-restart on failure: Configured

## Alternative Deployment Options

### Docker Compose (Local/VPS)
```bash
docker-compose up -d
```

### Manual Docker Deploy
```bash
# Build
docker build -t net-worth-estimator .

# Run
docker run -d -p 8000:8000 --name net-worth-app net-worth-estimator
```

## Optimization Notes

### Current Optimizations
- ✅ CPU-only PyTorch (smaller size, lower cost)
- ✅ Multi-stage build not needed (already slim)
- ✅ Proper layer caching
- ✅ Health checks included

### Potential Further Optimizations
- Consider model quantization for faster inference
- Implement model caching to reduce cold start times
- Add Redis for caching embeddings (future enhancement)

## Cost Estimates (Sevalla)
- **Memory**: High usage due to ML models
- **Storage**: ~9GB container + data
- **Network**: Moderate (mainly startup pulls)
- **Recommended Plan**: Professional tier or higher

## Troubleshooting

### Common Issues
1. **Long startup times**: Normal due to model loading
2. **Memory errors**: Ensure sufficient RAM allocated
3. **Port conflicts**: Sevalla handles port mapping automatically

### Monitoring
- Check `/health` endpoint after deployment
- Monitor startup logs for model loading progress
- Watch memory usage during initial requests
