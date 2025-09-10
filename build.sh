#!/bin/bash
# Build script for deployment platforms
# This ensures Docker build is used

echo "Starting Docker build process..."

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    echo "ERROR: Dockerfile not found!"
    exit 1
fi

# Build the Docker image
docker build -t net-worth-estimator .

echo "Docker build completed successfully!"
