#!/bin/bash

# Net Worth Estimator - Build and Run Script

set -e

echo "=== Net Worth Estimator - Build and Run ==="

# Function to display usage
usage() {
    echo "Usage: $0 [build|run|stop|logs|clean]"
    echo "  build  - Build the Docker image"
    echo "  run    - Run the container using docker-compose"
    echo "  stop   - Stop the running container"
    echo "  logs   - Show container logs"
    echo "  clean  - Stop and remove containers, networks, and images"
    exit 1
}

# Check if command is provided
if [ $# -eq 0 ]; then
    usage
fi

case "$1" in
    build)
        echo "Building Docker image..."
        docker-compose build --no-cache
        echo "Build completed!"
        ;;
    
    run)
        echo "Starting Net Worth Estimator service..."
        docker-compose up -d
        echo "Service started! Visit http://localhost:8000"
        echo "API documentation available at http://localhost:8000/docs"
        ;;
    
    stop)
        echo "Stopping service..."
        docker-compose down
        echo "Service stopped!"
        ;;
    
    logs)
        echo "Showing logs (press Ctrl+C to exit)..."
        docker-compose logs -f
        ;;
    
    clean)
        echo "Cleaning up containers, networks, and images..."
        docker-compose down --volumes --rmi all
        echo "Cleanup completed!"
        ;;
    
    *)
        echo "Unknown command: $1"
        usage
        ;;
esac
