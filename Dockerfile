FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

# Create uploads directory
RUN mkdir -p uploads temp

# Expose port
EXPOSE 8000

# Set Python path and memory optimization
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TORCH_THREAD=1
ENV OMP_NUM_THREADS=1

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
