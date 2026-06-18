#!/bin/bash

# Test script to verify Straylight Docker setup
echo "Testing Straylight Docker setup..."

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed"
    exit 1
fi

# Validate docker-compose file
echo "Validating docker-compose.yml..."
docker-compose config > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "✗ docker-compose.yml has errors"
    exit 1
fi

echo "✓ docker-compose.yml is valid"

# Build the image
echo "Building Docker image..."
docker-compose build > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "✗ Failed to build Docker image"
    exit 1
fi

echo "✓ Docker image built successfully"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Please create one with your LLM settings."
fi

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "Warning: models directory not found. Please ensure model files are in models/ directory."
fi

echo "All tests passed!"
echo "To start the service, run: docker-compose up -d"