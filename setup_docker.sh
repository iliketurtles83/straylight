#!/bin/bash

# Simple setup script for Straylight Docker deployment
echo "Setting up Straylight Docker environment..."

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Please create one with your LLM settings."
    exit 1
fi

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "Warning: models directory not found. Please ensure model files are in models/ directory."
fi

echo "Starting Straylight service..."
docker-compose up -d

if [ $? -eq 0 ]; then
    echo "✓ Straylight service started successfully"
    echo "To view logs: docker-compose logs -f"
    echo "To stop: docker-compose down"
else
    echo "✗ Failed to start Straylight service"
    exit 1
fi