#!/bin/bash
# Test script to verify Docker Compose setup

echo "Testing Straylight Docker setup..."

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed"
    exit 1
fi

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo "Error: docker-compose.yml not found"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Please create one from .env.example"
fi

# Test docker-compose configuration
echo "Testing docker-compose configuration..."
docker-compose config

echo "Setup complete. Run 'docker-compose up -d' to start the service."