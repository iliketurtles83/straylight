#!/bin/bash

# Test script to verify Straylight Docker setup
echo "Testing Straylight Docker setup..."

# Build the Docker image
echo "Building Docker image..."
docker build -t straylight .

# Check if the image was built successfully
if [ $? -eq 0 ]; then
    echo "✓ Docker image built successfully"
else
    echo "✗ Failed to build Docker image"
    exit 1
fi

# Test compose file syntax
echo "Validating docker-compose.yml..."
docker-compose config

if [ $? -eq 0 ]; then
    echo "✓ docker-compose.yml is valid"
else
    echo "✗ docker-compose.yml has errors"
    exit 1
fi

echo "All tests passed!"