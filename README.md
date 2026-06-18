# Straylight Docker Setup

## Overview

This repository contains the necessary files to containerize the Straylight voice service using Docker Compose. The service runs alongside an external llama.cpp server.

## Files Included

- `Dockerfile` - Container build instructions
- `docker-compose.yml` - Service configuration
- `README.md` - Setup instructions
- `setup_docker.sh` - Setup script
- `verify_setup.sh` - Verification script

## Prerequisites

Before using this setup:
1. Install Docker and Docker Compose
2. Have an external llama.cpp server running
3. Configure `.env` with your LLM settings
4. Place model files in the `models/` directory

## Quick Start

To build and run:

```bash
# Build the container
docker build -t straylight .

# Start the service
docker run --rm \
  --name straylight \
  --network host \
  --user 1000:1000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/shared:/app/shared:ro \
  -v $(pwd)/services/voice/cass_prompt.txt:/app/cass_prompt.txt:ro \
  -v $(pwd)/scripts:/app/scripts:ro \
  -v $(pwd)/exemplars.jsonl:/app/exemplars.jsonl:ro \
  -v /dev/snd:/dev/snd:rwm \
  -e $(grep -v '^#' .env | xargs) \
  straylight
```

## Docker Configuration

The setup uses:
- Host network mode for audio access
- Non-root user for security
- Read-only volume mounts
- Direct access to audio devices

## Environment Variables

The `.env` file should contain:
- `CASS_LLM_BASE_URL` - URL of external llama.cpp server
- `CASS_LLM_MODEL` - Model alias
- `LLAMA_MODEL` - Path to model file
- Other Straylight configuration variables

## Troubleshooting

### Audio Issues
Ensure audio device permissions are correct on the host system.

### Connection Issues
Verify the llama.cpp server is running and accessible at the URL specified in `CASS_LLM_BASE_URL`.

### Model Loading
Ensure model files are placed in the `models/` directory.