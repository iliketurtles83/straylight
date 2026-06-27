# Dockerfile for Straylight
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libasound2-dev \
    libpulse-dev \
    libsndfile1-dev \
    curl \
    gcc \
    g++ \
    make \
    pkg-config \
    libsndfile1 \
    libsndfile1-dev \
    libvorbis-dev \
    libflac-dev \
    libmp3lame-dev \
    libopus-dev \
    libogg-dev \
    libvorbisenc2 \
    libvorbisfile3 \
    libmpg123-0 \
    libmpg123-dev \
    libpulse0 \
    libpulse-mainloop-glib0 \
    libpulse-dev \
    libglib2.0-dev \
    libglib2.0-dev-bin \
    libgio-2.0-dev \
    libgio-2.0-dev-bin \
    libxml2 \
    libxml2-dev \
    libx11-dev \
    libx11-xcb-dev \
    libxcb1-dev \
    libxau-dev \
    libxdmcp-dev \
    libasound2 \
    libasound2-dev \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    portaudio19-dev \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY surfaces/voice/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install any additional dependencies that might be needed
RUN pip install --no-cache-dir \
    numpy \
    scikit-learn \
    sounddevice

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Set environment variables
ENV PYTHONPATH=/app
ENV HOME=/home/app

# Expose the port that llama-server will use (if needed)
EXPOSE 8080

# Command to run the application
CMD ["bash", "scripts/dev_gemma.sh"]