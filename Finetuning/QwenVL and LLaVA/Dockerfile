FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    texlive-full \
    python3-pip \
    python3-dev \
    libpq-dev \
    git \
    wget \
    unzip \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    google-generativeai \
    openai \
    groq \
    pandas \
    matplotlib \
    pdf2image \
    opencv-python-headless \
    pytesseract \
    python-docx

# Verify installations
RUN tex --version
RUN python3 --version
