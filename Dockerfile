# Use an official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    ffmpeg \
    git \
    curl \
    wget \
    libgl1 \
    && apt-get clean

# Create app directory
WORKDIR /app

# Copy project files
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download SAM model weights (optional if weights are not loaded dynamically)
# RUN mkdir -p /root/.segment_anything && \
#     wget -O /root/.segment_anything/sam_vit_h.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h.pth

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
