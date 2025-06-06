FROM python:3.10-slim

# Pre-install dlib dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Use pre-built dlib wheel
RUN pip install --no-cache-dir dlib==19.24.1 --find-links https://github.com/sachadee/Dlib-wheels/raw/master/dlib-19.24.1-cp310-cp310-linux_x86_64.whl

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .