# Dockerfile untuk Deployment Model Face Recognition ke RunPod GPU
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install dependency sistem untuk OpenCV dan FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Salin requirements dan instal dependency Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin script dan model AI
COPY main.py .
COPY facenet.onnx .
COPY data_wajah_facenet_mtcnn.npz .

# Expose port HTTP Web Dashboard & Stream
EXPOSE 8000

# Default command
CMD ["python", "main.py", "--port", "8000"]
