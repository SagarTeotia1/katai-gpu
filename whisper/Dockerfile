FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin ctranslate2 to CUDA 12.x compatible build, then faster-whisper on top
RUN pip3 install --no-cache-dir \
    "ctranslate2>=4.4.0,<5.0.0" \
    "faster-whisper>=1.0.3,<2.0.0" \
    "fastapi==0.115.0" \
    "uvicorn[standard]==0.30.0" \
    "pydantic==2.8.0"

COPY app.py .

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=10 \
    CMD curl -f http://localhost:9000/health | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('ready') else 1)" || exit 1

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
