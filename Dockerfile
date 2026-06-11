FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (smaller = faster build, less RAM)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements_app.txt .
RUN pip install --no-cache-dir -r requirements_app.txt

# Pre-download EasyOCR models during build (not at runtime)
# This avoids RAM spike on first request
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)" \
    || echo "EasyOCR model download attempted"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main_v6:app", "--host", "0.0.0.0", "--port", "8000"]
