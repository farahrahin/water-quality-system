FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "numpy<2"

RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements_app.txt .
RUN pip install --no-cache-dir -r requirements_app.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main_v6:app", "--host", "0.0.0.0", "--port", "8000"]
