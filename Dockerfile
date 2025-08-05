FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# --no-cache-dir: Reduces image size
# --upgrade pip: Good practice
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run will automatically map its external port to this container port
EXPOSE 8000

ENV PORT 8000
ENV PYTHONUNBUFFERED TRUE

# -w 4: Number of worker processes
# -k uvicorn.workers.UvicornWorker: Use Uvicorn for FastAPI
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "main:app"]
