# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Reduces image size
# --upgrade pip: Good practice
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Make port 8000 available to the world outside this container
# Cloud Run will automatically map its external port to this container port
EXPOSE 8000

# Define environment variable for the port (good practice, Cloud Run injects PORT)
ENV PORT 8000
ENV PYTHONUNBUFFERED TRUE # Ensures print statements appear in logs immediately

# Run main.py when the container launches using gunicorn
# -w 4: Number of worker processes (adjust based on your expected load and instance CPU)
# -k uvicorn.workers.UvicornWorker: Use Uvicorn for FastAPI
# main:app: 'main' is the Python file, 'app' is the FastAPI instance
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "main:app"]
