FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY server/requirements.txt ./server/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy all project files
COPY . .

# Expose port
EXPOSE 7860

# Start the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]