# -------------------------------
# Base image
# -------------------------------
FROM python:3.11.9

# -------------------------------
# Set working directory
# -------------------------------
WORKDIR /app

# -------------------------------
# Copy only required folders
# -------------------------------
COPY src/ src/
COPY output/ output/
COPY requirements-docker.txt .

# -------------------------------
# Install system dependencies (optional but safe)
# -------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Install Python dependencies
# -------------------------------
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

# -------------------------------
# Expose API port
# -------------------------------
EXPOSE 8000

# -------------------------------
# Run FastAPI app
# -------------------------------
CMD ["python", "src/app.py"]
