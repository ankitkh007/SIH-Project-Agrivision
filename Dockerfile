# Use lightweight Python slim image
FROM python:3.10-slim

WORKDIR /app

# Upgrade pip and install only required Python packages (no compilation)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

# Set environment variable for Railway
ENV PORT=8000
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "bend:app", "--host", "0.0.0.0", "--port", "8000"]
