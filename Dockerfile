# Base Image: Python 3.10 Slim (Enterprise Standard)
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="."

# Set the working directory
WORKDIR /app

# Install system dependencies (required for LightGBM and Streamlit)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . /app/

# Expose Streamlit default port
EXPOSE 8501

# Run the end-to-end pipeline to generate models & data, then launch the Streamlit dashboard
CMD ["sh", "-c", "python src/pipeline/run_all.py --fast-dev && streamlit run src/dashboard/app.py --server.port=8501 --server.address=0.0.0.0"]
