# ---------------------------
# 1. Base image
# ---------------------------
FROM python:3.11-slim
# Choose a lightweight Python runtime as the starting point

# ---------------------------
# 2. System dependencies
# ---------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*
# Install OS-level libs (libgomp1 is needed by xgboost)

# ---------------------------
# 3. Environment variables
# ---------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
# Control Python behavior + Streamlit defaults

# ---------------------------
# 4. Working directory
# ---------------------------
WORKDIR /app
# All subsequent commands run inside /app

# ---------------------------
# 5. Dependencies
# ---------------------------
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
# Copy only requirements first for caching, then install

# ---------------------------
# 6. Copy project files
# ---------------------------
COPY . /app
# Copy everything else (app.py, app/scripts, config, etc.)

# ---------------------------
# 7. Expose port
# ---------------------------
EXPOSE 8501
# Tell Docker the app listens on port 8501 (Streamlit default)

# ---------------------------
# 8. Entry point / command
# ---------------------------
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
# Default command to run the Streamlit app when the container starts
