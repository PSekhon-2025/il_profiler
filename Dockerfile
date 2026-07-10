# Container image for the IL Profiler Streamlit app (Fly.io deployment).
#
# The image contains ONLY the code + Python deps. The vector index and run
# snapshots live on a persistent Fly volume mounted at /app/data (see fly.toml),
# seeded once from the locally-built index — see DEPLOY.md. The raw (copyrighted)
# corpus is never shipped, so ingestion is disabled in the cloud (IL_PROFILER_CLOUD=1).
FROM python:3.11-slim

# Keep Python output unbuffered so Streamlit's subprocess log streaming is live,
# and force UTF-8 so tqdm/emoji output is identical to local runs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code (the .dockerignore keeps data/, .venv/, .env, .git out of the image).
COPY . .

EXPOSE 8080

# Behind Fly's proxy (and optionally Cloudflare Access): bind all interfaces,
# run headless, and relax XSRF/CORS so the websocket connects through the proxy.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
