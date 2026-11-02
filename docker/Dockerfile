# CPU image of the demo: FastAPI + Gradio on port 7860 (same port as Hugging Face Spaces).
# Build from the repo root: docker build -f docker/Dockerfile -t image-to-3d .
# Run: docker run --rm -p 7860:7860 -e WEIGHTS_REPO=iLogicus/image-to-3d-weights \
#        -v hf-cache:/home/user/.cache/huggingface image-to-3d
FROM python:3.12-slim

# Hugging Face Spaces runs the container as user 1000
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements-demo.txt .
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-demo.txt

COPY src src
COPY app app
COPY api api
RUN chown -R user /app

USER user
ENV HOME=/home/user U2NET_HOME=/home/user/.u2net WEIGHTS_DIR=/app/weights PYTHONUNBUFFERED=1
# Download the background removal model now, so the first request is fast
RUN python -c "from rembg import new_session; new_session('u2net')"
# Cache folder of the Hugging Face weights, owned by the user (a named volume mounted here keeps this owner)
RUN mkdir -p /home/user/.cache/huggingface

EXPOSE 7860
CMD ["python", "api/main.py"]
