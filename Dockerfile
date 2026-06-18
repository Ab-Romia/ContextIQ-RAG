FROM python:3.11-slim

# Hugging Face Spaces run containers as a non-root user with UID 1000. Matching that
# here avoids permission surprises with the model cache and uploaded files.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed \
    PORT=7860 \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Download the embedding and reranker models into the image so a cold start does not
# re-fetch them. bm25s needs no model. No C toolchain is installed: bm25s runs its
# scipy path and the rest ship as wheels.
RUN python -m app.warmup

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
