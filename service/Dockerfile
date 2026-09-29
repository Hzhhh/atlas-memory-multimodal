FROM python:3.11-slim

WORKDIR /app
COPY SimpleMem /app/SimpleMem
COPY service /app/service

RUN pip install --no-cache-dir -e /app/SimpleMem/OmniSimpleMem \
    faiss-cpu Pillow fastapi uvicorn python-multipart

WORKDIR /app/service
ENV DATA_DIR=/data PORT=8000 PYTHONUNBUFFERED=1
VOLUME /data

EXPOSE 8000
CMD ["python", "aml_mm_service.py"]
