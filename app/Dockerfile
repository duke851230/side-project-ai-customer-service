FROM python:3.11-slim

WORKDIR /workspace

COPY app/requirements.txt ./app/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.11.0+cpu \
    && pip install --no-cache-dir -r app/requirements.txt

COPY . .
