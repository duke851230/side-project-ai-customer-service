FROM python:3.11-slim

WORKDIR /workspace

COPY mock_server/requirements.txt ./mock_server/requirements.txt
RUN pip install --no-cache-dir -r mock_server/requirements.txt

COPY . .
