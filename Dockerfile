FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY rppg/ ./rppg/

WORKDIR /app/backend

ENV PORT=8080

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}