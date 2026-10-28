FROM python:3.12-slim
WORKDIR /app
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt
COPY src ./src
COPY web ./web
COPY lab ./lab
COPY results ./results
COPY tests ./tests
ENV HOST=0.0.0.0 PORT=8000 WORKERS=2
EXPOSE 8000
CMD ["sh", "-c", "gunicorn web.backend.wsgi:app --workers 1 --threads 8 --timeout 120 --bind 0.0.0.0:${PORT}"]
