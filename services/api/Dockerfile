FROM python:3.12-slim

WORKDIR /app

COPY services/api/requirements.txt /app/services/api/
RUN pip install --no-cache-dir -r services/api/requirements.txt

COPY services/__init__.py /app/services/
COPY services/api/ /app/services/api/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
