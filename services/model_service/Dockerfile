FROM python:3.12-slim

WORKDIR /app

# Install model service dependencies
COPY services/model_service/requirements.txt /app/services/model_service/
RUN pip install --no-cache-dir -r services/model_service/requirements.txt

# Copy the CLSA library and model service code
COPY clsa/ /app/clsa/
COPY services/__init__.py /app/services/
COPY services/model_service/ /app/services/model_service/

ENV PYTHONPATH=/app
ENV CLSA_DEVICE=cpu

EXPOSE 8001

CMD ["uvicorn", "services.model_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
