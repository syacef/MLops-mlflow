FROM python:3.11-slim-bookworm

WORKDIR /app
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY .env main.py model.py /app/

CMD ["python3", "main.py"]