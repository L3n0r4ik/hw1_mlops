FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/output /app/logs /app/models

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

VOLUME /app/output

CMD ["python", "./app/app.py"]
