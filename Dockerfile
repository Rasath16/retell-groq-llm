FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install uvloop

COPY . .

EXPOSE 8080

CMD ["python", "main.py"]
