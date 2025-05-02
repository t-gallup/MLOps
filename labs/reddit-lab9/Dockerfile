FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY reddit_model_pipeline.joblib .

COPY *.py .

RUN ls -la

EXPOSE 8000

CMD ["uvicorn", "redditApp:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]