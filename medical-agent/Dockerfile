FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install -e .
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
