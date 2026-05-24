FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
COPY rag_index.faiss .
COPY rag_meta.pkl .
COPY rich_users.csv .
COPY persona_cache/ ./persona_cache/
COPY item_cache/ ./item_cache/
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
