$env:LLM_ENDPOINT = "http://127.0.0.1:8001/v1"
$env:LLM_MODEL = "/models/Qwen2.5-1.5B-Instruct"
$env:RERANKER_DEVICE = "cpu"
$env:PYTORCH_ALLOC_CONF = "expandable_segments:True"

Set-Location "E:\PythonProject2"
E:\PythonProject2\.venv\Scripts\python -m uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 --workers 2
