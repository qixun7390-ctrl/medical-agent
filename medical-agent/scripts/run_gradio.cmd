@echo off
setlocal
set PYTHONPATH=E:\PythonProject2
set LLM_ENDPOINT=http://127.0.0.1:8001/v1
set LLM_MODEL=/models/Qwen2.5-1.5B-Instruct

python E:\PythonProject2\app\gradio_app.py
endlocal
