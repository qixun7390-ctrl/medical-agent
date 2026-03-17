@echo off
setlocal
set PYTHONPATH=E:\PythonProject2\medical-agent
set LLM_ENDPOINT=http://127.0.0.1:8001/v1
set LLM_MODEL=/models/Qwen2.5-1.5B-Instruct
set GEN_EVAL_SAMPLE=200
set USE_PRECOMPUTED_RETRIEVAL=0
set RERANKER_DISABLED=0
set REDIS_DISABLED=0
set PG_DISABLED=0

cd /d E:\PythonProject2\medical-agent\docker

docker compose -f docker-compose.eval.yml up -d

timeout /t 5 >nul

cd /d E:\PythonProject2\medical-agent
python scripts\run_generation_eval.py
python scripts\eval_generation.py
python scripts\eval_faithfulness_llm.py
python scripts\build_report_md.py
python scripts\build_report.py

endlocal
