@echo off
setlocal
set MODEL_DIR=E:\PythonProject2\medical-agent\model\qwen\Qwen2___5-1___5B-Instruct
if not exist "%MODEL_DIR%" (
  echo Model directory not found: %MODEL_DIR%
  echo Please download the model to this path before starting.
  exit /b 1
)

cd /d E:\PythonProject2\medical-agent\docker

docker compose -f docker-compose.vllm.yml up -d
endlocal
