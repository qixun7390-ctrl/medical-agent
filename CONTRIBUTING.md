# 贡献指南

感谢你愿意为本项目做贡献！

## 开发环境
- Python 3.11
- Windows / Linux / macOS

## 安装
```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## 运行
```bash
python -m uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
set USE_STREAM=1
python app/gradio_app.py
```

## 代码规范
- 保持模块职责清晰
- 不要在核心路径里引入重型依赖
- 新增规则时保持 JSON 结构一致

## 提交 PR
1. Fork 仓库并创建分支
2. 保持提交信息清晰（feat/fix/docs）
3. 提交前自测主要功能

## 报告问题
请在 issue 中提供：
- 复现步骤
- 期望结果与实际结果
- 相关日志或截图
