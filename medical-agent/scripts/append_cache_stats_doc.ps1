$doc = "E:\PythonProject2\medical-agent\FINAL_DOC.md"
Add-Content -Encoding UTF8 -Path $doc -Value "
## 缓存命中率统计（已落地）

**实现位置：**
- `app/utils/cache_stats.py`：命中统计
- `app/storage/cache_store.py`：读取/写入时累积 hit/miss
- `app/api/routes.py`：新增 /api/cache_stats
- Gradio 前端展示：Cache Stats 区域

**查看方式：**
- API: `GET /api/cache_stats`
- UI: 直接在前端面板查看
"
