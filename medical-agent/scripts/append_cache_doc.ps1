$doc = "E:\PythonProject2\medical-agent\FINAL_DOC.md"
Add-Content -Encoding UTF8 -Path $doc -Value "
## 高频问题缓存（已落地）

**实现方式：**
- 检索缓存：`Retriever` 中加入 `CacheStore`，命中后直接返回 top 文档 ID
- 生成缓存：`Generator` 中加入 `CacheStore`，命中后直接返回答案
- 缓存后端：Redis（不可用时内存 fallback）

**关键代码位置：**
- `app/storage/cache_store.py`
- `app/rag/retriever.py`
- `app/rag/generator.py`

**配置项：**
- `cache_enabled` / `cache_ttl_seconds`

**意义：**
- 高频问题延迟显著下降
- 降低 LLM 负载，提高并发能力
"
