import os
import re
from pathlib import Path

def get_config(file, var):
    if not os.path.exists(file): return "N/A"
    content = Path(file).read_text()
    match = re.search(f"{var} = [\"']?(.*?)[\"']?$", content, re.M)
    return match.group(1) if match else "Not Set"

# Audit Current Configs
embed_model = get_config('build_kb.py', 'EMBED_MODEL')
chunk_size = get_config('build_kb.py', 'CHUNK_SIZE')
sim_threshold = get_config('vault_query.py', 'SIM_THRESHOLD')

doc_content = f"""
# 🚀 LLM Vault: System Overview
*Auto-generated Status Report*

## 🛠 Project Configuration
- **Module 4 (LLM):** Using `{get_config('vault_query.py', 'CHAT_MODEL')}`
- **Module 3 (Vector DB):** `{embed_model}` with `{chunk_size}` char chunks
- **Retrieval Logic:** Cosine Similarity (Threshold: `{sim_threshold}`)

## 📚 Study Mapping (From Handwritten Notes)
| Module | Topic | Status | Note Reference |
| :--- | :--- | :--- | :--- |
| **Module 1** | RAG Architecture | ✅ Active | Retrieval + Generation split |
| **Module 2** | Keyword Search | ⚠️ Lacking | Needs BM25/TF-IDF "Plateau" logic |
| **Module 3** | Vector Retrieval | ✅ Active | Cosine Similarity & KNN loop |
| **Module 4** | LLM Generation | ✅ Active | Autoregressive token generation |

## 📂 Vault Stats
- **Total Chunks in `kb.json`:** (Processed from your resumes folder)
"""
Path("SYSTEM_OVERVIEW.md").write_text(doc_content)