# 🚀 LLM Vault: RAG Resume Assistant

> **Status:** Active Development  
> **Architecture:** Retrieval-Augmented Generation (RAG)  
> **Tech Stack:** Python, OpenAI, FastAPI, GitHub Actions

## 📖 Project Overview
**LLM Vault** is an intelligent "Resume Assistant" built on a Retrieval-Augmented Generation (RAG) architecture. It allows users to query a knowledge base of resumes using natural language to retrieve specific candidates, skills, or experience. 

The system separates concerns into discrete pipelines for data ingestion, vector retrieval, and LLM-based response generation, ensuring grounded and accurate answers.

---

## 🏗️ System Architecture

| Component | Functionality | Implementation File | Status |
| :--- | :--- | :--- | :--- |
| **Orchestration** | **RAG Pipeline** | `vault_query.py` | ✅ **Active** (Retrieval + Generation) |
| **API Layer** | **REST Interface** | `app.py` | ✅ **Active** (FastAPI Server) |
| **Vector Store** | **Similarity Search** | `build_kb.py` / `kb.json` | ✅ **Active** (Cosine Similarity) |
| **Retrieval** | **Keyword Search** | *Planned Feature* | ⚠️ **Roadmap** (BM25 Algorithm) |
| **DevOps** | **Auto-Documentation** | `.github/workflows/` | ✅ **Active** (CI/CD Pipeline) |

---

## 📂 Repository Structure

### 🧠 Core Engine
* **`resumes/`**: **Data Source.** Stores raw `.pdf` and `.docx` resume files.
* **`build_kb.py`**: **Indexer.** Handles document parsing, chunking (default: 400 chars), and vector embedding generation.
* **`kb.json`**: **Knowledge Base.** Lightweight JSON-based vector store containing text chunks and embeddings.
* **`vault_query.py`**: **Retriever & Generator.** Executes the vector search logic and prompts the LLM for the final response.

### 🛡️ Quality Assurance & CI/CD
* **`eval_data.json`**: **Evaluation Dataset.** Contains "Ground Truth" queries and expected results for regression testing.
* **`gen_docs.py`**: **System Auditor.** A script that scans the codebase for active configurations (e.g., chunk size, model version) to update documentation automatically.
* **`.github/workflows/auto_doc.yml`**: **Automation.** A GitHub Action that triggers `gen_docs.py` on every commit to ensure system reports remain up-to-date.

### ⚙️ Configuration
* **`.env`**: Environment variables (e.g., `OPENAI_API_KEY`). **(Not tracked in Git)**
* **`requirements.txt`**: Python dependencies (`openai`, `numpy`, `fastapi`, `pypdf2`, etc.).

---

## ⚡ Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with your API key
echo "OPENAI_API_KEY=your-key-here" > .env
