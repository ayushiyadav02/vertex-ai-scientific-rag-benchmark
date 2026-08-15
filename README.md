# 🔬 Production Scientific RAG Pipeline with Vertex AI & Gemini 2.5 Flash

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Google Cloud Vertex AI](https://img.shields.io/badge/Google%20Cloud-Vertex%20AI-4285F4?logo=googlecloud&logoColor=white)](https://cloud.google.com/vertex-ai)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-orange)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Retrieval-Augmented Generation (RAG) benchmark evaluating the extraction, synthesis, and grounding capabilities of **Google Cloud Vertex AI (`text-embedding-004`)** and **Gemini 2.5 Flash (`gemini-2.5-flash`)** over dense scientific literature.

---

## 📌 Project Overview

Parsing and querying academic literature with Large Language Models (LLMs) often suffers from **context starvation** and hallucination when relevant methodologies are buried deep within papers. 

This project implements a reproducible RAG pipeline tested against a 37-page dynamical systems research paper (*arXiv:2202.04944: "Supervised machine learning to estimate instabilities in chaotic systems"*). We analyze the engineering trade-offs between chunking overlap, embedding retrieval latency, and the retrieval depth ($k$) required for Gemini 2.5 Flash to extract complex machine learning architectures with zero hallucinations.

---

## 🏗️ Architecture & Pipeline Flow

```text
 ┌────────────────────────┐      ┌─────────────────────────┐
 │ arXiv 37-Page Research │ ───► │ Recursive Chunker       │
 │ Paper (PDF Ingestion)  │      │ (500 chars / 50 overlap)│
 └────────────────────────┘      └────────────┬────────────┘
                                              │
                                              ▼
 ┌────────────────────────┐      ┌─────────────────────────┐
 │ ChromaDB Vector Store  │ ◄─── │ Vertex AI Embeddings    │
 │ (In-Memory Index)      │      │ (text-embedding-004)    │
 └──────────┬─────────────┘      └─────────────────────────┘
            │
            │ Top-k Semantic Search (k=12)
            ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Gemini 2.5 Flash (`gemini-2.5-flash`)                   │
 │ Context-Grounded Synthesis & Strict Refusal Guardrails  │
 └─────────────────────────────────────────────────────────┘
```

* **Document Ingestion & Chunking:** Ingests the multi-page PDF via `pypdf` and recursively splits the text into 252 discrete segments (500 characters with a 50-character overlap) using LangChain.
* **Asymmetric Vector Embeddings:** Vectorizes document chunks in batches using Vertex AI's `text-embedding-004` (768 dimensions, `task_type="RETRIEVAL_DOCUMENT"`).
* **Vector Database Indexing:** Indexes chunk vectors and document metadata into an in-memory `ChromaDB` collection.
* **Grounded Generation & Guardrails:** Embeds user queries with `task_type="RETRIEVAL_QUERY"`, performs similarity retrieval ($k=12$), and passes the context to `gemini-2.5-flash` under strict anti-hallucination prompt constraints.

---

### 📊 Benchmark & Latency Results

| Dimension / Metric | Experimental Measurement | Notes |
| :--- | :--- | :--- |
| **Corpus Size** | 37 Pages / 252 Chunks | Source: arXiv:2202.04944 |
| **Vector Embedding Dimension** | 768 float32 | Vertex AI `text-embedding-004` |
| **Vector Retrieval Latency ($k=12$)** | **~140 ms** | ChromaDB cosine similarity search |
| **LLM Generation Latency** | **~9.9 s** | Gemini 2.5 Flash multi-chunk reasoning |
| **Extraction Precision** | **100%** | Zero hallucinations; full model coverage |

**Key Experimental Finding: Context Starvation**
* **Low Retrieval ($k=3$ to $k=8$):** The model retrieved high-level mentions from the abstract/introduction. Gemini strictly refused to fabricate missing architectural details, correctly stating that the excerpts lacked specific algorithm parameters.
* **Tuned Retrieval ($k=12$):** Surfaced deep methodology tables, enabling the LLM to successfully extract all four evaluated architectures:
  * **Multilayer Perceptrons (MLP):** Up to 10 dense layers, 200 neurons/layer in TensorFlow.
  * **Regression Trees (RT):** Tuned across depths $[1, 100]$ and leaf nodes $[5, 100]$ via Scikit-Learn.
  * **Convolutional Neural Networks (CNN):** 1D-convolution with max-pooling for 1D temporal series.
  * **Long Short-Term Memory (LSTM):** 1–3 recurrent layers with up to 100 units.
  * **Hyperparameter Optimization:** Bayesian optimization via Scikit-Optimize on Google Colab TPUs.

---

### 🚀 Quickstart Guide

**1. Clone the Repository**
```bash
git clone [https://github.com/ayushiyadav02/vertex-ai-scientific-rag-benchmark.git](https://github.com/ayushiyadav02/vertex-ai-scientific-rag-benchmark.git)
cd vertex-ai-scientific-rag-benchmark
```
2. Install Dependencies
```
Bash
pip install google-cloud-aiplatform chromadb pypdf langchain-text-splitters
```
3. Google Cloud Authentication & Setup
Ensure the Vertex AI API (aiplatform.googleapis.com) is enabled in your Google Cloud Project:

Python
```
import vertexai
from google.colab import auth

auth.authenticate_user()
vertexai.init(project="YOUR_GCP_PROJECT_ID", location="us-central1")
```
4. Run the Pipeline
Open and execute the included notebook:


vertex_ai_scientific_rag_benchmark.ipynb

🛠️ Tech Stack
* Large Language Model: Google Cloud Vertex AI gemini-2.5-flash
* Embedding Model: Google Cloud Vertex AI text-embedding-004
* Vector Store: ChromaDB
* Orchestration & Splitting: LangChain Text Splitters
*Document Parsing: PyPDF

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
