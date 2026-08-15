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
