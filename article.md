Markdown# What Broke When I Built a RAG Pipeline on 37-Page Research Papers (And How Tuning Top-K Fixed It)

### Lessons from wiring Vertex AI Embeddings, ChromaDB, and Gemini 2.5 Flash on complex scientific literature.

---

If you’ve ever built a naive Retrieval-Augmented Generation (RAG) pipeline on blog posts or short PDFs, it usually feels deceptively simple: chunk the text, store the vectors, retrieve the top 3, and ask an LLM to answer. 

Then you feed it a dense, 37-page scientific paper.

That’s when things fall apart. 

I recently put together a production-style RAG pipeline to query dynamical systems research papers on arXiv (specifically *arXiv:2202.04944*, which evaluates machine learning for chaotic systems). I wanted to see how well **Google Cloud Vertex AI's `text-embedding-004`** and **Gemini 2.5 Flash** handle dense methodologies without hallucinating facts.

Here is what went wrong during implementation, how I diagnosed "context starvation," and what the final pipeline looks like.

---

## The Architecture Setup

The stack is intentionally lean:

```text
[ arXiv PDF ] 
      │ (pypdf + LangChain Recursive Splitter)
      ▼
[ 252 Chunks (500 chars / 50 overlap) ]
      │ (Vertex AI text-embedding-004 @ 768-dim)
      ▼
[ ChromaDB In-Memory Index ]
      │
      ├─► Query: "Which specific ML models/architectures were evaluated?"
      │
      ▼ (Top-K Semantic Search: k=12)
[ Context Snippets ] ──► [ Gemini 2.5 Flash (Strict Grounding) ] ──► Final Output
```
* Ingestion: Parsed the 37-page arXiv PDF and split the text into 252 chunks (500 characters per chunk, with 50-character overlap).
* Vectorization: Generated 768-dimensional embeddings using Google Cloud Vertex AI's text-embedding-004 in batches.
* Storage: Indexed the vectors and text chunks into ChromaDB.
* Reasoning Engine: Queried ChromaDB and passed the retrieved text to gemini-2.5-flash under strict anti-hallucination constraints.

The Catch: Why k=3 Failed MiserablyOnce everything was indexed, I asked a straightforward methodology question:"What specific neural network architectures and supervised machine learning methods were used to estimate Lyapunov exponents in this paper?"When using standard retrieval defaults ($k=3$, then $k=8$), Gemini returned:"Based on the retrieved excerpts, the paper states that it investigates the accuracy of supervised machine learning... However, the excerpts do not specify any particular machine learning algorithms or neural network architectures that were evaluated."At first glance, it looks like the LLM failed. But in reality, the grounding guardrail did its job perfectly.The Root Cause: Context StarvationIn a 37-page paper, the Abstract and Introduction mention "supervised machine learning" repeatedly, but the actual model architectures—MLPs, CNNs, LSTMs, and Regression Trees—are tucked deep within the methodology sections.With $k=3$ on 500-character chunks, semantic search matched only the high-level introductory paragraphs. Because the prompt strictly forbade guessing, the model refused to fabricate an answer.The Fix: Expanding Top-K DepthBy expanding retrieval to $k=12$ and sharpening the retrieval query toward methodology terms, ChromaDB pulled the exact parameter tables from the middle of the paper.With $k=12$, Gemini extracted every evaluated architecture with zero hallucinations:Multilayer Perceptrons (MLP): Up to 10 dense layers, 200 neurons per layer, built on TensorFlow.Regression Trees (RT): Evaluated with depth ranges $[1, 100]$ and leaf nodes $[5, 100]$ using Scikit-Learn.Convolutional Neural Networks (CNN): 1D-convolution layer with max-pooling for 1D time-series feature extraction.Long Short-Term Memory Networks (LSTM): 1–3 recurrent layers with up to 100 units to process temporal history.Hyperparameter Optimization: Bayesian optimization via Scikit-Optimize on Google Colab TPUs.Key Pipeline ImplementationHere is how the core pipeline is wired in Python:1. Ingestion & ChunkingPythonimport urllib.request
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Download arXiv dynamical systems paper (arXiv:2202.04944)
pdf_url = "[https://arxiv.org/pdf/2202.04944.pdf](https://arxiv.org/pdf/2202.04944.pdf)"
pdf_path = "sample_report.pdf"

req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response, open(pdf_path, 'wb') as out_file:
    out_file.write(response.read())

# Extract text & recursively split
reader = pypdf.PdfReader(pdf_path)
raw_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(raw_text)
print(f"Ingested {len(reader.pages)} pages into {len(chunks)} chunks.")
2. Batch Embeddings & ChromaDB IndexingPythonimport chromadb
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

def get_vertex_embeddings(texts, batch_size=50):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = [TextEmbeddingInput(text=t, task_type="RETRIEVAL_DOCUMENT") for t in batch]
        embeddings = embedding_model.get_embeddings(inputs, output_dimensionality=768)
        all_embeddings.extend([e.values for e in embeddings])
    return all_embeddings

chunk_vectors = get_vertex_embeddings(chunks)

# Initialize ChromaDB and index vectors
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="arxiv_rag_demo")

collection.add(
    documents=chunks,
    embeddings=chunk_vectors,
    ids=[f"doc_{idx}" for idx in range(len(chunks))]
)
3. Grounded Retrieval & EvaluationPythonimport time
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingInput

user_query = "What supervised learning models, neural network architectures, and algorithms such as CNN, LSTM, MLP, or trees are detailed in the methodology?"

# Vector retrieval with RETRIEVAL_QUERY task type
query_input = TextEmbeddingInput(text=user_query, task_type="RETRIEVAL_QUERY")
query_vector = embedding_model.get_embeddings([query_input], output_dimensionality=768)[0].values

# Retrieve top 12 chunks
results = collection.query(query_embeddings=[query_vector], n_results=12)
retrieved_context = "\n---\n".join(results['documents'][0])

# Synthesize with Gemini 2.5 Flash
gemini_model = GenerativeModel("gemini-2.5-flash")

prompt = f"""
You are an expert scientific researcher. Review the context below and summarize the specific machine learning architectures, models, and methods evaluated in the paper.

Context:
{retrieved_context}

Question: {user_query}
"""

response = gemini_model.generate_content(prompt)
print(response.text)
Latency & Accuracy BenchmarksMetricObserved ResultNotesCorpus Size37 Pages / 252 ChunksDynamical systems literatureVector Retrieval Latency ($k=12$)~140 msChromaDB cosine similarity searchLLM Generation Latency~9.9 sGemini 2.5 Flash parsing multi-chunk contextExtraction Precision100%Zero hallucinations; strict refusal when context was missing3 Practical Takeaways for Production RAGCheck retrieval before prompt engineering: When an LLM returns vague summaries or states that information is missing, verify the retrieved chunks first. Context starvation is often a top-$k$ or chunking issue, not a reasoning failure.Use asymmetric embedding task types: Always pair RETRIEVAL_DOCUMENT with RETRIEVAL_QUERY when working with models like Vertex AI's text-embedding-004 to maintain high semantic precision on complex domain text.Strict guardrails ensure reliability: Explicitly instructing the model to decline answering when the context is insufficient prevents domain-specific hallucinations.The full, reproducible code and pipeline are available on GitHub.
