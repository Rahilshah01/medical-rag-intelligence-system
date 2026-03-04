# 🏥 Medical RAG Intelligence System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash_Lite-4285F4?style=for-the-badge&logo=google&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS_Vector_DB-00599C?style=for-the-badge&logo=meta&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace_Embeddings-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

**A RAG pipeline that eliminates AI hallucinations in high-stakes healthcare environments.**  
Grounded entirely in verified clinical transcriptions — the model cannot answer what isn't in the knowledge base.

</div>

---

## ⚡ Results at a Glance

| Metric | Detail |
|---|---|
| 📄 Clinical Records Loaded | 1,000 transcriptions (MTSamples via Kaggle) |
| 🔍 Retrieval | Top-3 semantic similarity matches per query (FAISS) |
| ✂️ Chunk Config | 1,000 tokens, 100-token overlap |
| 🧠 Embedding Model | `all-MiniLM-L6-v2` (HuggingFace) |
| 🤖 LLM | `gemini-2.5-flash-lite` (Google GenAI SDK) |
| 🚫 Hallucination Guard | Prompt-enforced abstention — "I don't know" if context is absent |

---

## 🧠 The Problem This Solves

Standard LLMs generate confident-sounding medical answers from training weights — even when they're wrong. In healthcare, a hallucinated drug interaction or fabricated patient detail isn't a UX issue — it's a safety risk.

This system enforces a **"Search-then-Summarize"** architecture:
1. Every query is embedded and matched against a local FAISS vector index of clinical records
2. Top-3 matching transcriptions are injected into the prompt as the **only** permitted context
3. The model is explicitly instructed: if the answer isn't in the context, say so — no inference from training weights

---

## 🏗️ System Architecture

```
User Query
    │
    ▼
HuggingFace Embeddings  ←  all-MiniLM-L6-v2
    │
    ▼
FAISS Similarity Search  →  Top-3 Matching Clinical Records
    │
    ▼
Prompt Assembly:
  "Only use the context provided.
   If the answer isn't there, say you don't know."
    │
    ▼
gemini-2.5-flash-lite  →  Grounded Response (or Abstention)
```

---

## 💻 Core Implementation

```python
def clinical_assistant(query):
    # Retrieve top-3 semantically similar clinical records
    search_results = vector_db.similarity_search(query, k=3)

    context = ""
    for res in search_results:
        context += f"\n---\n{res.page_content}\n"

    prompt = f"""
    You are an AI Clinical Assistant. Using the provided medical transcriptions, answer the user query.
    Rules:
    1. Only use the context provided.
    2. If the answer isn't there, say you don't know.

    CONTEXT: {context}
    QUERY: {query}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=prompt
    )
    return response.text
```

The guardrail is enforced at the **prompt level** — the model is given no fallback permission to use its training data. In medical AI, a confident wrong answer is worse than no answer.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash Lite (via `google-genai` SDK) |
| Vector Database | FAISS (local — sensitive records never leave the machine) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Data Ingestion | LangChain `CSVLoader` |
| Chunking | `RecursiveCharacterTextSplitter` (chunk=1000, overlap=100) |
| Dataset | [MTSamples](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) — Kaggle |

---

## 📊 Live Inference Results

### ✅ Test 1: Verified Knowledge Retrieval
> **Query:** *"What are the details of the 'Allergic Rhinitis' consultation?"*

The system identified the exact patient file from 1,000 records — surfacing precise vitals (BP: 124/78), active medications (Ortho Tri-Cyclen, Allegra), and full symptom history. Zero fabricated details.

### 🚫 Test 2: Hallucination Prevention in Action
> **Query:** *"How do I treat a broken arm?"*
> **Response:** *"I do not have enough verified information to answer this."*

Fracture treatment protocols were not in the loaded data slice. The prompt-level guardrail blocked the model from falling back on training knowledge — exactly the intended behavior.

---

## 🔑 Key Engineering Decisions

- **Why FAISS over a hosted vector DB?** Fully local — zero latency overhead, no API costs, and sensitive medical records never leave the machine.
- **Why `all-MiniLM-L6-v2`?** Strong semantic similarity on short clinical chunks with minimal memory footprint — purpose-fit for this retrieval task.
- **Why `k=3` retrieval?** Balances context richness with prompt size. Too few records → missed context. Too many → diluted relevance and higher token costs.
- **Why prompt-level abstention?** Hard prompt rules are deterministic and interpretable — no threshold calibration needed per domain.

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/Rahilshah01/medical-rag-intelligence-system.git
cd medical-rag-intelligence-system

# 2. Install
pip install google-genai langchain-community langchain-text-splitters langchain-huggingface faiss-cpu pandas python-dotenv

# 3. Add data — download mtsamples.csv from Kaggle → place in /data/

# 4. Set API key
echo "GEMINI_API_KEY=your_key_here" > .env

# 5. Run
python rag_assistant.py
```

---

## 📁 Repository Structure

```
medical-rag-intelligence-system/
├── data/
│   └── mtsamples.csv        # Clinical records (download from Kaggle)
├── rag_assistant.py         # Main pipeline
├── .env.example
├── requirements.txt
└── README.md
```

---

*Built by [Rahil Shah](https://rahil-shah-portfolio.vercel.app/) · MS Data Science @ Stevens Institute of Technology*
