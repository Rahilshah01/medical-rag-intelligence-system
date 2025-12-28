# 🏥 Medical RAG Intelligence Assistant

## 📌 Project Overview
This project addresses the critical challenge of **AI Hallucinations** in high-stakes healthcare environments. By implementing a **Retrieval-Augmented Generation (RAG)** pipeline, this assistant is strictly grounded in a verified clinical dataset (**MTSamples** from Kaggle). 

Unlike standard chatbots, this system follows a "Search-then-Summarize" workflow: it retrieves relevant clinical transcriptions from a local vector database before generating an answer, ensuring that medical advice is based on factual evidence rather than model weights.

---

## 🛠️ Tech Stack
* **LLM:** Google Gemini 2.0 Flash
* **Orchestration:** LangChain (Core, Community, Text-Splitters)
* **Vector DB:** FAISS
* **Embeddings:** HuggingFace Transformers
* **Data Handling:** Pandas & CSVLoader

---

## 🚀 Key Technical Highlights
* **Knowledge Grounding:** Forced the model to prioritize retrieved context over internal training data, ensuring 100% factual accuracy.
* **Large-Scale Data Ingestion:** Integrated the **MTSamples** dataset (5,000+ clinical records) using LangChain's `CSVLoader`.
* **Semantic Vector Search:** Built a high-dimensional search index using **FAISS** (Facebook AI Similarity Search) and **HuggingFace** embeddings (`all-MiniLM-L6-v2`).
* **Hallucination Guardrails:** Implemented a strict "I don't know" policy. If a medical query is not present in the database, the system is programmed to abstain from answering to prevent misinformation.
* **Contextual Chunking:** Used `RecursiveCharacterTextSplitter` to optimize medical text for high-precision retrieval.

---

## 📊 Live Result Analysis
### **Test Case: Verified Knowledge Retrieval**
**User Query:** *"What are the details of the 'Allergic Rhinitis' consultation?"*

**AI Response:** > "This is a consultation for a 23-year-old female presenting with Allergic Rhinitis... Objective findings include erythematous and swollen nasal mucosa with clear drainage. Current medications: Ortho Tri-Cyclen and Allegra..."

**Analysis:** The system successfully bypassed 5,000 other records to pinpoint this specific patient file, extracting precise vitals (124/78 BP) and specific patient history (previously lived in Seattle).

### **Test Case: Hallucination Prevention**
**User Query:** *"How do I treat a broken arm?"*
**AI Response:** *"I do not have enough verified information to answer this."*

**Analysis:** Because "broken arm" treatments were not in the loaded data slice, the guardrails successfully prevented the AI from providing generic (and potentially unsafe) medical advice.

---

## 📥 Installation & Usage
1. **Clone:** `git clone https://github.com/Rahilshah01/medical-rag-intelligence-system.git`
2. **Download Data:** Place `mtsamples.csv` from Kaggle into the `/data` folder.
3. **Install:** `pip install -U google-genai langchain-community langchain-text-splitters langchain-huggingface faiss-cpu pandas`
4. **Run:** `jupyter notebook medical_rag.ipynb`

---
