# Module 3: GenAI Support Assistant

A RAG-based support bot that answers questions about Zepto's policies. Built with LangGraph, FastAPI, and Docker.

## How it works

1. **Ingestion** — 8 policy documents (txt files) are loaded, one chunk per document, embedded using `sentence-transformers/all-MiniLM-L6-v2`, and stored in ChromaDB with cosine similarity
2. **Query routing** — LangGraph classifies the incoming question: if it's about a Zepto policy, it goes to retrieval; otherwise it gives a canned "I can only answer policy questions" response
3. **Retrieval + Answer** — Top 3 chunks are fetched from ChromaDB and used to generate the answer
4. **API** — FastAPI exposes `POST /ask` and `GET /health`

## Running it

```bash
pip install -r requirements.txt
python ingest.py          # builds the ChromaDB index
uvicorn main:app --port 7860
```

Example call:
```bash
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the delivery fee for small orders?"}'
```

Response:
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials...",
  "sources": ["doc_01_chunk_0", "doc_05_chunk_0", "doc_02_chunk_0"],
  "confidence": 1.0
}
```

For a general question like "What is the capital of France?":
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## MOCK_LLM mode

By default `MOCK_LLM=1` — no API key needed, no network calls to any LLM. The classification uses a keyword check and the answer is built from the top retrieved chunk. This is what gets graded.

Setting `MOCK_LLM=0` with a `GROQ_API_KEY` will use a real LLM, but that's optional.

## Docker

```bash
docker build -t zepto-assistant .
docker run --rm -p 7860:7860 zepto-assistant
```

The model and ChromaDB index are built into the image at build time (`RUN python ingest.py` in the Dockerfile), so the container starts up and serves requests immediately.

## LangGraph flow

```
START → classify_intent → (policy?) → retrieve_and_answer → END
                        → (general?) → direct_answer      → END
```
