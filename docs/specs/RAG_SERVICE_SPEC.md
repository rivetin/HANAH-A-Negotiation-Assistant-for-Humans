# RAG Service Specification

**Component:** LlamaIndex RAG Microservice  
**Version:** 1.0  
**Last Updated:** November 6, 2025

---

## Overview

The RAG (Retrieval-Augmented Generation) microservice is a standalone FastAPI application powered by LlamaIndex. It is responsible for ingesting negotiation knowledge (Chris Voss techniques + annotated transcripts), indexing content in Pinecone, and serving contextual negotiation advice via vector search and LLM synthesis.

**Key Responsibilities**:
- Ingest and chunk negotiation corpus with rich metadata
- Index embeddings in Pinecone for semantic search
- Accept query requests with session context
- Retrieve top-k relevant technique snippets
- Optionally synthesize plans + suggested replies using Bedrock
- Return provenance (sources, scores, metadata)

---

## Architecture

### Application Structure

```
hanah-rag-service/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── config.py                # Configuration
│   ├── models/
│   │   ├── ingestion.py         # Ingestion request/response schemas
│   │   ├── query.py             # Query request/response schemas
│   │   └── document.py          # Document and chunk models
│   ├── ingestion/
│   │   ├── chunker.py           # Text chunking logic
│   │   ├── metadata_builder.py # Metadata extraction
│   │   └── manifest.py          # Ingestion manifest tracker
│   ├── retrieval/
│   │   ├── vector_store.py      # Pinecone adapter
│   │   ├── retriever.py         # LlamaIndex retriever
│   │   └── reranker.py          # Metadata-based re-ranking
│   ├── synthesis/
│   │   ├── plan_generator.py    # Bedrock-based plan synthesis
│   │   └── prompts.py           # Synthesis prompts
│   ├── api/
│   │   ├── ingest_endpoint.py   # POST /ingest
│   │   └── query_endpoint.py    # POST /query
│   └── utils/
│       ├── bedrock_client.py    # AWS Bedrock wrapper
│       └── cache.py             # Redis/in-memory cache
├── data/
│   ├── voss_corpus/             # Never Split the Difference content
│   └── transcripts/             # Annotated negotiation transcripts
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Configuration (`config.py`)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    environment: str = "development"
    
    # Pinecone
    pinecone_api_key: str
    pinecone_environment: str
    pinecone_index_name: str = "hanah-negotiation"
    pinecone_namespace: str = "voss_rag"
    
    # AWS Bedrock
    aws_region: str = "us-east-1"
    aws_access_key_id: str
    aws_secret_access_key: str
    bedrock_embedding_model: str = "amazon.titan-embed-text-v1"
    bedrock_generation_model: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    
    # LlamaIndex
    chunk_size: int = 300
    chunk_overlap: int = 50
    top_k: int = 4
    similarity_threshold: float = 0.7
    
    # Synthesis
    enable_synthesis: bool = True
    max_tokens: int = 800
    
    # Caching
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600
    
    # Observability
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## Data Models

### Ingestion Models (`models/ingestion.py`)

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class IngestDocumentRequest(BaseModel):
    source_id: str                              # Unique identifier
    source_type: Literal["book", "transcript", "article"]
    title: str
    content: str                                # Full text
    metadata: dict = {}                         # Custom metadata
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

class IngestBatchRequest(BaseModel):
    documents: list[IngestDocumentRequest]
    namespace: str = "voss_rag"
    force_reindex: bool = False                 # Delete and re-ingest

class IngestResponse(BaseModel):
    status: Literal["success", "partial", "failed"]
    ingested_count: int
    chunk_count: int
    failed_documents: list[str] = []
    manifest_id: str                            # Reference to ingestion manifest
    elapsed_seconds: float
```

### Query Models (`models/query.py`)

```python
from pydantic import BaseModel
from typing import Optional, Literal

class SalaryValue(BaseModel):
    amount: float
    period: Literal["yearly", "monthly"]
    currency: str

class SessionContext(BaseModel):
    offered_salary: Optional[SalaryValue] = None
    desired_salary: Optional[SalaryValue] = None
    batna: Optional[str] = None
    parsed_texts: Optional[str] = None          # Extracted JD/offer text

class QueryPreferences(BaseModel):
    technique_priority: list[str] = []          # e.g., ["calibrated_question", "label"]
    response_length: Literal["short", "medium", "long"] = "short"
    include_synthesis: bool = True

class QueryRequest(BaseModel):
    session_id: str
    role_type: Literal["candidate", "recruiter"]
    user_message: str
    session_context: SessionContext
    preferences: QueryPreferences = QueryPreferences()
    top_k: Optional[int] = None                 # Override default

class RetrievedSnippet(BaseModel):
    id: str                                     # Pinecone vector ID
    snippet: str                                # Retrieved text
    technique: str                              # e.g., "calibrated_question"
    source: str                                 # e.g., "voss_book"
    score: float                                # Similarity score
    metadata: dict = {}

class SuggestedReply(BaseModel):
    text: str
    technique: str
    rationale: str
    confidence: Literal["high", "medium", "low"]

class SynthesizedPlan(BaseModel):
    plan: str
    suggested_replies: list[SuggestedReply]
    rationales: list[str]
    confidence: Literal["high", "medium", "low"]

class QueryResponse(BaseModel):
    retrieved: list[RetrievedSnippet]
    synthesized: Optional[SynthesizedPlan] = None
    debug: dict = {}                            # Scores, timings, etc.
```

### Document Models (`models/document.py`)

```python
from pydantic import BaseModel

class DocumentChunk(BaseModel):
    chunk_id: str
    source_id: str
    chunk_index: int
    text: str
    metadata: dict                              # technique, intent, source, etc.
    embedding: Optional[list[float]] = None     # Computed by Bedrock
```

---

## Ingestion Pipeline

### Chunking Strategy (`ingestion/chunker.py`)

**Goal**: Create semantically meaningful chunks (200–400 tokens) that preserve technique context.

```python
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

class NegotiationChunker:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator="\n\n"
        )
    
    def chunk_document(self, source_id: str, content: str, metadata: dict) -> list[DocumentChunk]:
        """Split document into chunks with metadata."""
        doc = Document(text=content, metadata=metadata)
        nodes = self.splitter.get_nodes_from_documents([doc])
        
        chunks = []
        for i, node in enumerate(nodes):
            chunk = DocumentChunk(
                chunk_id=f"{source_id}_chunk_{i}",
                source_id=source_id,
                chunk_index=i,
                text=node.get_content(),
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(nodes)
                }
            )
            chunks.append(chunk)
        
        return chunks
```

### Metadata Schema

Each chunk **must** include the following metadata:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `technique` | str | Voss technique name | `"calibrated_question"` |
| `intent` | str | Goal of technique | `"invite_collaboration"` |
| `source` | str | Source identifier | `"voss_book"`, `"transcript_042"` |
| `source_type` | str | Type of source | `"book"`, `"transcript"`, `"article"` |
| `language` | str | Language code | `"en"` |
| `excerpt` | bool | True if chunk is excerpt | `true` |
| `page` | int | Page number (for books) | `142` |
| `role_context` | str | Candidate/recruiter/both | `"candidate"` |

**Technique Taxonomy** (standardized values):

- `calibrated_question`
- `label`
- `mirror`
- `accusation_audit`
- `anchoring`
- `tactical_empathy`
- `black_swan`
- `loss_aversion`
- `no_oriented_questions`

### Metadata Builder (`ingestion/metadata_builder.py`)

```python
import re

def extract_technique(text: str) -> str:
    """Heuristic to extract technique from text."""
    text_lower = text.lower()
    
    if "calibrated question" in text_lower or "how" in text_lower or "what" in text_lower:
        return "calibrated_question"
    elif "label" in text_lower or "it sounds like" in text_lower:
        return "label"
    elif "mirror" in text_lower:
        return "mirror"
    elif "accusation audit" in text_lower:
        return "accusation_audit"
    elif "anchor" in text_lower:
        return "anchoring"
    else:
        return "general"

def build_metadata_from_source(source_type: str, title: str, content: str) -> dict:
    """Build metadata for a source document."""
    technique = extract_technique(content)
    
    return {
        "technique": technique,
        "intent": infer_intent(technique),
        "source": slugify(title),
        "source_type": source_type,
        "language": "en",
        "excerpt": True,
        "role_context": "both"  # Default to both candidate and recruiter
    }

def infer_intent(technique: str) -> str:
    """Map technique to intent."""
    intent_map = {
        "calibrated_question": "invite_collaboration",
        "label": "acknowledge_emotion",
        "mirror": "encourage_elaboration",
        "accusation_audit": "defuse_objection",
        "anchoring": "set_expectations"
    }
    return intent_map.get(technique, "general")
```

### Ingestion Endpoint (`api/ingest_endpoint.py`)

```python
from fastapi import APIRouter
from app.models.ingestion import IngestBatchRequest, IngestResponse
from app.ingestion.chunker import NegotiationChunker
from app.ingestion.metadata_builder import build_metadata_from_source
from app.retrieval.vector_store import PineconeVectorStore
from app.utils.bedrock_client import BedrockClient

router = APIRouter()

@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestBatchRequest):
    """Ingest batch of documents into vector store."""
    start_time = time.time()
    
    chunker = NegotiationChunker(
        chunk_size=request.documents[0].chunk_size or settings.chunk_size,
        chunk_overlap=request.documents[0].chunk_overlap or settings.chunk_overlap
    )
    
    vector_store = PineconeVectorStore(namespace=request.namespace)
    bedrock_client = BedrockClient()
    
    all_chunks = []
    failed_docs = []
    
    for doc in request.documents:
        try:
            # Build metadata
            metadata = {
                **build_metadata_from_source(doc.source_type, doc.title, doc.content),
                **doc.metadata
            }
            
            # Chunk document
            chunks = chunker.chunk_document(doc.source_id, doc.content, metadata)
            
            # Generate embeddings
            for chunk in chunks:
                embedding = await bedrock_client.generate_embedding(chunk.text)
                chunk.embedding = embedding
            
            all_chunks.extend(chunks)
        
        except Exception as e:
            logger.error(f"Failed to ingest {doc.source_id}: {e}")
            failed_docs.append(doc.source_id)
    
    # Upsert to Pinecone
    if request.force_reindex:
        vector_store.delete_all(namespace=request.namespace)
    
    vector_store.upsert_chunks(all_chunks)
    
    # Create ingestion manifest
    manifest_id = create_manifest(request, all_chunks)
    
    elapsed = time.time() - start_time
    
    return IngestResponse(
        status="success" if not failed_docs else "partial",
        ingested_count=len(request.documents) - len(failed_docs),
        chunk_count=len(all_chunks),
        failed_documents=failed_docs,
        manifest_id=manifest_id,
        elapsed_seconds=elapsed
    )
```

### Ingestion Manifest (`ingestion/manifest.py`)

**Purpose**: Track which chunks were ingested from which sources for reproducibility and debugging.

```python
from datetime import datetime
import json

class IngestionManifest:
    def __init__(self, manifest_id: str):
        self.manifest_id = manifest_id
        self.timestamp = datetime.utcnow()
        self.sources = []
    
    def add_source(self, source_id: str, chunk_ids: list[str]):
        self.sources.append({
            "source_id": source_id,
            "chunk_count": len(chunk_ids),
            "chunk_ids": chunk_ids
        })
    
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "manifest_id": self.manifest_id,
                "timestamp": self.timestamp.isoformat(),
                "sources": self.sources
            }, f, indent=2)

def create_manifest(request: IngestBatchRequest, chunks: list[DocumentChunk]) -> str:
    """Create and save ingestion manifest."""
    manifest_id = f"manifest_{int(time.time())}"
    manifest = IngestionManifest(manifest_id)
    
    # Group chunks by source
    source_chunks = {}
    for chunk in chunks:
        if chunk.source_id not in source_chunks:
            source_chunks[chunk.source_id] = []
        source_chunks[chunk.source_id].append(chunk.chunk_id)
    
    for source_id, chunk_ids in source_chunks.items():
        manifest.add_source(source_id, chunk_ids)
    
    manifest.save(f"manifests/{manifest_id}.json")
    return manifest_id
```

---

## Retrieval Pipeline

### Vector Store Adapter (`retrieval/vector_store.py`)

```python
import pinecone
from app.config import settings

class PineconeVectorStore:
    def __init__(self, namespace: str = "voss_rag"):
        pinecone.init(
            api_key=settings.pinecone_api_key,
            environment=settings.pinecone_environment
        )
        self.index = pinecone.Index(settings.pinecone_index_name)
        self.namespace = namespace
    
    def upsert_chunks(self, chunks: list[DocumentChunk]):
        """Upsert chunks to Pinecone."""
        vectors = [
            (
                chunk.chunk_id,
                chunk.embedding,
                chunk.metadata
            )
            for chunk in chunks
        ]
        
        # Batch upsert (Pinecone max 100 per request)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)
        
        logger.info(f"Upserted {len(chunks)} chunks to namespace {self.namespace}")
    
    def query(self, query_embedding: list[float], top_k: int = 4, filter_metadata: dict = None) -> list[dict]:
        """Query Pinecone for similar vectors."""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
            filter=filter_metadata
        )
        
        return results["matches"]
    
    def delete_all(self, namespace: str):
        """Delete all vectors in namespace."""
        self.index.delete(delete_all=True, namespace=namespace)
```

### Retriever (`retrieval/retriever.py`)

```python
from app.retrieval.vector_store import PineconeVectorStore
from app.retrieval.reranker import MetadataReranker
from app.utils.bedrock_client import BedrockClient
from app.models.query import RetrievedSnippet

class NegotiationRetriever:
    def __init__(self):
        self.vector_store = PineconeVectorStore()
        self.bedrock_client = BedrockClient()
        self.reranker = MetadataReranker()
    
    async def retrieve(
        self,
        query_text: str,
        top_k: int = 4,
        technique_priority: list[str] = None,
        role_context: str = None
    ) -> list[RetrievedSnippet]:
        """Retrieve relevant snippets."""
        # Generate query embedding
        query_embedding = await self.bedrock_client.generate_embedding(query_text)
        
        # Build metadata filter
        filter_metadata = {}
        if role_context:
            filter_metadata["role_context"] = {"$in": [role_context, "both"]}
        
        # Query vector store (retrieve more than top_k for re-ranking)
        matches = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Over-retrieve
            filter_metadata=filter_metadata if filter_metadata else None
        )
        
        # Convert to RetrievedSnippet
        snippets = [
            RetrievedSnippet(
                id=match["id"],
                snippet=match["metadata"].get("text", ""),
                technique=match["metadata"].get("technique", "general"),
                source=match["metadata"].get("source", "unknown"),
                score=match["score"],
                metadata=match["metadata"]
            )
            for match in matches
        ]
        
        # Re-rank if technique priority specified
        if technique_priority:
            snippets = self.reranker.rerank(snippets, technique_priority)
        
        # Return top_k after re-ranking
        return snippets[:top_k]
```

### Re-Ranker (`retrieval/reranker.py`)

```python
from app.models.query import RetrievedSnippet

class MetadataReranker:
    def rerank(self, snippets: list[RetrievedSnippet], technique_priority: list[str]) -> list[RetrievedSnippet]:
        """Re-rank snippets based on technique priority."""
        # Assign priority scores
        priority_scores = {tech: len(technique_priority) - i for i, tech in enumerate(technique_priority)}
        
        def score_snippet(snippet: RetrievedSnippet) -> float:
            base_score = snippet.score
            priority_boost = priority_scores.get(snippet.technique, 0) * 0.1
            return base_score + priority_boost
        
        # Sort by combined score
        return sorted(snippets, key=score_snippet, reverse=True)
```

---

## Synthesis Pipeline

### Plan Generator (`synthesis/plan_generator.py`)

```python
from app.utils.bedrock_client import BedrockClient
from app.models.query import SynthesizedPlan, SuggestedReply, RetrievedSnippet, SessionContext
from app.synthesis.prompts import build_synthesis_prompt

class PlanGenerator:
    def __init__(self):
        self.bedrock_client = BedrockClient()
    
    async def generate_plan(
        self,
        retrieved_snippets: list[RetrievedSnippet],
        session_context: SessionContext,
        user_message: str,
        role_type: str
    ) -> SynthesizedPlan:
        """Generate negotiation plan using Bedrock."""
        # Build prompt
        prompt = build_synthesis_prompt(
            retrieved_snippets=retrieved_snippets,
            session_context=session_context,
            user_message=user_message,
            role_type=role_type
        )
        
        # Call Bedrock
        response = await self.bedrock_client.generate_text(
            prompt=prompt,
            max_tokens=settings.max_tokens,
            temperature=0.7
        )
        
        # Parse response (assume structured JSON output)
        plan_data = parse_plan_response(response)
        
        return SynthesizedPlan(
            plan=plan_data["plan"],
            suggested_replies=[
                SuggestedReply(**reply) for reply in plan_data["suggested_replies"]
            ],
            rationales=plan_data["rationales"],
            confidence=plan_data["confidence"]
        )
```

### Synthesis Prompts (`synthesis/prompts.py`)

```python
def build_synthesis_prompt(
    retrieved_snippets: list[RetrievedSnippet],
    session_context: SessionContext,
    user_message: str,
    role_type: str
) -> str:
    """Build prompt for Bedrock synthesis."""
    
    # Format retrieved snippets
    snippets_text = "\n\n".join([
        f"**Snippet {i+1}** (Technique: {s.technique})\n{s.snippet}"
        for i, s in enumerate(retrieved_snippets)
    ])
    
    # Format session context
    context_text = f"""
Offered Salary: {format_salary(session_context.offered_salary)}
Desired Salary: {format_salary(session_context.desired_salary)}
BATNA: {session_context.batna or "None provided"}
"""
    
    prompt = f"""You are a negotiation coach specializing in Chris Voss's tactical empathy techniques. A {role_type} is preparing for a salary negotiation.

**Context:**
{context_text}

**User's Question:**
{user_message}

**Relevant Techniques:**
{snippets_text}

**Your Task:**
Generate a concise negotiation plan and 2-3 short suggested replies (≤25 words each). For each reply:
1. Specify the technique used
2. Provide a brief rationale (≤20 words)

**Output Format (JSON):**
{{
  "plan": "One-sentence plan",
  "suggested_replies": [
    {{
      "text": "Reply text here",
      "technique": "calibrated_question",
      "rationale": "Why this works",
      "confidence": "high|medium|low"
    }}
  ],
  "rationales": ["Overall reasoning for this approach"],
  "confidence": "high|medium|low"
}}
"""
    
    return prompt

def format_salary(salary: SalaryValue | None) -> str:
    if not salary:
        return "Not specified"
    return f"{salary.currency} {salary.amount:,.0f} ({salary.period})"
```

---

## Query Endpoint (`api/query_endpoint.py`)

```python
from fastapi import APIRouter, HTTPException
from app.models.query import QueryRequest, QueryResponse
from app.retrieval.retriever import NegotiationRetriever
from app.synthesis.plan_generator import PlanGenerator
from app.utils.cache import cache_get, cache_set
import time

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query RAG service for negotiation advice."""
    start_time = time.time()
    
    # Check cache
    cache_key = f"{request.session_id}_{hash(request.user_message)}"
    cached = cache_get(cache_key) if settings.enable_cache else None
    if cached:
        logger.info(f"Cache hit for session {request.session_id}")
        return cached
    
    # Retrieve snippets
    retriever = NegotiationRetriever()
    retrieved_snippets = await retriever.retrieve(
        query_text=request.user_message,
        top_k=request.top_k or settings.top_k,
        technique_priority=request.preferences.technique_priority,
        role_context=request.role_type
    )
    
    if not retrieved_snippets:
        raise HTTPException(status_code=404, detail="No relevant techniques found")
    
    # Synthesize plan (if requested)
    synthesized_plan = None
    if request.preferences.include_synthesis and settings.enable_synthesis:
        plan_generator = PlanGenerator()
        synthesized_plan = await plan_generator.generate_plan(
            retrieved_snippets=retrieved_snippets,
            session_context=request.session_context,
            user_message=request.user_message,
            role_type=request.role_type
        )
    
    elapsed = time.time() - start_time
    
    response = QueryResponse(
        retrieved=retrieved_snippets,
        synthesized=synthesized_plan,
        debug={
            "scores": [s.score for s in retrieved_snippets],
            "elapsed_ms": int(elapsed * 1000),
            "cached": False
        }
    )
    
    # Cache response
    if settings.enable_cache:
        cache_set(cache_key, response, ttl=settings.cache_ttl_seconds)
    
    # Emit metrics
    await emit_metric("rag_query_count", 1)
    await emit_metric("rag_query_latency_ms", elapsed * 1000)
    
    return response
```

---

## Bedrock Client (`utils/bedrock_client.py`)

```python
import boto3
import json
from app.config import settings

class BedrockClient:
    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key
        )
    
    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding using Bedrock Titan Embeddings."""
        response = self.client.invoke_model(
            modelId=settings.bedrock_embedding_model,
            body=json.dumps({"inputText": text})
        )
        
        result = json.loads(response["body"].read())
        return result["embedding"]
    
    async def generate_text(self, prompt: str, max_tokens: int = 800, temperature: float = 0.7) -> str:
        """Generate text using Bedrock Claude."""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        response = self.client.invoke_model(
            modelId=settings.bedrock_generation_model,
            body=json.dumps(body)
        )
        
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
```

---

## Caching Strategy (`utils/cache.py`)

```python
from functools import lru_cache
import hashlib
import json

# In-memory cache for MVP (can be replaced with Redis)
_cache = {}

def cache_key_hash(key: str) -> str:
    """Generate hash for cache key."""
    return hashlib.md5(key.encode()).hexdigest()

def cache_get(key: str):
    """Get value from cache."""
    hashed_key = cache_key_hash(key)
    if hashed_key in _cache:
        value, expiry = _cache[hashed_key]
        if time.time() < expiry:
            return value
        else:
            del _cache[hashed_key]
    return None

def cache_set(key: str, value, ttl: int):
    """Set value in cache with TTL."""
    hashed_key = cache_key_hash(key)
    expiry = time.time() + ttl
    _cache[hashed_key] = (value, expiry)
```

---

## Performance Optimizations

### 1. Batch Embedding Generation

When ingesting multiple chunks, batch embedding calls to reduce latency:

```python
async def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts in parallel."""
    tasks = [bedrock_client.generate_embedding(text) for text in texts]
    return await asyncio.gather(*tasks)
```

### 2. Query Result Caching

Cache common queries (identical session context + user message) with 1-hour TTL.

### 3. Pre-Warming

For demo, pre-compute embeddings for common scenarios and cache in memory.

---

## Observability

### Metrics

- `rag_query_count`: Total queries
- `rag_query_latency_ms`: Query duration
- `rag_retrieval_count`: Retrieved snippets per query
- `rag_cache_hit_rate`: Cache hit percentage
- `rag_synthesis_latency_ms`: Synthesis duration

### Logging

```python
logger.info(
    "rag_query",
    session_id=request.session_id,
    retrieved_count=len(retrieved_snippets),
    top_technique=retrieved_snippets[0].technique if retrieved_snippets else None,
    elapsed_ms=elapsed * 1000
)
```

---

## Testing Strategy

### Unit Tests

```python
@pytest.mark.asyncio
async def test_chunker():
    chunker = NegotiationChunker(chunk_size=100, chunk_overlap=20)
    content = "This is a long document..." * 50
    chunks = chunker.chunk_document("test-doc", content, {"technique": "label"})
    
    assert len(chunks) > 1
    assert all(chunk.metadata["technique"] == "label" for chunk in chunks)
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_query_endpoint():
    request = QueryRequest(
        session_id="test-123",
        role_type="candidate",
        user_message="How do I ask for more salary?",
        session_context=SessionContext(),
        preferences=QueryPreferences()
    )
    
    response = await query_rag(request)
    
    assert len(response.retrieved) > 0
    assert response.synthesized is not None
```

---

## Example Corpus Preparation

### Voss Book Excerpt (Sample)

```json
{
  "source_id": "voss_chapter_7",
  "source_type": "book",
  "title": "Never Split the Difference - Chapter 7",
  "content": "Calibrated questions start with 'What' or 'How' and avoid words like 'Why' that can sound accusatory. They invite the other party to solve the problem with you rather than against you. For example, instead of asking 'Why did you offer this amount?', ask 'How did you arrive at this number?' This shifts the conversation from confrontation to collaboration.",
  "metadata": {
    "technique": "calibrated_question",
    "page": 142,
    "chapter": 7
  }
}
```

### Annotated Transcript (Sample)

```json
{
  "source_id": "transcript_042",
  "source_type": "transcript",
  "title": "Candidate Negotiation - Tech Sector",
  "content": "Candidate: 'It sounds like the budget is constrained. What flexibility exists in the total compensation package?' — This label + calibrated question opened space for the recruiter to discuss equity and signing bonuses, which weren't part of the initial offer.",
  "metadata": {
    "technique": "label",
    "outcome": "success",
    "industry": "tech"
  }
}
```

---

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

### CI/CD Ingestion Job

```yaml
# .github/workflows/ingest.yml
name: Ingest Corpus

on:
  push:
    paths:
      - 'data/voss_corpus/**'
      - 'data/transcripts/**'

jobs:
  ingest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run ingestion
        run: python scripts/ingest.py --namespace voss_rag
        env:
          PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```

---

**End of RAG Service Specification**

