# HANAH System Architecture Specification

**Version:** 1.0  
**Last Updated:** November 6, 2025  
**Status:** Design / MVP Specification

---

## Executive Summary

HANAH (HANAH: A Negotiation Assistant for Humans) is a voice-first salary negotiation assistant built on a dual-agent architecture. The system combines real-time speech processing, document intelligence (OCR), and retrieval-augmented generation (RAG) to provide personalized negotiation coaching based on Chris Voss's tactical empathy principles.

### Key Capabilities

- **Voice-First Interaction**: Natural conversation via Deepgram ASR and AWS Polly TTS
- **Intelligent Context Collection**: Automated extraction of salary details, job information, and negotiation context from voice and documents
- **RAG-Powered Coaching**: Retrieval of negotiation techniques and generation of contextual tactics using LlamaIndex + Bedrock
- **Interactive Role-Play**: Simulated negotiation practice with real-time feedback
- **Privacy-First Design**: Ephemeral sessions, encrypted storage, user-controlled data deletion

### Target Users

- **Candidates**: Preparing for salary negotiation conversations
- **Recruiters**: Crafting empathetic, effective offer discussions

### MVP Scope

English-only, PDF upload support, salary negotiation focus, web-based interface.

---

## System Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (React + Vite)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Voice Input  │  │  Transcript  │  │ Negotiation  │      │
│  │   (Mic)      │  │   Display    │  │   Canvas     │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
└─────────┼──────────────────┼──────────────────┼──────────────┘
          │                  │                  │
          │ WebSocket        │ HTTP/WS          │ HTTP
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│         LangGraph Orchestrator (FastAPI Backend)            │
│                                                              │
│  ┌────────────────────┐         ┌─────────────────────┐    │
│  │  Collector Agent   │────────▶│  Negotiator Agent   │    │
│  │  (Context Builder) │         │  (Tactical Coach)   │    │
│  └─────────┬──────────┘         └──────────┬──────────┘    │
│            │                               │                │
│  ┌─────────▼──────────────────────────────▼──────────────┐ │
│  │          State Machine & Session Manager             │ │
│  └──────────────────────────┬───────────────────────────┘ │
│                             │                              │
└─────────────────────────────┼──────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐  ┌────────────────┐  ┌──────────────────┐
│   AWS Textract  │  │ RAG Service    │  │   Deepgram API   │
│   (OCR/PDF)     │  │ (LlamaIndex)   │  │   (ASR/STT)      │
└─────────────────┘  └────────┬───────┘  └──────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            ┌───────────────┐   ┌──────────────┐
            │   Pinecone    │   │ AWS Bedrock  │
            │  (Vector DB)  │   │ (LLM/Embed)  │
            └───────────────┘   └──────────────┘

Storage Layer:
┌──────────────┐  ┌──────────────┐
│   AWS S3     │  │  DynamoDB/   │
│  (Files)     │  │  PostgreSQL  │
└──────────────┘  └──────────────┘
```

---

## Component Responsibilities Matrix

| Component                  | Purpose                           | Key Responsibilities                                                                                                  | Technology Stack                               |
| -------------------------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| **Frontend**               | User interface and audio I/O      | Mic capture, WebSocket streaming, file upload, transcript display, negotiation plan visualization, provenance display | React, Vite, WebSocket, Deepgram SDK           |
| **LangGraph Orchestrator** | Agent orchestration and workflow  | Collector/Negotiator agent flows, state machine management, tool invocation, session persistence, WebSocket handling  | FastAPI, LangChain, LangGraph                  |
| **Collector Agent**        | Context gathering                 | Voice-based onboarding, file ingestion coordination, OCR orchestration, BATNA building, context validation            | LangGraph node, Deepgram client, Textract tool |
| **Negotiator Agent**       | Tactical coaching                 | RAG query coordination, plan synthesis, suggested reply generation, role-play simulation, feedback delivery           | LangGraph node, RAG client, Bedrock client     |
| **RAG Microservice**       | Knowledge retrieval and synthesis | Vector search, technique retrieval, context-aware plan generation, provenance tracking                                | LlamaIndex, FastAPI                            |
| **AWS Textract**           | Document intelligence             | PDF text extraction, table detection, layout analysis                                                                 | AWS Textract API                               |
| **Deepgram**               | Speech-to-text                    | Real-time ASR, interim transcripts, final transcripts                                                                 | Deepgram Streaming API                         |
| **AWS Polly**              | Text-to-speech                    | Plan vocalization, negotiation script playback                                                                        | AWS Polly API                                  |
| **Pinecone**               | Vector database                   | Embedding storage, similarity search, namespace isolation                                                             | Pinecone cloud                                 |
| **AWS Bedrock**            | LLM & embeddings                  | Text generation (Claude/Titan), embeddings (Titan Embeddings)                                                         | Bedrock API                                    |
| **AWS S3**                 | File storage                      | PDF storage, encrypted uploads, pre-signed URLs                                                                       | S3 with SSE                                    |
| **DynamoDB/Postgres**      | Session store                     | Ephemeral session data, transcript history, parsed context                                                            | DynamoDB or PostgreSQL                         |

---

## Data Flow Patterns

### Flow 1: Voice Onboarding (Collector Phase)

```
User speaks ──▶ Frontend (mic) ──▶ Deepgram API ──▶ Interim transcripts
                                                            │
                                                            ▼
                                                    LangGraph Orchestrator
                                                            │
                                                            ▼
                                                    Collector Agent
                                                    (accumulates context)
                                                            │
                                                            ▼
                                                    Session Store (DynamoDB)
```

### Flow 2: PDF Upload & OCR

```
User uploads PDF ──▶ Frontend ──▶ Pre-signed S3 URL ──▶ S3 Bucket
                                        │
                                        ▼
                            LangGraph Orchestrator
                                        │
                                        ▼
                            Textract StartDocumentAnalysis (async)
                                        │
                                        ▼
                            Poll/SNS callback ──▶ Raw OCR text
                                        │
                                        ▼
                            Bedrock extraction prompt ──▶ Normalized JSON
                                        │
                                        ▼
                            Collector reads back summary ──▶ User confirms
                                        │
                                        ▼
                            Session updated with parsed_data
```

### Flow 3: RAG Query & Plan Generation (Negotiator Phase)

```
Context ready ──▶ Negotiator Agent ──▶ RAG Microservice /query
                                                │
                                                ▼
                                        Pinecone vector search (top_k=4)
                                                │
                                                ▼
                                        Retrieved snippets (technique metadata)
                                                │
                                                ▼
                                        Bedrock synthesis ──▶ Plan + suggested replies
                                                │
                                                ▼
                                        Return to Negotiator
                                                │
                                                ▼
                                        TTS (Polly) ──▶ Vocalize plan
                                                │
                                                ▼
                                        Frontend displays replies + provenance
```

---

## Session Data Model

### Session Schema

```typescript
interface Session {
  session_id: string; // UUID v4
  role_type: "candidate" | "recruiter";
  user_profile: {
    name?: string;
  };
  company?: string;
  job_title?: string;
  offered_salary?: SalaryValue;
  desired_salary_min?: SalaryValue;
  desired_salary_max?: SalaryValue;
  batna?: string; // Best Alternative To Negotiated Agreement
  files: FileRecord[];
  transcript: TranscriptEntry[]; // Append-only
  state: SessionState;
  metadata: {
    created_at: string; // ISO 8601
    updated_at: string;
    user_ip?: string;
    consent_ocr: boolean;
    consent_storage: boolean;
  };
}

interface SalaryValue {
  amount: number; // Numeric value
  period: "yearly" | "monthly";
  currency: "INR" | "USD" | "EUR" | "GBP";
}

interface FileRecord {
  file_id: string;
  file_type: "resume" | "offer_letter" | "job_description";
  s3_url: string;
  upload_time: string;
  ocr_status: "pending" | "processing" | "completed" | "failed";
  parsed_data?: ParsedDocument;
}

interface ParsedDocument {
  offered_salary?: SalaryValue;
  company?: string;
  job_title?: string;
  benefits?: string[];
  raw_text?: string; // Full OCR output
}

interface TranscriptEntry {
  timestamp: string;
  speaker: "user" | "system";
  text: string;
  is_final: boolean;
}

type SessionState =
  | "start"
  | "identifying"
  | "collecting"
  | "upload_waiting"
  | "ocr_processing"
  | "confirm_context"
  | "context_ready"
  | "negotiating"
  | "done";
```

### Normalization Rules

1. **Salary Normalization**:

   - Always store as `{amount: number, period: "yearly"|"monthly", currency: string}`
   - Convert LPA (Lakhs Per Annum) → multiply by 100,000 for INR yearly
   - Convert monthly → yearly by multiplying by 12
   - Support common formats: "₹12 LPA", "$120k", "10L", "1.2 million"

2. **Currency Detection**:

   - Priority order: explicit currency symbol → country context → default to USD
   - Supported: INR (₹, Rs, Rupees, LPA), USD ($, dollars), EUR (€), GBP (£)

3. **BATNA Handling**:
   - If user has no BATNA, store `null` and flag `batna_built: false`
   - Guided BATNA builder stores structured text: other offers, minimum acceptable, timeline

---

## State Machine

### States and Transitions

```
┌─────────┐
│  start  │
└────┬────┘
     │ init_session
     ▼
┌──────────────┐
│ identifying  │ (Ask: "Are you a candidate or recruiter?")
└──────┬───────┘
       │ role_set
       ▼
┌──────────────┐
│  collecting  │ (Gather: offered_salary, desired, BATNA)
└──┬───────┬───┘
   │       │
   │       └─────────────┐
   │ file_upload         │ manual_entry / all_fields_collected
   ▼                     │
┌─────────────────┐      │
│ upload_waiting  │      │
└────────┬────────┘      │
         │ upload_complete│
         ▼               │
┌────────────────┐       │
│ ocr_processing │       │
└────────┬───────┘       │
         │ ocr_complete  │
         ▼               │
┌────────────────┐       │
│ confirm_context│◀──────┘
└────────┬───────┘
         │ user_confirms
         ▼
┌────────────────┐
│ context_ready  │ (Handoff to Negotiator)
└────────┬───────┘
         │ invoke_negotiator
         ▼
┌─────────────┐
│ negotiating │ (RAG → plan → role-play)
└──────┬──────┘
       │ user_done / export
       ▼
┌──────────┐
│   done   │
└──────────┘
```

### State Guard Conditions

| Transition                         | Guard Condition                                      |
| ---------------------------------- | ---------------------------------------------------- |
| `identifying → collecting`         | `role_type` is set                                   |
| `collecting → upload_waiting`      | User initiates file upload                           |
| `collecting → confirm_context`     | All required fields collected OR user says "proceed" |
| `upload_waiting → ocr_processing`  | S3 upload confirmed                                  |
| `ocr_processing → confirm_context` | Textract job completed + parsed                      |
| `confirm_context → context_ready`  | User confirms extracted data OR manually overrides   |
| `context_ready → negotiating`      | Context validation passed                            |
| `negotiating → done`               | User requests export OR session timeout              |

---

## Cross-Cutting Concerns

### Security

1. **Secrets Management**:

   - All API keys (Bedrock, Pinecone, Deepgram, AWS credentials) stored server-side
   - Use AWS Secrets Manager or environment variables (never in code)
   - Frontend receives no direct keys

2. **File Upload Security**:

   - Pre-signed S3 URLs with 5-minute expiration
   - Validate file type and size (PDF only, max 10 MB)
   - Private S3 bucket with SSE-S3 or SSE-KMS encryption
   - No public read access

3. **API Authentication**:

   - Internal services: mTLS or API tokens
   - Frontend ↔ Backend: JWT or session cookies with CSRF protection
   - Rate limiting on all endpoints (100 req/min per session)

4. **Data Sanitization**:
   - Escape all user input in logs
   - Redact PII (email, phone, SSN) in stored transcripts
   - No salary data in application logs (use hashed session IDs)

### Privacy

1. **User Consent**:

   - Explicit opt-in before OCR processing: "May I extract text from your document?"
   - Explicit consent for transcript storage
   - Local-only mode toggle (ephemeral, no persistence)

2. **Data Retention**:

   - Default: sessions expire after 24 hours
   - User-triggered deletion: immediate purge of session + files from S3
   - Audit log of deletion requests (compliance)

3. **Transparency**:

   - Display which retrieved snippets influenced each reply
   - "Show source" button to view original technique chunk
   - Provenance chain: user context → retrieved snippet → generated reply

4. **Right to Forget**:
   - One-click "Delete Session" button in UI
   - Background job removes S3 files, session record, and logs

### Observability

1. **Metrics (CloudWatch / Prometheus)**:

   - `session_start_count`, `session_completion_rate`
   - `ocr_success_rate`, `ocr_duration_seconds`
   - `rag_query_latency_seconds`, `rag_retrieval_count`
   - `agent_state_transitions` (by state)
   - `tts_generation_duration`, `asr_transcript_latency`

2. **Logging (Structured JSON)**:

   - Every state transition: `{session_id, from_state, to_state, timestamp}`
   - RAG queries: `{session_id, query_text_hash, retrieved_ids[], confidence}`
   - OCR jobs: `{session_id, file_id, textract_job_id, status, duration}`
   - User feedback: `{session_id, reply_id, feedback: "up"|"down"}`

3. **Tracing (OpenTelemetry / X-Ray)**:

   - End-to-end trace: frontend mic → ASR → agent → RAG → TTS → frontend
   - Span tags: `session_id`, `agent_name`, `tool_name`, `rag_top_k`

4. **Alerting**:
   - Alert if OCR success rate < 80% over 1 hour
   - Alert if RAG latency p95 > 2 seconds
   - Alert if session completion rate < 50%

### Error Handling

1. **Textract Failures**:

   - Inform user: "I couldn't read that document. Would you like to re-upload or tell me the details?"
   - Collector continues collecting other fields (non-blocking)
   - Log failure with `file_id` and error code

2. **RAG Service Timeout**:

   - Fallback to canned templates (5 pre-written scripts for common scenarios)
   - Display: "I'm using a general script while the service recovers."
   - Retry background request and update UI if successful

3. **Bedrock Rate Limits**:

   - Exponential backoff (3 retries, up to 10s delay)
   - If all retries fail, use canned fallback
   - Telemetry: increment `bedrock_rate_limit_errors`

4. **ASR Mis-Transcription**:

   - Show interim transcript to user in real-time
   - For critical fields (salary, company), confirm: "I heard ₹12 LPA — is that correct?"
   - Offer text input fallback if user says "no" repeatedly

5. **Session Timeout**:
   - If user silent for 60 seconds during collection, nudge: "Are you still there? What's your target salary?"
   - After 5 minutes idle, offer: "Would you like to continue with text input?"
   - After 10 minutes idle, save session and send resume link (if contact provided)

---

## Deployment Topology

### MVP Deployment

```
┌──────────────────────────────────────────────┐
│          AWS / Cloud Infrastructure          │
│                                              │
│  ┌────────────────────────────────────────┐ │
│  │   Frontend (S3 + CloudFront)           │ │
│  │   React build deployed as static site  │ │
│  └───────────────┬────────────────────────┘ │
│                  │                           │
│  ┌───────────────▼────────────────────────┐ │
│  │   LangGraph Orchestrator (ECS/Fargate) │ │
│  │   FastAPI app with WebSocket support   │ │
│  └───────────────┬────────────────────────┘ │
│                  │                           │
│  ┌───────────────▼────────────────────────┐ │
│  │   RAG Microservice (ECS/Fargate)       │ │
│  │   LlamaIndex + FastAPI                 │ │
│  └────────────────────────────────────────┘ │
│                                              │
│  ┌─────────────┐  ┌──────────────┐         │
│  │  DynamoDB   │  │  S3 Bucket   │         │
│  │  (Sessions) │  │  (PDFs)      │         │
│  └─────────────┘  └──────────────┘         │
│                                              │
└──────────────────────────────────────────────┘

External Services:
- Pinecone (managed cloud)
- Deepgram API (cloud)
- AWS Bedrock (regional)
- AWS Textract (regional)
```

### Scaling Considerations

1. **Frontend**: Static, scales automatically via CDN
2. **Orchestrator**: Horizontal scaling (multiple ECS tasks behind ALB)
3. **RAG Service**: Horizontal scaling; cache common queries in Redis
4. **DynamoDB**: On-demand billing, auto-scales
5. **S3**: No scaling needed
6. **Pinecone**: Serverless tier or pods (based on index size)

---

## Demo Scenario Walkthrough

### Pre-Demo Setup

1. Ingest Voss content into Pinecone:

   - 50 chunks from "Never Split the Difference"
   - 20 annotated negotiation transcripts
   - Metadata: `{technique, intent, source, language: "en"}`

2. Prepare sample PDFs:
   - `sample_offer_letter.pdf` (₹12 LPA, ABC Corp, Software Engineer)
   - `sample_resume.pdf`
   - `sample_jd.pdf`

### Demo Flow (5 minutes)

**Minute 0:00 – Start**

- User clicks "Start Negotiation"
- Collector: "Hello! Are you a candidate or a recruiter?"
- User (voice): "Candidate"

**Minute 0:20 – File Upload**

- Collector: "Great! Do you have an offer letter to upload, or should I ask you the details?"
- User (voice): "I'll upload it"
- User uploads `sample_offer_letter.pdf`
- UI shows: "Uploading… 100%" → "Extracting text…"
- Collector (continues): "While that processes, what salary are you hoping for?"

**Minute 0:45 – Collecting Context**

- User: "I want ₹15 lakhs per annum"
- Collector: "Got it. Do you have any other offers or alternatives?"
- User: "I have one other offer for ₹13 LPA, but I prefer this company"
- Collector: "That's a strong BATNA. One moment…"

**Minute 1:10 – OCR Completes**

- Textract finishes; Bedrock post-processing extracts:
  ```json
  {
    "offered_salary": {
      "amount": 1200000,
      "period": "yearly",
      "currency": "INR"
    },
    "company": "ABC Corp",
    "job_title": "Software Engineer",
    "benefits": ["Health insurance", "ESOP"]
  }
  ```
- Collector: "I found an offer of ₹12 lakhs per year from ABC Corp. Is that correct?"
- User: "Yes"

**Minute 1:30 – Context Ready**

- State transitions to `context_ready`
- Negotiator Agent invoked

**Minute 1:35 – RAG Query**

- Negotiator calls RAG service:
  ```json
  {
    "session_context": {
      "offered_salary": {
        "amount": 1200000,
        "currency": "INR",
        "period": "yearly"
      },
      "desired_salary": {
        "amount": 1500000,
        "currency": "INR",
        "period": "yearly"
      },
      "batna": "Other offer at ₹13 LPA"
    },
    "user_message": "How do I ask for ₹15 LPA when offered ₹12 LPA?"
  }
  ```
- RAG returns 4 snippets (techniques: calibrated_question, label, anchoring)
- Bedrock synthesizes plan

**Minute 1:50 – Plan Delivery**

- TTS (Polly): "Here's your plan: Anchor near ₹15 lakhs and use calibrated questions to explore their constraints."
- UI displays:

  ```
  📋 Negotiation Plan
  Anchor at ₹15 LPA and probe flexibility with empathy.

  💬 Suggested Replies:
  1. "Can you help me understand how you arrived at ₹12 LPA?"
     Technique: Calibrated Question
     Why: Invites problem-solving, not defensiveness.

  2. "It sounds like budget may be tight — what flexibility exists?"
     Technique: Label + Calibrated Question
     Why: Labels their constraint, opens space for solutions.

  📚 Sources: [Voss Ch. 7: Calibrated Questions], [Transcript: Candidate-123]
  ```

**Minute 2:30 – Practice Mode**

- User: "Can I practice?"
- Negotiator: "Sure! I'll play the recruiter. Ready?"
- (Role-play begins)
  - Negotiator (as recruiter): "We just don't have budget for ₹15 LPA right now."
  - User: "Can you help me understand what constraints you're facing?"
  - Negotiator (feedback): "Great! That's a calibrated question. You shifted to problem-solving mode."
  - (3 more turns)

**Minute 4:30 – Export**

- User clicks "Export Plan"
- System generates PDF with plan, replies, rationales, and sources
- User clicks "Delete Session"
- Confirmation: "Session and files deleted."

**Demo Complete** ✅

---

## Next Steps

1. **Implementation Phase**:

   - Set up repositories: `hanah-orchestrator`, `hanah-rag-service`, `hanah-frontend`
   - Define API contracts (OpenAPI specs)
   - Implement agent flows in LangGraph
   - Build RAG ingestion pipeline

2. **Testing Strategy**:

   - Unit tests for agent decision logic
   - Integration tests for tool invocations (mocked Textract, RAG)
   - E2E tests with sample audio and PDFs
   - Load testing: 100 concurrent sessions

3. **Deployment Checklist**:

   - Provision AWS resources (S3, DynamoDB, ECS)
   - Set up Pinecone index and ingest corpus
   - Configure Bedrock model access
   - Deploy frontend to S3 + CloudFront

4. **Observability Setup**:
   - CloudWatch dashboards for KPIs
   - X-Ray tracing for agent flows
   - Structured logging to CloudWatch Logs

---

## References

- **Chris Voss's "Never Split the Difference"**: Core negotiation principles
- **LangGraph Documentation**: Agent orchestration patterns
- **LlamaIndex Documentation**: RAG pipeline best practices
- **AWS Bedrock Models**: Claude 3 Sonnet, Titan Embeddings
- **Deepgram Streaming API**: Real-time ASR
- **AWS Textract**: Document analysis features

---

**End of Architecture Specification**
