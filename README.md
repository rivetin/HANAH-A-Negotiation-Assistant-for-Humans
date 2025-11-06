# 🤝 HANAH  
### *HANAH: A Negotiation Assistant for Humans*

---

## 🧭 Overview

**HANAH** is a voice-first, empathy-driven AI negotiation assistant designed to help **candidates** and **recruiters** prepare for, simulate, and master salary and offer negotiations.

Built on the principles of **tactical empathy** (inspired by Chris Voss’s *Never Split the Difference*), HANAH listens, understands, and coaches users to communicate with confidence and empathy.

> “Empathy meets intelligence in every negotiation.”

---

## 🧩 Core Concept

HANAH is powered by two specialized AI agents that collaborate through natural voice conversation:

| Agent | Role | Description |
|--------|------|-------------|
| 🎙️ **Collector Agent** | Context Builder | Gathers negotiation context through conversation — role, salary offer, target, BATNA — and extracts key info from uploaded PDFs (offer letters, résumés, job descriptions). |
| 🧠 **Negotiator Agent** | Tactical Coach | Retrieves negotiation strategies via a RAG (Retrieval-Augmented Generation) pipeline trained on Chris Voss’s negotiation methods, then generates tailored plans and response suggestions. |

---

## 🏗️ Architecture Overview
React + Vite (voice-first UI)
↓
Deepgram (ASR) → FastAPI backend (LangGraph)
↓
Collector Agent ──► AWS Textract (OCR)
│
▼
Negotiator Agent ──► LlamaIndex RAG Service
│
├─ AWS Bedrock (Embeddings + Generation)
└─ Pinecone (Vector DB)


---

## ⚙️ Component Breakdown

| Layer | Description | Technology |
|-------|--------------|-------------|
| 🎧 **Frontend** | Voice-first chat interface, transcripts, and file uploads | React + Vite + WebSocket |
| 🗣️ **Voice Input** | Real-time speech recognition | Deepgram |
| 💬 **Voice Output** | Conversational TTS playback | AWS Polly |
| 🧩 **Agent Orchestration** | Multi-agent flows & state management | LangGraph (LangChain) |
| 🧾 **OCR & Extraction** | PDF parsing for salary and job data | AWS Textract |
| 🧠 **Negotiation Brain (RAG)** | Retrieves techniques & examples | LlamaIndex + Pinecone |
| ☁️ **LLM & Embeddings** | Text generation & semantic embeddings | AWS Bedrock (Claude 3 / Titan) |
| 💾 **Storage** | Session data, uploads, context | AWS S3 + DynamoDB / Postgres |

---

## 🗣️ Conversational Flow

### Phase 1 — Collector Agent
1. **Identify user role** → “Are you a candidate or a recruiter?”
2. **Collect context** → Gather salary offer, target range, BATNA.
3. **File handling** → Upload offer letters, résumés, or job descriptions (PDF).
4. **OCR processing** → AWS Textract extracts structured data.
5. **Confirmation** → HANAH summarizes findings for voice confirmation.
6. **Handoff** → When context is complete, pass to the Negotiator Agent.

---

### Phase 2 — Negotiator Agent
1. **Context ingestion** → Load structured context + parsed data.
2. **RAG retrieval** → Query LlamaIndex for relevant negotiation techniques and examples.
3. **Plan synthesis** → Use Bedrock LLM to generate:
   - Negotiation plan (1–2 lines)
   - 2–3 short suggested replies (≤25 words)
   - Label each reply with its negotiation technique
   - Provide short rationale
4. **Voice coaching** → Speak plan via AWS Polly and display tactics visually.
5. **Practice mode** → Simulate recruiter conversation and give real-time feedback.
6. **Export** → Generate a personalized negotiation plan (PDF/text).

---

## 🧱 Data Flow Summary

| Stage | Purpose | Tools |
|--------|----------|-------|
| **ASR** | Convert user voice to text | Deepgram |
| **OCR** | Extract salary/company from PDFs | AWS Textract |
| **Context** | Store structured user data | DynamoDB / Postgres |
| **RAG Retrieval** | Fetch relevant negotiation tactics | LlamaIndex + Pinecone |
| **Generation** | Synthesize plan & replies | AWS Bedrock |
| **TTS Output** | Speak AI responses | AWS Polly |

---

## 🔐 Privacy & Ethics

- **Consent-first**: OCR and storage only with user permission  
- **Data privacy**: Files encrypted in S3; sessions ephemeral  
- **Empathy-first**: Never deceptive or manipulative  
- **Transparency**: Show which retrieved sources shaped the response  
- **Right to forget**: One-click session deletion and data purge  

---

## 🚀 MVP Roadmap

| Phase | Milestone | Deliverables |
|--------|------------|--------------|
| **Phase 1** | Voice onboarding + context collection | Collector Agent (LangGraph) + Deepgram + Textract integration |
| **Phase 2** | RAG engine | LlamaIndex ingestion + Pinecone + Bedrock embeddings |
| **Phase 3** | Negotiator logic | Bedrock-based tactical generation |
| **Phase 4** | Practice mode + TTS | Interactive role-play & audio output |
| **Phase 5** | Privacy + analytics | Feedback and deletion features |

---

## 💡 Example Interaction

**User:**  
> “They offered ₹12 LPA, but I’m hoping for ₹15 LPA. How do I respond?”

**HANAH:**  
> “Here’s your plan: Anchor near ₹15 LPA and use calibrated questions to explore flexibility.”  
> “Try: *‘Can you help me understand how ₹12 LPA was determined?’* — that’s a calibrated question.”  
> “Or: *‘It sounds like budget is tight — what flexibility might exist?’* — that’s labeling and empathy.”

**User:**  
> “Can I practice this?”

**HANAH:**  
> “Sure — I’ll play the recruiter. Ready? ‘We just don’t have the budget for ₹15 LPA right now…’”

---

## 🧠 RAG Corpus Design

| Attribute | Description |
|------------|-------------|
| **Source** | Curated summaries from *Never Split the Difference* + annotated transcripts |
| **Chunk size** | 200–400 tokens |
| **Metadata tags** | `{technique, intent, role, source}` |
| **Retrieval config** | top_k = 4, re-rank by technique |
| **Stored provenance** | Include source ID and snippet offset for transparency |

---

## 🧭 Key Principles

1. **Voice-first** – Speak naturally; type only if preferred.  
2. **Empathetic tone** – Coach, not command.  
3. **Transparency** – Always show “why” behind every suggestion.  
4. **Human + AI collaboration** – AI assists, humans decide.  
5. **Composable architecture** – LangGraph (control) + LlamaIndex (knowledge).  

---

## 🧰 Tech Stack Summary

| Layer | Technology |
|--------|-------------|
| Frontend | React + Vite |
| ASR | Deepgram |
| TTS | AWS Polly |
| Backend | FastAPI + LangGraph |
| OCR | AWS Textract |
| RAG | LlamaIndex + Pinecone |
| LLM | AWS Bedrock |
| Storage | AWS S3 |
| Context DB | DynamoDB / PostgreSQL |

---

## 💬 Brand Identity

- **Full Name:** HANAH: A Negotiation Assistant for Humans  
- **Tagline:** *“Empathy meets intelligence in every negotiation.”*  
- **Tone:** Calm, balanced, humanistic, empowering  
- **Core Traits:** Empathy, Clarity, Trust, Confidence  
- **Logo concept:** Palindrome symmetry representing balance & mirroring (a key Voss technique)  

---

## 🏁 Quick Summary

| Property | Value |
|-----------|--------|
| **Name** | HANAH |
| **Expansion** | HANAH: A Negotiation Assistant for Humans |
| **Mission** | Make every salary conversation empathetic, informed, and confident |
| **Architecture** | LangGraph (orchestration) + LlamaIndex (RAG) + Bedrock + Pinecone |
| **Interface** | Voice-first (Deepgram + Polly) |
| **Focus** | Dual-agent flow: Collector → Negotiator |
| **Status** | MVP in design, architecture finalized |

---

## 📜 License
This project is released under the **MIT License**.

---

## ✨ Credits
- Inspired by *Never Split the Difference* by Chris Voss  
- Built with ❤️ using LangGraph, LlamaIndex, AWS Bedrock, and Pinecone  

---
