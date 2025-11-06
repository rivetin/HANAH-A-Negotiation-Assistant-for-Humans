# LangGraph Orchestrator Specification

**Component:** FastAPI Backend with LangGraph Agent Orchestration  
**Version:** 1.0  
**Last Updated:** November 6, 2025

---

## Overview

The LangGraph Orchestrator is the central backend service responsible for coordinating the Collector and Negotiator agents, managing session state, invoking external tools (Deepgram, Textract, RAG service), and handling real-time WebSocket communication with the frontend.

**Key Responsibilities**:

- Host and execute Collector and Negotiator agent workflows
- Maintain session state machine
- Provide tools for ASR, OCR, and RAG queries
- Manage WebSocket connections for real-time audio and transcript streaming
- Persist session data to DynamoDB/PostgreSQL

---

## Architecture

### Application Structure

```
hanah-orchestrator/
├── app/
│   ├── main.py                 # FastAPI app entry point
│   ├── config.py               # Environment config
│   ├── models/
│   │   ├── session.py          # Session Pydantic models
│   │   ├── messages.py         # WebSocket message schemas
│   │   └── agent_state.py      # LangGraph state schemas
│   ├── agents/
│   │   ├── collector.py        # Collector agent graph
│   │   ├── negotiator.py       # Negotiator agent graph
│   │   └── tools/
│   │       ├── deepgram.py     # Deepgram ASR tool
│   │       ├── textract.py     # AWS Textract tool
│   │       ├── rag_client.py   # RAG service HTTP client
│   │       └── tts.py          # AWS Polly TTS tool
│   ├── services/
│   │   ├── session_store.py    # DynamoDB/Postgres adapter
│   │   ├── s3_service.py       # S3 upload/download
│   │   └── websocket_manager.py# WebSocket connection pool
│   ├── api/
│   │   ├── routes.py           # HTTP endpoints
│   │   └── websocket.py        # WebSocket handler
│   └── utils/
│       ├── normalization.py    # Salary normalization
│       └── logging.py          # Structured logging
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## FastAPI Application

### Main Entry Point (`main.py`)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes, websocket
from app.config import settings

app = FastAPI(
    title="HANAH Orchestrator",
    version="1.0.0",
    docs_url="/docs" if settings.environment != "production" else None
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP routes
app.include_router(routes.router, prefix="/api/v1")

# WebSocket endpoint
app.add_websocket_route("/ws/{session_id}", websocket.websocket_endpoint)

@app.on_event("startup")
async def startup():
    # Initialize services
    await session_store.connect()
    logger.info("Orchestrator started")

@app.on_event("shutdown")
async def shutdown():
    await session_store.disconnect()
```

### Configuration (`config.py`)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    environment: str = "development"
    allowed_origins: list[str] = ["http://localhost:5173"]

    # AWS
    aws_region: str = "us-east-1"
    aws_access_key_id: str
    aws_secret_access_key: str
    s3_bucket_name: str

    # Deepgram
    deepgram_api_key: str

    # Bedrock
    bedrock_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"

    # RAG Service
    rag_service_url: str = "http://rag-service:8001"
    rag_service_timeout: int = 5  # seconds

    # Session Store
    session_store_type: str = "dynamodb"  # or "postgres"
    dynamodb_table_name: str = "hanah_sessions"
    postgres_url: str | None = None

    # Observability
    log_level: str = "INFO"
    sentry_dsn: str | None = None

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## Session Models

### Session Schema (`models/session.py`)

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal, Optional
from uuid import UUID, uuid4

class SalaryValue(BaseModel):
    amount: float
    period: Literal["yearly", "monthly"]
    currency: str  # ISO 4217 codes

class FileRecord(BaseModel):
    file_id: str = Field(default_factory=lambda: str(uuid4()))
    file_type: Literal["resume", "offer_letter", "job_description"]
    s3_url: str
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    ocr_status: Literal["pending", "processing", "completed", "failed"] = "pending"
    parsed_data: Optional[dict] = None

class TranscriptEntry(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    speaker: Literal["user", "system"]
    text: str
    is_final: bool = True

SessionState = Literal[
    "start",
    "identifying",
    "collecting",
    "upload_waiting",
    "ocr_processing",
    "confirm_context",
    "context_ready",
    "negotiating",
    "done"
]

class Session(BaseModel):
    session_id: UUID = Field(default_factory=uuid4)
    role_type: Optional[Literal["candidate", "recruiter"]] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    offered_salary: Optional[SalaryValue] = None
    desired_salary_min: Optional[SalaryValue] = None
    desired_salary_max: Optional[SalaryValue] = None
    batna: Optional[str] = None
    files: list[FileRecord] = Field(default_factory=list)
    transcript: list[TranscriptEntry] = Field(default_factory=list)
    state: SessionState = "start"
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
```

### WebSocket Message Schemas (`models/messages.py`)

```python
from pydantic import BaseModel
from typing import Literal, Optional

class WSMessage(BaseModel):
    type: str

# Client → Server
class AudioChunkMessage(WSMessage):
    type: Literal["audio_chunk"] = "audio_chunk"
    data: str  # base64 encoded audio
    format: str = "pcm_16khz_mono"

class UserTextMessage(WSMessage):
    type: Literal["user_message"] = "user_message"
    text: str

class FileUploadedMessage(WSMessage):
    type: Literal["file_uploaded"] = "file_uploaded"
    file_id: str
    s3_url: str

class UserActionMessage(WSMessage):
    type: Literal["user_action"] = "user_action"
    action: str  # "confirm_extraction", "start_practice", "export_plan"
    payload: dict = {}

# Server → Client
class TranscriptInterimMessage(WSMessage):
    type: Literal["transcript_interim"] = "transcript_interim"
    utterance_id: str
    text: str
    confidence: float

class TranscriptFinalMessage(WSMessage):
    type: Literal["transcript_final"] = "transcript_final"
    utterance_id: str
    text: str
    confidence: float

class AgentMessage(WSMessage):
    type: Literal["agent_message"] = "agent_message"
    speaker: Literal["collector", "negotiator"]
    text: str
    audio_url: Optional[str] = None

class StateUpdateMessage(WSMessage):
    type: Literal["state_update"] = "state_update"
    state: str
    progress: Optional[int] = None  # 0-100
    message: Optional[str] = None

class NegotiationPlanMessage(WSMessage):
    type: Literal["negotiation_plan"] = "negotiation_plan"
    plan: dict  # Full plan object

class ErrorMessage(WSMessage):
    type: Literal["error"] = "error"
    code: str
    message: str
```

---

## Collector Agent

### Agent Purpose

The Collector Agent is responsible for:

1. Identifying user role (candidate or recruiter)
2. Gathering required negotiation context (offered salary, desired salary, BATNA)
3. Orchestrating file uploads and OCR processing
4. Validating and confirming extracted data with the user
5. Determining when context is complete and handing off to Negotiator

### Agent State Schema

```python
from typing import TypedDict, Optional
from app.models.session import Session, SalaryValue

class CollectorState(TypedDict):
    session: Session
    current_message: Optional[str]  # Latest user input
    pending_confirmation: Optional[dict]  # Data awaiting user confirmation
    ocr_jobs: dict[str, str]  # file_id → textract_job_id
    attempts: dict[str, int]  # field_name → retry_count
```

### LangGraph Flow

```python
from langgraph.graph import StateGraph, END
from app.agents.tools.deepgram import DeepgramTool
from app.agents.tools.textract import TextractTool

def create_collector_graph():
    graph = StateGraph(CollectorState)

    # Nodes
    graph.add_node("identify_role", identify_role_node)
    graph.add_node("collect_offer", collect_offer_node)
    graph.add_node("collect_desired", collect_desired_node)
    graph.add_node("collect_batna", collect_batna_node)
    graph.add_node("handle_file_upload", handle_file_upload_node)
    graph.add_node("process_ocr", process_ocr_node)
    graph.add_node("confirm_context", confirm_context_node)
    graph.add_node("finalize", finalize_node)

    # Edges
    graph.set_entry_point("identify_role")

    graph.add_conditional_edges(
        "identify_role",
        router_after_identify,
        {
            "collect_offer": "collect_offer",
            "wait": END  # Wait for next user input
        }
    )

    graph.add_conditional_edges(
        "collect_offer",
        router_after_collect_offer,
        {
            "file_upload": "handle_file_upload",
            "collect_desired": "collect_desired",
            "wait": END
        }
    )

    graph.add_conditional_edges(
        "handle_file_upload",
        router_after_file_upload,
        {
            "process_ocr": "process_ocr",
            "collect_desired": "collect_desired"
        }
    )

    graph.add_conditional_edges(
        "process_ocr",
        router_after_ocr,
        {
            "confirm_context": "confirm_context",
            "collect_desired": "collect_desired"
        }
    )

    graph.add_conditional_edges(
        "collect_desired",
        router_after_collect_desired,
        {
            "collect_batna": "collect_batna",
            "wait": END
        }
    )

    graph.add_conditional_edges(
        "collect_batna",
        router_after_collect_batna,
        {
            "confirm_context": "confirm_context",
            "wait": END
        }
    )

    graph.add_conditional_edges(
        "confirm_context",
        router_after_confirm,
        {
            "finalize": "finalize",
            "collect_offer": "collect_offer",  # Re-collect if user rejects
            "wait": END
        }
    )

    graph.add_edge("finalize", END)

    return graph.compile()
```

### Node Implementations (Key Examples)

#### Identify Role Node

```python
async def identify_role_node(state: CollectorState) -> CollectorState:
    session = state["session"]

    if session.role_type is not None:
        # Role already set, skip
        return state

    if state["current_message"]:
        # Parse user message to extract role
        message = state["current_message"].lower()
        if "candidate" in message or "job seeker" in message:
            session.role_type = "candidate"
            await send_agent_message(
                session.session_id,
                "collector",
                "Great! Let's prepare you for your negotiation. Do you have an offer letter to upload?"
            )
        elif "recruiter" in message or "hiring" in message:
            session.role_type = "recruiter"
            await send_agent_message(
                session.session_id,
                "collector",
                "Perfect! Let's discuss the role you're hiring for."
            )
        else:
            # Unclear, ask again
            state["attempts"]["role_type"] = state["attempts"].get("role_type", 0) + 1
            if state["attempts"]["role_type"] < 3:
                await send_agent_message(
                    session.session_id,
                    "collector",
                    "I didn't catch that. Are you a candidate looking for a job, or a recruiter hiring?"
                )
            else:
                # Offer text input fallback
                await send_agent_message(
                    session.session_id,
                    "collector",
                    "Let me offer you a choice: type 'candidate' or 'recruiter'."
                )

        # Update session state
        session.state = "identifying"
        await session_store.update(session)

    state["session"] = session
    return state
```

#### Handle File Upload Node

```python
from app.agents.tools.textract import start_textract_job

async def handle_file_upload_node(state: CollectorState) -> CollectorState:
    session = state["session"]

    # Find pending file uploads
    pending_files = [f for f in session.files if f.ocr_status == "pending"]

    for file_record in pending_files:
        # Start Textract job
        job_id = await start_textract_job(file_record.s3_url)
        state["ocr_jobs"][file_record.file_id] = job_id
        file_record.ocr_status = "processing"

        # Send progress update to frontend
        await send_state_update(
            session.session_id,
            "ocr_processing",
            progress=0,
            message="Reading text from document…"
        )

    # Continue collecting while OCR runs (non-blocking)
    await send_agent_message(
        session.session_id,
        "collector",
        "Got it! While I process that, what salary are you hoping for?"
    )

    session.state = "ocr_processing"
    await session_store.update(session)
    state["session"] = session
    return state
```

#### Process OCR Node

```python
from app.agents.tools.textract import get_textract_result
from app.agents.tools.bedrock import extract_salary_data

async def process_ocr_node(state: CollectorState) -> CollectorState:
    session = state["session"]

    for file_id, job_id in state["ocr_jobs"].items():
        # Poll Textract job status
        status, raw_text = await get_textract_result(job_id)

        if status == "SUCCEEDED":
            # Post-process with Bedrock
            parsed_data = await extract_salary_data(raw_text)

            # Update file record
            file_record = next(f for f in session.files if f.file_id == file_id)
            file_record.ocr_status = "completed"
            file_record.parsed_data = parsed_data

            # Store in pending_confirmation for user validation
            state["pending_confirmation"] = parsed_data

            # Read back summary
            if parsed_data.get("offered_salary"):
                salary = parsed_data["offered_salary"]
                formatted = format_salary(salary)
                await send_agent_message(
                    session.session_id,
                    "collector",
                    f"I found an offer of {formatted} from {parsed_data.get('company', 'the company')}. Is that correct?"
                )

            # Remove from pending jobs
            del state["ocr_jobs"][file_id]

        elif status == "FAILED":
            file_record = next(f for f in session.files if f.file_id == file_id)
            file_record.ocr_status = "failed"
            await send_error_message(
                session.session_id,
                "OCR_FAILED",
                "I couldn't read that document. Would you like to re-upload or tell me the details?"
            )
            del state["ocr_jobs"][file_id]

    await session_store.update(session)
    state["session"] = session
    return state
```

### Router Logic (Decision Functions)

```python
def router_after_identify(state: CollectorState) -> str:
    if state["session"].role_type is not None:
        return "collect_offer"
    return "wait"

def router_after_collect_offer(state: CollectorState) -> str:
    session = state["session"]

    # Check if user initiated file upload
    if any(f.ocr_status == "pending" for f in session.files):
        return "file_upload"

    # Check if offered_salary is set
    if session.offered_salary is not None:
        return "collect_desired"

    return "wait"

def router_after_confirm(state: CollectorState) -> str:
    session = state["session"]

    # Check if all required fields are present
    required_fields = [
        session.role_type,
        session.offered_salary or session.state == "context_ready",
        session.desired_salary_min or session.desired_salary_max
    ]

    if all(required_fields):
        return "finalize"

    # User rejected, re-collect
    if state["current_message"] and "no" in state["current_message"].lower():
        return "collect_offer"

    return "wait"
```

---

## Negotiator Agent

### Agent Purpose

The Negotiator Agent is responsible for:

1. Querying the RAG service with session context
2. Synthesizing a negotiation plan and suggested replies
3. Vocalizing the plan via TTS
4. Facilitating role-play practice mode
5. Providing micro-feedback on user responses

### Agent State Schema

```python
class NegotiatorState(TypedDict):
    session: Session
    rag_response: Optional[dict]
    plan: Optional[dict]
    role_play_mode: bool
    role_play_turn: int
    role_play_history: list[dict]
```

### LangGraph Flow

```python
def create_negotiator_graph():
    graph = StateGraph(NegotiatorState)

    graph.add_node("query_rag", query_rag_node)
    graph.add_node("synthesize_plan", synthesize_plan_node)
    graph.add_node("deliver_plan", deliver_plan_node)
    graph.add_node("role_play", role_play_node)
    graph.add_node("provide_feedback", provide_feedback_node)

    graph.set_entry_point("query_rag")

    graph.add_edge("query_rag", "synthesize_plan")
    graph.add_edge("synthesize_plan", "deliver_plan")

    graph.add_conditional_edges(
        "deliver_plan",
        router_after_deliver,
        {
            "role_play": "role_play",
            "end": END
        }
    )

    graph.add_edge("role_play", "provide_feedback")

    graph.add_conditional_edges(
        "provide_feedback",
        router_after_feedback,
        {
            "role_play": "role_play",  # Continue role-play
            "end": END
        }
    )

    return graph.compile()
```

### Node Implementations

#### Query RAG Node

```python
from app.agents.tools.rag_client import query_rag_service

async def query_rag_node(state: NegotiatorState) -> NegotiatorState:
    session = state["session"]

    # Build RAG query payload
    query_payload = {
        "session_id": str(session.session_id),
        "role_type": session.role_type,
        "user_message": f"User offered {session.offered_salary}, wants {session.desired_salary_min}",
        "session_context": {
            "offered_salary": session.offered_salary.dict() if session.offered_salary else None,
            "desired_salary": session.desired_salary_min.dict() if session.desired_salary_min else None,
            "batna": session.batna,
            "parsed_texts": extract_parsed_texts(session.files)
        },
        "preferences": {
            "technique_priority": ["calibrated_question", "label"],
            "response_length": "short"
        }
    }

    # Call RAG service
    try:
        rag_response = await query_rag_service(query_payload)
        state["rag_response"] = rag_response

        # Send state update
        await send_state_update(
            session.session_id,
            "negotiating",
            message="Plan ready!"
        )

    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        # Fallback to canned template
        state["rag_response"] = get_fallback_plan(session)

    return state
```

#### Synthesize Plan Node

```python
async def synthesize_plan_node(state: NegotiatorState) -> NegotiatorState:
    rag_response = state["rag_response"]

    # RAG service already synthesized plan (includes retrieved + generated)
    plan = rag_response.get("synthesized", {})

    state["plan"] = {
        "plan_text": plan.get("plan", ""),
        "suggested_replies": plan.get("suggested_replies", []),
        "sources": rag_response.get("retrieved", []),
        "confidence": plan.get("confidence", "medium")
    }

    return state
```

#### Deliver Plan Node

```python
from app.agents.tools.tts import generate_tts

async def deliver_plan_node(state: NegotiatorState) -> NegotiatorState:
    session = state["session"]
    plan = state["plan"]

    # Generate TTS audio for plan text
    audio_url = await generate_tts(plan["plan_text"])

    # Send plan to frontend
    await send_negotiation_plan(
        session.session_id,
        plan,
        audio_url
    )

    # Log for observability
    logger.info(
        "Plan delivered",
        extra={
            "session_id": str(session.session_id),
            "replies_count": len(plan["suggested_replies"]),
            "sources_count": len(plan["sources"]),
            "confidence": plan["confidence"]
        }
    )

    return state
```

#### Role-Play Node

```python
async def role_play_node(state: NegotiatorState) -> NegotiatorState:
    session = state["session"]

    if not state["role_play_mode"]:
        # Initialize role-play
        state["role_play_mode"] = True
        state["role_play_turn"] = 0
        state["role_play_history"] = []

        # First recruiter utterance (use Bedrock to generate)
        recruiter_response = await generate_recruiter_utterance(
            session,
            tone="neutral",
            context="opening"
        )

        await send_agent_message(
            session.session_id,
            "negotiator",
            f"[Role-play mode] Recruiter: \"{recruiter_response}\""
        )

        state["role_play_history"].append({
            "speaker": "recruiter",
            "text": recruiter_response
        })
    else:
        # Generate next recruiter response based on user's last message
        user_message = state["session"].transcript[-1].text
        recruiter_response = await generate_recruiter_utterance(
            session,
            tone="neutral",
            context=user_message
        )

        await send_agent_message(
            session.session_id,
            "negotiator",
            f"Recruiter: \"{recruiter_response}\""
        )

        state["role_play_history"].append({
            "speaker": "recruiter",
            "text": recruiter_response
        })

    state["role_play_turn"] += 1
    return state
```

---

## Tools

### Deepgram Tool (`tools/deepgram.py`)

```python
from deepgram import Deepgram
from app.config import settings

class DeepgramTool:
    def __init__(self):
        self.client = Deepgram(settings.deepgram_api_key)

    async def transcribe_stream(self, audio_chunk: bytes) -> dict:
        """Send audio chunk to Deepgram, return interim/final transcript."""
        response = await self.client.transcription.prerecorded(
            {"buffer": audio_chunk, "mimetype": "audio/wav"},
            {"punctuate": True, "interim_results": True}
        )
        return response

    async def start_live_stream(self, websocket):
        """Start live streaming transcription."""
        # Connect to Deepgram live API
        dg_socket = await self.client.transcription.live({
            "punctuate": True,
            "interim_results": True,
            "language": "en"
        })

        return dg_socket
```

### Textract Tool (`tools/textract.py`)

```python
import boto3
from app.config import settings

textract_client = boto3.client(
    "textract",
    region_name=settings.aws_region,
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key
)

async def start_textract_job(s3_url: str) -> str:
    """Start async Textract document analysis job."""
    # Parse S3 URL
    bucket, key = parse_s3_url(s3_url)

    response = textract_client.start_document_analysis(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}},
        FeatureTypes=["TABLES", "FORMS"]
    )

    job_id = response["JobId"]
    logger.info(f"Textract job started: {job_id}")
    return job_id

async def get_textract_result(job_id: str) -> tuple[str, str]:
    """Poll Textract job status and retrieve result."""
    response = textract_client.get_document_analysis(JobId=job_id)

    status = response["JobStatus"]

    if status == "SUCCEEDED":
        # Extract text from blocks
        blocks = response["Blocks"]
        text = extract_text_from_blocks(blocks)
        return "SUCCEEDED", text

    elif status in ["FAILED", "PARTIAL_SUCCESS"]:
        return "FAILED", ""

    else:  # IN_PROGRESS
        return "IN_PROGRESS", ""

def extract_text_from_blocks(blocks: list) -> str:
    """Extract raw text from Textract blocks."""
    lines = []
    for block in blocks:
        if block["BlockType"] == "LINE":
            lines.append(block["Text"])
    return "\n".join(lines)
```

### RAG Client Tool (`tools/rag_client.py`)

```python
import httpx
from app.config import settings

async def query_rag_service(payload: dict) -> dict:
    """Call RAG microservice /query endpoint."""
    async with httpx.AsyncClient(timeout=settings.rag_service_timeout) as client:
        response = await client.post(
            f"{settings.rag_service_url}/query",
            json=payload
        )
        response.raise_for_status()
        return response.json()
```

### TTS Tool (`tools/tts.py`)

```python
import boto3
from app.config import settings

polly_client = boto3.client(
    "polly",
    region_name=settings.aws_region,
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key
)

async def generate_tts(text: str, voice_id: str = "Joanna") -> str:
    """Generate TTS audio and upload to S3, return public URL."""
    response = polly_client.synthesize_speech(
        Text=text,
        OutputFormat="mp3",
        VoiceId=voice_id,
        Engine="neural"
    )

    # Upload audio stream to S3
    audio_stream = response["AudioStream"].read()
    audio_key = f"tts/{uuid4()}.mp3"
    s3_client.put_object(
        Bucket=settings.s3_bucket_name,
        Key=audio_key,
        Body=audio_stream,
        ContentType="audio/mpeg"
    )

    # Generate public URL (or pre-signed URL)
    audio_url = f"https://{settings.s3_bucket_name}.s3.amazonaws.com/{audio_key}"
    return audio_url
```

---

## State Machine Implementation

The state machine is implicit in the LangGraph flows, but explicit state transitions are tracked in the session:

```python
async def transition_state(session: Session, new_state: SessionState):
    """Transition session to new state with logging."""
    old_state = session.state
    session.state = new_state
    session.updated_at = datetime.utcnow()

    await session_store.update(session)

    # Log transition
    logger.info(
        "State transition",
        extra={
            "session_id": str(session.session_id),
            "from": old_state,
            "to": new_state
        }
    )

    # Send state update to frontend
    await send_state_update(session.session_id, new_state)
```

---

## WebSocket Communication

### WebSocket Manager (`services/websocket_manager.py`)

```python
from fastapi import WebSocket
from typing import Dict

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected: {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected: {session_id}")

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            await websocket.send_json(message)

    async def broadcast(self, message: dict):
        """Send to all connected clients."""
        for websocket in self.active_connections.values():
            await websocket.send_json(message)

ws_manager = WebSocketManager()
```

### WebSocket Handler (`api/websocket.py`)

```python
from fastapi import WebSocket, WebSocketDisconnect
from app.services.websocket_manager import ws_manager
from app.agents.collector import collector_graph
from app.agents.negotiator import negotiator_graph

async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await ws_manager.connect(session_id, websocket)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "audio_chunk":
                await handle_audio_chunk(session_id, data)

            elif message_type == "user_message":
                await handle_user_message(session_id, data["text"])

            elif message_type == "file_uploaded":
                await handle_file_uploaded(session_id, data)

            elif message_type == "user_action":
                await handle_user_action(session_id, data)

    except WebSocketDisconnect:
        ws_manager.disconnect(session_id)

async def handle_user_message(session_id: str, text: str):
    """Process user text message through agent."""
    session = await session_store.get(session_id)

    # Add to transcript
    session.transcript.append(TranscriptEntry(
        speaker="user",
        text=text,
        is_final=True
    ))
    await session_store.update(session)

    # Route to appropriate agent
    if session.state in ["start", "identifying", "collecting", "upload_waiting", "ocr_processing", "confirm_context"]:
        # Collector agent
        state = {"session": session, "current_message": text, "ocr_jobs": {}, "attempts": {}}
        result = await collector_graph.ainvoke(state)

    elif session.state in ["context_ready", "negotiating"]:
        # Negotiator agent
        state = {"session": session, "role_play_mode": False, "role_play_turn": 0, "role_play_history": []}
        result = await negotiator_graph.ainvoke(state)
```

---

## Error Handling and Fallback Strategies

### Textract Failure

```python
async def handle_ocr_failure(session_id: str, file_id: str):
    """Handle OCR failure gracefully."""
    await send_error_message(
        session_id,
        "OCR_FAILED",
        "I couldn't read that document. Would you like to re-upload or tell me the details?"
    )

    # Continue collecting other fields (non-blocking)
    await send_agent_message(
        session_id,
        "collector",
        "Let's continue. What salary are you hoping for?"
    )
```

### RAG Timeout

```python
def get_fallback_plan(session: Session) -> dict:
    """Return canned fallback plan for common scenario."""
    return {
        "synthesized": {
            "plan": "Ask calibrated questions to understand their constraints.",
            "suggested_replies": [
                {
                    "text": "Can you help me understand your budget constraints?",
                    "technique": "calibrated_question",
                    "rationale": "Invites collaboration"
                },
                {
                    "text": "What flexibility might exist in the total compensation?",
                    "technique": "calibrated_question",
                    "rationale": "Opens alternatives"
                }
            ],
            "confidence": "low"
        },
        "retrieved": []
    }
```

### Bedrock Rate Limit

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
async def call_bedrock_with_retry(prompt: str) -> str:
    """Call Bedrock with exponential backoff."""
    response = await bedrock_client.invoke_model(
        modelId=settings.bedrock_model_id,
        body={"prompt": prompt, "max_tokens": 500}
    )
    return response["completion"]
```

---

## Timeouts and User Nudges

```python
import asyncio

async def start_idle_timer(session_id: str):
    """Start idle timer, nudge user if silent too long."""
    await asyncio.sleep(60)  # 60 seconds

    session = await session_store.get(session_id)

    # Check if user has been silent
    last_transcript = session.transcript[-1] if session.transcript else None
    if last_transcript and last_transcript.speaker == "system":
        # System spoke last, user hasn't responded
        await send_agent_message(
            session_id,
            "collector",
            "Are you still there? Let me know when you're ready to continue."
        )

        # Start second timer
        await asyncio.sleep(300)  # 5 minutes

        session = await session_store.get(session_id)
        # Offer text input fallback
        await send_agent_message(
            session_id,
            "collector",
            "Would you prefer to type your responses instead?"
        )
```

---

## Session Store Service (`services/session_store.py`)

```python
from abc import ABC, abstractmethod
from app.models.session import Session

class SessionStore(ABC):
    @abstractmethod
    async def get(self, session_id: str) -> Session:
        pass

    @abstractmethod
    async def create(self, session: Session) -> Session:
        pass

    @abstractmethod
    async def update(self, session: Session) -> Session:
        pass

    @abstractmethod
    async def delete(self, session_id: str):
        pass

class DynamoDBSessionStore(SessionStore):
    def __init__(self, table_name: str):
        self.table = boto3.resource("dynamodb").Table(table_name)

    async def get(self, session_id: str) -> Session:
        response = self.table.get_item(Key={"session_id": session_id})
        if "Item" not in response:
            raise ValueError(f"Session {session_id} not found")
        return Session(**response["Item"])

    async def create(self, session: Session) -> Session:
        self.table.put_item(Item=session.dict())
        return session

    async def update(self, session: Session) -> Session:
        session.updated_at = datetime.utcnow()
        self.table.put_item(Item=session.dict())
        return session

    async def delete(self, session_id: str):
        self.table.delete_item(Key={"session_id": session_id})

# Factory
def create_session_store() -> SessionStore:
    if settings.session_store_type == "dynamodb":
        return DynamoDBSessionStore(settings.dynamodb_table_name)
    elif settings.session_store_type == "postgres":
        return PostgresSessionStore(settings.postgres_url)
    else:
        raise ValueError(f"Unknown session store type: {settings.session_store_type}")

session_store = create_session_store()
```

---

## Observability

### Structured Logging

```python
import structlog

logger = structlog.get_logger()

# Example usage
logger.info(
    "agent_invoked",
    agent="collector",
    session_id=str(session_id),
    state=session.state
)
```

### Metrics (CloudWatch)

```python
import boto3

cloudwatch = boto3.client("cloudwatch")

async def emit_metric(name: str, value: float, unit: str = "Count"):
    cloudwatch.put_metric_data(
        Namespace="HANAH/Orchestrator",
        MetricData=[{
            "MetricName": name,
            "Value": value,
            "Unit": unit
        }]
    )

# Example
await emit_metric("session_started", 1)
await emit_metric("ocr_duration", duration_seconds, "Seconds")
```

---

## Testing Strategy

### Unit Tests

```python
import pytest
from app.agents.collector import identify_role_node

@pytest.mark.asyncio
async def test_identify_role_candidate():
    state = {
        "session": Session(),
        "current_message": "I'm a candidate",
        "attempts": {}
    }
    result = await identify_role_node(state)
    assert result["session"].role_type == "candidate"
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_file_upload_to_ocr():
    # Mock Textract
    with patch("app.agents.tools.textract.start_textract_job") as mock_textract:
        mock_textract.return_value = "job-123"

        session = Session(files=[FileRecord(s3_url="s3://bucket/file.pdf")])
        state = {"session": session, "ocr_jobs": {}}

        result = await handle_file_upload_node(state)

        assert len(result["ocr_jobs"]) == 1
        assert result["session"].files[0].ocr_status == "processing"
```

---

**End of LangGraph Orchestrator Specification**
