# Frontend Specification

**Component:** React + Vite Frontend  
**Version:** 1.0  
**Last Updated:** November 6, 2025

---

## Overview

The HANAH frontend is a voice-first, single-page application (SPA) built with React and Vite. It provides an intuitive conversational interface for salary negotiation preparation, emphasizing real-time audio streaming, transparent AI interactions, and user control over privacy.

### Design Principles

1. **Voice-First, Not Voice-Only**: Microphone is primary input; text fallback always available
2. **Progressive Disclosure**: Show complexity only when needed
3. **Transparency**: Always reveal AI reasoning and data sources
4. **Privacy by Design**: Clear controls, explicit consent, one-click deletion
5. **Accessibility**: WCAG 2.1 AA compliance

---

## UI/UX Requirements

### 1. Voice-First Design

**Primary Interaction Pattern**:
- Large, prominent microphone button (center of initial view)
- Button states:
  - **Idle**: Gray, "Tap to speak"
  - **Listening**: Pulsing red, "Listening…"
  - **Processing**: Spinner, "Thinking…"
  - **Error**: Red X, "Mic unavailable"
- Voice activity indicator (waveform visualization during speech)
- Automatic silence detection (stops listening after 2s of silence)

**Fallback to Text**:
- Small "Type instead" button below mic
- Text input expands when clicked
- User can switch mid-conversation

### 2. Core Layout

```
┌─────────────────────────────────────────────────────────┐
│  Header: HANAH logo | Session ID | [Delete Session]     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │         Transcript Panel (scrollable)          │    │
│  │  ┌──────────────────────────────────────────┐  │    │
│  │  │ 🎙️ You: "I want ₹15 LPA"                │  │    │
│  │  └──────────────────────────────────────────┘  │    │
│  │  ┌──────────────────────────────────────────┐  │    │
│  │  │ 🤖 HANAH: "Got it. Do you have any       │  │    │
│  │  │             other offers?"                │  │    │
│  │  └──────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │         Negotiation Canvas (conditionally)     │    │
│  │  📋 Plan: Anchor at ₹15 LPA, use calibrated…  │    │
│  │  💬 Suggested Replies: [Card] [Card] [Card]   │    │
│  │  📚 Sources: [Voss Ch.7] [Transcript-42]      │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
├─────────────────────────────────────────────────────────┤
│  Footer: 🎤 [Mic Button] | 📄 [Upload] | ⚙️ [Settings] │
└─────────────────────────────────────────────────────────┘
```

### 3. Component Structure

#### 3.1 MicButton Component

```typescript
interface MicButtonProps {
  status: "idle" | "listening" | "processing" | "error";
  onStart: () => void;
  onStop: () => void;
}
```

**Behavior**:
- Click to start recording (request mic permission if not granted)
- Click again to stop (or auto-stop after silence)
- Visual feedback: pulsing animation during listening
- Error state: show permission instructions or browser compatibility message

**Accessibility**:
- ARIA label: "Start voice input" / "Stop voice input"
- Keyboard shortcut: `Space` to toggle (when focused)
- Screen reader announces status changes

#### 3.2 TranscriptPanel Component

```typescript
interface TranscriptEntry {
  id: string;
  speaker: "user" | "system";
  text: string;
  timestamp: Date;
  is_final: boolean;  // false for interim ASR results
}

interface TranscriptPanelProps {
  entries: TranscriptEntry[];
  onEdit?: (id: string, newText: string) => void;  // Edit critical values
}
```

**Behavior**:
- Auto-scroll to latest message
- Interim transcripts render with lower opacity (30%)
- Final transcripts render solid
- User messages show edit icon for numbers/critical data
- System messages can have embedded UI (e.g., "Confirm this value?" with buttons)

**Styling**:
- User messages: right-aligned, blue background
- System messages: left-aligned, gray background
- Timestamps: small, muted, shown on hover

#### 3.3 FileUpload Component

```typescript
interface FileUploadProps {
  onUpload: (file: File) => void;
  allowedTypes: string[];  // ["application/pdf"]
  maxSizeMB: number;       // 10
  status: "idle" | "uploading" | "processing" | "success" | "error";
}
```

**Behavior**:
- Click to open file picker (or drag-and-drop)
- Validate file type and size client-side before upload
- Show progress bar during upload
- After upload, show "OCR in progress…" state
- On completion, briefly highlight extracted summary in transcript

**UX Flow**:
1. User clicks "Upload Document"
2. File picker opens (filtered to PDFs)
3. User selects file → immediate upload to pre-signed S3 URL
4. Progress bar: "Uploading… 75%"
5. Upload completes → UI shows "Reading document…"
6. Meanwhile, Collector continues voice conversation (non-blocking)
7. OCR completes → "I found ₹12 LPA. Is that correct?" appears in transcript

**Error Handling**:
- File too large: "Please upload a PDF under 10 MB"
- Wrong type: "Only PDF files are supported"
- Upload fails: "Upload failed. Try again?"

#### 3.4 NegotiationCanvas Component

```typescript
interface NegotiationPlan {
  plan_text: string;                 // "Anchor at ₹15 LPA…"
  suggested_replies: SuggestedReply[];
  sources: ProvenanceSource[];
}

interface SuggestedReply {
  id: string;
  text: string;                      // "Can you help me understand…"
  technique: string;                 // "calibrated_question"
  rationale: string;                 // "Invites problem-solving"
  confidence: "high" | "medium" | "low";
}

interface ProvenanceSource {
  id: string;
  title: string;                     // "Voss Ch. 7"
  snippet: string;                   // First 100 chars
  onClick: () => void;               // Show full snippet modal
}
```

**Layout**:
```
┌────────────────────────────────────────────┐
│  📋 Your Negotiation Plan                  │
│  ─────────────────────────────────────────  │
│  Anchor at ₹15 LPA and use calibrated      │
│  questions to explore flexibility.         │
│                                            │
│  💬 Suggested Replies                      │
│  ┌──────────────────────────────────────┐ │
│  │ "Can you help me understand how      │ │
│  │  you arrived at ₹12 LPA?"            │ │
│  │ 🏷️ Calibrated Question              │ │
│  │ 💡 Invites problem-solving           │ │
│  │ [Copy] [Practice]                    │ │
│  └──────────────────────────────────────┘ │
│  ┌──────────────────────────────────────┐ │
│  │ "It sounds like budget is tight —    │ │
│  │  what flexibility exists?"           │ │
│  │ 🏷️ Label + Calibrated Question      │ │
│  │ 💡 Opens space for solutions         │ │
│  │ [Copy] [Practice]                    │ │
│  └──────────────────────────────────────┘ │
│                                            │
│  📚 Sources                                │
│  • Voss Ch. 7: Calibrated Questions [View]│
│  • Transcript-42: Candidate negotiation    │
└────────────────────────────────────────────┘
```

**Behavior**:
- Appears only after Negotiator Agent generates plan
- Smooth slide-in animation from bottom
- "Copy" button copies reply text to clipboard
- "Practice" button enters role-play mode with that reply pre-loaded
- "View" source opens modal with full snippet and metadata

**Accessibility**:
- Each reply card is a focusable element
- Arrow keys navigate between replies
- Screen reader announces technique and rationale

#### 3.5 ProvenanceModal Component

```typescript
interface ProvenanceModalProps {
  source: {
    title: string;
    full_text: string;
    metadata: {
      technique: string;
      source_book?: string;
      page?: number;
      intent: string;
    };
  };
  onClose: () => void;
}
```

**Content**:
```
┌───────────────────────────────────────────────┐
│  📖 Source: Voss Ch. 7                    [X] │
├───────────────────────────────────────────────┤
│  Technique: Calibrated Question               │
│  Intent: Invite collaboration                 │
│                                               │
│  Full Text:                                   │
│  "Calibrated questions start with 'What' or   │
│   'How' and invite the other party to solve   │
│   the problem with you. They shift the        │
│   dynamic from confrontation to               │
│   collaboration."                             │
│                                               │
│  📄 Source: Never Split the Difference, p.142 │
└───────────────────────────────────────────────┘
```

### 4. Real-Time Feedback Patterns

#### 4.1 Interim Transcripts

- Deepgram sends interim results via WebSocket
- Render in transcript panel with 30% opacity and italic style
- When final result arrives, replace interim with final (solid, normal weight)
- Prevents UI flicker by updating in place (keyed by utterance ID)

#### 4.2 OCR Progress Indicator

```
┌────────────────────────────────────────────┐
│  📄 Processing offer_letter.pdf            │
│  ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░ 60%                 │
│  Reading text from document…               │
└────────────────────────────────────────────┘
```

**States**:
1. **Uploading**: "Uploading… X%"
2. **OCR Queued**: "Waiting to process…"
3. **OCR Processing**: "Reading text… (this may take 30s)"
4. **Parsing**: "Extracting salary details…"
5. **Complete**: "✓ Found ₹12 LPA" (auto-dismiss after 3s)

#### 4.3 RAG "Preparing Plan" State

```
┌────────────────────────────────────────────┐
│  🧠 Preparing your negotiation plan…       │
│  [Animated dots or spinner]                │
└────────────────────────────────────────────┘
```

**Trigger**: When Negotiator Agent invokes RAG service
**Duration**: Typically 1–3 seconds
**Fallback**: If > 5 seconds, show "Taking longer than usual…"

### 5. Transparency and Provenance Display

**Requirement**: Every suggested reply must show:
1. The negotiation technique used (e.g., "Calibrated Question")
2. A brief rationale (≤20 words)
3. Link to source snippet(s) that informed it

**Implementation**:
- Small "Sources" section at bottom of NegotiationCanvas
- Each source is clickable → opens ProvenanceModal
- Modal shows full retrieved chunk + metadata (technique, book, page)

**Why**: Builds trust; user can verify AI isn't fabricating advice

### 6. Accessibility Requirements

**WCAG 2.1 AA Compliance**:

| Requirement | Implementation |
|-------------|----------------|
| **Keyboard Navigation** | All interactive elements focusable; logical tab order; Esc closes modals |
| **Screen Reader Support** | ARIA labels on all buttons; live regions for transcript updates; role announcements |
| **Color Contrast** | Minimum 4.5:1 for text; 3:1 for UI components |
| **Focus Indicators** | Visible focus ring (2px solid blue) on all interactive elements |
| **Text Resizing** | UI remains usable at 200% zoom; no horizontal scroll |
| **Captions/Transcripts** | Audio playback has text alternative (transcript panel) |

**Voice Input Alternatives**:
- Always provide text input fallback
- Allow editing of interim transcripts for mis-recognized speech
- Keyboard shortcut to toggle mic (`Space` or `Ctrl+M`)

**Visual Impairment**:
- High-contrast mode toggle in settings
- Larger text option (16px → 20px base)

**Motor Impairment**:
- Large click targets (min 44x44px)
- Voice-only mode (no mouse required)

### 7. Privacy Controls and Session Management

#### 7.1 Consent Modal (First Visit)

```
┌───────────────────────────────────────────────────────┐
│  Welcome to HANAH                                 [X] │
├───────────────────────────────────────────────────────┤
│  Before we start:                                     │
│                                                       │
│  ☐ I consent to transcription of my voice            │
│  ☐ I consent to OCR processing of uploaded files     │
│  ☐ I consent to temporary storage of session data    │
│     (deleted after 24 hours or on request)           │
│                                                       │
│  You can change these settings anytime.               │
│  [Privacy Policy]                                     │
│                                                       │
│  [Cancel]                      [Accept & Continue]    │
└───────────────────────────────────────────────────────┘
```

**Behavior**:
- Appears on first session load
- All checkboxes must be checked to proceed (hard requirement for MVP)
- Future enhancement: allow partial consent (e.g., no file upload)

#### 7.2 Settings Panel

**Access**: Gear icon (⚙️) in footer

**Options**:
```
┌────────────────────────────────────────┐
│  ⚙️ Settings                           │
├────────────────────────────────────────┤
│  Privacy                               │
│  ☐ Local-only mode (no server storage)│
│  ☐ Delete transcripts after session   │
│                                        │
│  Accessibility                         │
│  ☐ High-contrast mode                 │
│  ☐ Large text                          │
│  ☐ Reduce motion                       │
│                                        │
│  Audio                                 │
│  Mic sensitivity: [───●──]             │
│  TTS speed: [──●────] 1.0x             │
│  ☐ Auto-play agent responses           │
│                                        │
│  [Save]                    [Cancel]    │
└────────────────────────────────────────┘
```

#### 7.3 Delete Session Button

**Location**: Top-right header

**Behavior**:
1. User clicks "Delete Session"
2. Confirmation modal:
   ```
   ┌────────────────────────────────────────┐
   │  Delete this session?                  │
   ├────────────────────────────────────────┤
   │  This will permanently delete:         │
   │  • Your transcript                     │
   │  • Uploaded files                      │
   │  • Generated plans                     │
   │                                        │
   │  This cannot be undone.                │
   │                                        │
   │  [Cancel]           [Yes, Delete]      │
   └────────────────────────────────────────┘
   ```
3. On confirm, send DELETE request to backend
4. Show success toast: "Session deleted" (3s)
5. Redirect to home screen

**Backend Side Effect**:
- Remove DynamoDB session record
- Delete all S3 files associated with session
- Log deletion in audit trail

---

## WebSocket / Streaming Integration

### WebSocket Protocol

**Endpoint**: `wss://api.hanah.example.com/ws/{session_id}`

**Authentication**: JWT token in query param or header

**Message Types** (JSON):

#### Client → Server

1. **Audio Chunk** (binary or base64)
   ```json
   {
     "type": "audio_chunk",
     "data": "<base64_audio>",
     "format": "pcm_16khz_mono"
   }
   ```

2. **Text Input**
   ```json
   {
     "type": "user_message",
     "text": "I want ₹15 LPA"
   }
   ```

3. **File Upload Complete**
   ```json
   {
     "type": "file_uploaded",
     "file_id": "abc-123",
     "s3_url": "s3://bucket/..."
   }
   ```

4. **User Action**
   ```json
   {
     "type": "user_action",
     "action": "confirm_extraction" | "start_practice" | "export_plan",
     "payload": {}
   }
   ```

#### Server → Client

1. **Interim Transcript**
   ```json
   {
     "type": "transcript_interim",
     "utterance_id": "utt-001",
     "text": "I want fifteen",
     "confidence": 0.87
   }
   ```

2. **Final Transcript**
   ```json
   {
     "type": "transcript_final",
     "utterance_id": "utt-001",
     "text": "I want ₹15 LPA",
     "confidence": 0.95
   }
   ```

3. **Agent Message**
   ```json
   {
     "type": "agent_message",
     "speaker": "collector" | "negotiator",
     "text": "Got it. Do you have any other offers?",
     "audio_url": "https://cdn.../audio.mp3"  // Optional TTS
   }
   ```

4. **State Update**
   ```json
   {
     "type": "state_update",
     "state": "ocr_processing",
     "progress": 60,
     "message": "Reading document…"
   }
   ```

5. **Negotiation Plan**
   ```json
   {
     "type": "negotiation_plan",
     "plan": {
       "plan_text": "Anchor at ₹15 LPA…",
       "suggested_replies": [
         {
           "id": "reply-1",
           "text": "Can you help me understand…",
           "technique": "calibrated_question",
           "rationale": "Invites problem-solving",
           "confidence": "high"
         }
       ],
       "sources": [
         {
           "id": "src-1",
           "title": "Voss Ch. 7",
           "snippet": "Calibrated questions…",
           "metadata": {"page": 142}
         }
       ]
     }
   }
   ```

6. **Error**
   ```json
   {
     "type": "error",
     "code": "OCR_FAILED" | "RAG_TIMEOUT" | "INVALID_INPUT",
     "message": "Could not read document. Please re-upload."
   }
   ```

### Connection Lifecycle

1. **Open**: Frontend connects on session start
2. **Ping/Pong**: Send ping every 30s to keep alive
3. **Reconnect**: If connection drops, attempt reconnect with exponential backoff (1s, 2s, 4s, 8s, 16s max)
4. **Close**: Graceful close on session end or user navigation away

---

## Component Interaction Patterns

### Pattern 1: Voice Input Flow

```
User clicks mic
  ↓
Frontend requests mic permission
  ↓
Start audio capture (MediaRecorder API)
  ↓
Send audio chunks via WebSocket (every 500ms)
  ↓
Backend forwards to Deepgram
  ↓
Deepgram returns interim transcripts
  ↓
Frontend renders interim in TranscriptPanel (low opacity)
  ↓
User stops speaking (2s silence)
  ↓
Frontend sends end-of-utterance marker
  ↓
Deepgram returns final transcript
  ↓
Frontend updates interim → final (solid, normal)
  ↓
Backend processes message (Collector/Negotiator)
  ↓
Backend sends agent_message
  ↓
Frontend renders in TranscriptPanel + plays TTS (if enabled)
```

### Pattern 2: File Upload Flow

```
User clicks "Upload Document"
  ↓
File picker opens (PDF only)
  ↓
User selects file
  ↓
Frontend validates (type, size)
  ↓
Frontend requests pre-signed S3 URL from backend (HTTP POST)
  ↓
Backend returns pre-signed URL + file_id
  ↓
Frontend uploads file to S3 directly (with progress bar)
  ↓
Upload completes
  ↓
Frontend sends file_uploaded message via WebSocket
  ↓
Backend triggers Textract job
  ↓
Backend sends state_update: "ocr_processing" with progress
  ↓
OCR completes
  ↓
Backend sends agent_message: "I found ₹12 LPA. Is that correct?"
  ↓
Frontend shows confirmation buttons in transcript
```

### Pattern 3: Plan Generation Flow

```
Context ready (Collector completes)
  ↓
Backend sends state_update: "negotiating"
  ↓
Frontend shows "Preparing plan…" indicator
  ↓
Backend Negotiator calls RAG service
  ↓
RAG service returns plan + replies + sources
  ↓
Backend sends negotiation_plan message
  ↓
Frontend renders NegotiationCanvas (slide-in animation)
  ↓
Frontend plays TTS of plan_text (if auto-play enabled)
  ↓
User clicks "View" on a source
  ↓
Frontend opens ProvenanceModal with full snippet
```

---

## Error Handling in UI

| Error Scenario | User-Facing Message | Recovery Action |
|----------------|---------------------|-----------------|
| **Mic permission denied** | "Microphone access required. Please enable in browser settings." | Show text input; link to help doc |
| **WebSocket disconnected** | "Connection lost. Reconnecting…" | Auto-reconnect; show retry button if fails |
| **File upload failed** | "Upload failed. Try again?" | [Retry] button |
| **OCR failed** | "Couldn't read document. You can re-upload or tell me the details." | Continue voice conversation; allow re-upload |
| **RAG timeout** | "Using a general template while service recovers." | Show fallback script; retry in background |
| **Bedrock rate limit** | "AI service is busy. Trying again…" | Exponential backoff; show fallback after 3 retries |
| **Invalid input** | "I didn't catch that. Could you rephrase?" | Ask again; offer text input |

**Error UI Component**:
```typescript
interface ErrorBannerProps {
  type: "warning" | "error" | "info";
  message: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  dismissible: boolean;
}
```

**Example**:
```
┌────────────────────────────────────────────────────┐
│ ⚠️ Connection lost. Reconnecting… [Retry] [X]      │
└────────────────────────────────────────────────────┘
```

---

## Performance Considerations

1. **Audio Streaming**:
   - Buffer size: 4096 samples
   - Send frequency: every 500ms (avoid too frequent small chunks)
   - Use WebSocket binary frames (not base64) for efficiency

2. **Transcript Rendering**:
   - Virtualize transcript panel if > 100 messages (react-window)
   - Debounce interim transcript updates (50ms)

3. **File Upload**:
   - Direct S3 upload (bypass backend for large files)
   - Show progress bar; allow cancellation

4. **TTS Playback**:
   - Preload audio URLs when plan arrives
   - Cache common phrases (e.g., "Got it.")

5. **Lazy Loading**:
   - Code-split NegotiationCanvas and ProvenanceModal
   - Load only when needed

---

## Testing Strategy

### Unit Tests

- Component rendering (Jest + React Testing Library)
- WebSocket message handling (mocked)
- File upload validation logic

### Integration Tests

- Full voice input flow (mocked Deepgram)
- File upload → OCR progress → confirmation
- Plan generation → display → provenance modal

### E2E Tests (Playwright)

1. **Happy path**:
   - Start session → upload PDF → collect context → generate plan → export
2. **Error scenarios**:
   - Mic permission denied
   - File upload fails
   - OCR fails → fallback to manual entry
3. **Accessibility**:
   - Keyboard-only navigation
   - Screen reader testing (NVDA/JAWS)

---

## Configuration and Environment Variables

**Frontend `.env`**:

```bash
VITE_API_BASE_URL=https://api.hanah.example.com
VITE_WS_URL=wss://api.hanah.example.com/ws
VITE_DEEPGRAM_API_KEY=<client-side-key>  # Optional, if using client-side Deepgram
VITE_SENTRY_DSN=<sentry-dsn>              # Error tracking
VITE_ENVIRONMENT=production
```

**Feature Flags** (fetched from backend):

```json
{
  "features": {
    "enable_role_play": true,
    "enable_export_pdf": true,
    "enable_local_mode": false,
    "max_file_size_mb": 10,
    "supported_currencies": ["INR", "USD", "EUR", "GBP"]
  }
}
```

---

## Deployment

### Build

```bash
npm run build
# Output: dist/
```

### Hosting

- **S3**: Upload `dist/` to S3 bucket
- **CloudFront**: CDN in front of S3 for global low latency
- **Custom Domain**: Route53 DNS → CloudFront → S3

### CI/CD

1. PR → GitHub Actions runs tests
2. Merge to `main` → build + deploy to staging
3. Manual approval → deploy to production

---

## Metrics and Observability

**Frontend Metrics** (sent to backend analytics endpoint):

- `page_load_time_ms`: Time to interactive
- `mic_permission_granted`: Boolean, tracks permission success rate
- `file_upload_duration_ms`: Time from select to S3 upload complete
- `ws_connection_errors`: Count of WebSocket failures
- `plan_display_time_ms`: Time from context_ready to plan rendered
- `user_feedback`: Thumbs up/down on suggested replies

**Error Tracking** (Sentry):

- JavaScript exceptions
- WebSocket errors
- Failed API calls

---

## Future Enhancements (Post-MVP)

1. **Multi-Language Support**: Hindi, Spanish, Mandarin
2. **Mobile App**: React Native version
3. **Offline Mode**: IndexedDB cache for plans; sync when online
4. **Advanced Role-Play**: Video-based recruiter avatar
5. **Team Plans**: Share negotiation plans with mentors/friends
6. **Integration**: Export to Google Docs, Notion, email

---

**End of Frontend Specification**

