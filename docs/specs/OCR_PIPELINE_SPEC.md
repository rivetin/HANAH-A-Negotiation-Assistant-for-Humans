# OCR Pipeline Specification

**Component:** AWS Textract Integration & Data Extraction  
**Version:** 1.0  
**Last Updated:** November 6, 2025

---

## Overview

The OCR (Optical Character Recognition) pipeline is responsible for extracting structured data from PDF documents uploaded by users. It leverages AWS Textract for document analysis, applies post-processing heuristics to normalize salary and job information, and validates results with the user before incorporating them into the negotiation context.

**Key Goals**:
- Extract salary details (amount, currency, period) from offer letters
- Extract company name, job title, and benefits from various document types
- Normalize extracted data into structured format
- Confirm accuracy with the user before proceeding
- Handle failures gracefully with fallback to manual entry

---

## Architecture

### Flow Diagram

```
User uploads PDF
       ↓
Frontend → Pre-signed S3 URL (from backend)
       ↓
Upload to S3 (direct from frontend)
       ↓
Notify backend via WebSocket
       ↓
Backend → Start Textract job (async)
       ↓
Poll/SNS notification → Job complete
       ↓
Retrieve Textract results
       ↓
Extract raw text + tables
       ↓
Apply regex heuristics (salary patterns, company, title)
       ↓
Post-process with Bedrock (normalize to JSON)
       ↓
Validate & confirm with user
       ↓
Update session context
```

---

## AWS Textract Integration

### Document Types Supported

- **PDF only** (MVP scope)
- Max file size: **10 MB**
- Max pages: **50 pages** (for performance)

### Textract API Selection

| API | Use Case | Justification |
|-----|----------|---------------|
| `StartDocumentAnalysis` | Offer letters, JDs with tables | Extracts tables (salary breakdowns, benefits) |
| `StartDocumentTextDetection` | Simple resumes | Faster, text-only extraction |

**Decision Rule**: Use `StartDocumentAnalysis` by default for offer letters; use `StartDocumentTextDetection` for resumes.

### Textract Job Flow

#### 1. Start Job

```python
import boto3

textract_client = boto3.client("textract", region_name="us-east-1")

def start_textract_job(s3_bucket: str, s3_key: str, document_type: str) -> str:
    """Start async Textract job."""
    if document_type in ["offer_letter", "job_description"]:
        response = textract_client.start_document_analysis(
            DocumentLocation={
                "S3Object": {
                    "Bucket": s3_bucket,
                    "Name": s3_key
                }
            },
            FeatureTypes=["TABLES", "FORMS"]
        )
    else:  # resume
        response = textract_client.start_document_text_detection(
            DocumentLocation={
                "S3Object": {
                    "Bucket": s3_bucket,
                    "Name": s3_key
                }
            }
        )
    
    job_id = response["JobId"]
    logger.info(f"Textract job started: {job_id} for {s3_key}")
    return job_id
```

#### 2. Poll Job Status

```python
import asyncio

async def wait_for_textract_job(job_id: str, max_wait_seconds: int = 60) -> str:
    """Poll Textract job until complete or timeout."""
    start_time = time.time()
    
    while time.time() - start_time < max_wait_seconds:
        response = textract_client.get_document_analysis(JobId=job_id)
        status = response["JobStatus"]
        
        if status == "SUCCEEDED":
            logger.info(f"Textract job {job_id} succeeded")
            return "SUCCEEDED"
        elif status == "FAILED":
            logger.error(f"Textract job {job_id} failed")
            return "FAILED"
        elif status == "PARTIAL_SUCCESS":
            logger.warning(f"Textract job {job_id} partially succeeded")
            return "PARTIAL_SUCCESS"
        
        # Still in progress, wait and retry
        await asyncio.sleep(2)
    
    # Timeout
    logger.warning(f"Textract job {job_id} timed out")
    return "TIMEOUT"
```

#### 3. Retrieve Results

```python
def get_textract_results(job_id: str) -> dict:
    """Retrieve Textract analysis results."""
    response = textract_client.get_document_analysis(JobId=job_id)
    
    if response["JobStatus"] != "SUCCEEDED":
        raise ValueError(f"Job {job_id} not successful: {response['JobStatus']}")
    
    blocks = response["Blocks"]
    
    # Handle pagination (if document > 1000 blocks)
    next_token = response.get("NextToken")
    while next_token:
        response = textract_client.get_document_analysis(
            JobId=job_id,
            NextToken=next_token
        )
        blocks.extend(response["Blocks"])
        next_token = response.get("NextToken")
    
    return {"blocks": blocks, "metadata": response.get("DocumentMetadata", {})}
```

---

## Text Extraction

### Block Types

Textract returns blocks of various types:

| Block Type | Description | Usage |
|------------|-------------|-------|
| `PAGE` | Page boundary | Pagination |
| `LINE` | Line of text | Extract raw text |
| `WORD` | Individual word | Fine-grained parsing |
| `TABLE` | Table structure | Extract salary breakdowns |
| `CELL` | Table cell | Parse structured data |
| `KEY_VALUE_SET` | Form field | Extract labeled fields |

### Extract Raw Text

```python
def extract_text_from_blocks(blocks: list) -> str:
    """Extract all text from LINE blocks."""
    lines = []
    for block in blocks:
        if block["BlockType"] == "LINE":
            lines.append(block["Text"])
    
    return "\n".join(lines)
```

### Extract Tables

```python
def extract_tables_from_blocks(blocks: list) -> list[dict]:
    """Extract table data from TABLE and CELL blocks."""
    tables = []
    
    # Build block map for relationships
    block_map = {block["Id"]: block for block in blocks}
    
    for block in blocks:
        if block["BlockType"] == "TABLE":
            table_data = parse_table(block, block_map)
            tables.append(table_data)
    
    return tables

def parse_table(table_block: dict, block_map: dict) -> dict:
    """Parse a table block into rows and columns."""
    rows = {}
    
    if "Relationships" not in table_block:
        return {"rows": []}
    
    for relationship in table_block["Relationships"]:
        if relationship["Type"] == "CHILD":
            for child_id in relationship["Ids"]:
                cell_block = block_map.get(child_id)
                if cell_block and cell_block["BlockType"] == "CELL":
                    row_index = cell_block.get("RowIndex", 0)
                    col_index = cell_block.get("ColumnIndex", 0)
                    
                    if row_index not in rows:
                        rows[row_index] = {}
                    
                    rows[row_index][col_index] = get_cell_text(cell_block, block_map)
    
    # Convert to list of lists
    table_rows = []
    for row_index in sorted(rows.keys()):
        row_data = [rows[row_index].get(col, "") for col in sorted(rows[row_index].keys())]
        table_rows.append(row_data)
    
    return {"rows": table_rows}

def get_cell_text(cell_block: dict, block_map: dict) -> str:
    """Get text content of a cell."""
    text_parts = []
    
    if "Relationships" in cell_block:
        for relationship in cell_block["Relationships"]:
            if relationship["Type"] == "CHILD":
                for child_id in relationship["Ids"]:
                    word_block = block_map.get(child_id)
                    if word_block and word_block["BlockType"] == "WORD":
                        text_parts.append(word_block["Text"])
    
    return " ".join(text_parts)
```

---

## Extraction Heuristics

### Salary Pattern Detection

**Goal**: Extract salary amounts from various formats.

#### Common Patterns

| Format | Regex | Example Match |
|--------|-------|---------------|
| Indian Rupees (LPA) | `₹?\s*(\d+(?:\.\d+)?)\s*LPA` | `₹12 LPA` → 12 |
| Indian Rupees (Lakhs) | `₹?\s*(\d+(?:\.\d+)?)\s*lakhs?` | `15 lakhs` → 15 |
| Indian Rupees (absolute) | `₹\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)` | `₹1,200,000` → 1200000 |
| USD (K notation) | `\$\s*(\d+)k` | `$120k` → 120000 |
| USD (absolute) | `\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)` | `$120,000` → 120000 |
| CTC | `CTC:?\s*₹?\s*(\d+(?:\.\d+)?)\s*(?:LPA\|lakhs)?` | `CTC: 12 LPA` → 12 |
| Per month | `₹?\s*(\d+(?:,\d{3})*)\s*(?:per month\|monthly)` | `₹100,000 per month` → 100000 |

#### Extraction Function

```python
import re

def extract_salary_patterns(text: str) -> list[dict]:
    """Extract all salary mentions from text."""
    patterns = [
        (r'₹?\s*(\d+(?:\.\d+)?)\s*LPA', 'INR', 'yearly', lambda x: float(x) * 100000),
        (r'₹?\s*(\d+(?:\.\d+)?)\s*lakhs?', 'INR', 'yearly', lambda x: float(x) * 100000),
        (r'₹\s*(\d{1,3}(?:,\d{3})*)', 'INR', 'unknown', lambda x: float(x.replace(',', ''))),
        (r'\$\s*(\d+)k', 'USD', 'yearly', lambda x: float(x) * 1000),
        (r'\$\s*(\d{1,3}(?:,\d{3})*)', 'USD', 'unknown', lambda x: float(x.replace(',', ''))),
    ]
    
    found_salaries = []
    
    for pattern, currency, period, converter in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            amount = converter(match.group(1))
            found_salaries.append({
                "amount": amount,
                "currency": currency,
                "period": period,
                "matched_text": match.group(0),
                "confidence": 0.8
            })
    
    return found_salaries
```

### Company Name Extraction

**Heuristics**:
1. Look for "Company:", "Employer:", "Organization:" labels
2. Extract text from first few lines (often header/letterhead)
3. Use NER (Named Entity Recognition) if available

```python
def extract_company_name(text: str, tables: list) -> str | None:
    """Extract company name from text."""
    lines = text.split('\n')
    
    # Pattern 1: Explicit label
    for line in lines[:10]:  # Check first 10 lines
        match = re.search(r'(?:Company|Employer|Organization):\s*(.+)', line, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Pattern 2: First non-empty line (often company name)
    for line in lines[:3]:
        if line.strip() and len(line.strip()) > 3:
            return line.strip()
    
    return None
```

### Job Title Extraction

```python
def extract_job_title(text: str) -> str | None:
    """Extract job title from offer letter."""
    lines = text.split('\n')
    
    # Pattern 1: Explicit label
    for line in lines:
        match = re.search(r'(?:Position|Role|Job Title|Designation):\s*(.+)', line, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Pattern 2: Common titles in text
    common_titles = [
        "Software Engineer", "Senior Software Engineer", "Staff Engineer",
        "Product Manager", "Data Scientist", "Designer", "Analyst"
    ]
    
    for title in common_titles:
        if title.lower() in text.lower():
            return title
    
    return None
```

### Benefits Extraction

```python
def extract_benefits(text: str, tables: list) -> list[str]:
    """Extract benefits from text and tables."""
    benefits = []
    
    # Common benefit keywords
    benefit_keywords = [
        "health insurance", "medical insurance", "dental",
        "401k", "PF", "provident fund", "ESOP", "equity",
        "bonus", "signing bonus", "relocation", "remote work",
        "paid time off", "PTO", "vacation", "parental leave"
    ]
    
    for keyword in benefit_keywords:
        if keyword.lower() in text.lower():
            benefits.append(keyword.title())
    
    # Check tables for benefit rows
    for table in tables:
        for row in table["rows"]:
            row_text = " ".join(row).lower()
            for keyword in benefit_keywords:
                if keyword in row_text and keyword.title() not in benefits:
                    benefits.append(keyword.title())
    
    return list(set(benefits))  # Deduplicate
```

---

## Post-Processing with Bedrock

**Problem**: Regex heuristics can be brittle and miss edge cases.

**Solution**: Use Bedrock LLM to normalize extracted data into structured JSON.

### Normalization Prompt

```python
def build_normalization_prompt(raw_text: str, extracted_salaries: list, extracted_company: str, extracted_title: str) -> str:
    """Build prompt for Bedrock normalization."""
    
    prompt = f"""You are a data extraction assistant. Given OCR output from a job offer letter or related document, extract and normalize the following fields:

1. **Offered Salary**: Annual salary amount, currency, and period (yearly/monthly)
2. **Company**: Company or employer name
3. **Job Title**: Position or role title
4. **Benefits**: List of benefits mentioned

**OCR Text:**
{raw_text[:2000]}  # Limit to first 2000 chars

**Preliminary Extractions:**
- Salary mentions: {extracted_salaries}
- Company: {extracted_company}
- Job title: {extracted_title}

**Instructions:**
- Choose the most likely annual salary (prefer base salary over total CTC if both mentioned)
- Normalize currency to ISO codes (INR, USD, EUR, GBP)
- Convert all salaries to yearly amounts
- If salary is in LPA (Lakhs Per Annum), multiply by 100,000 for INR
- Output JSON only, no explanation

**Output Format:**
{{
  "offered_salary": {{
    "amount": 1200000,
    "currency": "INR",
    "period": "yearly"
  }},
  "company": "ABC Corp",
  "job_title": "Software Engineer",
  "benefits": ["Health Insurance", "ESOP", "Relocation Bonus"]
}}

If a field cannot be determined, set it to null.
"""
    
    return prompt
```

### Bedrock Normalization Call

```python
from app.utils.bedrock_client import BedrockClient

async def normalize_with_bedrock(raw_text: str, heuristic_data: dict) -> dict:
    """Use Bedrock to normalize extracted data."""
    bedrock_client = BedrockClient()
    
    prompt = build_normalization_prompt(
        raw_text=raw_text,
        extracted_salaries=heuristic_data.get("salaries", []),
        extracted_company=heuristic_data.get("company"),
        extracted_title=heuristic_data.get("title")
    )
    
    response = await bedrock_client.generate_text(
        prompt=prompt,
        max_tokens=500,
        temperature=0.1  # Low temperature for factual extraction
    )
    
    # Parse JSON from response
    try:
        normalized_data = json.loads(response)
        return normalized_data
    except json.JSONDecodeError:
        logger.error(f"Failed to parse Bedrock response: {response}")
        # Fallback to heuristics
        return fallback_normalization(heuristic_data)
```

---

## Normalization Rules

### Salary Normalization

**Rule**: All salaries stored as `{amount: float, period: "yearly", currency: ISO_CODE}`

| Input | Normalized Output |
|-------|-------------------|
| `₹12 LPA` | `{"amount": 1200000, "period": "yearly", "currency": "INR"}` |
| `15 lakhs per year` | `{"amount": 1500000, "period": "yearly", "currency": "INR"}` |
| `₹100,000 per month` | `{"amount": 1200000, "period": "yearly", "currency": "INR"}` (converted) |
| `$120k` | `{"amount": 120000, "period": "yearly", "currency": "USD"}` |
| `$10,000/month` | `{"amount": 120000, "period": "yearly", "currency": "USD"}` |

### Currency Detection

**Priority Order**:
1. Explicit symbol (₹ → INR, $ → USD, € → EUR, £ → GBP)
2. Context keywords (LPA/lakhs → INR)
3. Default to USD if ambiguous

### Period Detection

**Heuristics**:
- Keywords: `per annum`, `yearly`, `annual` → yearly
- Keywords: `per month`, `monthly`, `pm` → monthly
- If amount > 100,000 and currency = INR → likely yearly
- If amount < 50,000 and currency = USD → likely monthly
- Default: yearly

**Conversion**: Monthly → yearly by multiplying by 12

---

## User Confirmation Flow

### Confirmation Message

After OCR and normalization, the Collector Agent reads back a summary:

```python
async def send_confirmation_message(session_id: str, parsed_data: dict):
    """Send confirmation to user."""
    salary = parsed_data.get("offered_salary")
    company = parsed_data.get("company")
    title = parsed_data.get("job_title")
    
    if salary:
        formatted_salary = format_salary_for_voice(salary)
        message = f"I found an offer of {formatted_salary}"
        
        if company:
            message += f" from {company}"
        
        if title:
            message += f" for the {title} position"
        
        message += ". Is that correct?"
    else:
        message = "I couldn't extract salary details from the document. Could you tell me the offered amount?"
    
    await send_agent_message(session_id, "collector", message)
    
    # Show editable fields in frontend
    await send_state_update(
        session_id,
        "confirm_context",
        data=parsed_data
    )

def format_salary_for_voice(salary: dict) -> str:
    """Format salary for TTS readability."""
    amount = salary["amount"]
    currency = salary["currency"]
    
    if currency == "INR":
        # Convert to lakhs for readability
        lakhs = amount / 100000
        if lakhs == int(lakhs):
            return f"₹{int(lakhs)} lakhs per year"
        else:
            return f"₹{lakhs:.1f} lakhs per year"
    
    elif currency == "USD":
        if amount >= 1000:
            k_amount = amount / 1000
            return f"${int(k_amount)}k per year"
        else:
            return f"${amount} per year"
    
    return f"{currency} {amount:,.0f} per year"
```

### User Response Handling

```python
async def handle_confirmation_response(session_id: str, user_message: str, parsed_data: dict):
    """Handle user's confirmation or correction."""
    message_lower = user_message.lower()
    
    if any(word in message_lower for word in ["yes", "correct", "right", "yep", "yeah"]):
        # User confirms
        session = await session_store.get(session_id)
        
        # Update session with parsed data
        if parsed_data.get("offered_salary"):
            session.offered_salary = SalaryValue(**parsed_data["offered_salary"])
        if parsed_data.get("company"):
            session.company = parsed_data["company"]
        if parsed_data.get("job_title"):
            session.job_title = parsed_data["job_title"]
        
        await session_store.update(session)
        
        # Transition to context_ready or continue collecting
        await transition_state(session, "context_ready")
    
    elif any(word in message_lower for word in ["no", "wrong", "incorrect", "not quite"]):
        # User rejects, ask for correction
        await send_agent_message(
            session_id,
            "collector",
            "No problem. What should I correct?"
        )
        # Stay in confirm_context state, wait for correction
    
    else:
        # User provides corrected value
        # Parse correction and update
        corrected_salary = extract_salary_from_message(user_message)
        if corrected_salary:
            parsed_data["offered_salary"] = corrected_salary
            await send_confirmation_message(session_id, parsed_data)
```

---

## Error Handling

### Textract Job Failures

**Possible Causes**:
- Document too complex
- Unsupported format
- AWS service error

**Recovery**:

```python
async def handle_textract_failure(session_id: str, file_id: str, error_reason: str):
    """Handle OCR failure."""
    # Update file status
    session = await session_store.get(session_id)
    file_record = next(f for f in session.files if f.file_id == file_id)
    file_record.ocr_status = "failed"
    await session_store.update(session)
    
    # Notify user
    await send_error_message(
        session_id,
        "OCR_FAILED",
        "I couldn't read that document. Would you like to re-upload or tell me the details manually?"
    )
    
    # Offer alternatives
    await send_agent_message(
        session_id,
        "collector",
        "Let's continue anyway. What salary did the offer letter mention?"
    )
    
    # Log for debugging
    logger.error(
        "textract_failed",
        session_id=session_id,
        file_id=file_id,
        reason=error_reason
    )
```

### Low-Confidence Extraction

If Bedrock or heuristics produce low-confidence results:

```python
def check_extraction_confidence(parsed_data: dict) -> str:
    """Assess confidence of extraction."""
    confidence_score = 0
    
    if parsed_data.get("offered_salary"):
        salary = parsed_data["offered_salary"]
        if salary["amount"] > 0 and salary["currency"] in ["INR", "USD", "EUR", "GBP"]:
            confidence_score += 3
    
    if parsed_data.get("company") and len(parsed_data["company"]) > 2:
        confidence_score += 2
    
    if parsed_data.get("job_title") and len(parsed_data["job_title"]) > 3:
        confidence_score += 1
    
    # Scale: 0-2 = low, 3-4 = medium, 5-6 = high
    if confidence_score <= 2:
        return "low"
    elif confidence_score <= 4:
        return "medium"
    else:
        return "high"

async def handle_low_confidence_extraction(session_id: str, parsed_data: dict):
    """Handle low-confidence extraction."""
    await send_agent_message(
        session_id,
        "collector",
        "I found some information, but I'm not very confident. Let me read it back and you can correct me."
    )
    
    # Read back with emphasis on uncertainty
    await send_confirmation_message(session_id, parsed_data)
```

### Ambiguous Data

If multiple salary values found:

```python
async def handle_multiple_salaries(session_id: str, salaries: list[dict]):
    """Handle multiple salary mentions."""
    if len(salaries) == 1:
        return salaries[0]
    
    # Heuristic: prefer yearly over monthly
    yearly_salaries = [s for s in salaries if s["period"] == "yearly"]
    if yearly_salaries:
        return max(yearly_salaries, key=lambda s: s["amount"])
    
    # Ask user to clarify
    await send_agent_message(
        session_id,
        "collector",
        f"I found {len(salaries)} salary values in the document. Which one is the base salary offer?"
    )
    
    # Present options in frontend
    return None
```

---

## S3 Upload and File Management

### Pre-Signed URL Generation

```python
import boto3
from datetime import timedelta

s3_client = boto3.client("s3")

def generate_presigned_upload_url(file_id: str, file_type: str, expires_in: int = 300) -> dict:
    """Generate pre-signed URL for file upload."""
    key = f"uploads/{file_id}.pdf"
    
    presigned_url = s3_client.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": settings.s3_bucket_name,
            "Key": key,
            "ContentType": "application/pdf"
        },
        ExpiresIn=expires_in
    )
    
    return {
        "upload_url": presigned_url,
        "s3_url": f"s3://{settings.s3_bucket_name}/{key}",
        "expires_in": expires_in
    }
```

### File Cleanup

```python
async def cleanup_session_files(session_id: str):
    """Delete all files associated with a session."""
    session = await session_store.get(session_id)
    
    for file_record in session.files:
        # Parse S3 URL
        bucket, key = parse_s3_url(file_record.s3_url)
        
        # Delete from S3
        s3_client.delete_object(Bucket=bucket, Key=key)
        logger.info(f"Deleted file {key} from S3")
    
    # Remove from session
    session.files = []
    await session_store.update(session)
```

---

## Testing Strategy

### Unit Tests

```python
def test_salary_extraction():
    text = "Your annual CTC is ₹12 LPA."
    salaries = extract_salary_patterns(text)
    
    assert len(salaries) == 1
    assert salaries[0]["amount"] == 1200000
    assert salaries[0]["currency"] == "INR"

def test_company_extraction():
    text = "Company: Acme Corp\nOffer Letter"
    company = extract_company_name(text, [])
    
    assert company == "Acme Corp"
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_textract_flow():
    # Upload test PDF
    s3_key = "test/sample_offer.pdf"
    upload_test_file(s3_key)
    
    # Start Textract
    job_id = start_textract_job(settings.s3_bucket_name, s3_key, "offer_letter")
    
    # Wait for completion
    status = await wait_for_textract_job(job_id, max_wait_seconds=30)
    assert status == "SUCCEEDED"
    
    # Retrieve and parse
    results = get_textract_results(job_id)
    text = extract_text_from_blocks(results["blocks"])
    
    assert "salary" in text.lower() or "compensation" in text.lower()
```

---

## Performance Considerations

1. **Parallel Processing**: If multiple files uploaded, start Textract jobs in parallel
2. **Polling Interval**: Poll every 2 seconds (Textract jobs typically complete in 5-30s)
3. **Timeout**: Set max wait time of 60 seconds; if exceeded, continue with manual entry
4. **Caching**: Cache normalization results for identical raw texts

---

## Observability

### Metrics

- `ocr_job_started`: Count of Textract jobs started
- `ocr_job_duration_seconds`: Time from start to completion
- `ocr_success_rate`: Percentage of successful extractions
- `ocr_confidence_distribution`: Distribution of confidence levels (high/medium/low)
- `user_confirmation_rate`: Percentage of users who confirm vs. correct

### Logging

```python
logger.info(
    "ocr_completed",
    session_id=session_id,
    file_id=file_id,
    job_id=job_id,
    duration_seconds=duration,
    extracted_fields=list(parsed_data.keys()),
    confidence=confidence
)
```

---

## Security

1. **S3 Bucket**: Private, no public access
2. **Pre-signed URLs**: Short expiration (5 minutes)
3. **File Validation**: Enforce PDF type, max size, max pages
4. **Data Retention**: Delete files after session completion or user request
5. **Encryption**: Enable SSE-S3 or SSE-KMS for S3 objects

---

**End of OCR Pipeline Specification**

