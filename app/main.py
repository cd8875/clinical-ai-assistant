from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
import time
from dotenv import load_dotenv

from .models import (
    SummaryRequest, SummaryResponse, QuestionRequest, 
    QuestionResponse, UploadResponse, MedicalReport
)
from .services.summarizer import MedicalSummarizer
from .services.qa_engine import QAEngine
from .services.entity_extractor import MedicalEntityExtractor
from .services.vector_store import VectorStore
from .utils.pdf_parser import DocumentParser

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Clinical Report AI Assistant",
    description="AI-powered medical report summarization and conversational Q&A",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services (ALL FREE)
vector_store = VectorStore()
summarizer = MedicalSummarizer()  # Uses FREE Gemini
qa_engine = QAEngine(vector_store)  # Uses FREE Gemini
entity_extractor = MedicalEntityExtractor()  # FREE local model

# In-memory storage for reports (use database in production)
reports_db = {}

# Constants
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/reports")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Clinical Report AI Assistant",
        "version": "1.0.0"
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_report(file: UploadFile = File(...)):
    """Upload and process a medical report"""
    try:
        # Validate file type
        allowed_extensions = ['.pdf', '.docx', '.txt']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(400, f"Unsupported file type. Allowed: {allowed_extensions}")
        
        # Generate unique report ID
        report_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, f"{report_id}{file_ext}")
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Parse document
        parser = DocumentParser()
        parsed = parser.parse_document(file_path, file_ext[1:])  # Remove dot from extension
        
        text_content = parsed['text']
        metadata = parsed.get('metadata', {})
        metadata['filename'] = file.filename
        metadata['upload_time'] = time.time()
        
        # Store in vector database
        vector_store.add_report(report_id, text_content, metadata)
        
        # Store report metadata
        reports_db[report_id] = {
            'id': report_id,
            'filename': file.filename,
            'content': text_content,
            'metadata': metadata,
            'file_path': file_path
        }
        
        return UploadResponse(
            report_id=report_id,
            filename=file.filename,
            status="success",
            message="Report uploaded and indexed successfully",
            extracted_text_length=len(text_content)
        )
    
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_report(request: SummaryRequest):
    """Generate AI summary of a medical report"""
    try:
        # Get report from database
        if request.report_id not in reports_db:
            raise HTTPException(404, "Report not found")
        
        report = reports_db[request.report_id]
        
        start_time = time.time()
        
        # Generate summary
        summary_result = summarizer.summarize(
            report['content'],
            report.get('type', 'general'),
            request.summary_type
        )
        
        # Extract entities
        entities = entity_extractor.extract_entities(report['content'])
        
        processing_time = time.time() - start_time
        
        return SummaryResponse(
            report_id=request.report_id,
            summary=summary_result['summary'],
            key_findings=summary_result['key_findings'],
            extracted_entities=entities[:20],  # Top 20 entities
            confidence_score=summary_result['confidence_score'],
            processing_time=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Summarization failed: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about medical reports using RAG"""
    try:
        # Answer question
        result = qa_engine.answer_question(
            request.question,
            request.report_ids,
            request.context_window
        )
        
        # Extract entities from the question
        question_entities = entity_extractor.extract_entities(request.question)
        
        return QuestionResponse(
            question=request.question,
            answer=result['answer'],
            sources=result['sources'],
            confidence=result['confidence'],
            related_entities=question_entities
        )
    
    except Exception as e:
        raise HTTPException(500, f"Q&A failed: {str(e)}")

@app.get("/reports")
async def list_reports():
    """List all uploaded reports"""
    reports = []
    for report_id, report in reports_db.items():
        reports.append({
            'report_id': report_id,
            'filename': report['filename'],
            'upload_time': report['metadata'].get('upload_time'),
            'text_length': len(report['content'])
        })
    return {"reports": reports, "total": len(reports)}

@app.get("/reports/{report_id}")
async def get_report(report_id: str):
    """Get full report details"""
    if report_id not in reports_db:
        raise HTTPException(404, "Report not found")
    
    report = reports_db[report_id]
    return {
        'report_id': report_id,
        'filename': report['filename'],
        'content_preview': report['content'][:500] + "...",
        'full_length': len(report['content']),
        'metadata': report['metadata']
    }

@app.delete("/reports/{report_id}")
async def delete_report(report_id: str):
    """Delete a report"""
    if report_id not in reports_db:
        raise HTTPException(404, "Report not found")
    
    # Remove from database
    report = reports_db.pop(report_id)
    
    # Delete file
    if os.path.exists(report['file_path']):
        os.remove(report['file_path'])
    
    # Remove from vector store
    vector_store.delete_report(report_id)
    
    return {"status": "deleted", "report_id": report_id}

@app.get("/suggest-questions/{report_id}")
async def suggest_questions(report_id: str, num_questions: int = 5):
    """Generate suggested questions for a report"""
    try:
        questions = qa_engine.suggest_questions(report_id, num_questions)
        return {"report_id": report_id, "suggested_questions": questions}
    except Exception as e:
        raise HTTPException(500, f"Failed to generate questions: {str(e)}")

@app.get("/entities/{report_id}")
async def extract_entities(report_id: str):
    """Extract medical entities from a report"""
    if report_id not in reports_db:
        raise HTTPException(404, "Report not found")
    
    report = reports_db[report_id]
    entities = entity_extractor.extract_entities(report['content'])
    structured = entity_extractor.extract_structured_data(report['content'])
    
    return {
        "report_id": report_id,
        "entities": [e.dict() for e in entities],
        "structured_data": structured,
        "total_entities": len(entities)
    }

@app.get("/stats")
async def get_statistics():
    
    vector_stats = vector_store.get_stats()
    
    return {
        "total_reports": len(reports_db),
        "vector_store": vector_stats,
        "llm_provider": os.getenv("DEFAULT_LLM", "openai")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)