from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum

class ReportType(str, Enum):
    LAB = "lab"
    RADIOLOGY = "radiology"
    PATHOLOGY = "pathology"
    DISCHARGE = "discharge"
    GENERAL = "general"

class ClinicalEntity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0

class MedicalReport(BaseModel):
    report_id: str = Field(..., description="Unique report identifier")
    patient_id: Optional[str] = None
    report_type: ReportType = ReportType.GENERAL
    content: str = Field(..., min_length=1)
    upload_date: datetime = Field(default_factory=datetime.now)
    metadata: Dict = Field(default_factory=dict)

class SummaryRequest(BaseModel):
    report_id: str
    summary_type: str = "comprehensive"
    focus_areas: Optional[List[str]] = None

class SummaryResponse(BaseModel):
    report_id: str
    summary: str
    key_findings: List[str]
    extracted_entities: List[ClinicalEntity]
    confidence_score: float
    processing_time: float

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3)
    report_ids: Optional[List[str]] = None
    context_window: int = 3

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict]
    confidence: float
    related_entities: List[ClinicalEntity]

class UploadResponse(BaseModel):
    report_id: str
    filename: str
    status: str
    message: str
    extracted_text_length: int