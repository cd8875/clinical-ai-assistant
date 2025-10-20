import re
from typing import List, Dict
from ..models import ClinicalEntity

class MedicalEntityExtractor:
    
    def __init__(self):
        print("✅ Using regex-based medical entity extraction")
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict:
        return {
            'MEDICATION': [
                r'\b\w+mycin\b',
                r'\b\w+cillin\b',
                r'\b\w+prazole\b',
                r'\b\w+statin\b',
                r'\b\w+olol\b',
                r'\b\w+pine\b',
                r'\baspirin\b',
                r'\bmetformin\b',
                r'\blisinopril\b',
                r'\bibuprofen\b',
                r'\binsulin\b',
            ],
            'LAB_VALUE': [
                r'\b(?:HbA1c|A1C)[:\s]*(\d+\.?\d*)\s*%',
                r'\b(?:BP|Blood Pressure)[:\s]*(\d+)/(\d+)',
                r'\b(?:glucose|sugar)[:\s]*(\d+)\s*mg/dL',
                r'\b(?:creatinine)[:\s]*(\d+\.?\d*)\s*mg/dL',
                r'\b(?:hemoglobin|Hb)[:\s]*(\d+\.?\d*)\s*g/dL',
                r'\b(?:WBC|white blood cell)[:\s]*(\d+\.?\d*)',
                r'\b(?:platelet)[:\s]*(\d+\.?\d*)',
            ],
            'VITAL_SIGNS': [
                r'\b(?:temperature|temp)[:\s]*(\d+\.?\d*)\s*[°]?[FC]',
                r'\b(?:heart rate|HR|pulse)[:\s]*(\d+)\s*bpm',
                r'\b(?:respiratory rate|RR)[:\s]*(\d+)',
                r'\b(?:SpO2|oxygen saturation)[:\s]*(\d+)\s*%',
            ]
        }
    
    def extract_entities(self, text: str) -> List[ClinicalEntity]:
        entities = []
        
        for label, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append(ClinicalEntity(
                        text=match.group(0),
                        label=label,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.85
                    ))
        
        # Remove duplicates
        entities = self._deduplicate_entities(entities)
        return entities
    
    def _deduplicate_entities(self, entities: List[ClinicalEntity]) -> List[ClinicalEntity]:
        if not entities:
            return []
        
        sorted_entities = sorted(entities, key=lambda e: (e.start, -e.confidence))
        
        deduplicated = []
        for entity in sorted_entities:
            overlap = False
            for existing in deduplicated:
                if (entity.start < existing.end and entity.end > existing.start):
                    overlap = True
                    break
            
            if not overlap:
                deduplicated.append(entity)
        
        return deduplicated
    
    def extract_structured_data(self, text: str) -> Dict:
        entities = self.extract_entities(text)
        
        structured = {
            'medications': [],
            'lab_values': [],
            'vital_signs': [],
            'diagnoses': [],
            'procedures': [],
            'symptoms': []
        }
        
        for entity in entities:
            category = entity.label.lower()
            if category == 'medication':
                structured['medications'].append({
                    'text': entity.text,
                    'confidence': entity.confidence
                })
            elif category == 'lab_value':
                structured['lab_values'].append({
                    'text': entity.text,
                    'confidence': entity.confidence
                })
            elif category == 'vital_signs':
                structured['vital_signs'].append({
                    'text': entity.text,
                    'confidence': entity.confidence
                })
        
        return structured
    