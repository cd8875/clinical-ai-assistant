from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from typing import Dict, List
import time
import os
from ..utils.prompts import SUMMARIZATION_PROMPT, BRIEF_SUMMARY_PROMPT, INSIGHTS_PROMPT
from ..models import ClinicalEntity

class MedicalSummarizer:
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=float(os.getenv("TEMPERATURE", 0.3)),
            convert_system_message_to_human=True
        )
        
        self.summary_chain = LLMChain(
            llm=self.llm,
            prompt=SUMMARIZATION_PROMPT
        )
        
        self.brief_chain = LLMChain(
            llm=self.llm,
            prompt=BRIEF_SUMMARY_PROMPT
        )
    
    def summarize(
        self,
        report_content: str,
        report_type: str = "general",
        summary_type: str = "comprehensive"
    ) -> Dict:
        start_time = time.time()
        
        try:
            if summary_type == "brief":
                result = self.brief_chain.invoke({"report_content": report_content})
                summary = result['text']
                key_findings = self._extract_bullet_points(summary)
            else:
                result = self.summary_chain.invoke({
                    "report_type": report_type,
                    "report_content": report_content
                })
                summary = result['text']
                key_findings = self._extract_key_findings(summary)
            
            processing_time = time.time() - start_time
            
            return {
                "summary": summary,
                "key_findings": key_findings,
                "processing_time": processing_time,
                "confidence_score": self._calculate_confidence(summary, report_content)
            }
        
        except Exception as e:
            raise Exception(f"Summarization failed: {str(e)}")
    
    def _extract_key_findings(self, summary: str) -> List[str]:
        findings = []
        
        lines = summary.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.')):
                finding = line.lstrip('•-*123456789. ')
                if len(finding) > 10:
                    findings.append(finding)
        
        if not findings:
            import re
            medical_keywords = [
                'diagnosis', 'diagnosed', 'abnormal', 'elevated', 'decreased',
                'positive', 'negative', 'findings', 'showed', 'revealed'
            ]
            
            sentences = re.split(r'[.!?]+', summary)
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in medical_keywords):
                    findings.append(sentence.strip())
        
        return findings[:5]
    
    def _extract_bullet_points(self, text: str) -> List[str]:
        points = []
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith(('•', '-', '*')):
                points.append(line.lstrip('•-* '))
        return points
    
    def _calculate_confidence(self, summary: str, original: str) -> float:
        summary_lower = summary.lower()
        original_lower = original.lower()
        
        medical_terms = set()
        import re
        words = re.findall(r'\b[A-Z][a-z]+\b', original)
        medical_terms.update(words[:20])
        
        if len(medical_terms) == 0:
            return 0.5
        
        matched = sum(1 for term in medical_terms if term.lower() in summary_lower)
        confidence = min(matched / len(medical_terms), 1.0)
        return round(confidence, 2)