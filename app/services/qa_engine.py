from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Optional
import os
from .vector_store import VectorStore
from ..utils.prompts import QA_PROMPT

class QAEngine:
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2,
            convert_system_message_to_human=True
        )
    
    def answer_question(
        self,
        question: str,
        report_ids: Optional[List[str]] = None,
        k: int = 5
    ) -> Dict:
        
        if report_ids:
            relevant_docs = []
            for report_id in report_ids:
                docs = self.vector_store.search_by_report_ids(question, [report_id], k=k)
                relevant_docs.extend(docs)
        else:
            relevant_docs = self.vector_store.search(question, k=k)
        
        if not relevant_docs:
            return {
                "answer": "I couldn't find relevant information in the reports to answer this question.",
                "sources": [],
                "confidence": 0.0
            }
        
        context = self._prepare_context(relevant_docs)
        
        prompt = QA_PROMPT.format_messages(
            question=question,
            context=context
        )
        
        response = self.llm.invoke(prompt)
        answer = response.content
        
        avg_similarity = sum(doc['similarity_score'] for doc in relevant_docs) / len(relevant_docs)
        
        return {
            "answer": answer,
            "sources": self._format_sources(relevant_docs),
            "confidence": round(avg_similarity, 2),
            "num_sources": len(relevant_docs)
        }
    
    def suggest_questions(self, report_id: str, num_suggestions: int = 5) -> List[str]:
        chunks = self.vector_store.get_report_chunks(report_id)
        
        if not chunks:
            return []
        
        sample_content = " ".join(chunks[:3])[:2000]
        
        prompt = f"""Based on this medical report excerpt, suggest {num_suggestions} important questions a clinician might ask:

Report excerpt:
{sample_content}

Generate specific, clinically relevant questions. Format as a numbered list."""
        
        response = self.llm.invoke(prompt)
        
        questions = []
        for line in response.content.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                question = line.lstrip('0123456789.-) ')
                if len(question) > 10:
                    questions.append(question)
        
        return questions[:num_suggestions]
    
    def _prepare_context(self, documents: List[Dict]) -> str:
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            report_id = doc['metadata'].get('report_id', 'Unknown')
            similarity = doc['similarity_score']
            content = doc['content']
            
            context_parts.append(
                f"[Source {i} - Report ID: {report_id} | Relevance: {similarity:.2f}]\n{content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _format_sources(self, documents: List[Dict]) -> List[Dict]:
        sources = []
        for doc in documents:
            sources.append({
                "report_id": doc['metadata'].get('report_id'),
                "chunk_id": doc['metadata'].get('chunk_id'),
                "similarity": doc['similarity_score'],
                "excerpt": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
            })
        return sources