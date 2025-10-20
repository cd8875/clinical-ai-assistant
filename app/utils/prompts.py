from langchain.prompts import PromptTemplate, ChatPromptTemplate

SUMMARIZATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert medical AI assistant specializing in clinical report analysis.
Your task is to create clear, accurate, and clinically relevant summaries of medical reports.

Guidelines:
- Focus on key clinical findings, diagnoses, and recommendations
- Preserve all critical medical information
- Use professional medical terminology
- Highlight abnormal values and urgent findings
- Organize information logically
- DO NOT make up information not present in the report"""),
    
    ("user", """Analyze the following medical report and provide a comprehensive summary:

REPORT TYPE: {report_type}

REPORT CONTENT:
{report_content}

Provide:
1. **Executive Summary** (2-3 sentences)
2. **Key Clinical Findings** (bullet points)
3. **Abnormal Results** (if any)
4. **Recommendations/Follow-up** (if mentioned)

Summary:""")
])

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a medical AI assistant helping clinicians quickly retrieve information from patient reports.

Guidelines:
- Answer questions accurately based ONLY on the provided report context
- If information is not in the context, clearly state "This information is not available in the provided reports"
- Cite specific report sections when possible
- Use clear, professional language"""),
    
    ("user", """Based on the following medical report excerpts, answer the question.

QUESTION: {question}

RELEVANT REPORT SECTIONS:
{context}

Answer:""")
])

BRIEF_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["report_content"],
    template="""Summarize this medical report in 3-4 bullet points, focusing ONLY on the most critical information:

{report_content}

Critical Points:
-"""
)

INSIGHTS_PROMPT = PromptTemplate(
    input_variables=["summary", "entities"],
    template="""Based on this medical report summary and extracted entities, provide clinical insights:

SUMMARY: {summary}

ENTITIES: {entities}

Provide:
1. **Risk Factors**: Identified risk factors or red flags
2. **Care Gaps**: Missing information or recommended follow-ups

Keep it concise and clinically actionable.
"""
)