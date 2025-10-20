import streamlit as st
import requests
from datetime import datetime

st.set_page_config(
    page_title="Clinical AI Assistant",
    page_icon="🏥",
    layout="wide"
)

st.markdown('<h1 style="text-align: center;">🏥 Clinical AI Assistant</h1>', unsafe_allow_html=True)
st.markdown("**Powered by Google Gemini & LangChain**")

API_URL = "http://localhost:8000"

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_report_id' not in st.session_state:
    st.session_state.current_report_id = None

with st.sidebar:
    st.header("📊 System Status")
    
    try:
        health = requests.get(f"{API_URL}/").json()
        st.success("✅ API Online")
        
        stats = requests.get(f"{API_URL}/stats").json()
        st.metric("Total Reports", stats['total_reports'])
    except:
        st.error("❌ API Offline")
    
    st.divider()
    
    st.header("📁 Uploaded Reports")
    try:
        reports_response = requests.get(f"{API_URL}/reports")
        reports = reports_response.json().get('reports', [])
        
        if reports:
            for report in reports:
                with st.expander(f"📄 {report['filename'][:30]}"):
                    st.text(f"ID: {report['report_id'][:12]}...")
                    st.text(f"Size: {report['text_length']} chars")
        else:
            st.info("No reports uploaded yet")
    except:
        st.error("Could not load reports")

tab1, tab2, tab3 = st.tabs(["📤 Upload & Summarize", "💬 Ask Questions", "🔬 Entity Extraction"])

with tab1:
    st.header("Upload Medical Report")
    
    uploaded_file = st.file_uploader(
        "Choose a medical report (PDF, TXT, DOCX)",
        type=['pdf', 'txt', 'docx']
    )
    
    summary_type = st.selectbox("Summary Type", ["comprehensive", "brief"])
    
    if uploaded_file:
        if st.button("🚀 Process Report", type="primary"):
            with st.spinner("⏳ Processing..."):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    upload_response = requests.post(f"{API_URL}/upload", files=files)
                    
                    if upload_response.status_code == 200:
                        upload_data = upload_response.json()
                        report_id = upload_data['report_id']
                        st.session_state.current_report_id = report_id
                        
                        st.success(f"✅ Uploaded! Report ID: `{report_id[:16]}...`")
                        
                        with st.spinner("🤖 Generating AI summary..."):
                            summary_response = requests.post(
                                f"{API_URL}/summarize",
                                json={'report_id': report_id, 'summary_type': summary_type}
                            )
                            
                            if summary_response.status_code == 200:
                                summary_data = summary_response.json()
                                
                                st.subheader("📋 AI-Generated Summary")
                                st.markdown(summary_data['summary'])
                                
                                st.subheader("🔍 Key Findings")
                                for i, finding in enumerate(summary_data['key_findings'], 1):
                                    st.markdown(f"{i}. {finding}")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("⏱️ Time", f"{summary_data['processing_time']:.2f}s")
                                with col2:
                                    st.metric("✅ Confidence", f"{summary_data['confidence_score']*100:.0f}%")
                                with col3:
                                    st.metric("🏷️ Entities", len(summary_data['extracted_entities']))
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with tab2:
    st.header("💬 Ask Questions")
    
    question = st.text_input("Ask a question:", placeholder="e.g., What medications were prescribed?")
    
    if st.button("🔍 Ask", type="primary", disabled=not question):
        with st.spinner("🤔 Thinking..."):
            try:
                qa_response = requests.post(f"{API_URL}/ask", json={'question': question})
                
                if qa_response.status_code == 200:
                    qa_data = qa_response.json()
                    
                    st.session_state.chat_history.append({
                        'question': question,
                        'answer': qa_data['answer'],
                        'confidence': qa_data['confidence'],
                        'time': datetime.now().strftime("%H:%M:%S")
                    })
                    
                    st.rerun()
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    if st.session_state.chat_history:
        st.subheader("💬 Conversation History")
        
        for chat in reversed(st.session_state.chat_history[-5:]):
            st.markdown(f"**Q:** {chat['question']}")
            st.markdown(f"**A:** {chat['answer']}")
            st.caption(f"⏰ {chat['time']} | Confidence: {chat['confidence']*100:.0f}%")
            st.divider()

with tab3:
    st.header("🔬 Medical Entity Extraction")
    
    if st.session_state.current_report_id:
        if st.button("🔍 Extract Entities", type="primary"):
            with st.spinner("Extracting..."):
                try:
                    entities_response = requests.get(f"{API_URL}/entities/{st.session_state.current_report_id}")
                    
                    if entities_response.status_code == 200:
                        entities_data = entities_response.json()
                        
                        st.success(f"✅ Found {entities_data['total_entities']} entities")
                        
                        structured = entities_data['structured_data']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.subheader("💊 Medications")
                            for med in structured.get('medications', [])[:5]:
                                st.text(f"• {med['text']}")
                        
                        with col2:
                            st.subheader("🩺 Diagnoses")
                            for diag in structured.get('diagnoses', [])[:5]:
                                st.text(f"• {diag['text']}")
                        
                        with col3:
                            st.subheader("🧪 Lab Values")
                            for lab in structured.get('lab_values', [])[:5]:
                                st.text(f"• {lab['text']}")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.info("📤 Upload a report first")