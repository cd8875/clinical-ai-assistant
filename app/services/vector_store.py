from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Dict
import pickle
import os

class VectorStore:
    """Manages vector embeddings and semantic search using FAISS"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_db_path = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.vector_store = None
        self.metadata_store = {}  # Store report metadata
        
        # Load existing vector store if available
        self._load_vector_store()
    
    def add_report(self, report_id: str, content: str, metadata: Dict = None):
        """Add a medical report to the vector store"""
        # Split text into chunks
        chunks = self.text_splitter.split_text(content)
        
        # Add metadata to each chunk
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "report_id": report_id,
                "chunk_id": i,
                "chunk_total": len(chunks),
                **(metadata or {})
            }
            metadatas.append(chunk_metadata)
        
        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(
                chunks,
                self.embeddings,
                metadatas=metadatas
            )
        else:
            new_vectors = FAISS.from_texts(
                chunks,
                self.embeddings,
                metadatas=metadatas
            )
            self.vector_store.merge_from(new_vectors)
        
        # Store metadata
        self.metadata_store[report_id] = metadata or {}
        
        # Save to disk
        self._save_vector_store()
        
        return {
            "report_id": report_id,
            "chunks_added": len(chunks),
            "status": "success"
        }
    
    def search(self, query: str, k: int = 5, filter_by: Dict = None) -> List[Dict]:
        """Semantic search across all reports"""
        if self.vector_store is None:
            return []
        
        # Perform similarity search
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Format results
        formatted_results = []
        for doc, score in results:
            # Apply filters if provided
            if filter_by:
                skip = False
                for key, value in filter_by.items():
                    if doc.metadata.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue
            
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(1 - score),  # Convert distance to similarity
                "report_id": doc.metadata.get("report_id"),
            })
        
        return formatted_results
    
    def search_by_report_ids(self, query: str, report_ids: List[str], k: int = 5) -> List[Dict]:
        """Search within specific reports only"""
        return self.search(
            query,
            k=k,
            filter_by={"report_id": report_ids[0]} if len(report_ids) == 1 else None
        )
    
    def get_report_chunks(self, report_id: str) -> List[str]:
        """Retrieve all chunks for a specific report"""
        if self.vector_store is None:
            return []
        
        # Search with high k to get all chunks
        all_docs = self.vector_store.similarity_search("", k=1000)
        
        chunks = []
        for doc in all_docs:
            if doc.metadata.get("report_id") == report_id:
                chunks.append({
                    "content": doc.page_content,
                    "chunk_id": doc.metadata.get("chunk_id")
                })
        
        # Sort by chunk_id
        chunks.sort(key=lambda x: x.get("chunk_id", 0))
        return [c["content"] for c in chunks]
    
    def delete_report(self, report_id: str):
        """Remove a report from the vector store"""
        # Note: FAISS doesn't support deletion easily
        # Best approach is to rebuild the store without the report
        if report_id in self.metadata_store:
            del self.metadata_store[report_id]
            self._save_vector_store()
            return {"status": "marked_for_deletion", "note": "Full deletion requires rebuild"}
        return {"status": "not_found"}
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        if self.vector_store is None:
            return {
                "total_reports": 0,
                "total_chunks": 0,
                "reports": []
            }
        
        return {
            "total_reports": len(self.metadata_store),
            "total_chunks": self.vector_store.index.ntotal,
            "reports": list(self.metadata_store.keys())
        }
    
    def _save_vector_store(self):
        """Save vector store to disk"""
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        if self.vector_store:
            self.vector_store.save_local(self.vector_db_path)
        
        # Save metadata separately
        metadata_path = os.path.join(self.vector_db_path, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata_store, f)
    
    def _load_vector_store(self):
        """Load vector store from disk"""
        try:
            if os.path.exists(self.vector_db_path):
                self.vector_store = FAISS.load_local(
                    self.vector_db_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Load metadata
                metadata_path = os.path.join(self.vector_db_path, "metadata.pkl")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'rb') as f:
                        self.metadata_store = pickle.load(f)
                
                print(f"✅ Loaded vector store with {self.vector_store.index.ntotal} vectors")
        except Exception as e:
            print(f"⚠️ Could not load vector store: {e}")
            self.vector_store = None