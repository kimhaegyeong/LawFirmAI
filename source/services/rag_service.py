"""
RAG Service
Retrieval-Augmented Generation 서비스
"""

import logging
from typing import List, Dict, Any, Optional
from ..models.model_manager import ModelManager
from ..data.vector_store import VectorStore
from ..data.database import DatabaseManager
from ..utils.config import Config

logger = logging.getLogger(__name__)


class RAGService:
    """RAG 서비스 클래스"""
    
    def __init__(self, config: Config, model_manager: ModelManager, 
                 vector_store: VectorStore, database: DatabaseManager):
        """RAG 서비스 초기화"""
        self.config = config
        self.model_manager = model_manager
        self.vector_store = vector_store
        self.database = database
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("RAGService initialized")
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """관련 문서 검색"""
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.model_manager.encode_text(query)
            if not query_embedding:
                self.logger.warning("Failed to encode query")
                return []
            
            # 벡터 저장소에서 유사한 문서 검색
            import numpy as np
            query_array = np.array(query_embedding)
            similar_docs = self.vector_store.search_similar(query_array, top_k)
            
            # 결과 포맷팅
            results = []
            for idx, similarity, metadata in similar_docs:
                result = {
                    "document_id": metadata.get("document_id"),
                    "title": metadata.get("title", ""),
                    "content": metadata.get("content", ""),
                    "similarity": similarity,
                    "source": metadata.get("source", ""),
                    "chunk_index": metadata.get("chunk_index", 0)
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {e}")
            return []
    
    def generate_context(self, query: str, retrieved_docs: List[Dict[str, Any]], 
                        max_context_length: int = 2000) -> str:
        """검색된 문서로부터 컨텍스트 생성"""
        try:
            if not retrieved_docs:
                return ""
            
            context_parts = []
            current_length = 0
            
            for doc in retrieved_docs:
                content = doc.get("content", "")
                title = doc.get("title", "")
                similarity = doc.get("similarity", 0.0)
                
                # 문서 정보 추가
                doc_info = f"[문서: {title} (유사도: {similarity:.3f})]\n{content}\n"
                
                if current_length + len(doc_info) > max_context_length:
                    break
                
                context_parts.append(doc_info)
                current_length += len(doc_info)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating context: {e}")
            return ""
    
    def process_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """RAG 쿼리 처리"""
        try:
            # 1. 관련 문서 검색
            retrieved_docs = self.retrieve_relevant_documents(query, top_k)
            
            # 2. 컨텍스트 생성
            context = self.generate_context(query, retrieved_docs)
            
            # 3. 응답 생성
            response = self.model_manager.generate_response(query, context)
            
            # 4. 소스 정보 추가
            sources = []
            for doc in retrieved_docs:
                source_info = {
                    "title": doc.get("title", ""),
                    "similarity": doc.get("similarity", 0.0),
                    "chunk_index": doc.get("chunk_index", 0)
                }
                sources.append(source_info)
            
            response["sources"] = sources
            response["retrieved_docs_count"] = len(retrieved_docs)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing RAG query: {e}")
            return {
                "response": "죄송합니다. 질문을 처리하는 중 오류가 발생했습니다.",
                "confidence": 0.0,
                "sources": [],
                "retrieved_docs_count": 0
            }
    
    def add_document(self, document: Dict[str, Any]) -> bool:
        """문서 추가"""
        try:
            # 문서 임베딩 생성
            doc_embeddings = self.model_manager.encode_documents([document])
            if not doc_embeddings:
                self.logger.warning("Failed to encode document")
                return False
            
            # 벡터 저장소에 추가
            for embedding_data in doc_embeddings:
                embedding = embedding_data["embedding"]
                metadata = {
                    "document_id": embedding_data["document_id"],
                    "title": embedding_data["title"],
                    "content": embedding_data["content"],
                    "chunk_index": 0
                }
                
                self.vector_store.add_embedding(embedding, metadata)
            
            self.logger.info(f"Document added: {document.get('title', 'Unknown')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding document: {e}")
            return False
    
    def get_rag_stats(self) -> Dict[str, Any]:
        """RAG 서비스 통계"""
        try:
            vector_stats = self.vector_store.get_stats()
            
            return {
                "vector_store": vector_stats,
                "model_status": self.model_manager.get_model_status(),
                "total_documents": vector_stats.get("total_embeddings", 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting RAG stats: {e}")
            return {"error": str(e)}
