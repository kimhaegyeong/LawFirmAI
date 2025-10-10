# -*- coding: utf-8 -*-
"""
LangChain RAG Service
LangChain 기반 RAG 서비스 구현
"""

import logging
import time
import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

try:
    from langchain.chains import RetrievalQA
    from langchain_community.vectorstores import FAISS, Chroma
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain.retrievers import VectorStoreRetriever
    LANCHAIN_AVAILABLE = True
except ImportError:
    LANCHAIN_AVAILABLE = False
    # Mock classes for when LangChain is not available
    class FAISS:
        @staticmethod
        def from_documents(*args, **kwargs):
            return None
    
    class SentenceTransformerEmbeddings:
        def __init__(self, *args, **kwargs):
            pass
    
    class RetrievalQA:
        def __init__(self, *args, **kwargs):
            pass
        def run(self, *args, **kwargs):
            return "Mock response"

from .langfuse_client import LangfuseClient, MetricsCollector, RAGMetrics
from .document_processor import LegalDocumentProcessor, DocumentChunk
from .context_manager import ContextManager, ContextWindow
from .answer_generator import AnswerGenerator, AnswerResult
from utils.langchain_config import LangChainConfig, langchain_config
from data.vector_store import LegalVectorStore
from data.database import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """RAG 결과 데이터 클래스"""
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    retrieved_docs: List[Dict[str, Any]]
    response_time: float
    tokens_used: int
    trace_id: Optional[str]
    metadata: Dict[str, Any]


class LangChainRAGService:
    """LangChain 기반 RAG 서비스"""
    
    def __init__(self, config: Optional[LangChainConfig] = None):
        """RAG 서비스 초기화"""
        self.config = config or langchain_config
        self.logger = logging.getLogger(__name__)
        
        # 컴포넌트 초기화
        self.langfuse_client = LangfuseClient(self.config)
        self.document_processor = LegalDocumentProcessor(self.config)
        self.context_manager = ContextManager(self.config)
        self.answer_generator = AnswerGenerator(self.config, self.langfuse_client)
        self.metrics_collector = MetricsCollector(self.langfuse_client)
        
        # 벡터 저장소 초기화
        self.vector_store = None
        self.embeddings = None
        self.retriever = None
        
        # 기존 벡터 저장소와 데이터베이스 연결
        self.legal_vector_store = None
        self.database_manager = None
        
        # 통계
        self.stats = {
            'total_queries': 0,
            'total_documents': 0,
            'avg_response_time': 0.0,
            'avg_confidence': 0.0
        }
        
        # 초기화
        self._initialize_components()
        
        self.logger.info("LangChainRAGService initialized successfully")
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        try:
            # 임베딩 모델 초기화
            if LANCHAIN_AVAILABLE:
                self.embeddings = SentenceTransformerEmbeddings(
                    model_name=self.config.embedding_model
                )
                self.logger.info(f"Initialized embeddings: {self.config.embedding_model}")
            
            # 기존 벡터 저장소 연결 시도
            self._connect_existing_vector_store()
            
            # LangChain 벡터 저장소 초기화
            self._initialize_vector_store()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
    
    def _connect_existing_vector_store(self):
        """기존 벡터 저장소 연결"""
        try:
            # 기존 LegalVectorStore와 DatabaseManager 연결 시도
            from data.vector_store import LegalVectorStore
            from data.database import DatabaseManager
            
            # 설정에서 모델명과 인덱스 경로 가져오기
            model_name = self.config.embedding_model
            index_path = self.config.vector_store_path
            
            # LegalVectorStore를 올바른 모델로 초기화
            self.legal_vector_store = LegalVectorStore(model_name=model_name)
            
            # 기존 인덱스 로드 시도
            if os.path.exists(index_path):
                success = self.legal_vector_store.load_index(index_path)
                if success:
                    self.logger.info(f"Loaded existing vector store from {index_path}")
                else:
                    self.logger.warning(f"Failed to load vector store from {index_path}")
            else:
                self.logger.warning(f"Vector store path does not exist: {index_path}")
            
            self.database_manager = DatabaseManager()
            
            self.logger.info("Connected to existing vector store and database")
            
        except Exception as e:
            self.logger.warning(f"Could not connect to existing stores: {e}")
    
    def _initialize_vector_store(self):
        """LangChain 벡터 저장소 초기화"""
        if not LANCHAIN_AVAILABLE or not self.embeddings:
            self.logger.warning("LangChain or embeddings not available")
            return
        
        try:
            vector_store_path = self.config.vector_store_path
            
            if os.path.exists(vector_store_path):
                # 기존 벡터 저장소 로드
                if self.config.vector_store_type.value == "faiss":
                    self.vector_store = FAISS.load_local(
                        vector_store_path, 
                        self.embeddings
                    )
                elif self.config.vector_store_type.value == "chroma":
                    self.vector_store = Chroma(
                        persist_directory=vector_store_path,
                        embedding_function=self.embeddings
                    )
                
                self.logger.info(f"Loaded existing vector store from {vector_store_path}")
            else:
                # 새 벡터 저장소 생성
                self.logger.info("No existing vector store found, will create new one")
            
            # 리트리버 초기화
            if self.vector_store:
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": self.config.search_k}
                )
                self.logger.info("Initialized retriever")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {e}")
    
    def process_query(self, query: str, session_id: Optional[str] = None,
                     template_type: str = "legal_qa") -> RAGResult:
        """RAG 쿼리 처리"""
        start_time = time.time()
        
        try:
            # 세션에 쿼리 추가
            if session_id:
                self.context_manager.add_query_to_session(session_id, query)
            
            # 1. 문서 검색
            retrieved_docs = self._retrieve_documents(query)
            
            # 2. 컨텍스트 윈도우 구축
            context_windows = self.context_manager.build_context_window(
                retrieved_docs, query, session_id
            )
            
            # 3. 컨텍스트 생성
            context = self._build_context(context_windows)
            
            # 4. 답변 생성
            answer_result = self.answer_generator.generate_answer(
                query=query,
                context=context,
                template_type=template_type,
                session_id=session_id
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # 5. 메트릭 수집
            metrics = self.metrics_collector.collect_rag_metrics(
                query=query,
                response=answer_result.answer,
                retrieved_docs=retrieved_docs,
                response_time=response_time
            )
            
            # 6. 결과 생성
            result = RAGResult(
                answer=answer_result.answer,
                confidence=answer_result.confidence,
                sources=answer_result.sources,
                retrieved_docs=retrieved_docs,
                response_time=response_time,
                tokens_used=answer_result.tokens_used,
                trace_id=self.langfuse_client.get_current_trace_id() if self.langfuse_client else None,
                metadata={
                    'template_type': template_type,
                    'session_id': session_id,
                    'context_length': len(context),
                    'retrieved_docs_count': len(retrieved_docs),
                    'context_windows_count': len(context_windows),
                    'metrics': metrics
                }
            )
            
            # 통계 업데이트
            self._update_stats(result)
            
            self.logger.info(f"Processed RAG query in {response_time:.2f}s with confidence {answer_result.confidence:.2f}")
            return result
            
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            
            # 오류 추적
            if self.langfuse_client and self.langfuse_client.is_enabled():
                self.langfuse_client.track_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={"query": query, "session_id": session_id}
                )
            
            self.logger.error(f"Failed to process RAG query: {e}")
            
            # 오류 결과 반환
            return RAGResult(
                answer="죄송합니다. 질문을 처리하는 중 오류가 발생했습니다.",
                confidence=0.0,
                sources=[],
                retrieved_docs=[],
                response_time=response_time,
                tokens_used=0,
                trace_id=None,
                metadata={'error': str(e)}
            )
    
    def _retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """문서 검색"""
        retrieved_docs = []
        
        try:
            # LangChain 리트리버 사용
            if self.retriever and LANCHAIN_AVAILABLE:
                langchain_docs = self.retriever.get_relevant_documents(query)
                
                for doc in langchain_docs:
                    retrieved_docs.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'similarity': 0.8,  # 기본값
                        'source': doc.metadata.get('source', 'unknown'),
                        'chunk_id': doc.metadata.get('chunk_id', 'unknown')
                    })
            
            # 기존 벡터 저장소 사용 (백업)
            if not retrieved_docs and self.legal_vector_store:
                similar_docs = self.legal_vector_store.search(query, self.config.search_k)
                
                for doc in similar_docs:
                    retrieved_docs.append({
                        'content': doc.get('text', ''),
                        'metadata': doc.get('metadata', {}),
                        'similarity': doc.get('score', 0.0),
                        'source': doc.get('metadata', {}).get('law_name', 'unknown'),
                        'chunk_id': doc.get('metadata', {}).get('chunk_id', 'unknown')
                    })
            
            # 유사도 임계값 필터링
            filtered_docs = [
                doc for doc in retrieved_docs 
                if doc.get('similarity', 0.0) >= self.config.similarity_threshold
            ]
            
            self.logger.info(f"Retrieved {len(filtered_docs)} documents for query")
            return filtered_docs
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve documents: {e}")
            return []
    
    def _build_context(self, context_windows: List[ContextWindow]) -> str:
        """컨텍스트 구축"""
        if not context_windows:
            return ""
        
        context_parts = []
        current_length = 0
        
        for cw in context_windows:
            # 컨텍스트 길이 확인
            if current_length + len(cw.content) > self.config.max_context_length:
                break
            
            context_part = f"[문서: {cw.source_document}]\n{cw.content}"
            context_parts.append(context_part)
            current_length += len(context_part)
        
        return "\n\n".join(context_parts)
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """문서 추가"""
        try:
            if not LANCHAIN_AVAILABLE:
                self.logger.error("LangChain is not available")
                return False
            
            # 문서 처리
            processed_docs = []
            for doc_data in documents:
                from langchain.schema import Document
                doc = Document(
                    page_content=doc_data.get('content', ''),
                    metadata=doc_data.get('metadata', {})
                )
                processed_docs.append(doc)
            
            # 문서 청킹
            all_chunks = []
            for doc in processed_docs:
                chunks = self.document_processor.split_document(doc)
                all_chunks.extend(chunks)
            
            # 벡터 저장소에 추가
            if self.vector_store and self.embeddings:
                # LangChain 문서로 변환
                langchain_docs = []
                for chunk in all_chunks:
                    from langchain.schema import Document
                    langchain_doc = Document(
                        page_content=chunk.content,
                        metadata=chunk.metadata
                    )
                    langchain_docs.append(langchain_doc)
                
                # 벡터 저장소 업데이트
                if self.config.vector_store_type.value == "faiss":
                    # FAISS는 새로 생성해야 함
                    if os.path.exists(self.config.vector_store_path):
                        existing_store = FAISS.load_local(
                            self.config.vector_store_path, 
                            self.embeddings
                        )
                        existing_store.add_documents(langchain_docs)
                        existing_store.save_local(self.config.vector_store_path)
                    else:
                        new_store = FAISS.from_documents(langchain_docs, self.embeddings)
                        new_store.save_local(self.config.vector_store_path)
                    
                    # 리트리버 재초기화
                    self._initialize_vector_store()
                
                elif self.config.vector_store_type.value == "chroma":
                    self.vector_store.add_documents(langchain_docs)
            
            # 기존 벡터 저장소에도 추가
            if self.legal_vector_store:
                for chunk in all_chunks:
                    doc_data = {
                        'text': chunk.content,
                        'metadata': chunk.metadata
                    }
                    self.legal_vector_store.add_documents([doc_data])
            
            self.stats['total_documents'] += len(all_chunks)
            self.logger.info(f"Added {len(all_chunks)} document chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            return False
    
    def _update_stats(self, result: RAGResult):
        """통계 업데이트"""
        self.stats['total_queries'] += 1
        
        # 평균 응답 시간 업데이트
        if self.stats['total_queries'] == 1:
            self.stats['avg_response_time'] = result.response_time
            self.stats['avg_confidence'] = result.confidence
        else:
            # 이동 평균 계산
            alpha = 0.1
            self.stats['avg_response_time'] = (
                (1 - alpha) * self.stats['avg_response_time'] + 
                alpha * result.response_time
            )
            self.stats['avg_confidence'] = (
                (1 - alpha) * self.stats['avg_confidence'] + 
                alpha * result.confidence
            )
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """서비스 통계 반환"""
        return {
            "rag_stats": self.stats,
            "vector_store_stats": self._get_vector_store_stats(),
            "context_stats": self.context_manager.get_context_statistics(),
            "generator_stats": self.answer_generator.get_generator_statistics(),
            "langfuse_enabled": self.langfuse_client.is_enabled() if self.langfuse_client else False,
            "langchain_available": LANCHAIN_AVAILABLE,
            "embeddings_model": self.config.embedding_model,
            "llm_model": self.config.llm_model
        }
    
    def _get_vector_store_stats(self) -> Dict[str, Any]:
        """벡터 저장소 통계"""
        stats = {
            "vector_store_type": self.config.vector_store_type.value,
            "vector_store_path": self.config.vector_store_path,
            "embeddings_available": self.embeddings is not None,
            "retriever_available": self.retriever is not None
        }
        
        # 기존 벡터 저장소 통계 추가
        if self.legal_vector_store:
            try:
                legal_stats = self.legal_vector_store.get_stats()
                stats.update(legal_stats)
            except Exception as e:
                stats['legal_vector_store_error'] = str(e)
        
        return stats
    
    def clear_session(self, session_id: str) -> bool:
        """세션 삭제"""
        return self.context_manager.clear_session(session_id)
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """오래된 세션 정리"""
        self.context_manager.cleanup_old_sessions(max_age_hours)
    
    def validate_configuration(self) -> List[str]:
        """설정 유효성 검사"""
        errors = []
        
        # 설정 검증
        config_errors = self.config.validate()
        errors.extend(config_errors)
        
        # 컴포넌트 검증
        if not LANCHAIN_AVAILABLE:
            errors.append("LangChain is not available")
        
        if not self.embeddings:
            errors.append("Embeddings model is not initialized")
        
        if not self.vector_store and not self.legal_vector_store:
            errors.append("No vector store is available")
        
        if not self.answer_generator.llm:
            errors.append("LLM is not initialized")
        
        return errors
