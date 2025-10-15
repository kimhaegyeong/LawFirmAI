# -*- coding: utf-8 -*-
"""
RAG Service (ML Enhanced)
Retrieval-Augmented Generation 서비스 - ML 강화 버전
"""

import logging
from typing import List, Dict, Any, Optional
from source.models.model_manager import LegalModelManager
from source.data.vector_store import LegalVectorStore
from source.data.database import DatabaseManager
from source.utils.config import Config

logger = logging.getLogger(__name__)


class MLEnhancedRAGService:
    """ML 강화 RAG 서비스 클래스"""
    
    def __init__(self, config: Config, model_manager: LegalModelManager, 
                 vector_store: LegalVectorStore, database: DatabaseManager):
        """ML 강화 RAG 서비스 초기화"""
        self.config = config
        self.model_manager = model_manager
        self.vector_store = vector_store
        self.database = database
        self.logger = logging.getLogger(__name__)
        
        # ML 강화 기능 설정
        self.use_ml_enhanced_search = True
        self.quality_threshold = 0.7  # 파싱 품질 임계값
        self.supplementary_weight = 0.8  # 부칙 가중치
        
        self.logger.info("MLEnhancedRAGService initialized")
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 5, 
                                   filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """ML 강화 관련 문서 검색 (Assembly 데이터 포함)"""
        try:
            # 기본 필터 설정
            search_filters = filters or {}
            
            # ML 강화 검색이 활성화된 경우 품질 필터 추가
            if self.use_ml_enhanced_search:
                search_filters['ml_enhanced'] = True
            
            # 벡터 저장소에서 유사한 문서 검색
            similar_docs = self.vector_store.search(query, top_k * 2, search_filters)  # 더 많은 결과 가져오기
            
            # Assembly 데이터베이스에서 추가 검색
            assembly_docs = self._search_assembly_documents(query, top_k)
            
            # 모든 결과 통합
            all_docs = similar_docs + assembly_docs
            
            # ML 강화 결과 필터링 및 스코어링
            filtered_docs = self._filter_and_score_documents(all_docs, query)
            
            # 결과 포맷팅
            results = []
            for doc in filtered_docs[:top_k]:
                metadata = doc.get("metadata", {})
                result = {
                    "document_id": metadata.get("document_id", metadata.get("law_id", "")),
                    "title": metadata.get("law_name", metadata.get("title", "")),
                    "content": doc.get("text", doc.get("content", "")),
                    "similarity": doc.get("score", 0.0),
                    "source": metadata.get("document_type", metadata.get("source", "assembly")),
                    "chunk_index": metadata.get("chunk_id", 0),
                    "article_number": metadata.get("article_number", ""),
                    "article_title": metadata.get("article_title", ""),
                    "article_type": metadata.get("article_type", "main"),
                    "is_supplementary": metadata.get("is_supplementary", False),
                    "ml_confidence_score": metadata.get("ml_confidence_score"),
                    "parsing_method": metadata.get("parsing_method", "ml_enhanced"),
                    "quality_score": metadata.get("parsing_quality_score", metadata.get("quality_score", 0.0))
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _search_assembly_documents(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Assembly 데이터베이스에서 문서 검색"""
        try:
            # Assembly 데이터베이스에서 검색
            assembly_results = self.database.search_assembly_documents(query, top_k)
            
            # 결과를 벡터 검색 결과와 동일한 형식으로 변환
            formatted_results = []
            for result in assembly_results:
                formatted_result = {
                    "text": result.get("content", ""),
                    "score": result.get("relevance_score", 0.0),
                    "metadata": {
                        "law_id": result.get("law_id", ""),
                        "law_name": result.get("law_name", ""),
                        "article_number": result.get("article_number", ""),
                        "article_title": result.get("article_title", ""),
                        "article_type": result.get("article_type", "main"),
                        "is_supplementary": result.get("is_supplementary", False),
                        "ml_confidence_score": result.get("ml_confidence_score"),
                        "parsing_method": result.get("parsing_method", "ml_enhanced"),
                        "parsing_quality_score": result.get("quality_score", 0.0),
                        "source": "assembly",
                        "document_type": "assembly_law"
                    }
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error searching assembly documents: {e}")
            return []
    
    def _filter_and_score_documents(self, documents: List[Dict[str, Any]], 
                                   query: str) -> List[Dict[str, Any]]:
        """ML 강화 문서 필터링 및 스코어링"""
        try:
            filtered_docs = []
            
            for doc in documents:
                metadata = doc.get("metadata", {})
                
                # 품질 필터링
                quality_score = metadata.get("parsing_quality_score", 0.0)
                if quality_score < self.quality_threshold:
                    continue
                
                # 부칙 가중치 적용
                is_supplementary = metadata.get("is_supplementary", False)
                if is_supplementary:
                    doc["score"] *= self.supplementary_weight
                
                # ML 신뢰도 점수 고려
                ml_confidence = metadata.get("ml_confidence_score")
                if ml_confidence is not None:
                    doc["score"] *= (0.5 + 0.5 * ml_confidence)  # 0.5~1.0 범위로 조정
                
                filtered_docs.append(doc)
            
            # 스코어 기준으로 정렬
            filtered_docs.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            
            return filtered_docs
            
        except Exception as e:
            self.logger.error(f"Error filtering documents: {e}")
            return documents
    
    def generate_context(self, query: str, retrieved_docs: List[Dict[str, Any]], 
                        max_context_length: int = 2000) -> str:
        """ML 강화 컨텍스트 생성"""
        try:
            if not retrieved_docs:
                return ""
            
            context_parts = []
            current_length = 0
            
            # 문서를 품질 점수와 유사도로 정렬
            sorted_docs = sorted(retrieved_docs, 
                               key=lambda x: (x.get("quality_score", 0.0), x.get("similarity", 0.0)), 
                               reverse=True)
            
            for doc in sorted_docs:
                content = doc.get("content", "")
                title = doc.get("title", "")
                similarity = doc.get("similarity", 0.0)
                article_number = doc.get("article_number", "")
                article_title = doc.get("article_title", "")
                article_type = doc.get("article_type", "main")
                quality_score = doc.get("quality_score", 0.0)
                
                # 문서 정보 구성 (ML 강화 정보 포함)
                doc_header = f"[문서: {title}"
                if article_number:
                    doc_header += f" - {article_number}"
                    if article_title:
                        doc_header += f"({article_title})"
                doc_header += f" | 유사도: {similarity:.3f} | 품질: {quality_score:.3f}"
                if article_type == "supplementary":
                    doc_header += " | 부칙"
                doc_header += "]"
                
                doc_info = f"{doc_header}\n{content}\n"
                
                if current_length + len(doc_info) > max_context_length:
                    break
                
                context_parts.append(doc_info)
                current_length += len(doc_info)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating context: {e}")
            return ""
    
    def process_query(self, query: str, top_k: int = 5, 
                     filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """ML 강화 RAG 쿼리 처리"""
        try:
            # 1. 관련 문서 검색 (ML 강화)
            retrieved_docs = self.retrieve_relevant_documents(query, top_k, filters)
            
            # 2. 컨텍스트 생성 (ML 강화)
            context = self.generate_context(query, retrieved_docs)
            
            # 3. 응답 생성
            response = self.model_manager.generate_response(query, context)
            
            # 4. 소스 정보 추가 (ML 강화 정보 포함)
            sources = []
            for doc in retrieved_docs:
                source_info = {
                    "title": doc.get("title", ""),
                    "article_number": doc.get("article_number", ""),
                    "article_title": doc.get("article_title", ""),
                    "article_type": doc.get("article_type", "main"),
                    "similarity": doc.get("similarity", 0.0),
                    "quality_score": doc.get("quality_score", 0.0),
                    "ml_confidence_score": doc.get("ml_confidence_score"),
                    "parsing_method": doc.get("parsing_method", "ml_enhanced"),
                    "is_supplementary": doc.get("is_supplementary", False),
                    "chunk_index": doc.get("chunk_index", 0)
                }
                sources.append(source_info)
            
            # 5. ML 강화 통계 추가
            ml_stats = self._calculate_ml_stats(retrieved_docs)
            
            response["sources"] = sources
            response["retrieved_docs_count"] = len(retrieved_docs)
            response["ml_enhanced"] = True
            response["ml_stats"] = ml_stats
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing RAG query: {e}")
            return {
                "response": "죄송합니다. 질문을 처리하는 중 오류가 발생했습니다.",
                "confidence": 0.0,
                "sources": [],
                "retrieved_docs_count": 0,
                "ml_enhanced": False,
                "ml_stats": {}
            }
    
    def _calculate_ml_stats(self, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ML 강화 통계 계산"""
        try:
            if not retrieved_docs:
                return {}
            
            total_docs = len(retrieved_docs)
            main_articles = sum(1 for doc in retrieved_docs if not doc.get("is_supplementary", False))
            supplementary_articles = sum(1 for doc in retrieved_docs if doc.get("is_supplementary", False))
            
            avg_quality_score = sum(doc.get("quality_score", 0.0) for doc in retrieved_docs) / total_docs
            avg_similarity = sum(doc.get("similarity", 0.0) for doc in retrieved_docs) / total_docs
            
            ml_confidence_scores = [doc.get("ml_confidence_score") for doc in retrieved_docs 
                                   if doc.get("ml_confidence_score") is not None]
            avg_ml_confidence = sum(ml_confidence_scores) / len(ml_confidence_scores) if ml_confidence_scores else 0.0
            
            return {
                "total_documents": total_docs,
                "main_articles": main_articles,
                "supplementary_articles": supplementary_articles,
                "avg_quality_score": round(avg_quality_score, 3),
                "avg_similarity": round(avg_similarity, 3),
                "avg_ml_confidence": round(avg_ml_confidence, 3),
                "ml_enhanced_docs": sum(1 for doc in retrieved_docs if doc.get("parsing_method") == "ml_enhanced")
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating ML stats: {e}")
            return {}
    
    def add_document(self, document: Dict[str, Any]) -> bool:
        """ML 강화 문서 추가"""
        try:
            # 벡터 저장소에 문서 추가
            success = self.vector_store.add_documents([document])
            
            if success:
                self.logger.info(f"ML-enhanced document added: {document.get('law_name', document.get('case_name', 'Unknown'))}")
            else:
                self.logger.warning("Failed to add document to vector store")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error adding document: {e}")
            return False
    
    def search_by_article_type(self, query: str, article_type: str = "main", top_k: int = 5) -> List[Dict[str, Any]]:
        """조문 유형별 검색 (본칙/부칙)"""
        try:
            filters = {"article_type": article_type}
            return self.retrieve_relevant_documents(query, top_k, filters)
            
        except Exception as e:
            self.logger.error(f"Error searching by article type: {e}")
            return []
    
    def search_by_quality_threshold(self, query: str, min_quality: float = 0.8, top_k: int = 5) -> List[Dict[str, Any]]:
        """품질 임계값 기반 검색"""
        try:
            # 임시로 품질 임계값 설정
            original_threshold = self.quality_threshold
            self.quality_threshold = min_quality
            
            results = self.retrieve_relevant_documents(query, top_k)
            
            # 원래 임계값 복원
            self.quality_threshold = original_threshold
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching by quality threshold: {e}")
            return []
    
    def get_rag_stats(self) -> Dict[str, Any]:
        """ML 강화 RAG 서비스 통계"""
        try:
            vector_stats = self.vector_store.get_stats()
            
            return {
                "vector_store": vector_stats,
                "model_status": self.model_manager.get_model_status(),
                "total_documents": vector_stats.get("documents_count", 0),
                "ml_enhanced": True,
                "quality_threshold": self.quality_threshold,
                "supplementary_weight": self.supplementary_weight,
                "use_ml_enhanced_search": self.use_ml_enhanced_search
            }
            
        except Exception as e:
            self.logger.error(f"Error getting RAG stats: {e}")
            return {"error": str(e)}


# 레거시 호환성을 위한 기존 RAGService 클래스
class RAGService(MLEnhancedRAGService):
    """레거시 호환성을 위한 RAG 서비스 클래스"""
    
    def __init__(self, config: Config, model_manager: LegalModelManager, 
                 vector_store: LegalVectorStore, database: DatabaseManager):
        """레거시 RAG 서비스 초기화"""
        super().__init__(config, model_manager, vector_store, database)
        self.logger.info("Legacy RAGService initialized (using MLEnhancedRAGService)")
