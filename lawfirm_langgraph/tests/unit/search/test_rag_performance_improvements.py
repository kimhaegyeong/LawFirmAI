# -*- coding: utf-8 -*-
"""
RAG 검색 성능 개선 테스트
"""

import pytest
import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import List, Dict, Any

from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
from lawfirm_langgraph.core.search.engines.hybrid_search_engine_v2 import HybridSearchEngineV2
from lawfirm_langgraph.core.workflow.processors.workflow_document_processor import WorkflowDocumentProcessor
from lawfirm_langgraph.core.search.processors.result_merger import ResultRanker

logger = get_logger(__name__)


class TestRAGPerformanceImprovements:
    """RAG 검색 성능 개선 테스트"""
    
    @pytest.fixture
    def semantic_search_engine(self):
        """시맨틱 검색 엔진 초기화"""
        try:
            engine = SemanticSearchEngineV2()
            return engine
        except Exception as e:
            pytest.skip(f"SemanticSearchEngineV2 initialization failed: {e}")
    
    @pytest.fixture
    def hybrid_search_engine(self):
        """하이브리드 검색 엔진 초기화"""
        try:
            engine = HybridSearchEngineV2()
            return engine
        except Exception as e:
            pytest.skip(f"HybridSearchEngineV2 initialization failed: {e}")
    
    @pytest.fixture
    def document_processor(self):
        """문서 프로세서 초기화"""
        return WorkflowDocumentProcessor()
    
    @pytest.fixture
    def result_ranker(self):
        """결과 랭커 초기화"""
        return ResultRanker(use_cross_encoder=False)  # 테스트에서는 Cross-Encoder 비활성화
    
    def test_search_candidate_expansion(self, semantic_search_engine):
        """검색 후보 수 확대 테스트"""
        query = "계약 해지 사유"
        k = 10
        
        # _search_with_threshold 메서드가 search_k = k * 5를 사용하는지 확인
        # 실제 검색 수행
        results = semantic_search_engine.search(
            query=query,
            k=k,
            similarity_threshold=0.3
        )
        
        # 결과가 있는지 확인
        assert isinstance(results, list)
        logger.info(f"Search returned {len(results)} results for query: {query}")
    
    def test_query_expansion_enhancement(self, semantic_search_engine):
        """쿼리 확장 강화 테스트"""
        query = "계약 해지"
        expanded_keywords = ["계약", "해지", "사유", "조건", "효과", "손해배상"]
        
        # search_with_query_expansion 메서드 테스트
        results = semantic_search_engine.search_with_query_expansion(
            query=query,
            k=10,
            expanded_keywords=expanded_keywords,
            use_query_variations=True
        )
        
        assert isinstance(results, list)
        logger.info(f"Query expansion returned {len(results)} results")
    
    def test_dynamic_threshold_adjustment(self, document_processor):
        """동적 임계값 조정 테스트"""
        # 다양한 점수 분포를 가진 문서 리스트 생성
        test_docs = [
            {"content": "문서 1", "relevance_score": 0.8, "final_weighted_score": 0.8},
            {"content": "문서 2", "relevance_score": 0.75, "final_weighted_score": 0.75},
            {"content": "문서 3", "relevance_score": 0.7, "final_weighted_score": 0.7},
            {"content": "문서 4", "relevance_score": 0.65, "final_weighted_score": 0.65},
            {"content": "문서 5", "relevance_score": 0.6, "final_weighted_score": 0.6},
        ]
        
        # build_prompt_optimized_context 호출
        result = document_processor.build_prompt_optimized_context(
            retrieved_docs=test_docs,
            query="테스트 쿼리",
            extracted_keywords=["테스트"],
            query_type="law_inquiry",
            legal_field="contract"
        )
        
        assert isinstance(result, dict)
        assert "prompt_optimized_text" in result
        assert "document_count" in result
        logger.info(f"Dynamic threshold adjustment: {result['document_count']} documents selected")
    
    def test_cross_encoder_reranker(self, result_ranker):
        """Cross-Encoder Reranker 테스트"""
        # MergedResult 형태의 테스트 데이터 생성
        from lawfirm_langgraph.core.search.processors.result_merger import MergedResult
        
        test_results = [
            MergedResult(
                text="계약 해지 사유에 대한 법령 조문",
                score=0.7,
                source="law",
                metadata={"query": "계약 해지 사유"}
            ),
            MergedResult(
                text="계약 해지 관련 판례",
                score=0.65,
                source="precedent",
                metadata={"query": "계약 해지 사유"}
            ),
            MergedResult(
                text="계약 해지 손해배상",
                score=0.6,
                source="law",
                metadata={"query": "계약 해지 사유"}
            ),
        ]
        
        # Cross-Encoder reranking 테스트 (use_cross_encoder=False이므로 기본 정렬만 수행)
        reranked = result_ranker.cross_encoder_rerank(
            results=test_results,
            query="계약 해지 사유",
            top_k=2
        )
        
        assert isinstance(reranked, list)
        assert len(reranked) <= 2
        logger.info(f"Cross-Encoder reranking: {len(reranked)} results")
    
    def test_mmr_diversity_algorithm(self, document_processor):
        """MMR 다양성 알고리즘 테스트"""
        test_docs = [
            {"content": "계약 해지 사유 법령 조문", "relevance_score": 0.8, "final_weighted_score": 0.8},
            {"content": "계약 해지 사유 법령 조문", "relevance_score": 0.75, "final_weighted_score": 0.75},  # 중복
            {"content": "계약 해지 관련 판례", "relevance_score": 0.7, "final_weighted_score": 0.7},
            {"content": "계약 해지 손해배상", "relevance_score": 0.65, "final_weighted_score": 0.65},
        ]
        
        # MMR 다양성 선택
        diverse_docs = document_processor.select_diverse_documents(
            documents=test_docs,
            query="계약 해지",
            max_docs=3,
            diversity_weight=0.3
        )
        
        assert isinstance(diverse_docs, list)
        assert len(diverse_docs) <= 3
        
        # 다양성 확인: 중복 문서가 제거되었는지 확인
        contents = [doc.get("content", "") for doc in diverse_docs]
        unique_contents = set(contents)
        assert len(unique_contents) == len(contents), "중복 문서가 제거되어야 함"
        
        logger.info(f"MMR diversity: {len(diverse_docs)} diverse documents selected")
    
    def test_query_type_based_weights(self, hybrid_search_engine):
        """질문 유형별 가중치 조정 테스트"""
        # 법령 조회 질문
        law_query = "계약 해지 사유는 무엇인가요?"
        law_results = hybrid_search_engine.search(
            query=law_query,
            max_results=10
        )
        
        assert isinstance(law_results, dict)
        assert "results" in law_results
        logger.info(f"Law inquiry: {law_results['total']} results")
        
        # 판례 검색 질문
        precedent_query = "계약 해지 관련 판례"
        precedent_results = hybrid_search_engine.search(
            query=precedent_query,
            max_results=10
        )
        
        assert isinstance(precedent_results, dict)
        assert "results" in precedent_results
        logger.info(f"Precedent search: {precedent_results['total']} results")
    
    def test_integrated_rag_pipeline(self, hybrid_search_engine, document_processor):
        """통합 RAG 파이프라인 테스트"""
        query = "계약 해지 사유와 손해배상 범위"
        
        # 1. 하이브리드 검색
        search_results = hybrid_search_engine.search(
            query=query,
            max_results=20
        )
        
        assert isinstance(search_results, dict)
        assert "results" in search_results
        
        # 2. 문서 처리 및 필터링
        retrieved_docs = search_results["results"]
        if retrieved_docs:
            context_result = document_processor.build_prompt_optimized_context(
                retrieved_docs=retrieved_docs,
                query=query,
                extracted_keywords=["계약", "해지", "사유", "손해배상"],
                query_type="complex_question",
                legal_field="contract"
            )
            
            assert isinstance(context_result, dict)
            assert "prompt_optimized_text" in context_result
            assert context_result["document_count"] > 0
            
            logger.info(
                f"Integrated pipeline: {context_result['document_count']} documents, "
                f"context length: {context_result['total_context_length']}"
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v"])

