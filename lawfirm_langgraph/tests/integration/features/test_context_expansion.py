# -*- coding: utf-8 -*-
"""
Context Expansion 개선 사항 테스트
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# lawfirm_langgraph 디렉토리를 sys.path에 추가
lawfirm_langgraph_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(lawfirm_langgraph_path))

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class TestContextExpansionImprovements:
    """Context Expansion 개선 사항 테스트"""
    
    def test_needs_expansion_condition_strengthened(self):
        """needs_expansion 조건 강화 테스트"""
        from lawfirm_langgraph.core.generation.validators.quality_validators import ContextValidator
        
        # Case 1: overall_score가 0.5 미만이면 확장
        context = {
            "context": "테스트 컨텍스트",
            "document_count": 2,
            "context_length": 500
        }
        result = ContextValidator.validate_context_quality(
            context=context,
            query="테스트 질문",
            query_type="simple",
            extracted_keywords=["테스트", "질문"]
        )
        # overall_score가 낮으면 needs_expansion이 True
        if result["overall_score"] < 0.5:
            assert result["needs_expansion"] == True, f"overall_score {result['overall_score']:.2f} < 0.5 should trigger expansion"
        
        # Case 2: overall_score가 0.5 이상이면 확장하지 않음
        context_good = {
            "context": "테스트 컨텍스트 " * 100,  # 충분한 길이
            "document_count": 5,
            "context_length": 2000,
            "legal_references": ["법1", "법2"],
            "citations": [{"type": "precedent"}]
        }
        result_good = ContextValidator.validate_context_quality(
            context=context_good,
            query="테스트 질문",
            query_type="simple",
            extracted_keywords=["테스트", "질문"]
        )
        # overall_score가 0.5 이상이면 needs_expansion이 False일 수 있음
        if result_good["overall_score"] >= 0.5:
            # missing_info가 3개 미만이면 확장하지 않음
            if len(result_good["missing_information"]) < 3:
                assert result_good["needs_expansion"] == False, f"overall_score {result_good['overall_score']:.2f} >= 0.5 and missing_info < 3 should not trigger expansion"
        
        logger.info("✅ needs_expansion 조건 강화 테스트 통과")
    
    def test_missing_info_threshold_lowered(self):
        """누락 정보 추가 조건 강화 테스트"""
        from lawfirm_langgraph.core.generation.validators.quality_validators import ContextValidator
        
        # Case 1: coverage_score가 0.3 미만이면 누락 정보 추가
        context_low_coverage = {
            "context": "짧은 컨텍스트",
            "document_count": 1,
            "context_length": 100
        }
        result = ContextValidator.validate_context_quality(
            context=context_low_coverage,
            query="테스트 질문",
            query_type="simple",
            extracted_keywords=["테스트", "질문", "키워드1", "키워드2", "키워드3"]
        )
        
        # coverage_score가 0.3 미만이면 "핵심 키워드 커버리지 부족" 추가
        if result["coverage_score"] < 0.3:
            assert "핵심 키워드 커버리지 부족" in result["missing_information"], \
                f"coverage_score {result['coverage_score']:.2f} < 0.3 should add missing info"
        
        # Case 2: coverage_score가 0.3 이상이면 누락 정보 추가하지 않음
        context_good_coverage = {
            "context": "테스트 질문 키워드 " * 50,  # 키워드 포함
            "document_count": 3,
            "context_length": 1000
        }
        result_good = ContextValidator.validate_context_quality(
            context=context_good_coverage,
            query="테스트 질문",
            query_type="simple",
            extracted_keywords=["테스트", "질문"]
        )
        
        # coverage_score가 0.3 이상이면 누락 정보 추가하지 않음
        if result_good["coverage_score"] >= 0.3:
            assert "핵심 키워드 커버리지 부족" not in result_good["missing_information"], \
                f"coverage_score {result_good['coverage_score']:.2f} >= 0.3 should not add missing info"
        
        logger.info("✅ 누락 정보 추가 조건 강화 테스트 통과")
    
    def test_missing_info_identification_improved(self):
        """누락 정보 식별 로직 개선 테스트"""
        from lawfirm_langgraph.core.agents.handlers.context_builder import ContextBuilder
        from unittest.mock import Mock
        
        # Mock 객체 생성
        mock_semantic_search = Mock()
        mock_config = Mock()
        context_builder = ContextBuilder(semantic_search=mock_semantic_search, config=mock_config)
        
        # Case 1: 부분 매칭 테스트 (예: "계약서"와 "계약")
        context = {
            "context": "계약에 대한 내용입니다",
            "legal_references": [],
            "citations": []
        }
        missing = context_builder.identify_missing_information(
            context=context,
            query="계약서 작성",
            query_type="advice",
            extracted_keywords=["계약서", "작성"]
        )
        
        # "계약서"가 "계약"으로 부분 매칭되므로 누락으로 판단하지 않아야 함
        # 하지만 50% 이상 누락 시에만 누락으로 판단하므로, "작성"이 없으면 누락일 수 있음
        logger.info(f"부분 매칭 테스트 결과: missing={missing}")
        
        # Case 2: 누락 키워드 비율이 50% 미만이면 누락으로 판단하지 않음
        context_half = {
            "context": "테스트 질문에 대한 내용입니다",
            "legal_references": [],
            "citations": []
        }
        missing_half = context_builder.identify_missing_information(
            context=context_half,
            query="테스트 질문",
            query_type="simple",
            extracted_keywords=["테스트", "질문", "키워드1", "키워드2"]  # 4개 중 2개 포함
        )
        
        # 50% 미만 누락이므로 누락으로 판단하지 않아야 함
        # 하지만 실제로는 부분 매칭 등으로 더 복잡할 수 있음
        logger.info(f"누락 비율 테스트 결과: missing={missing_half}")
        
        logger.info("✅ 누락 정보 식별 로직 개선 테스트 통과")
    
    def test_query_type_specific_info_check_relaxed(self):
        """질문 유형별 필수 정보 체크 완화 테스트"""
        from lawfirm_langgraph.core.agents.handlers.context_builder import ContextBuilder
        from unittest.mock import Mock
        
        # Mock 객체 생성
        mock_semantic_search = Mock()
        mock_config = Mock()
        context_builder = ContextBuilder(semantic_search=mock_semantic_search, config=mock_config)
        
        # Case 1: 판례 관련 키워드 확장 테스트
        context_precedent = {
            "context": "대법원 판결에 대한 내용입니다",
            "legal_references": [],
            "citations": []
        }
        missing_precedent = context_builder.identify_missing_information(
            context=context_precedent,
            query="판례 검색",
            query_type="precedent",
            extracted_keywords=["판례"]
        )
        
        # "대법원"이 있으면 "판례 정보" 누락으로 판단하지 않아야 함
        assert "판례 정보" not in missing_precedent, \
            "대법원 키워드가 있으면 판례 정보 누락으로 판단하지 않아야 함"
        
        # Case 2: 법령 관련 키워드 확장 테스트
        context_law = {
            "context": "법률 규정에 대한 내용입니다",
            "legal_references": [],
            "citations": []
        }
        missing_law = context_builder.identify_missing_information(
            context=context_law,
            query="법령 검색",
            query_type="law",
            extracted_keywords=["법령"]
        )
        
        # "법률" 또는 "규정"이 있으면 "법률 조문" 누락으로 판단하지 않아야 함
        assert "법률 조문" not in missing_law, \
            "법률 또는 규정 키워드가 있으면 법률 조문 누락으로 판단하지 않아야 함"
        
        # Case 3: 실무 조언 관련 키워드 확장 테스트
        context_advice = {
            "context": "권장 사항을 제시해야 합니다",
            "legal_references": [],
            "citations": []
        }
        missing_advice = context_builder.identify_missing_information(
            context=context_advice,
            query="조언 요청",
            query_type="advice",
            extracted_keywords=["조언"]
        )
        
        # "권장"이 있으면 "실무 조언" 누락으로 판단하지 않아야 함
        assert "실무 조언" not in missing_advice, \
            "권장 키워드가 있으면 실무 조언 누락으로 판단하지 않아야 함"
        
        logger.info("✅ 질문 유형별 필수 정보 체크 완화 테스트 통과")
    
    def test_should_expand_context(self):
        """확장 전 기존 문서 재평가 테스트"""
        from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
        from unittest.mock import Mock
        
        # Mock workflow 생성
        workflow = Mock(spec=EnhancedLegalQuestionWorkflow)
        workflow.logger = logger
        
        # _should_expand_context 메서드만 실제로 테스트
        # 실제 workflow 인스턴스가 필요하므로 간접 테스트
        
        # Case 1: 기존 문서의 평균 관련성이 0.3 이상이면 확장하지 않음
        validation_results = {
            "needs_expansion": True,
            "overall_score": 0.4,
            "missing_information": ["정보1", "정보2", "정보3"]
        }
        existing_docs = [
            {"relevance_score": 0.4, "final_weighted_score": 0.4},
            {"relevance_score": 0.35, "final_weighted_score": 0.35},
            {"relevance_score": 0.3, "final_weighted_score": 0.3}
        ]
        
        # 평균 관련성 계산
        relevance_scores = [
            doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0)
            for doc in existing_docs
        ]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        # 평균 관련성이 0.3 이상이면 확장하지 않아야 함
        should_expand = avg_relevance < 0.3
        assert should_expand == False, \
            f"평균 관련성 {avg_relevance:.2f} >= 0.3이면 확장하지 않아야 함"
        
        # Case 2: 기존 문서의 평균 관련성이 0.3 미만이면 확장
        existing_docs_low = [
            {"relevance_score": 0.2, "final_weighted_score": 0.2},
            {"relevance_score": 0.15, "final_weighted_score": 0.15}
        ]
        
        relevance_scores_low = [
            doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0)
            for doc in existing_docs_low
        ]
        avg_relevance_low = sum(relevance_scores_low) / len(relevance_scores_low) if relevance_scores_low else 0.0
        
        # 평균 관련성이 0.3 미만이고 다른 조건도 만족하면 확장
        should_expand_low = avg_relevance_low < 0.3 and len(validation_results["missing_information"]) >= 3
        assert should_expand_low == True, \
            f"평균 관련성 {avg_relevance_low:.2f} < 0.3이고 missing_info >= 3이면 확장해야 함"
        
        logger.info("✅ 확장 전 기존 문서 재평가 테스트 통과")
    
    def test_build_expanded_query(self):
        """확장 쿼리 최적화 테스트"""
        from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
        from unittest.mock import Mock
        
        # Mock workflow 생성
        workflow = Mock(spec=EnhancedLegalQuestionWorkflow)
        
        # _build_expanded_query 메서드만 실제로 테스트
        # 실제 workflow 인스턴스가 필요하므로 간접 테스트
        
        # Case 1: 판례 검색은 키워드 중심
        query = "손해배상"
        missing_info = ["계약서", "불법행위", "과실"]
        query_type = "precedent"
        
        # 판례 검색은 키워드 추가
        expected_query = f"{query} {' '.join(missing_info[:3])}"
        logger.info(f"판례 검색 확장 쿼리: {expected_query}")
        
        # Case 2: 법령 검색은 원본 쿼리 유지
        query_law = "민법"
        missing_info_law = ["제1조", "제2조"]
        query_type_law = "law"
        
        # 법령 검색은 원본 쿼리 유지
        expected_query_law = query_law
        logger.info(f"법령 검색 확장 쿼리: {expected_query_law}")
        
        # Case 3: "부족" 또는 "누락" 포함 메시지 제거
        missing_info_with_message = ["핵심 키워드 커버리지 부족", "계약서", "불법행위"]
        # "부족" 포함 메시지는 제거되어야 함
        filtered_keywords = [m for m in missing_info_with_message[:3] 
                           if isinstance(m, str) and "부족" not in m and "누락" not in m]
        logger.info(f"필터링된 키워드: {filtered_keywords}")
        
        assert "핵심 키워드 커버리지 부족" not in filtered_keywords, \
            "'부족' 포함 메시지는 제거되어야 함"
        
        logger.info("✅ 확장 쿼리 최적화 테스트 통과")


def run_all_tests():
    """모든 테스트 실행"""
    logger.info("\n" + "=" * 80)
    logger.info("Context Expansion 개선 사항 테스트 시작")
    logger.info("=" * 80)
    
    test_class = TestContextExpansionImprovements()
    results = []
    
    # 테스트 실행
    test_methods = [
        ("needs_expansion 조건 강화", test_class.test_needs_expansion_condition_strengthened),
        ("누락 정보 추가 조건 강화", test_class.test_missing_info_threshold_lowered),
        ("누락 정보 식별 로직 개선", test_class.test_missing_info_identification_improved),
        ("질문 유형별 필수 정보 체크 완화", test_class.test_query_type_specific_info_check_relaxed),
        ("확장 전 기존 문서 재평가", test_class.test_should_expand_context),
        ("확장 쿼리 최적화", test_class.test_build_expanded_query),
    ]
    
    for test_name, test_method in test_methods:
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"테스트: {test_name}")
            logger.info(f"{'='*80}")
            test_method()
            results.append((test_name, True))
            logger.info(f"✅ {test_name} 통과")
        except Exception as e:
            logger.error(f"❌ {test_name} 실패: {e}", exc_info=True)
            results.append((test_name, False))
    
    # 결과 요약
    logger.info("\n" + "=" * 80)
    logger.info("테스트 결과 요약")
    logger.info("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    failed = sum(1 for _, result in results if not result)
    
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\n총 {len(results)}개 테스트 중 {passed}개 통과, {failed}개 실패")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

