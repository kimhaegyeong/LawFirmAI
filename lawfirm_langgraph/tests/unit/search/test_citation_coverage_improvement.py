# -*- coding: utf-8 -*-
"""
Citation Coverage 개선 효과 테스트

이 테스트는 Citation Coverage 개선이 검색된 문서의 Citation을 얼마나 더 많이 인용하는지 확인합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
unit_dir = script_dir.parent
tests_dir = unit_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

import logging
import pytest
from typing import List, Dict, Any

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from lawfirm_langgraph.core.generation.validators.quality_validators import AnswerValidator
except ImportError:
    try:
        from core.generation.validators.quality_validators import AnswerValidator
    except ImportError:
        AnswerValidator = None


class TestCitationCoverageImprovement:
    """Citation Coverage 개선 효과 테스트"""
    
    @pytest.fixture
    def test_cases(self):
        """테스트 케이스 데이터"""
        return [
            {
                "query": "계약 해지 사유에 대해 알려주세요",
                "retrieved_docs": [
                    {
                        "id": "doc1",
                        "content": "계약 해제는 당사자 간의 합의로 이루어질 수 있습니다.",
                        "source": "민법 제543조",
                        "type": "statute_article"
                    },
                    {
                        "id": "doc2",
                        "content": "계약 해지권은 계약 위반 시 발생합니다.",
                        "source": "민법 제544조",
                        "type": "statute_article"
                    },
                    {
                        "id": "doc3",
                        "content": "계약 해지 절차 및 효과에 대해 설명합니다.",
                        "source": "민법 제545조",
                        "type": "statute_article"
                    },
                    {
                        "id": "doc4",
                        "content": "계약 해지와 관련된 손해배상 청구권에 대한 판례가 있습니다.",
                        "source": "대법원 2020다12345",
                        "type": "precedent"
                    },
                    {
                        "id": "doc5",
                        "content": "계약 해지 사유는 법정 해지사유와 약정 해지사유로 나뉩니다.",
                        "source": "민법 제546조",
                        "type": "statute_article"
                    }
                ],
                "answer_without_citations": "계약 해제는 당사자 간의 합의로 이루어질 수 있습니다.",
                "answer_with_doc_citations": "계약 해제는 당사자 간의 합의로 이루어질 수 있습니다. [문서 1] 계약 해지권은 계약 위반 시 발생합니다. [문서 2]",
                "answer_with_full_citations": "계약 해제는 당사자 간의 합의로 이루어질 수 있습니다. [문서 1] 민법 제543조에 따르면, 계약 해지권은 계약 위반 시 발생합니다. [문서 2] 민법 제544조에 따르면, 계약 해지 절차 및 효과에 대해 설명합니다. [문서 3] 대법원 2020다12345 판례에 따르면, 계약 해지와 관련된 손해배상 청구권이 있습니다. [문서 4]",
                "expected_coverage": 0.5,  # 문서 인용 포함 시 최소 0.5 이상
                "description": "5개 문서 중 법령/판례 인용 포함 테스트"
            },
            {
                "query": "손해배상 청구 요건은 무엇인가요?",
                "retrieved_docs": [
                    {
                        "id": "doc1",
                        "content": "손해 배상 책임을 지려면 불법행위의 구성요건을 충족해야 합니다.",
                        "source": "민법 제750조",
                        "type": "statute_article"
                    },
                    {
                        "id": "doc2",
                        "content": "배상책임의 요건으로는 고의 또는 과실이 필요합니다.",
                        "source": "민법 제751조",
                        "type": "statute_article"
                    },
                    {
                        "id": "doc3",
                        "content": "손해전보를 위한 청구권이 발생합니다.",
                        "source": "민법 제752조",
                        "type": "statute_article"
                    }
                ],
                "answer_without_citations": "손해배상 청구 요건은 불법행위의 구성요건을 충족해야 합니다.",
                "answer_with_doc_citations": "손해배상 청구 요건은 불법행위의 구성요건을 충족해야 합니다. [문서 1] 고의 또는 과실이 필요합니다. [문서 2]",
                "answer_with_full_citations": "손해배상 청구 요건은 불법행위의 구성요건을 충족해야 합니다. [문서 1] 민법 제750조에 따르면, 배상책임의 요건으로는 고의 또는 과실이 필요합니다. [문서 2] 민법 제751조에 따르면, 손해전보를 위한 청구권이 발생합니다. [문서 3] 민법 제752조에 따르면",
                "expected_coverage": 0.5,  # 문서 인용 포함 시 최소 0.5 이상
                "description": "3개 문서 중 법령 인용 포함 테스트"
            }
        ]
    
    def test_citation_coverage_improvement(self, test_cases):
        """Citation Coverage 개선 효과 테스트"""
        if not AnswerValidator:
            pytest.skip("AnswerValidator를 import할 수 없습니다")
        
        results = []
        
        for case in test_cases:
            query = case["query"]
            retrieved_docs = case["retrieved_docs"]
            answer_no_citations = case["answer_without_citations"]
            answer_with_docs = case.get("answer_with_doc_citations", "")
            answer_with_full = case.get("answer_with_full_citations", "")
            expected_coverage = case["expected_coverage"]
            description = case["description"]
            
            logger.info(f"\n{'='*80}")
            logger.info(f"테스트 케이스: {description}")
            logger.info(f"질문: {query}")
            logger.info(f"검색된 문서 수: {len(retrieved_docs)}")
            
            # 컨텍스트 생성
            context_text = " ".join([doc.get("content", "") for doc in retrieved_docs])
            legal_references = [doc.get("source", "") for doc in retrieved_docs if doc.get("type") == "statute_article"]
            citations = [{"text": doc.get("source", ""), "type": doc.get("type", "")} for doc in retrieved_docs]
            
            context = {
                "context": context_text,
                "legal_references": legal_references,
                "citations": citations
            }
            
            # Citation 없이 답변 검증
            logger.info(f"\n--- Citation 없이 답변 검증 ---")
            validation_no_citations = AnswerValidator.validate_answer_uses_context(
                answer=answer_no_citations,
                context=context,
                query=query,
                retrieved_docs=retrieved_docs
            )
            
            citation_coverage_no = validation_no_citations.get("citation_coverage", 0.0)
            logger.info(f"Citation Coverage: {citation_coverage_no:.2f}")
            
            # 문서 인용만 있는 답변 검증
            if answer_with_docs:
                logger.info(f"\n--- 문서 인용만 있는 답변 검증 ---")
                validation_with_docs = AnswerValidator.validate_answer_uses_context(
                    answer=answer_with_docs,
                    context=context,
                    query=query,
                    retrieved_docs=retrieved_docs
                )
                
                citation_coverage_docs = validation_with_docs.get("citation_coverage", 0.0)
                logger.info(f"Citation Coverage: {citation_coverage_docs:.2f}")
                
                improvement_docs = citation_coverage_docs - citation_coverage_no
                logger.info(f"개선 효과: {improvement_docs:.2f} 증가")
            
            # 문서 인용 + 법령/판례 인용이 있는 답변 검증
            if answer_with_full:
                logger.info(f"\n--- 문서 인용 + 법령/판례 인용이 있는 답변 검증 ---")
                validation_with_full = AnswerValidator.validate_answer_uses_context(
                    answer=answer_with_full,
                    context=context,
                    query=query,
                    retrieved_docs=retrieved_docs
                )
                
                citation_coverage_full = validation_with_full.get("citation_coverage", 0.0)
                logger.info(f"Citation Coverage: {citation_coverage_full:.2f}")
                
                improvement_full = citation_coverage_full - citation_coverage_no
                logger.info(f"개선 효과: {improvement_full:.2f} 증가")
                
                results.append({
                    "case": description,
                    "query": query,
                    "doc_count": len(retrieved_docs),
                    "coverage_no_citations": citation_coverage_no,
                    "coverage_with_docs": citation_coverage_docs if answer_with_docs else None,
                    "coverage_with_full": citation_coverage_full,
                    "improvement": improvement_full,
                    "expected_coverage": expected_coverage,
                    "passed": citation_coverage_full >= expected_coverage
                })
                
                # 검증: Citation Coverage가 목표값 이상인지 확인
                assert citation_coverage_full >= expected_coverage, (
                    f"Citation Coverage가 목표값보다 낮습니다. "
                    f"현재: {citation_coverage_full:.2f}, 목표: {expected_coverage:.2f}"
                )
        
        # 전체 결과 요약
        logger.info(f"\n{'='*80}")
        logger.info("전체 테스트 결과 요약")
        logger.info(f"{'='*80}")
        
        total_cases = len(results)
        passed_cases = sum(1 for r in results if r.get("passed", False))
        avg_coverage = sum(r.get("coverage_with_full", 0) for r in results) / total_cases if total_cases > 0 else 0.0
        avg_improvement = sum(r.get("improvement", 0) for r in results) / total_cases if total_cases > 0 else 0.0
        
        logger.info(f"총 테스트 케이스: {total_cases}")
        logger.info(f"통과한 케이스: {passed_cases}")
        logger.info(f"평균 Citation Coverage: {avg_coverage:.2f}")
        logger.info(f"평균 개선 효과: {avg_improvement:.2f}")
        
        for result in results:
            status = "✅ 통과" if result.get("passed", False) else "❌ 실패"
            logger.info(f"{status} | {result['case']}: {result.get('coverage_with_full', 0):.2f}")
        
        # 최소 50% 이상 통과해야 함
        pass_rate = passed_cases / total_cases if total_cases > 0 else 0.0
        assert pass_rate >= 0.5, (
            f"테스트 통과율이 50% 미만입니다. 통과율: {pass_rate*100:.1f}%"
        )
        
        # 평균 Citation Coverage가 0.5 이상이어야 함 (문서 인용 포함 시)
        assert avg_coverage >= 0.5, (
            f"평균 Citation Coverage가 0.5 미만입니다. 평균: {avg_coverage:.2f}"
        )


if __name__ == "__main__":
    # 직접 실행 시
    pytest.main([__file__, "-v", "-s"])

