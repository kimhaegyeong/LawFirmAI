# -*- coding: utf-8 -*-
"""
문서 활용도 개선 효과 테스트

이 테스트는 문서 활용도 개선이 검색된 문서를 얼마나 더 많이 사용하는지 확인합니다.
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
import re
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

try:
    from lawfirm_langgraph.core.services.unified_prompt_manager import UnifiedPromptManager
except ImportError:
    try:
        from core.services.unified_prompt_manager import UnifiedPromptManager
    except ImportError:
        UnifiedPromptManager = None


class TestDocumentUsageImprovement:
    """문서 활용도 개선 효과 테스트"""
    
    @pytest.fixture
    def test_cases(self):
        """테스트 케이스 데이터"""
        return [
            {
                "query": "계약 해지 사유에 대해 알려주세요",
                "retrieved_docs": [
                    {"id": "doc1", "content": "계약 해제는 당사자 간의 합의로 이루어질 수 있습니다.", "source": "민법 제543조"},
                    {"id": "doc2", "content": "계약 해지권은 계약 위반 시 발생합니다.", "source": "민법 제544조"},
                    {"id": "doc3", "content": "계약 해지 절차 및 효과에 대해 설명합니다.", "source": "민법 제545조"},
                    {"id": "doc4", "content": "계약 해지와 관련된 손해배상 청구권에 대한 판례가 있습니다.", "source": "대법원 2020다12345"},
                    {"id": "doc5", "content": "계약 해지 사유는 법정 해지사유와 약정 해지사유로 나뉩니다.", "source": "민법 제546조"}
                ],
                "answer_with_1_doc": "계약 해제는 당사자 간의 합의로 이루어질 수 있습니다. [문서 1]",
                "answer_with_3_docs": "계약 해지 사유는 다음과 같습니다. [문서 1] 계약 해제는 당사자 간의 합의로 이루어질 수 있습니다. [문서 2] 계약 해지권은 계약 위반 시 발생합니다. [문서 3] 계약 해지 절차 및 효과에 대해 설명합니다.",
                "min_required": 3,
                "description": "5개 문서 중 최소 3개 인용 테스트"
            },
            {
                "query": "손해배상 청구 요건은 무엇인가요?",
                "retrieved_docs": [
                    {"id": "doc1", "content": "손해 배상 책임을 지려면 불법행위의 구성요건을 충족해야 합니다.", "source": "민법 제750조"},
                    {"id": "doc2", "content": "배상책임의 요건으로는 고의 또는 과실이 필요합니다.", "source": "민법 제751조"},
                    {"id": "doc3", "content": "손해전보를 위한 청구권이 발생합니다.", "source": "민법 제752조"}
                ],
                "answer_with_1_doc": "손해배상 청구 요건은 불법행위의 구성요건을 충족해야 합니다. [문서 1]",
                "answer_with_2_docs": "손해배상 청구 요건은 다음과 같습니다. [문서 1] 불법행위의 구성요건을 충족해야 합니다. [문서 2] 고의 또는 과실이 필요합니다.",
                "min_required": 2,
                "description": "3개 문서 중 최소 2개 인용 테스트"
            }
        ]
    
    def test_document_usage_validation(self, test_cases):
        """문서 활용도 검증 테스트"""
        if not AnswerValidator:
            pytest.skip("AnswerValidator를 import할 수 없습니다")
        
        results = []
        
        for case in test_cases:
            query = case["query"]
            retrieved_docs = case["retrieved_docs"]
            answer_1_doc = case["answer_with_1_doc"]
            answer_multiple_docs = case.get("answer_with_3_docs") or case.get("answer_with_2_docs", "")
            min_required = case["min_required"]
            description = case["description"]
            
            logger.info(f"\n{'='*80}")
            logger.info(f"테스트 케이스: {description}")
            logger.info(f"질문: {query}")
            logger.info(f"검색된 문서 수: {len(retrieved_docs)}")
            logger.info(f"최소 인용 요구: {min_required}개")
            
            # 컨텍스트 생성
            context_text = " ".join([doc.get("content", "") for doc in retrieved_docs])
            context = {
                "context": context_text,
                "legal_references": [],
                "citations": []
            }
            
            # 1개 문서만 사용한 답변 검증
            logger.info(f"\n--- 1개 문서만 사용한 답변 검증 ---")
            validation_1_doc = AnswerValidator.validate_answer_uses_context(
                answer=answer_1_doc,
                context=context,
                query=query,
                retrieved_docs=retrieved_docs
            )
            
            doc_usage_rate_1 = validation_1_doc.get("document_usage_rate", 0.0)
            used_doc_count_1 = validation_1_doc.get("used_doc_count", 0)
            has_sufficient_1 = validation_1_doc.get("document_usage_sufficient", False)
            
            logger.info(f"문서 활용도: {doc_usage_rate_1:.1%} ({used_doc_count_1}/{len(retrieved_docs)})")
            logger.info(f"충분한 문서 활용: {has_sufficient_1}")
            
            # 여러 문서를 사용한 답변 검증
            if answer_multiple_docs:
                logger.info(f"\n--- 여러 문서를 사용한 답변 검증 ---")
                validation_multiple = AnswerValidator.validate_answer_uses_context(
                    answer=answer_multiple_docs,
                    context=context,
                    query=query,
                    retrieved_docs=retrieved_docs
                )
                
                doc_usage_rate_multiple = validation_multiple.get("document_usage_rate", 0.0)
                used_doc_count_multiple = validation_multiple.get("used_doc_count", 0)
                has_sufficient_multiple = validation_multiple.get("document_usage_sufficient", False)
                
                logger.info(f"문서 활용도: {doc_usage_rate_multiple:.1%} ({used_doc_count_multiple}/{len(retrieved_docs)})")
                logger.info(f"충분한 문서 활용: {has_sufficient_multiple}")
                
                # 개선 효과 확인
                improvement = doc_usage_rate_multiple - doc_usage_rate_1
                logger.info(f"개선 효과: {improvement:.1%} 증가")
                
                results.append({
                    "case": description,
                    "query": query,
                    "doc_count": len(retrieved_docs),
                    "min_required": min_required,
                    "usage_1_doc": doc_usage_rate_1,
                    "usage_multiple": doc_usage_rate_multiple,
                    "improvement": improvement,
                    "sufficient_1": has_sufficient_1,
                    "sufficient_multiple": has_sufficient_multiple,
                    "passed": has_sufficient_multiple and used_doc_count_multiple >= min_required
                })
                
                # 검증: 여러 문서를 사용한 경우 최소 요구사항 충족
                assert has_sufficient_multiple, (
                    f"여러 문서를 사용한 답변이 최소 요구사항을 충족하지 않습니다. "
                    f"사용된 문서: {used_doc_count_multiple}, 최소 요구: {min_required}"
                )
                assert used_doc_count_multiple >= min_required, (
                    f"사용된 문서 수가 최소 요구사항보다 적습니다. "
                    f"사용된 문서: {used_doc_count_multiple}, 최소 요구: {min_required}"
                )
            else:
                results.append({
                    "case": description,
                    "query": query,
                    "doc_count": len(retrieved_docs),
                    "min_required": min_required,
                    "usage_1_doc": doc_usage_rate_1,
                    "usage_multiple": None,
                    "improvement": None,
                    "sufficient_1": has_sufficient_1,
                    "sufficient_multiple": None,
                    "passed": False
                })
        
        # 전체 결과 요약
        logger.info(f"\n{'='*80}")
        logger.info("전체 테스트 결과 요약")
        logger.info(f"{'='*80}")
        
        total_cases = len(results)
        passed_cases = sum(1 for r in results if r.get("passed", False))
        avg_improvement = sum(r.get("improvement", 0) for r in results if r.get("improvement") is not None) / max(1, sum(1 for r in results if r.get("improvement") is not None))
        
        logger.info(f"총 테스트 케이스: {total_cases}")
        logger.info(f"통과한 케이스: {passed_cases}")
        if avg_improvement > 0:
            logger.info(f"평균 개선 효과: {avg_improvement:.1%}")
        
        for result in results:
            status = "✅ 통과" if result.get("passed", False) else "❌ 실패"
            logger.info(f"{status} | {result['case']}: {result.get('usage_multiple', 'N/A'):.1%}")
        
        # 최소 50% 이상 통과해야 함
        pass_rate = passed_cases / total_cases if total_cases > 0 else 0.0
        assert pass_rate >= 0.5, (
            f"테스트 통과율이 50% 미만입니다. 통과율: {pass_rate*100:.1f}%"
        )
    
    def test_prompt_minimum_citations(self, test_cases):
        """프롬프트에 최소 인용 수가 포함되는지 확인"""
        if not UnifiedPromptManager:
            pytest.skip("UnifiedPromptManager를 import할 수 없습니다")
        
        prompt_manager = UnifiedPromptManager()
        
        # 첫 번째 테스트 케이스 사용
        case = test_cases[0]
        documents = case["retrieved_docs"]
        
        logger.info(f"\n{'='*80}")
        logger.info("프롬프트 최소 인용 수 테스트")
        logger.info(f"문서 수: {len(documents)}")
        
        # 문서 섹션 생성
        documents_section = prompt_manager._build_documents_section(documents, case["query"])
        
        logger.info(f"\n생성된 문서 섹션:")
        logger.info(documents_section[:500] + "..." if len(documents_section) > 500 else documents_section)
        
        # 실제 생성된 프롬프트 확인
        logger.info(f"\n생성된 문서 섹션 (처음 500자):\n{documents_section[:500]}")
        
        # "최소 N개 이상" 패턴을 우선적으로 찾기 (가장 정확한 패턴)
        min_pattern = re.search(r'최소\s*(\d+)\s*개\s*이상', documents_section)
        
        assert min_pattern is not None, (
            f"프롬프트에 '최소 N개 이상' 패턴이 포함되어 있지 않습니다. "
            f"문서 섹션: {documents_section[:500]}"
        )
        
        min_citations_in_prompt = int(min_pattern.group(1))
        expected_min = case["min_required"]
        
        logger.info(f"프롬프트의 최소 인용 수: {min_citations_in_prompt}")
        logger.info(f"예상 최소 인용 수: {expected_min}")
        
        # 문서 수에 따라 올바른 범위인지 확인
        doc_count = len(documents)
        if doc_count >= 5:
            assert min_citations_in_prompt == 3, (
                f"5개 이상 문서의 경우 최소 인용 수는 3이어야 합니다. "
                f"현재: {min_citations_in_prompt}, 문서 수: {doc_count}"
            )
        elif doc_count >= 3:
            assert min_citations_in_prompt == 2, (
                f"3-4개 문서의 경우 최소 인용 수는 2이어야 합니다. "
                f"현재: {min_citations_in_prompt}, 문서 수: {doc_count}"
            )
        else:
            assert min_citations_in_prompt == 1, (
                f"1-2개 문서의 경우 최소 인용 수는 1이어야 합니다. "
                f"현재: {min_citations_in_prompt}, 문서 수: {doc_count}"
            )
        
        logger.info("✅ 프롬프트에 최소 인용 수가 올바르게 포함되어 있습니다")


if __name__ == "__main__":
    # 직접 실행 시
    pytest.main([__file__, "-v", "-s"])

