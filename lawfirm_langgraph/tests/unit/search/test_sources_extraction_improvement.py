# -*- coding: utf-8 -*-
"""
Sources 추출 개선 효과 테스트

이 테스트는 Sources 추출 개선이 retrieved_docs에서 Sources를 얼마나 더 잘 추출하는지 확인합니다.
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
    from lawfirm_langgraph.core.agents.handlers.answer_formatter import AnswerFormatterHandler
except ImportError:
    try:
        from core.agents.handlers.answer_formatter import AnswerFormatterHandler
    except ImportError:
        AnswerFormatterHandler = None


class TestSourcesExtractionImprovement:
    """Sources 추출 개선 효과 테스트"""
    
    @pytest.fixture
    def test_cases(self):
        """테스트 케이스 데이터"""
        return [
            {
                "answer": "계약 해제는 당사자 간의 합의로 이루어질 수 있습니다. [문서 1] 계약 해지권은 계약 위반 시 발생합니다. [문서 2]",
                "retrieved_docs": [
                    {
                        "id": "doc1",
                        "content": "계약 해제는 당사자 간의 합의로 이루어질 수 있습니다.",
                        "source": "민법 제543조",
                        "type": "statute_article",
                        "metadata": {"article_no": "543", "law_name": "민법"}
                    },
                    {
                        "id": "doc2",
                        "content": "계약 해지권은 계약 위반 시 발생합니다.",
                        "source": "민법 제544조",
                        "type": "statute_article",
                        "metadata": {"article_no": "544", "law_name": "민법"}
                    },
                    {
                        "id": "doc3",
                        "content": "계약 해지 절차 및 효과에 대해 설명합니다.",
                        "source": "민법 제545조",
                        "type": "statute_article",
                        "metadata": {"article_no": "545", "law_name": "민법"}
                    }
                ],
                "expected_sources_count": 2,  # [문서 1], [문서 2]만 사용됨
                "description": "답변에서 사용된 문서만 Sources 추출 테스트"
            },
            {
                "answer": "손해배상 청구 요건은 불법행위의 구성요건을 충족해야 합니다.",
                "retrieved_docs": [
                    {
                        "id": "doc1",
                        "content": "손해 배상 책임을 지려면 불법행위의 구성요건을 충족해야 합니다.",
                        "source": "민법 제750조",
                        "type": "statute_article",
                        "metadata": {"article_no": "750", "law_name": "민법"}
                    },
                    {
                        "id": "doc2",
                        "content": "배상책임의 요건으로는 고의 또는 과실이 필요합니다.",
                        "source": "민법 제751조",
                        "type": "statute_article",
                        "metadata": {"article_no": "751", "law_name": "민법"}
                    }
                ],
                "expected_sources_count": 0,  # 문서 인용이 없으면 추출된 번호는 0개 (fallback에서 모든 문서 처리)
                "description": "문서 인용이 없을 때 모든 문서에서 Sources 추출 테스트"
            }
        ]
    
    def test_extract_used_document_numbers(self, test_cases):
        """답변에서 사용된 문서 번호 추출 테스트"""
        if not AnswerFormatterHandler:
            pytest.skip("AnswerFormatterHandler를 import할 수 없습니다")
        
        formatter = AnswerFormatterHandler(None, None, None, None, None)
        
        results = []
        
        for case in test_cases:
            answer = case["answer"]
            expected_count = case["expected_sources_count"]
            description = case["description"]
            
            logger.info(f"\n{'='*80}")
            logger.info(f"테스트 케이스: {description}")
            logger.info(f"답변: {answer[:100]}...")
            
            # 사용된 문서 번호 추출
            used_doc_numbers = formatter._extract_used_document_numbers(answer)
            
            logger.info(f"사용된 문서 번호: {sorted(used_doc_numbers)}")
            logger.info(f"예상 Sources 수: {expected_count}")
            
            results.append({
                "case": description,
                "answer": answer,
                "used_doc_numbers": sorted(used_doc_numbers),
                "expected_count": expected_count,
                "passed": len(used_doc_numbers) == expected_count if expected_count <= 2 else len(used_doc_numbers) >= 0
            })
            
            # 검증: 문서 번호가 올바르게 추출되었는지 확인
            assert len(used_doc_numbers) == expected_count, (
                f"사용된 문서 번호 수가 예상과 다릅니다. "
                f"추출된 번호: {sorted(used_doc_numbers)}, 예상: {expected_count}"
            )
        
        # 전체 결과 요약
        logger.info(f"\n{'='*80}")
        logger.info("전체 테스트 결과 요약")
        logger.info(f"{'='*80}")
        
        total_cases = len(results)
        passed_cases = sum(1 for r in results if r.get("passed", False))
        
        logger.info(f"총 테스트 케이스: {total_cases}")
        logger.info(f"통과한 케이스: {passed_cases}")
        
        for result in results:
            status = "✅ 통과" if result.get("passed", False) else "❌ 실패"
            logger.info(f"{status} | {result['case']}: {len(result['used_doc_numbers'])}개 문서 번호 추출")
        
        # 최소 50% 이상 통과해야 함
        pass_rate = passed_cases / total_cases if total_cases > 0 else 0.0
        assert pass_rate >= 0.5, (
            f"테스트 통과율이 50% 미만입니다. 통과율: {pass_rate*100:.1f}%"
        )
    
    def test_sources_extraction_from_used_documents(self, test_cases):
        """사용된 문서에서만 Sources 추출 테스트"""
        if not AnswerFormatterHandler:
            pytest.skip("AnswerFormatterHandler를 import할 수 없습니다")
        
        formatter = AnswerFormatterHandler(None, None, None, None, None)
        
        # 첫 번째 테스트 케이스 사용 (문서 인용이 있는 경우)
        case = test_cases[0]
        answer = case["answer"]
        retrieved_docs = case["retrieved_docs"]
        
        logger.info(f"\n{'='*80}")
        logger.info("사용된 문서에서만 Sources 추출 테스트")
        logger.info(f"답변: {answer}")
        logger.info(f"검색된 문서 수: {len(retrieved_docs)}")
        
        # 사용된 문서 번호 추출
        used_doc_numbers = formatter._extract_used_document_numbers(answer)
        logger.info(f"사용된 문서 번호: {sorted(used_doc_numbers)}")
        
        # 사용된 문서만 필터링
        used_docs = []
        for doc_index, doc in enumerate(retrieved_docs, 1):
            if doc_index in used_doc_numbers:
                used_docs.append(doc)
        
        logger.info(f"사용된 문서 수: {len(used_docs)}")
        
        # 검증: 사용된 문서만 필터링되었는지 확인
        assert len(used_docs) == len(used_doc_numbers), (
            f"사용된 문서 수가 일치하지 않습니다. "
            f"필터링된 문서: {len(used_docs)}, 사용된 번호: {len(used_doc_numbers)}"
        )
        
        # 각 문서가 올바른 번호인지 확인
        for i, doc_index in enumerate(sorted(used_doc_numbers)):
            assert doc_index <= len(retrieved_docs), (
                f"문서 번호 {doc_index}가 retrieved_docs 범위를 벗어났습니다. "
                f"총 문서 수: {len(retrieved_docs)}"
            )
        
        logger.info("✅ 사용된 문서만 올바르게 필터링되었습니다")


if __name__ == "__main__":
    # 직접 실행 시
    pytest.main([__file__, "-v", "-s"])

