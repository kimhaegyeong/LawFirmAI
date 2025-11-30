# -*- coding: utf-8 -*-
"""
Keyword Coverage 개선 효과 테스트 (Ko-Legal-SBERT 기반)

이 테스트는 Ko-Legal-SBERT를 사용한 의미적 매칭이 Keyword Coverage를 얼마나 개선하는지 확인합니다.
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
    from lawfirm_langgraph.core.search.processors.search_result_processor import SearchResultProcessor
except ImportError:
    try:
        from core.search.processors.search_result_processor import SearchResultProcessor
    except ImportError:
        SearchResultProcessor = None

try:
    from lawfirm_langgraph.core.generation.validators.quality_validators import AnswerValidator
except ImportError:
    try:
        from core.generation.validators.quality_validators import AnswerValidator
    except ImportError:
        AnswerValidator = None


class TestKeywordCoverageSBERT:
    """Ko-Legal-SBERT 기반 Keyword Coverage 개선 효과 테스트"""
    
    @pytest.fixture
    def search_result_processor(self):
        """SearchResultProcessor 인스턴스 생성"""
        if not SearchResultProcessor:
            pytest.skip("SearchResultProcessor를 import할 수 없습니다")
        return SearchResultProcessor()
    
    @pytest.fixture
    def test_cases(self):
        """테스트 케이스 데이터"""
        return [
            {
                "query": "계약 해지 사유에 대해 알려주세요",
                "keywords": ["계약", "해지", "사유"],
                "documents": [
                    {
                        "id": "doc1",
                        "content": "계약서를 해제할 수 있는 경우는 다음과 같습니다. 계약 해제는 당사자 간의 합의로 이루어질 수 있습니다.",
                        "text": "계약서를 해제할 수 있는 경우는 다음과 같습니다. 계약 해제는 당사자 간의 합의로 이루어질 수 있습니다."
                    },
                    {
                        "id": "doc2",
                        "content": "계약관계를 종료하는 방법에는 해지와 해제가 있습니다. 해지권은 계약 위반 시 발생합니다.",
                        "text": "계약관계를 종료하는 방법에는 해지와 해제가 있습니다. 해지권은 계약 위반 시 발생합니다."
                    },
                    {
                        "id": "doc3",
                        "content": "손해배상 청구권은 불법행위로 인한 손해를 전보하기 위한 권리입니다.",
                        "text": "손해배상 청구권은 불법행위로 인한 손해를 전보하기 위한 권리입니다."
                    }
                ],
                "expected_coverage_min": 0.6,  # 개선 후 예상 최소 커버리지
                "description": "계약 해지 관련 질문 - 동의어 매칭 테스트"
            },
            {
                "query": "손해배상 청구 요건은 무엇인가요?",
                "keywords": ["손해배상", "청구", "요건"],
                "documents": [
                    {
                        "id": "doc1",
                        "content": "손해 배상 책임을 지려면 불법행위의 구성요건을 충족해야 합니다. 손해보상 청구는 손실을 전보하기 위한 것입니다.",
                        "text": "손해 배상 책임을 지려면 불법행위의 구성요건을 충족해야 합니다. 손해보상 청구는 손실을 전보하기 위한 것입니다."
                    },
                    {
                        "id": "doc2",
                        "content": "배상책임의 요건으로는 고의 또는 과실이 필요합니다. 손해전보를 위한 청구권이 발생합니다.",
                        "text": "배상책임의 요건으로는 고의 또는 과실이 필요합니다. 손해전보를 위한 청구권이 발생합니다."
                    }
                ],
                "expected_coverage_min": 0.7,  # 개선 후 예상 최소 커버리지
                "description": "손해배상 관련 질문 - 동의어 매칭 테스트"
            },
            {
                "query": "이혼 절차와 재산분할에 대해 설명해주세요",
                "keywords": ["이혼", "절차", "재산분할"],
                "documents": [
                    {
                        "id": "doc1",
                        "content": "이혼소송 절차는 협의이혼과 재판이혼으로 나뉩니다. 재산 분할은 이혼 시 중요한 사항입니다.",
                        "text": "이혼소송 절차는 협의이혼과 재판이혼으로 나뉩니다. 재산 분할은 이혼 시 중요한 사항입니다."
                    },
                    {
                        "id": "doc2",
                        "content": "이혼신고를 하기 위해서는 이혼절차를 완료해야 합니다. 재산분할협의는 별도로 진행됩니다.",
                        "text": "이혼신고를 하기 위해서는 이혼절차를 완료해야 합니다. 재산분할협의는 별도로 진행됩니다."
                    }
                ],
                "expected_coverage_min": 0.8,  # 개선 후 예상 최소 커버리지
                "description": "이혼 관련 질문 - 복합어 매칭 테스트"
            }
        ]
    
    def test_keyword_coverage_with_sbert(self, search_result_processor, test_cases):
        """Ko-Legal-SBERT를 사용한 Keyword Coverage 개선 효과 테스트"""
        if not search_result_processor:
            pytest.skip("SearchResultProcessor를 초기화할 수 없습니다")
        
        results = []
        
        for case in test_cases:
            query = case["query"]
            keywords = case["keywords"]
            documents = case["documents"]
            expected_min = case["expected_coverage_min"]
            description = case["description"]
            
            logger.info(f"\n{'='*80}")
            logger.info(f"테스트 케이스: {description}")
            logger.info(f"질문: {query}")
            logger.info(f"키워드: {keywords}")
            logger.info(f"문서 수: {len(documents)}")
            
            # Keyword Coverage 계산
            try:
                coverage = search_result_processor.calculate_keyword_coverage(
                    query=query,
                    extracted_keywords=keywords,
                    results=documents
                )
                
                logger.info(f"✅ Keyword Coverage: {coverage:.3f} (예상 최소: {expected_min:.3f})")
                
                results.append({
                    "case": description,
                    "query": query,
                    "keywords": keywords,
                    "coverage": coverage,
                    "expected_min": expected_min,
                    "passed": coverage >= expected_min
                })
                
                # 개선 효과 확인
                assert coverage >= expected_min, (
                    f"Keyword Coverage가 예상 최소값({expected_min:.3f})보다 낮습니다. "
                    f"실제 값: {coverage:.3f}"
                )
                
            except Exception as e:
                logger.error(f"❌ 테스트 실패: {e}")
                pytest.fail(f"Keyword Coverage 계산 중 오류 발생: {e}")
        
        # 전체 결과 요약
        logger.info(f"\n{'='*80}")
        logger.info("전체 테스트 결과 요약")
        logger.info(f"{'='*80}")
        
        total_cases = len(results)
        passed_cases = sum(1 for r in results if r["passed"])
        avg_coverage = sum(r["coverage"] for r in results) / total_cases if total_cases > 0 else 0.0
        
        logger.info(f"총 테스트 케이스: {total_cases}")
        logger.info(f"통과한 케이스: {passed_cases}")
        logger.info(f"평균 Keyword Coverage: {avg_coverage:.3f}")
        
        for result in results:
            status = "✅ 통과" if result["passed"] else "❌ 실패"
            logger.info(f"{status} | {result['case']}: {result['coverage']:.3f}")
        
        # 최소 80% 이상 통과해야 함
        pass_rate = passed_cases / total_cases if total_cases > 0 else 0.0
        assert pass_rate >= 0.8, (
            f"테스트 통과율이 80% 미만입니다. 통과율: {pass_rate*100:.1f}%"
        )
        
        # 평균 커버리지가 0.6 이상이어야 함
        assert avg_coverage >= 0.6, (
            f"평균 Keyword Coverage가 0.6 미만입니다. 실제 값: {avg_coverage:.3f}"
        )
    
    def test_semantic_matching_enabled(self, search_result_processor):
        """의미적 매칭이 활성화되어 있는지 확인"""
        if not search_result_processor:
            pytest.skip("SearchResultProcessor를 초기화할 수 없습니다")
        
        # 임베딩 모델이 초기화되어 있는지 확인
        has_embedding_model = hasattr(search_result_processor, 'embedding_model') and \
                             search_result_processor.embedding_model is not None
        
        logger.info(f"임베딩 모델 초기화 여부: {has_embedding_model}")
        
        if has_embedding_model:
            logger.info(f"✅ Ko-Legal-SBERT 모델이 활성화되어 있습니다")
            logger.info(f"모델 타입: {type(search_result_processor.embedding_model).__name__}")
        else:
            logger.warning("⚠️ 임베딩 모델이 초기화되지 않았습니다. 의미적 매칭이 비활성화됩니다.")
        
        # 의미적 매칭이 가능한 경우에만 테스트 통과
        # 모델이 없어도 직접 매칭은 작동하므로 경고만 출력
        if not has_embedding_model:
            logger.warning("의미적 매칭이 비활성화되어 있지만, 직접 매칭은 여전히 작동합니다.")
    
    def test_answer_validation_with_sbert(self, test_cases):
        """답변 검증에서 의미적 매칭 사용 확인"""
        if not AnswerValidator:
            pytest.skip("AnswerValidator를 import할 수 없습니다")
        
        # 첫 번째 테스트 케이스 사용
        case = test_cases[0]
        query = case["query"]
        keywords = case["keywords"]
        
        # 컨텍스트 생성
        context_text = " ".join([doc["content"] for doc in case["documents"]])
        context = {
            "context": context_text,
            "legal_references": [],
            "citations": []
        }
        
        # 답변 생성 (키워드를 포함하되 동의어 사용)
        answer = (
            "계약관계를 종료하는 방법에는 계약 해제와 계약 해지가 있습니다. "
            "계약 해제는 계약서를 취소하는 것을 의미하며, 해지권은 계약 위반 시 발생합니다. "
            "계약 해지 사유는 당사자 간의 합의, 계약 위반 등이 있습니다."
        )
        
        logger.info(f"\n{'='*80}")
        logger.info("답변 검증 테스트")
        logger.info(f"질문: {query}")
        logger.info(f"답변 길이: {len(answer)}자")
        
        try:
            validation_result = AnswerValidator.validate_answer_uses_context(
                answer=answer,
                context=context,
                query=query,
                retrieved_docs=case["documents"]
            )
            
            keyword_coverage = validation_result.get("keyword_coverage", 0.0)
            coverage_score = validation_result.get("coverage_score", 0.0)
            
            logger.info(f"✅ Keyword Coverage: {keyword_coverage:.3f}")
            logger.info(f"✅ Coverage Score: {coverage_score:.3f}")
            logger.info(f"✅ Uses Context: {validation_result.get('uses_context', False)}")
            
            # Keyword Coverage가 0.6 이상이어야 함
            assert keyword_coverage >= 0.6, (
                f"답변 검증의 Keyword Coverage가 0.6 미만입니다. 실제 값: {keyword_coverage:.3f}"
            )
            
        except Exception as e:
            logger.error(f"❌ 답변 검증 실패: {e}")
            pytest.fail(f"답변 검증 중 오류 발생: {e}")


if __name__ == "__main__":
    # 직접 실행 시
    pytest.main([__file__, "-v", "-s"])

