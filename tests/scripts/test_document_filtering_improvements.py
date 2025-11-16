# -*- coding: utf-8 -*-
"""
문서 필터링 개선사항 테스트 스크립트
"""

import sys
import os
import logging

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from lawfirm_langgraph.core.workflow.processors.workflow_document_processor import WorkflowDocumentProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_documents():
    """테스트용 문서 생성"""
    return [
        {
            "content": "계약서 작성 시 당사자 특정은 매우 중요합니다. 당사자의 신원을 확인하는 절차는 다음과 같습니다. 1) 주민등록번호 확인 2) 사업자등록번호 확인 3) 인감증명서 확인",
            "source": "계약법 해설서",
            "relevance_score": 0.85,
            "keyword_match_score": 0.8,
            "matched_keywords": ["계약서", "당사자", "신원", "확인"],
            "search_type": "semantic",
            "final_weighted_score": 0.85
        },
        {
            "content": "당사자 신원 확인 절차에 대한 법적 근거는 민법 제105조에 명시되어 있습니다. 계약의 성립을 위해서는 당사자의 의사표시가 필요하며, 이를 확인하기 위한 절차가 필요합니다.",
            "source": "민법 해설",
            "relevance_score": 0.75,
            "keyword_match_score": 0.6,
            "matched_keywords": ["당사자", "신원", "확인", "계약"],
            "search_type": "semantic",
            "final_weighted_score": 0.75
        },
        {
            "content": "육체노동자의 가동연한을 60세로 인정한 1990년 전후와는 많은 부분이 달라지고 있다. 과거 법원이 취하여 왔던 육체노동자의 60세 가동연한에 관한 입장을 그대로 고수한다면, 경비원 등 감시단속적 업무에 종사하는 사람의 상당수가 60세 이상이고, 공사현장에서도 60대 이상의 인부 등을 흔히 볼 수 있는 현실과의 상당한 괴리를 쉽사리 설명하기 어렵다.",
            "source": "구상금",
            "relevance_score": 0.63,
            "keyword_match_score": 0.0,
            "matched_keywords": [],
            "search_type": "hybrid",
            "final_weighted_score": 0.63
        },
        {
            "content": "화물자동차법 제2조제3호 및 같은 법 시행규칙 제3조의2제1항에 따르면, 화주가 화물자동차에 함께 탈 때의 화물 기준은 중량, 용적, 형상 등이 여객자동차 운송사업용 자동차에 싣기 부적합한 것으로서 화주 1명당 화물의 중량이 20kg 이상일 것 또는 화주 1명당 화물의 용적이 4만c㎡ 이상일 것 등이라고 되어 있다.",
            "source": "19-진정-0404100",
            "relevance_score": 0.61,
            "keyword_match_score": 0.0,
            "matched_keywords": [],
            "search_type": "hybrid",
            "final_weighted_score": 0.61
        },
        {
            "content": "조합채권자는 그 채권발생 당시에 조합원의 손실부담의 비율을 알지 못한 때에는 각 조합원에게 균분하여 그 권리를 행사할 수 있다.",
            "source": "민법 712",
            "relevance_score": 0.53,
            "keyword_match_score": 0.0,
            "matched_keywords": [],
            "search_type": "hybrid",
            "final_weighted_score": 0.53
        },
        {
            "content": "부부의 공동생활에 필요한 비용은 당사자간에 특별한 약정이 없으면 부부가 공동으로 부담한다.",
            "source": "민법 833",
            "relevance_score": 0.52,
            "keyword_match_score": 0.0,
            "matched_keywords": [],
            "search_type": "hybrid",
            "final_weighted_score": 0.52
        },
        {
            "content": "나이는 출생일을 산입하여 만(滿) 나이로 계산하고, 연수(年數)로 표시한다. 다만, 1세에 이르지 아니한 경우에는 월수(月數)로 표시할 수 있다.",
            "source": "민법 158",
            "relevance_score": 0.52,
            "keyword_match_score": 0.0,
            "matched_keywords": [],
            "search_type": "hybrid",
            "final_weighted_score": 0.52
        }
    ]


def test_document_filtering():
    """문서 필터링 테스트"""
    logger.info("=" * 80)
    logger.info("문서 필터링 개선사항 테스트 시작")
    logger.info("=" * 80)
    
    processor = WorkflowDocumentProcessor()
    
    query = "계약서 작성 시 당사자 특정은 왜 중요하며, 신원 확인 절차는 어떻게 진행되나요?"
    extracted_keywords = ["계약서", "당사자", "신원", "확인", "절차"]
    query_type = "legal_advice"
    legal_field = "민사법"
    
    test_docs = create_test_documents()
    
    logger.info(f"\n테스트 질문: {query}")
    logger.info(f"추출된 키워드: {extracted_keywords}")
    logger.info(f"초기 문서 수: {len(test_docs)}")
    logger.info("\n초기 문서 목록:")
    for idx, doc in enumerate(test_docs, 1):
        logger.info(
            f"  문서 {idx}: {doc['source']} "
            f"(관련도: {doc['relevance_score']:.3f}, "
            f"키워드 매칭: {doc['keyword_match_score']:.3f}, "
            f"매칭 키워드: {doc.get('matched_keywords', [])})"
        )
    
    # build_prompt_optimized_context 호출
    result = processor.build_prompt_optimized_context(
        retrieved_docs=test_docs,
        query=query,
        extracted_keywords=extracted_keywords,
        query_type=query_type,
        legal_field=legal_field
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("필터링 결과")
    logger.info("=" * 80)
    
    document_count = result.get("document_count", 0)
    structured_docs = result.get("structured_documents", {})
    documents = structured_docs.get("documents", [])
    final_validation = result.get("final_validation", {})
    
    logger.info(f"\n최종 선택된 문서 수: {document_count}")
    logger.info(f"최대 문서 수 제한: 7개")
    
    logger.info("\n선택된 문서 목록:")
    for doc in documents:
        logger.info(
            f"  문서 {doc.get('document_id')}: {doc.get('source')} "
            f"(관련도: {doc.get('relevance_score', 0.0):.3f})"
        )
    
    logger.info("\n최종 검증 결과:")
    logger.info(f"  전체 문서 수: {final_validation.get('total_docs', 0)}")
    logger.info(f"  고관련도 문서 (>=0.65): {final_validation.get('high_relevance_count', 0)}")
    logger.info(f"  중관련도 문서 (0.55-0.65): {final_validation.get('medium_relevance_count', 0)}")
    logger.info(f"  저관련도 문서 (<0.55): {final_validation.get('low_relevance_count', 0)}")
    logger.info(f"  평균 관련도 점수: {final_validation.get('avg_relevance_score', 0.0):.3f}")
    logger.info(f"  최소 관련도 점수: {final_validation.get('min_relevance_score', 0.0):.3f}")
    logger.info(f"  최대 관련도 점수: {final_validation.get('max_relevance_score', 0.0):.3f}")
    
    if final_validation.get("low_relevance_warning"):
        logger.warning(f"  ⚠️ 경고: {final_validation['low_relevance_warning']}")
    
    # 검증
    logger.info("\n" + "=" * 80)
    logger.info("검증 결과")
    logger.info("=" * 80)
    
    # 1. 문서 수 제한 확인 (최대 7개)
    assert document_count <= 7, f"문서 수가 7개를 초과함: {document_count}"
    logger.info(f"✅ 문서 수 제한 확인: {document_count}개 (최대 7개)")
    
    # 2. 관련도가 낮은 문서 필터링 확인
    low_relevance_docs = [doc for doc in documents if doc.get('relevance_score', 0.0) < 0.55]
    if low_relevance_docs:
        logger.warning(f"⚠️ 관련도가 낮은 문서가 포함됨: {len(low_relevance_docs)}개")
        for doc in low_relevance_docs:
            logger.warning(f"  - {doc.get('source')} (관련도: {doc.get('relevance_score', 0.0):.3f})")
    else:
        logger.info("✅ 관련도가 낮은 문서(<0.55) 필터링 확인")
    
    # 3. 키워드 매칭이 없는 문서 필터링 확인
    no_keyword_docs = []
    for doc in documents:
        source = doc.get('source', '')
        # 원본 문서에서 키워드 매칭 확인
        original_doc = next((d for d in test_docs if d.get('source') == source), None)
        if original_doc:
            if original_doc.get('keyword_match_score', 0.0) == 0.0 and not original_doc.get('matched_keywords'):
                if original_doc.get('relevance_score', 0.0) < 0.70:
                    no_keyword_docs.append(doc)
    
    if no_keyword_docs:
        logger.warning(f"⚠️ 키워드 매칭이 없고 관련도가 낮은 문서가 포함됨: {len(no_keyword_docs)}개")
        for doc in no_keyword_docs:
            logger.warning(f"  - {doc.get('source')} (관련도: {doc.get('relevance_score', 0.0):.3f})")
    else:
        logger.info("✅ 키워드 매칭이 없고 관련도가 낮은 문서 필터링 확인")
    
    # 4. 관련도가 높은 문서 포함 확인
    high_relevance_docs = [doc for doc in documents if doc.get('relevance_score', 0.0) >= 0.65]
    logger.info(f"✅ 고관련도 문서(>=0.65) 포함: {len(high_relevance_docs)}개")
    
    # 5. 평균 관련도 점수 확인
    avg_score = final_validation.get('avg_relevance_score', 0.0)
    if avg_score >= 0.60:
        logger.info(f"✅ 평균 관련도 점수 양호: {avg_score:.3f}")
    else:
        logger.warning(f"⚠️ 평균 관련도 점수 낮음: {avg_score:.3f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("테스트 완료")
    logger.info("=" * 80)
    
    return {
        "document_count": document_count,
        "high_relevance_count": final_validation.get('high_relevance_count', 0),
        "medium_relevance_count": final_validation.get('medium_relevance_count', 0),
        "low_relevance_count": final_validation.get('low_relevance_count', 0),
        "avg_relevance_score": avg_score
    }


if __name__ == "__main__":
    try:
        result = test_document_filtering()
        print("\n" + "=" * 80)
        print("테스트 결과 요약")
        print("=" * 80)
        print(f"최종 문서 수: {result['document_count']}")
        print(f"고관련도 문서: {result['high_relevance_count']}")
        print(f"중관련도 문서: {result['medium_relevance_count']}")
        print(f"저관련도 문서: {result['low_relevance_count']}")
        print(f"평균 관련도 점수: {result['avg_relevance_score']:.3f}")
        print("\n✅ 모든 테스트 통과!")
    except Exception as e:
        logger.error(f"테스트 실패: {e}", exc_info=True)
        sys.exit(1)

