# -*- coding: utf-8 -*-
"""
법적 근거 제시 시스템 사용 예시
구현된 법적 근거 제시 기능들의 사용법을 보여주는 예시
"""

import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from source.services.legal_basis_integration_service import LegalBasisIntegrationService
from source.services.answer_structure_enhancer import QuestionType


def demonstrate_legal_basis_system():
    """법적 근거 제시 시스템 사용 예시"""
    
    print("=" * 80)
    print("법적 근거 제시 시스템 사용 예시")
    print("=" * 80)
    
    # 통합 서비스 초기화
    legal_service = LegalBasisIntegrationService()
    
    # 예시 쿼리와 답변
    examples = [
        {
            "query": "계약 해지 시 손해배상 범위는 어떻게 되나요?",
            "answer": """
            계약 해지 시 손해배상 범위는 민법 제543조와 제544조에 따라 결정됩니다.
            
            제543조에 따르면, 계약의 해지로 인한 손해배상은 계약 이행으로 얻을 수 있었던 이익의 상실을 의미합니다.
            제544조에서는 해지로 인한 손해배상의 범위를 제한하고 있습니다.
            
            대법원 2023다12345 판례에서는 계약 해지 시 손해배상 범위를 구체적으로 명시하고 있습니다.
            """
        },
        {
            "query": "부동산 매매계약에서 하자담보책임 관련 판례를 찾아주세요",
            "answer": """
            부동산 매매계약에서 하자담보책임과 관련된 주요 판례는 다음과 같습니다:
            
            1. 대법원 2022다56789 판례: 부동산의 하자로 인한 손해배상 범위
            2. 서울고등법원 2023나12345 판례: 하자 발견 시 계약 해지 요건
            
            민법 제580조와 제581조가 하자담보책임의 법적 근거가 됩니다.
            """
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n[예시 {i}]")
        print("-" * 40)
        print(f"질문: {example['query']}")
        print(f"원본 답변: {example['answer'][:100]}...")
        
        # 법적 근거 강화 처리
        result = legal_service.process_query_with_legal_basis(
            example['query'], 
            example['answer']
        )
        
        print(f"\n처리 결과:")
        print(f"- 질문 유형: {result['question_type']}")
        print(f"- 신뢰도: {result['confidence']:.2f}")
        print(f"- 법적 근거 검증: {'통과' if result['is_legally_sound'] else '실패'}")
        
        # 법적 근거 요약
        legal_basis = result['legal_basis']
        citations = legal_basis.get('citations', {})
        citation_count = citations.get('citation_count', 0)
        
        print(f"- 발견된 법적 인용: {citation_count}개")
        
        if citation_count > 0:
            summary = citations.get('legal_basis_summary', {})
            laws = summary.get('laws_referenced', [])
            precedents = summary.get('precedents_referenced', [])
            
            if laws:
                print(f"- 관련 법령: {len(laws)}개")
                for law in laws[:3]:  # 최대 3개만 표시
                    print(f"  * {law['formatted']}")
            
            if precedents:
                print(f"- 관련 판례: {len(precedents)}개")
                for precedent in precedents[:3]:  # 최대 3개만 표시
                    print(f"  * {precedent['formatted']}")
        
        # 강화된 답변 미리보기
        enhanced_answer = result['enhanced_answer']
        print(f"\n강화된 답변 미리보기:")
        print(f"{enhanced_answer[:200]}...")
        
        print("\n" + "=" * 80)


def demonstrate_citation_enhancement():
    """법적 인용 강화 기능 예시"""
    
    print("\n" + "=" * 80)
    print("법적 인용 강화 기능 예시")
    print("=" * 80)
    
    from source.services.legal_citation_enhancer import LegalCitationEnhancer
    
    citation_enhancer = LegalCitationEnhancer()
    
    # 예시 텍스트
    sample_text = """
    계약 해지 시 손해배상 범위는 민법 제543조와 제544조에 따라 결정됩니다.
    대법원 2023다12345 판례에서는 이러한 원칙을 명확히 하고 있습니다.
    또한 근로기준법 제28조에 따른 임금 체불 시 대응 방법도 고려해야 합니다.
    """
    
    print(f"원본 텍스트:")
    print(sample_text)
    
    # 인용 강화 처리
    result = citation_enhancer.enhance_text_with_citations(sample_text)
    
    print(f"\n처리 결과:")
    print(f"- 발견된 인용: {result['citation_count']}개")
    print(f"- 인용 유형별 통계: {result['citation_stats']}")
    
    print(f"\n강화된 텍스트:")
    print(result['enhanced_text'])
    
    print(f"\n법적 근거 요약:")
    summary = result['legal_basis_summary']
    if summary.get('laws_referenced'):
        print("관련 법령:")
        for law in summary['laws_referenced']:
            print(f"  - {law['formatted']} (신뢰도: {law['confidence']:.2f})")
    
    if summary.get('precedents_referenced'):
        print("관련 판례:")
        for precedent in summary['precedents_referenced']:
            print(f"  - {precedent['formatted']} (신뢰도: {precedent['confidence']:.2f})")


def demonstrate_api_usage():
    """API 사용 예시"""
    
    print("\n" + "=" * 80)
    print("API 사용 예시")
    print("=" * 80)
    
    print("""
    법적 근거 제시 시스템은 다음과 같은 API 엔드포인트를 제공합니다:
    
    1. 법적 근거 강화 엔드포인트
       POST /api/legal-basis/enhance
       
       요청 예시:
       {
         "query": "계약 해지 시 손해배상 범위는 어떻게 되나요?",
         "answer": "계약 해지 시 손해배상 범위는 민법 제543조에 따라...",
         "question_type": "law_inquiry",
         "include_validation": true,
         "include_citations": true
       }
    
    2. 법적 인용 검증 엔드포인트
       POST /api/legal-citations/validate
       
       요청 예시:
       {
         "text": "민법 제543조와 대법원 2023다12345 판례에 따르면...",
         "include_validation": true
       }
    
    3. 법적 근거 통계 조회 엔드포인트
       GET /api/legal-basis/statistics?days=30
       
    4. 기존 답변 강화 엔드포인트
       POST /api/legal-basis/enhance-existing
       
       요청 예시:
       {
         "query": "질문 내용",
         "answer": "기존 답변 내용"
       }
    
    모든 엔드포인트는 JSON 형태로 응답하며, 법적 근거 검증 결과와
    신뢰도 점수를 포함합니다.
    """)


def main():
    """메인 함수"""
    try:
        # 법적 근거 제시 시스템 예시
        demonstrate_legal_basis_system()
        
        # 법적 인용 강화 기능 예시
        demonstrate_citation_enhancement()
        
        # API 사용 예시
        demonstrate_api_usage()
        
        print("\n" + "=" * 80)
        print("예시 실행 완료!")
        print("=" * 80)
        
    except Exception as e:
        print(f"예시 실행 중 오류 발생: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
