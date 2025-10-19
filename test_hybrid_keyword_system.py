#!/usr/bin/env python3
"""
하이브리드 키워드 관리 시스템 테스트
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from source.services.domain_specific_extractor import DomainSpecificExtractor, LegalDomain
from source.services.hybrid_keyword_manager import ExpansionStrategy

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_hybrid_keyword_system():
    """하이브리드 키워드 시스템 테스트"""
    print("=" * 60)
    print("하이브리드 키워드 관리 시스템 테스트")
    print("=" * 60)
    
    try:
        # 1. 하이브리드 전략으로 초기화
        print("\n1. 하이브리드 키워드 추출기 초기화...")
        extractor = DomainSpecificExtractor(
            data_dir="data",
            cache_dir="data/cache",
            min_keyword_threshold=20,
            expansion_strategy=ExpansionStrategy.HYBRID
        )
        print("✓ 초기화 완료")
        
        # 2. 도메인별 키워드 통계 조회
        print("\n2. 도메인별 키워드 통계 조회...")
        stats = await extractor.get_domain_statistics()
        
        for domain, stat in stats.items():
            print(f"  {domain}: {stat['keyword_count']}개 키워드 (소스: {stat['source']}, 신뢰도: {stat['confidence']:.2f})")
        
        # 3. 테스트 텍스트로 도메인 분류 테스트
        test_texts = [
            "계약서를 작성할 때 손해배상 조항을 포함해야 합니다.",
            "살인죄의 구성요건과 형의 종류에 대해 알고 싶습니다.",
            "이혼 시 양육권과 양육비에 대한 법적 절차는 어떻게 되나요?",
            "회사의 주주총회에서 이사 선임 절차를 진행하고 있습니다.",
            "근로계약서에 최저임금과 근로시간을 명시해야 합니다."
        ]
        
        print("\n3. 도메인 분류 테스트...")
        for i, text in enumerate(test_texts, 1):
            print(f"\n  테스트 {i}: {text}")
            
            # 주요 도메인 식별
            primary_domain, confidence = await extractor.get_primary_domain(text)
            print(f"    주요 도메인: {primary_domain.value} (신뢰도: {confidence:.3f})")
            
            # 도메인별 점수
            domain_scores = await extractor.classify_domain_by_keywords(text)
            top_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"    상위 3개 도메인:")
            for domain, score in top_domains:
                print(f"      {domain.value}: {score:.3f}")
        
        # 4. 키워드 확장 테스트
        print("\n4. 키워드 확장 테스트...")
        
        # 민사법 도메인 키워드 확장 테스트
        civil_keywords = await extractor._get_domain_keywords(LegalDomain.CIVIL_LAW)
        print(f"  민사법 현재 키워드 수: {len(civil_keywords)}")
        
        # 확장이 필요한지 확인하고 실행
        expanded = await extractor.expand_domain_keywords_if_needed(LegalDomain.CIVIL_LAW)
        if expanded:
            print("  ✓ 민사법 키워드 확장 완료")
        else:
            print("  - 민사법 키워드 확장 불필요")
        
        # 확장 후 키워드 수 확인
        civil_keywords_after = await extractor._get_domain_keywords(LegalDomain.CIVIL_LAW)
        print(f"  민사법 확장 후 키워드 수: {len(civil_keywords_after)}")
        
        # 5. 캐시 통계 조회
        print("\n5. 캐시 통계 조회...")
        cache_stats = extractor.get_cache_statistics()
        print(f"  메모리 캐시: {cache_stats['memory_cache_count']}개")
        print(f"  파일 캐시: {cache_stats['file_cache_count']}개")
        print(f"  총 캐시 크기: {cache_stats['total_cache_size_bytes']} bytes")
        
        # 6. 용어 추출 테스트
        print("\n6. 용어 추출 테스트...")
        test_text = "계약서 작성 시 손해배상 조항과 위약금에 대한 명시가 필요하며, 소유권 이전 등기 절차를 거쳐야 합니다."
        
        # 도메인별 용어 추출
        all_domain_terms = extractor.extract_all_domain_terms(test_text)
        print(f"  추출된 도메인별 용어:")
        for domain, terms in all_domain_terms.items():
            if terms:
                print(f"    {domain.value}: {', '.join(terms[:5])}{'...' if len(terms) > 5 else ''}")
        
        # 용어 추출 강화
        enhanced_result = await extractor.enhance_term_extraction(test_text, [])
        print(f"  강화된 용어 추출 결과:")
        print(f"    주요 도메인: {enhanced_result['primary_domain'].value}")
        print(f"    도메인 신뢰도: {enhanced_result['domain_confidence']:.3f}")
        print(f"    총 용어 수: {len(enhanced_result['enhanced_terms'])}")
        print(f"    상위 10개 용어: {', '.join(enhanced_result['enhanced_terms'][:10])}")
        
        print("\n" + "=" * 60)
        print("하이브리드 키워드 시스템 테스트 완료!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

async def test_expansion_strategies():
    """확장 전략별 테스트"""
    print("\n" + "=" * 60)
    print("확장 전략별 테스트")
    print("=" * 60)
    
    strategies = [
        (ExpansionStrategy.DATABASE_ONLY, "데이터베이스만 사용"),
        (ExpansionStrategy.AI_ONLY, "AI만 사용"),
        (ExpansionStrategy.HYBRID, "하이브리드 (권장)"),
        (ExpansionStrategy.FALLBACK, "폴백 방식")
    ]
    
    for strategy, description in strategies:
        print(f"\n{description} 테스트...")
        
        try:
            extractor = DomainSpecificExtractor(
                data_dir="data",
                cache_dir="data/cache",
                min_keyword_threshold=10,  # 낮은 임계값으로 설정
                expansion_strategy=strategy
            )
            
            # 민사법 도메인 테스트
            keywords = await extractor._get_domain_keywords(LegalDomain.CIVIL_LAW)
            print(f"  민사법 키워드 수: {len(keywords)}")
            
            # 상위 5개 키워드 출력
            print(f"  상위 5개 키워드: {', '.join(keywords[:5])}")
            
        except Exception as e:
            print(f"  오류: {e}")

if __name__ == "__main__":
    # 메인 테스트 실행
    asyncio.run(test_hybrid_keyword_system())
    
    # 확장 전략별 테스트 실행
    asyncio.run(test_expansion_strategies())
