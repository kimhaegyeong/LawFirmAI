#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
하이브리드 키워드 매니저 캐시 초기화 및 테스트 스크립트
"""

import sys
import os
import asyncio
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.services.hybrid_keyword_manager import HybridKeywordManager
from source.services.domain_specific_extractor import DomainSpecificExtractor

async def test_updated_keywords():
    """업데이트된 키워드로 테스트"""
    try:
        print("하이브리드 키워드 매니저 캐시 초기화 및 테스트")
        print("="*60)
        
        # 하이브리드 키워드 매니저 초기화
        keyword_manager = HybridKeywordManager()
        
        # 캐시 초기화 (캐시 디렉토리 삭제)
        import shutil
        cache_dir = Path("data/cache")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print("키워드 캐시 디렉토리 삭제 완료")
        else:
            print("캐시 디렉토리가 존재하지 않음")
        
        # 모든 도메인 키워드 조회 (강제 새로고침)
        print("\n도메인별 키워드 수 (캐시 새로고침 후):")
        all_keywords = await keyword_manager.get_all_domain_keywords(force_refresh=True)
        
        total_keywords = 0
        for domain, keywords in all_keywords.items():
            count = len(keywords)
            total_keywords += count
            print(f"  {domain}: {count:,}개")
        
        print(f"\n총 키워드 수: {total_keywords:,}개")
        
        # 도메인별 특화 추출기 테스트
        print("\n도메인별 특화 추출기 테스트:")
        extractor = DomainSpecificExtractor()
        
        # 테스트 텍스트들
        test_texts = {
            "민사법": "계약서에 명시된 손해배상 조항에 따라 채권자는 채무자에게 손해배상을 청구할 수 있습니다.",
            "형사법": "살인죄는 형법상 가장 중한 범죄 중 하나이며, 절도죄와 사기죄도 중요한 범죄입니다.",
            "가족법": "이혼 시 양육권이 중요한 문제가 되며, 혼인무효와 혼인취소는 다른 개념입니다.",
            "지적재산권법": "특허권 침해 시 정지청구를 할 수 있으며, 상표권과 저작권도 보호받습니다.",
            "형사소송법": "수사기관의 수사 절차가 중요하며, 변호인 접견권이 보장됩니다."
        }
        
        for domain_name, text in test_texts.items():
            print(f"\n{domain_name} 테스트:")
            print(f"  텍스트: {text}")
            
            # 주 도메인 분류
            primary_domain, confidence = await extractor.get_primary_domain(text)
            print(f"  주 도메인: {primary_domain.value} (신뢰도: {confidence:.2f})")
            
            # 도메인별 점수
            domain_scores = await extractor.get_enhanced_domain_classification(text)
            print(f"  도메인별 점수:")
            for domain, score in sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"    {domain.value}: {score:.3f}")
        
        print("\n테스트 완료!")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 함수"""
    asyncio.run(test_updated_keywords())

if __name__ == "__main__":
    main()