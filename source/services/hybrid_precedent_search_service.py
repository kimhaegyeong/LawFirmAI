# -*- coding: utf-8 -*-
"""
하이브리드 판례 검색 서비스
로컬 DB와 API를 결합한 하이브리드 판례 검색 시스템
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from .dynamic_precedent_search_service import DynamicPrecedentSearchService, PrecedentResult
from .precedent_api_service import PrecedentAPIService, PrecedentAPIRecord

logger = logging.getLogger(__name__)

@dataclass
class HybridPrecedentResult:
    """하이브리드 판례 검색 결과"""
    case_number: str
    case_name: str
    decision_date: str
    court: str
    summary: str
    key_point: str
    relevance_score: float
    field: str
    source: str  # "database" 또는 "api"
    detail_url: str = ""

class HybridPrecedentSearchService:
    """하이브리드 판례 검색 서비스"""
    
    def __init__(self, db_manager, enable_api: bool = True):
        self.db_manager = db_manager
        self.enable_api = enable_api
        self.logger = logging.getLogger(__name__)
        
        # 로컬 DB 검색 서비스
        self.db_service = DynamicPrecedentSearchService(db_manager)
        
        # API 검색 서비스
        self.api_service = PrecedentAPIService() if enable_api else None
        
        # 캐시 설정
        self.cache_duration = timedelta(hours=1)  # 1시간 캐시
        self.search_cache = {}
        
        self.logger.info("HybridPrecedentSearchService 초기화 완료")
    
    async def get_related_precedents(self, 
                                   law_name: str, 
                                   article_number: str, 
                                   content: str = "", 
                                   limit: int = 3,
                                   use_api: bool = True) -> List[HybridPrecedentResult]:
        """
        하이브리드 방식으로 관련 판례 검색
        
        Args:
            law_name: 법령명
            article_number: 조문번호
            content: 조문 내용
            limit: 최대 결과 수
            use_api: API 사용 여부
        """
        self.logger.info(f"하이브리드 판례 검색: {law_name} 제{article_number}조")
        
        # 1. 로컬 DB에서 먼저 검색
        db_results = self.db_service.get_related_precedents(
            law_name, article_number, content, limit
        )
        
        hybrid_results = []
        
        # DB 결과를 하이브리드 형식으로 변환
        for result in db_results:
            hybrid_result = HybridPrecedentResult(
                case_number=result.case_number,
                case_name=result.case_name,
                decision_date=result.decision_date,
                court=result.court,
                summary=result.summary,
                key_point=result.key_point,
                relevance_score=result.relevance_score,
                field=result.field,
                source="database"
            )
            hybrid_results.append(hybrid_result)
        
        # 2. DB 결과가 부족하고 API가 활성화된 경우 API 검색
        if len(db_results) < limit and use_api and self.enable_api and self.api_service:
            try:
                api_results = await self._search_api_precedents(
                    law_name, article_number, content, limit - len(db_results)
                )
                
                # API 결과를 하이브리드 형식으로 변환
                for result in api_results:
                    hybrid_result = HybridPrecedentResult(
                        case_number=result.case_number,
                        case_name=result.case_name,
                        decision_date=result.decision_date,
                        court=result.court,
                        summary=result.summary,
                        key_point=result.key_point,
                        relevance_score=result.relevance_score,
                        field=result.case_type,
                        source="api",
                        detail_url=result.detail_url
                    )
                    hybrid_results.append(hybrid_result)
                
                # API 결과를 로컬 DB에 캐시 저장
                await self._cache_api_results(api_results)
                
            except Exception as e:
                self.logger.error(f"API 판례 검색 실패: {e}")
        
        # 3. 중복 제거 및 관련도 순 정렬
        unique_results = self._deduplicate_and_rank(hybrid_results)
        
        return unique_results[:limit]
    
    async def _search_api_precedents(self, 
                                   law_name: str, 
                                   article_number: str, 
                                   content: str, 
                                   limit: int) -> List[PrecedentAPIRecord]:
        """API를 통한 판례 검색"""
        try:
            async with self.api_service as api_service:
                # 1. 조문별 직접 검색
                law_results = await api_service.search_by_law_article(law_name, article_number)
                
                # 2. 조문 내용에서 키워드 추출하여 검색
                keywords = self._extract_keywords_from_content(content)
                keyword_results = []
                if keywords:
                    keyword_results = await api_service.search_by_keywords(keywords, limit=3)
                
                # 3. 결과 통합
                all_results = law_results + keyword_results
                
                # 중복 제거
                unique_results = self._deduplicate_api_results(all_results)
                
                return unique_results[:limit]
                
        except Exception as e:
            self.logger.error(f"API 판례 검색 중 오류: {e}")
            return []
    
    def _extract_keywords_from_content(self, content: str) -> List[str]:
        """조문 내용에서 핵심 키워드 추출"""
        if not content:
            return []
        
        keywords = []
        
        # 법률 용어 패턴
        legal_patterns = [
            r'[가-힣]{2,6}(?:권|의무|책임|효력|효과|요건|조건)',
            r'[가-힣]{2,6}(?:행위|처분|계약|합의|약정)',
            r'[가-힣]{2,6}(?:손해|배상|보상|지급|반환)',
            r'[가-힣]{2,6}(?:고의|과실|위법|불법)',
            r'[가-힣]{2,6}(?:소유권|점유권|사용권|수익권)',
            r'[가-힣]{2,6}(?:범죄|처벌|형량|벌금)',
            r'[가-힣]{2,6}(?:상속|유언|상속인)',
            r'[가-힣]{2,6}(?:이혼|혼인|부부|가족)'
        ]
        
        import re
        for pattern in legal_patterns:
            matches = re.findall(pattern, content)
            keywords.extend(matches)
        
        # 중복 제거 및 상위 5개 반환
        return list(dict.fromkeys(keywords))[:5]
    
    def _deduplicate_api_results(self, results: List[PrecedentAPIRecord]) -> List[PrecedentAPIRecord]:
        """API 결과 중복 제거"""
        unique_dict = {}
        
        for result in results:
            if result.case_id not in unique_dict:
                unique_dict[result.case_id] = result
            else:
                # 더 높은 관련도로 업데이트
                if result.relevance_score > unique_dict[result.case_id].relevance_score:
                    unique_dict[result.case_id] = result
        
        return list(unique_dict.values())
    
    def _deduplicate_and_rank(self, results: List[HybridPrecedentResult]) -> List[HybridPrecedentResult]:
        """하이브리드 결과 중복 제거 및 관련도 순 정렬"""
        unique_dict = {}
        
        for result in results:
            # 사건번호로 중복 판단
            if result.case_number not in unique_dict:
                unique_dict[result.case_number] = result
            else:
                # DB 결과를 우선시 (더 신뢰할 수 있음)
                if result.source == "database" and unique_dict[result.case_number].source == "api":
                    unique_dict[result.case_number] = result
                elif result.relevance_score > unique_dict[result.case_number].relevance_score:
                    unique_dict[result.case_number] = result
        
        # 관련도 순으로 정렬
        sorted_results = sorted(unique_dict.values(), 
                              key=lambda x: x.relevance_score, reverse=True)
        
        return sorted_results
    
    async def _cache_api_results(self, api_results: List[PrecedentAPIRecord]):
        """API 결과를 로컬 DB에 캐시 저장"""
        try:
            for result in api_results:
                # 이미 존재하는지 확인
                existing = self.db_manager.execute_query(
                    "SELECT case_id FROM precedent_cases WHERE case_id = ?",
                    (result.case_id,)
                )
                
                if not existing:
                    # 새 판례 데이터 삽입
                    self.db_manager.execute_update(
                        """
                        INSERT OR REPLACE INTO precedent_cases 
                        (case_id, category, case_name, case_number, decision_date, field, court, detail_url, full_text, searchable_text)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            result.case_id,
                            "API",
                            result.case_name,
                            result.case_number,
                            result.decision_date,
                            result.case_type,
                            result.court,
                            result.detail_url,
                            result.summary,
                            f"{result.case_name} {result.case_number} {result.case_type}"
                        )
                    )
                    
                    self.logger.info(f"API 판례 캐시 저장: {result.case_name}")
                    
        except Exception as e:
            self.logger.error(f"API 결과 캐시 저장 실패: {e}")
    
    async def get_precedent_statistics(self) -> Dict[str, Any]:
        """하이브리드 판례 통계 조회"""
        stats = {
            "local_db": self.db_service.get_precedent_statistics(),
            "api_enabled": self.enable_api,
            "cache_size": len(self.search_cache)
        }
        
        if self.enable_api and self.api_service:
            try:
                async with self.api_service as api_service:
                    api_stats = await api_service.get_precedent_statistics()
                    stats["api"] = api_stats
            except Exception as e:
                stats["api"] = {"error": str(e)}
        
        return stats
    
    async def search_recent_precedents(self, days: int = 30, limit: int = 5) -> List[HybridPrecedentResult]:
        """최근 판례 검색 (하이브리드)"""
        # 로컬 DB에서 최근 판례 검색
        db_results = self.db_service.get_recent_precedents(days, limit)
        
        hybrid_results = []
        for result in db_results:
            hybrid_result = HybridPrecedentResult(
                case_number=result.case_number,
                case_name=result.case_name,
                decision_date=result.decision_date,
                court=result.court,
                summary=result.summary,
                key_point=result.key_point,
                relevance_score=result.relevance_score,
                field=result.field,
                source="database"
            )
            hybrid_results.append(hybrid_result)
        
        return hybrid_results
    
    def clear_cache(self):
        """검색 캐시 초기화"""
        self.search_cache.clear()
        self.logger.info("하이브리드 검색 캐시 초기화 완료")

# 테스트 함수
async def test_hybrid_precedent_search():
    """하이브리드 판례 검색 테스트"""
    print("🚀 하이브리드 판례 검색 테스트")
    print("=" * 50)
    
    from source.data.database import DatabaseManager
    
    db_manager = DatabaseManager(db_path="data/lawfirm.db")
    hybrid_service = HybridPrecedentSearchService(db_manager, enable_api=True)
    
    # 1. 하이브리드 검색 테스트
    print("\n1. 하이브리드 검색 테스트:")
    results = await hybrid_service.get_related_precedents(
        law_name="민법",
        article_number="750",
        content="고의 또는 과실로 인한 위법행위로 타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있다.",
        limit=5
    )
    
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.case_name} ({result.case_number})")
        print(f"     소스: {result.source}, 관련도: {result.relevance_score:.2f}")
        print(f"     요약: {result.summary[:100]}...")
        print()
    
    # 2. 통계 조회
    print("\n2. 하이브리드 통계:")
    stats = await hybrid_service.get_precedent_statistics()
    print(f"  로컬 DB 판례 수: {stats['local_db'].get('total_count', 0)}")
    print(f"  API 활성화: {stats['api_enabled']}")
    print(f"  캐시 크기: {stats['cache_size']}")
    
    print("\n✅ 하이브리드 판례 검색 테스트 완료")

if __name__ == "__main__":
    asyncio.run(test_hybrid_precedent_search())
