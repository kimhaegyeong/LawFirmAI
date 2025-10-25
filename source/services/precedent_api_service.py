# -*- coding: utf-8 -*-
"""
판례 API 연동 서비스
법원 판례 API를 통해 실시간 판례 검색 및 조회 기능 제공
"""

import logging
import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

@dataclass
class PrecedentAPIRecord:
    """판례 API 검색 결과 데이터 클래스"""
    case_id: str
    case_name: str
    case_number: str
    decision_date: str
    court: str
    case_type: str
    decision_type: str
    data_source: str
    detail_url: str
    summary: str = ""
    key_point: str = ""
    relevance_score: float = 0.0

@dataclass
class PrecedentAPIDetail:
    """판례 API 상세 조회 결과 데이터 클래스"""
    case_id: str
    case_name: str
    case_number: str
    decision_date: str
    court: str
    case_type: str
    decision_type: str
    summary: str
    key_point: str
    referenced_articles: str
    referenced_precedents: str
    full_content: str

class PrecedentAPIService:
    """판례 API 연동 서비스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_oc = os.getenv("LAW_OPEN_API_OC", "test")
        self.base_url = "http://www.law.go.kr/DRF"
        self.session = None
        self.logger.info("PrecedentAPIService 초기화 완료")
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    async def search_precedents(self, 
                              query: str, 
                              search_type: int = 1,
                              display: int = 20,
                              page: int = 1,
                              court: str = "",
                              law_reference: str = "",
                              sort: str = "ddes") -> List[PrecedentAPIRecord]:
        """
        판례 목록 검색
        
        Args:
            query: 검색어
            search_type: 검색범위 (1: 판례명, 2: 본문검색)
            display: 검색 결과 개수 (max=100)
            page: 페이지 번호
            court: 법원명 (대법원, 서울고등법원 등)
            law_reference: 참조법령명 (형법, 민법 등)
            sort: 정렬옵션 (ddes: 선고일자 내림차순)
        """
        try:
            params = {
                "OC": self.api_oc,
                "target": "prec",
                "type": "JSON",
                "search": search_type,
                "query": query,
                "display": min(display, 100),
                "page": page,
                "sort": sort
            }
            
            if court:
                params["curt"] = court
            if law_reference:
                params["JO"] = law_reference
            
            url = f"{self.base_url}/lawSearch.do"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_search_results(data)
                else:
                    self.logger.error(f"판례 검색 API 오류: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"판례 검색 실패: {e}")
            return []
    
    async def get_precedent_detail(self, case_id: str) -> Optional[PrecedentAPIDetail]:
        """
        판례 상세 정보 조회
        
        Args:
            case_id: 판례 일련번호
        """
        try:
            params = {
                "OC": self.api_oc,
                "target": "prec",
                "type": "JSON",
                "ID": case_id
            }
            
            url = f"{self.base_url}/lawService.do"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_detail_result(data)
                else:
                    self.logger.error(f"판례 상세 조회 API 오류: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"판례 상세 조회 실패: {e}")
            return None
    
    def _parse_search_results(self, data: Dict[str, Any]) -> List[PrecedentAPIRecord]:
        """API 검색 결과 파싱"""
        precedents = []
        
        try:
            if "prec" in data:
                prec_data = data["prec"]
                if isinstance(prec_data, list):
                    for item in prec_data:
                        precedent = PrecedentAPIRecord(
                            case_id=str(item.get("판례일련번호", "")),
                            case_name=item.get("사건명", ""),
                            case_number=item.get("사건번호", ""),
                            decision_date=item.get("선고일자", ""),
                            court=item.get("법원명", ""),
                            case_type=item.get("사건종류명", ""),
                            decision_type=item.get("판결유형", ""),
                            data_source=item.get("데이터출처명", ""),
                            detail_url=item.get("판례상세링크", ""),
                            summary=item.get("사건명", ""),  # 기본 요약
                            relevance_score=0.8  # API 결과는 높은 관련도
                        )
                        precedents.append(precedent)
                        
        except Exception as e:
            self.logger.error(f"검색 결과 파싱 실패: {e}")
        
        return precedents
    
    def _parse_detail_result(self, data: Dict[str, Any]) -> Optional[PrecedentAPIDetail]:
        """API 상세 결과 파싱"""
        try:
            if "prec" in data:
                item = data["prec"]
                if isinstance(item, dict):
                    return PrecedentAPIDetail(
                        case_id=str(item.get("판례정보일련번호", "")),
                        case_name=item.get("사건명", ""),
                        case_number=item.get("사건번호", ""),
                        decision_date=str(item.get("선고일자", "")),
                        court=item.get("법원명", ""),
                        case_type=item.get("사건종류명", ""),
                        decision_type=item.get("판결유형", ""),
                        summary=item.get("판시사항", ""),
                        key_point=item.get("판결요지", ""),
                        referenced_articles=item.get("참조조문", ""),
                        referenced_precedents=item.get("참조판례", ""),
                        full_content=item.get("판례내용", "")
                    )
                    
        except Exception as e:
            self.logger.error(f"상세 결과 파싱 실패: {e}")
        
        return None
    
    async def search_by_law_article(self, law_name: str, article_number: str) -> List[PrecedentAPIRecord]:
        """
        특정 법령 조문에 관련된 판례 검색
        
        Args:
            law_name: 법령명 (민법, 형법 등)
            article_number: 조문번호
        """
        # 조문별 검색 키워드 생성
        search_queries = [
            f"{law_name} 제{article_number}조",
            f"{law_name} {article_number}조",
            f"제{article_number}조"
        ]
        
        all_precedents = []
        
        for query in search_queries:
            precedents = await self.search_precedents(
                query=query,
                search_type=1,  # 판례명 검색
                display=10,
                law_reference=law_name
            )
            all_precedents.extend(precedents)
        
        # 중복 제거 및 관련도 순 정렬
        unique_precedents = self._deduplicate_precedents(all_precedents)
        return unique_precedents[:5]  # 상위 5개만 반환
    
    async def search_by_keywords(self, keywords: List[str], limit: int = 5) -> List[PrecedentAPIRecord]:
        """
        키워드 리스트로 판례 검색
        
        Args:
            keywords: 검색 키워드 리스트
            limit: 최대 결과 수
        """
        all_precedents = []
        
        for keyword in keywords[:3]:  # 상위 3개 키워드만 사용
            precedents = await self.search_precedents(
                query=keyword,
                search_type=2,  # 본문 검색
                display=5
            )
            all_precedents.extend(precedents)
        
        # 중복 제거 및 관련도 순 정렬
        unique_precedents = self._deduplicate_precedents(all_precedents)
        return unique_precedents[:limit]
    
    def _deduplicate_precedents(self, precedents: List[PrecedentAPIRecord]) -> List[PrecedentAPIRecord]:
        """중복 제거 및 관련도 순 정렬"""
        unique_dict = {}
        
        for precedent in precedents:
            if precedent.case_id not in unique_dict:
                unique_dict[precedent.case_id] = precedent
            else:
                # 더 높은 관련도로 업데이트
                if precedent.relevance_score > unique_dict[precedent.case_id].relevance_score:
                    unique_dict[precedent.case_id] = precedent
        
        # 관련도 순으로 정렬
        sorted_precedents = sorted(unique_dict.values(), 
                                 key=lambda x: x.relevance_score, reverse=True)
        
        return sorted_precedents
    
    async def get_precedent_statistics(self) -> Dict[str, Any]:
        """판례 API 통계 정보 조회"""
        try:
            # 기본 검색으로 전체 통계 파악
            stats = await self.search_precedents(
                query="민법",
                display=100,
                page=1
            )
            
            return {
                "total_searchable": len(stats),
                "api_status": "active",
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"판례 API 통계 조회 실패: {e}")
            return {
                "total_searchable": 0,
                "api_status": "error",
                "error": str(e)
            }

# 테스트 함수
async def test_precedent_api():
    """판례 API 테스트"""
    print("🚀 판례 API 연동 테스트")
    print("=" * 50)
    
    async with PrecedentAPIService() as api_service:
        # 1. 기본 검색 테스트
        print("\n1. 기본 검색 테스트:")
        results = await api_service.search_precedents("민법 제750조", display=5)
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.case_name} ({result.case_number})")
            print(f"     법원: {result.court}, 날짜: {result.decision_date}")
        
        # 2. 조문별 검색 테스트
        print("\n2. 조문별 검색 테스트:")
        law_results = await api_service.search_by_law_article("민법", "750")
        for i, result in enumerate(law_results):
            print(f"  {i+1}. {result.case_name} ({result.case_number})")
        
        # 3. 키워드 검색 테스트
        print("\n3. 키워드 검색 테스트:")
        keyword_results = await api_service.search_by_keywords(["불법행위", "손해배상"], limit=3)
        for i, result in enumerate(keyword_results):
            print(f"  {i+1}. {result.case_name} ({result.case_number})")
        
        # 4. 상세 조회 테스트 (첫 번째 결과가 있는 경우)
        if results:
            print("\n4. 상세 조회 테스트:")
            detail = await api_service.get_precedent_detail(results[0].case_id)
            if detail:
                print(f"  사건명: {detail.case_name}")
                print(f"  판시사항: {detail.summary[:100]}...")
                print(f"  판결요지: {detail.key_point[:100]}...")
        
        # 5. API 통계 조회
        print("\n5. API 통계:")
        stats = await api_service.get_precedent_statistics()
        print(f"  검색 가능한 판례 수: {stats.get('total_searchable', 0)}")
        print(f"  API 상태: {stats.get('api_status', 'unknown')}")
    
    print("\n✅ 판례 API 연동 테스트 완료")

if __name__ == "__main__":
    asyncio.run(test_precedent_api())
