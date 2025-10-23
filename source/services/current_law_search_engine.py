# -*- coding: utf-8 -*-
"""
Current Law Search Engine
현행법령 전용 검색 엔진
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import re

from ..data.vector_store import LegalVectorStore
from ..data.database import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class CurrentLawSearchResult:
    """현행법령 검색 결과"""
    law_id: str
    law_name_korean: str
    law_name_abbreviation: Optional[str]
    promulgation_date: int
    promulgation_number: int
    amendment_type: str
    ministry_name: str
    law_type: str
    effective_date: int
    law_detail_link: str
    detailed_info: str
    similarity_score: float
    search_type: str  # 'vector', 'fts', 'exact', 'hybrid'
    matched_content: str  # 매칭된 구체적 내용
    article_content: Optional[str] = None  # 특정 조문 내용


class CurrentLawSearchEngine:
    """현행법령 전용 검색 엔진"""
    
    def __init__(self, 
                 db_path: str = "data/lawfirm.db",
                 vector_store: Optional[LegalVectorStore] = None):
        """현행법령 검색 엔진 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 데이터베이스 연결
        self.db_manager = DatabaseManager(db_path)
        
        # 벡터 스토어
        self.vector_store = vector_store
        
        # 검색 설정
        self.search_config = {
            "max_results": 20,
            "similarity_threshold": 0.3,
            "fts_weight": 0.4,
            "vector_weight": 0.6,
            "exact_weight": 0.8
        }
        
        # 법령명 매핑 (약칭 -> 정식명칭)
        self.law_name_mapping = {
            "민법": "민법",
            "형법": "형법", 
            "상법": "상법",
            "노동법": "근로기준법",
            "가족법": "가족관계등록법",
            "행정법": "행정절차법",
            "헌법": "대한민국헌법",
            "민사소송법": "민사소송법",
            "형사소송법": "형사소송법"
        }
    
    def search_current_laws(self, 
                           query: str, 
                           search_type: str = 'hybrid',
                           top_k: int = 10) -> List[CurrentLawSearchResult]:
        """
        현행법령 검색 실행
        
        Args:
            query: 검색 쿼리
            search_type: 검색 유형 ('vector', 'fts', 'exact', 'hybrid')
            top_k: 반환할 결과 수
            
        Returns:
            List[CurrentLawSearchResult]: 검색 결과 리스트
        """
        try:
            self.logger.info(f"Searching current laws: query='{query}', type='{search_type}'")
            
            results = []
            
            # 하이브리드 검색
            if search_type == 'hybrid':
                # 벡터 검색
                vector_results = self._search_vector(query, top_k)
                results.extend(vector_results)
                
                # FTS 검색
                fts_results = self._search_fts(query, top_k)
                results.extend(fts_results)
                
                # 정확 검색
                exact_results = self._search_exact(query, top_k)
                results.extend(exact_results)
                
            elif search_type == 'vector':
                results = self._search_vector(query, top_k)
            elif search_type == 'fts':
                results = self._search_fts(query, top_k)
            elif search_type == 'exact':
                results = self._search_exact(query, top_k)
            
            # 결과 통합 및 중복 제거
            merged_results = self._merge_and_deduplicate_results(results)
            
            # 점수 기반 정렬
            sorted_results = self._rank_results(merged_results, query)
            
            # 상위 결과 반환
            final_results = sorted_results[:top_k]
            
            self.logger.info(f"Found {len(final_results)} current law results")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error searching current laws: {e}")
            return []
    
    def search_by_law_article(self, 
                             law_name: str, 
                             article_number: str) -> Optional[CurrentLawSearchResult]:
        """
        특정 법령의 특정 조문 검색
        
        Args:
            law_name: 법령명 (예: "민법")
            article_number: 조문번호 (예: "750")
            
        Returns:
            CurrentLawSearchResult: 검색 결과
        """
        try:
            self.logger.info(f"Searching law article: {law_name} 제{article_number}조")
            
            # 법령명 정규화 및 정확한 매칭
            normalized_law_name = self._normalize_law_name(law_name)
            
            # 정확한 법령명으로 검색
            exact_results = self._search_exact_by_name(normalized_law_name, 1)
            
            if not exact_results:
                # 부분 매칭으로 시도
                exact_results = self._search_exact_by_name(law_name, 1)
            
            if exact_results:
                result = exact_results[0]
                # 조문 내용 추출
                article_content = self._extract_article_content(
                    result.detailed_info, article_number
                )
                result.article_content = article_content
                result.matched_content = article_content or result.detailed_info[:500]
                return result
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error searching law article: {e}")
            return None
    
    def _normalize_law_name(self, law_name: str) -> str:
        """법령명 정규화"""
        # 법령명 매핑
        law_mapping = {
            "민법": "민법",
            "형법": "형법", 
            "상법": "상법",
            "노동법": "근로기준법",
            "가족법": "가족관계등록법",
            "행정법": "행정절차법",
            "헌법": "대한민국헌법",
            "민사소송법": "민사소송법",
            "형사소송법": "형사소송법"
        }
        
        return law_mapping.get(law_name, law_name)
    
    def _search_exact_by_name(self, law_name: str, top_k: int) -> List[CurrentLawSearchResult]:
        """법령명으로 정확 검색"""
        try:
            # 법령명으로 정확 검색 (LIKE 검색)
            query = """
                SELECT * FROM current_laws 
                WHERE law_name_korean LIKE ? OR law_name_abbreviation LIKE ?
                ORDER BY 
                    CASE 
                        WHEN law_name_korean = ? THEN 1
                        WHEN law_name_abbreviation = ? THEN 2
                        WHEN law_name_korean LIKE ? THEN 3
                        ELSE 4
                    END,
                    effective_date DESC
                LIMIT ?
            """
            
            exact_pattern = f"%{law_name}%"
            results = self.db_manager.execute_query(query, (
                exact_pattern, exact_pattern, law_name, law_name, exact_pattern, top_k
            ))
            
            search_results = []
            for result in results:
                current_law_result = CurrentLawSearchResult(
                    law_id=result['law_id'],
                    law_name_korean=result['law_name_korean'],
                    law_name_abbreviation=result.get('law_name_abbreviation'),
                    promulgation_date=result['promulgation_date'],
                    promulgation_number=result['promulgation_number'],
                    amendment_type=result['amendment_type'],
                    ministry_name=result['ministry_name'],
                    law_type=result['law_type'],
                    effective_date=result['effective_date'],
                    law_detail_link=result['law_detail_link'],
                    detailed_info=result['detailed_info'],
                    similarity_score=1.0 if result['law_name_korean'] == law_name else 0.9,
                    search_type="exact_name",
                    matched_content=result.get('detailed_info', '')
                )
                search_results.append(current_law_result)
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Exact name search error: {e}")
            return []
    
    def search_by_law_name(self, law_name: str, top_k: int = 5) -> List[CurrentLawSearchResult]:
        """법령명으로 검색"""
        try:
            # 정확한 법령명 매칭
            results = self._search_exact(law_name, top_k)
            
            if not results:
                # 부분 매칭 시도
                results = self._search_fts(law_name, top_k)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching by law name: {e}")
            return []
    
    def search_by_ministry(self, ministry_name: str, top_k: int = 10) -> List[CurrentLawSearchResult]:
        """소관부처별 검색"""
        try:
            query = f"ministry:{ministry_name}"
            return self._search_fts(query, top_k)
            
        except Exception as e:
            self.logger.error(f"Error searching by ministry: {e}")
            return []
    
    def search_by_effective_date(self, 
                               start_date: int, 
                               end_date: int, 
                               top_k: int = 10) -> List[CurrentLawSearchResult]:
        """시행일 범위로 검색"""
        try:
            # 데이터베이스에서 시행일 범위 검색
            db_results = self.db_manager.execute_query("""
                SELECT * FROM current_laws
                WHERE effective_date BETWEEN ? AND ?
                ORDER BY effective_date DESC
                LIMIT ?
            """, (start_date, end_date, top_k))
            
            results = []
            for result in db_results:
                current_law_result = CurrentLawSearchResult(
                    law_id=result['law_id'],
                    law_name_korean=result['law_name_korean'],
                    law_name_abbreviation=result.get('law_name_abbreviation'),
                    promulgation_date=result['promulgation_date'],
                    promulgation_number=result['promulgation_number'],
                    amendment_type=result['amendment_type'],
                    ministry_name=result['ministry_name'],
                    law_type=result['law_type'],
                    effective_date=result['effective_date'],
                    law_detail_link=result['law_detail_link'],
                    detailed_info=result['detailed_info'],
                    similarity_score=1.0,
                    search_type="date_range",
                    matched_content=result['detailed_info'][:500]
                )
                results.append(current_law_result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching by effective date: {e}")
            return []
    
    def get_latest_laws(self, top_k: int = 10) -> List[CurrentLawSearchResult]:
        """최신 시행 법령 조회"""
        try:
            # 데이터베이스에서 최신 법령 조회
            db_results = self.db_manager.execute_query("""
                SELECT * FROM current_laws
                ORDER BY effective_date DESC
                LIMIT ?
            """, (top_k,))
            
            results = []
            for result in db_results:
                current_law_result = CurrentLawSearchResult(
                    law_id=result['law_id'],
                    law_name_korean=result['law_name_korean'],
                    law_name_abbreviation=result.get('law_name_abbreviation'),
                    promulgation_date=result['promulgation_date'],
                    promulgation_number=result['promulgation_number'],
                    amendment_type=result['amendment_type'],
                    ministry_name=result['ministry_name'],
                    law_type=result['law_type'],
                    effective_date=result['effective_date'],
                    law_detail_link=result['law_detail_link'],
                    detailed_info=result['detailed_info'],
                    similarity_score=1.0,
                    search_type="latest",
                    matched_content=result['detailed_info'][:500]
                )
                results.append(current_law_result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting latest laws: {e}")
            return []
    
    def _search_vector(self, query: str, top_k: int) -> List[CurrentLawSearchResult]:
        """벡터 유사도 검색"""
        try:
            if not self.vector_store:
                return []
            
            # 벡터 검색 실행
            vector_results = self.vector_store.search_current_laws(query, top_k)
            
            results = []
            for result in vector_results:
                # 데이터베이스에서 상세 정보 조회
                law_info = self.db_manager.get_current_law_by_id(result.get('law_id'))
                if not law_info:
                    continue
                
                current_law_result = CurrentLawSearchResult(
                    law_id=law_info['law_id'],
                    law_name_korean=law_info['law_name_korean'],
                    law_name_abbreviation=law_info.get('law_name_abbreviation'),
                    promulgation_date=law_info['promulgation_date'],
                    promulgation_number=law_info['promulgation_number'],
                    amendment_type=law_info['amendment_type'],
                    ministry_name=law_info['ministry_name'],
                    law_type=law_info['law_type'],
                    effective_date=law_info['effective_date'],
                    law_detail_link=law_info['law_detail_link'],
                    detailed_info=law_info['detailed_info'],
                    similarity_score=result.get('similarity_score', 0.0),
                    search_type="vector",
                    matched_content=result.get('content', '')
                )
                
                results.append(current_law_result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Vector search error: {e}")
            return []
    
    def _search_fts(self, query: str, top_k: int) -> List[CurrentLawSearchResult]:
        """FTS 키워드 검색"""
        try:
            # FTS 검색 실행
            fts_results = self.db_manager.search_current_laws_fts(query, top_k)
            
            results = []
            for result in fts_results:
                current_law_result = CurrentLawSearchResult(
                    law_id=result['law_id'],
                    law_name_korean=result['law_name_korean'],
                    law_name_abbreviation=result.get('law_name_abbreviation'),
                    promulgation_date=result['promulgation_date'],
                    promulgation_number=result['promulgation_number'],
                    amendment_type=result['amendment_type'],
                    ministry_name=result['ministry_name'],
                    law_type=result['law_type'],
                    effective_date=result['effective_date'],
                    law_detail_link=result['law_detail_link'],
                    detailed_info=result['detailed_info'],
                    similarity_score=0.8,  # FTS는 고정 점수
                    search_type="fts",
                    matched_content=result.get('detailed_info', '')
                )
                
                results.append(current_law_result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"FTS search error: {e}")
            return []
    
    def _search_exact(self, query: str, top_k: int) -> List[CurrentLawSearchResult]:
        """정확 매칭 검색"""
        try:
            # 법령명으로 정확 검색
            exact_results = self.db_manager.search_current_laws_by_keyword(query, top_k)
            
            results = []
            for result in exact_results:
                current_law_result = CurrentLawSearchResult(
                    law_id=result['law_id'],
                    law_name_korean=result['law_name_korean'],
                    law_name_abbreviation=result.get('law_name_abbreviation'),
                    promulgation_date=result['promulgation_date'],
                    promulgation_number=result['promulgation_number'],
                    amendment_type=result['amendment_type'],
                    ministry_name=result['ministry_name'],
                    law_type=result['law_type'],
                    effective_date=result['effective_date'],
                    law_detail_link=result['law_detail_link'],
                    detailed_info=result['detailed_info'],
                    similarity_score=1.0,  # 정확 매칭은 최고 점수
                    search_type="exact",
                    matched_content=result.get('detailed_info', '')
                )
                
                results.append(current_law_result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Exact search error: {e}")
            return []
    
    def _extract_article_content(self, detailed_info: str, article_number: str) -> str:
        """조문 내용 추출"""
        try:
            if not detailed_info:
                return ""
            
            # 조문 패턴 검색 (다양한 형태 지원)
            patterns = [
                f"제{article_number}조[\\s\\S]*?(?=제\\d+조|$)",
                f"제\\s*{article_number}\\s*조[\\s\\S]*?(?=제\\d+조|$)",
                f"{article_number}조[\\s\\S]*?(?=제\\d+조|$)",
                f"제{article_number}조.*?(?=제\\d+조|$)",
                f"제\\s*{article_number}\\s*조.*?(?=제\\d+조|$)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, detailed_info, re.MULTILINE | re.DOTALL)
                if match:
                    content = match.group(0).strip()
                    # 조문 내용이 너무 길면 앞부분만 반환
                    if len(content) > 1000:
                        return content[:1000] + "..."
                    return content
            
            # 조문을 찾지 못한 경우 전체 내용의 앞부분 반환
            return detailed_info[:500] + "..." if len(detailed_info) > 500 else detailed_info
            
        except Exception as e:
            self.logger.error(f"Error extracting article content: {e}")
            return detailed_info[:500] + "..." if len(detailed_info) > 500 else detailed_info
    
    def _merge_and_deduplicate_results(self, results: List[CurrentLawSearchResult]) -> List[CurrentLawSearchResult]:
        """결과 통합 및 중복 제거"""
        seen_laws = set()
        merged_results = []
        
        for result in results:
            if result.law_id not in seen_laws:
                seen_laws.add(result.law_id)
                merged_results.append(result)
            else:
                # 중복된 법령의 경우 더 높은 점수로 업데이트
                for i, existing in enumerate(merged_results):
                    if existing.law_id == result.law_id and result.similarity_score > existing.similarity_score:
                        merged_results[i] = result
                        break
        
        return merged_results
    
    def _rank_results(self, results: List[CurrentLawSearchResult], query: str) -> List[CurrentLawSearchResult]:
        """결과 점수 기반 정렬 및 가중치 적용"""
        try:
            # 현재 날짜 기준 최신성 보너스 계산
            current_date = int(datetime.now().timestamp())
            
            for result in results:
                # 최신성 보너스 (1년 이내 법령에 보너스)
                days_diff = (current_date - result.effective_date) / 86400
                recency_bonus = max(0, 1 - (days_diff / 365)) * 0.1
                
                # 검색 타입별 가중치 적용
                type_weight = {
                    'exact': 1.0,
                    'vector': 0.8,
                    'fts': 0.6,
                    'date_range': 0.9,
                    'latest': 0.95
                }.get(result.search_type, 0.5)
                
                # 최종 점수 계산
                result.similarity_score = (result.similarity_score * type_weight) + recency_bonus
                result.similarity_score = min(1.0, result.similarity_score)  # 최대 1.0으로 제한
            
            # 점수 기반 정렬 (내림차순)
            return sorted(results, key=lambda x: x.similarity_score, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error ranking results: {e}")
            return sorted(results, key=lambda x: x.similarity_score, reverse=True)
