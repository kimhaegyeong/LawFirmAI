# -*- coding: utf-8 -*-
"""
Legal Term Search Service
법령용어 검색 및 추천 서비스
"""

import re
import logging
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
from ..data.database import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class LegalTermMatch:
    """법령용어 매칭 결과"""
    term_id: str
    term_name: str
    match_type: str  # 'exact', 'partial', 'related'
    confidence: float
    context: Optional[str] = None


class LegalTermSearchService:
    """법령용어 검색 및 추천 서비스"""
    
    def __init__(self, db_manager: DatabaseManager):
        """법령용어 검색 서비스 초기화"""
        self.db_manager = db_manager
        self.cache = {}  # 캐싱을 위한 메모리 저장소
        self.term_cache = {}  # 용어별 캐시
        self._initialize_term_cache()
        
        logger.info("LegalTermSearchService 초기화 완료")
    
    def _initialize_term_cache(self):
        """용어 캐시 초기화"""
        try:
            # 전체 법령용어를 메모리에 로드 (성능 최적화)
            all_terms = self.db_manager.execute_query(
                "SELECT 법령용어ID, 법령용어명, 비고 FROM base_legal_term_lists LIMIT 1000"
            )
            
            for term in all_terms:
                term_name = term['법령용어명']
                self.term_cache[term_name] = term
            
            logger.info(f"법령용어 캐시 초기화 완료: {len(self.term_cache)}개 용어")
            
        except Exception as e:
            logger.warning(f"용어 캐시 초기화 실패: {e}")
    
    def search_legal_terms(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """법령용어 검색"""
        try:
            # 캐시 확인
            cache_key = f"search:{query}:{limit}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # 데이터베이스에서 검색
            results = self.db_manager.search_base_legal_terms(query, limit)
            
            # 캐시 저장
            self.cache[cache_key] = results
            
            logger.debug(f"법령용어 검색 완료: '{query}' -> {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"법령용어 검색 실패: {e}")
            return []
    
    def get_term_suggestions(self, partial_term: str, limit: int = 5) -> List[str]:
        """법령용어 자동완성"""
        try:
            if len(partial_term) < 2:
                return []
            
            # 캐시 확인
            cache_key = f"suggest:{partial_term}:{limit}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # 데이터베이스에서 자동완성 검색
            suggestions = self.db_manager.get_legal_term_suggestions(partial_term, limit)
            
            # 캐시 저장
            self.cache[cache_key] = suggestions
            
            logger.debug(f"법령용어 자동완성 완료: '{partial_term}' -> {len(suggestions)}개 제안")
            return suggestions
            
        except Exception as e:
            logger.error(f"법령용어 자동완성 실패: {e}")
            return []
    
    def get_related_terms(self, term: str, limit: int = 5) -> List[Dict[str, Any]]:
        """관련 법령용어 추천"""
        try:
            # 캐시 확인
            cache_key = f"related:{term}:{limit}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # 데이터베이스에서 관련 용어 검색
            related_terms = self.db_manager.get_related_legal_terms(term, limit)
            
            # 캐시 저장
            self.cache[cache_key] = related_terms
            
            logger.debug(f"관련 법령용어 검색 완료: '{term}' -> {len(related_terms)}개 관련 용어")
            return related_terms
            
        except Exception as e:
            logger.error(f"관련 법령용어 검색 실패: {e}")
            return []
    
    def extract_legal_terms_from_text(self, text: str) -> List[Dict[str, Any]]:
        """텍스트에서 법령용어 추출"""
        try:
            if not text or len(text.strip()) < 2:
                return []
            
            # 캐시 확인
            cache_key = f"extract:{hash(text)}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # 텍스트를 단어로 분리
            words = self._tokenize_korean_text(text)
            found_terms = []
            
            # 각 단어에 대해 법령용어 검색
            for word in words:
                if len(word) >= 2:  # 최소 2글자 이상
                    terms = self.search_legal_terms(word, limit=3)
                    found_terms.extend(terms)
            
            # 중복 제거 및 정렬
            unique_terms = self._deduplicate_terms(found_terms)
            
            # 캐시 저장
            self.cache[cache_key] = unique_terms
            
            logger.debug(f"텍스트에서 법령용어 추출 완료: {len(unique_terms)}개 용어")
            return unique_terms
            
        except Exception as e:
            logger.error(f"텍스트에서 법령용어 추출 실패: {e}")
            return []
    
    def _tokenize_korean_text(self, text: str) -> List[str]:
        """한국어 텍스트 토큰화"""
        # 간단한 한국어 토큰화 (공백, 구두점 기준)
        tokens = re.findall(r'[가-힣]+', text)
        
        # 추가로 의미있는 단어 추출 (2글자 이상)
        meaningful_tokens = []
        for token in tokens:
            if len(token) >= 2:
                meaningful_tokens.append(token)
                # 긴 단어에서 부분 단어도 추출
                if len(token) > 3:
                    for i in range(len(token) - 1):
                        substring = token[i:i+2]
                        if substring not in meaningful_tokens:
                            meaningful_tokens.append(substring)
        
        return meaningful_tokens
    
    def _deduplicate_terms(self, terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """중복 용어 제거"""
        seen_ids = set()
        unique_terms = []
        
        for term in terms:
            term_id = term.get('법령용어ID', '')
            if term_id and term_id not in seen_ids:
                seen_ids.add(term_id)
                unique_terms.append(term)
        
        return unique_terms
    
    def get_term_definition(self, term_name: str) -> Optional[Dict[str, Any]]:
        """특정 법령용어의 정의 조회"""
        try:
            return self.db_manager.get_legal_term_definition(term_name)
        except Exception as e:
            logger.error(f"법령용어 정의 조회 실패: {e}")
            return None
    
    def get_domain_keywords(self, domain: str) -> List[str]:
        """도메인별 키워드 추출"""
        domain_seeds = {
            "civil_law": ["민법", "계약", "손해배상", "채권", "채무", "소유권", "물권"],
            "criminal_law": ["형법", "범죄", "처벌", "형량", "구성요건", "고의", "과실"],
            "family_law": ["이혼", "상속", "양육권", "친권", "위자료", "재산분할", "유언"],
            "commercial_law": ["상법", "회사", "주식", "이사", "주주", "회사설립", "합병"],
            "labor_law": ["노동법", "근로", "임금", "해고", "근로시간", "휴게시간", "연차"],
            "real_estate": ["부동산", "매매", "임대차", "등기", "소유권이전", "전세", "월세"],
            "general": ["법률", "법령", "조문", "법원", "판례", "소송"]
        }
        
        seeds = domain_seeds.get(domain, [])
        all_keywords = []
        
        for seed in seeds:
            terms = self.search_legal_terms(seed, limit=10)
            keywords = [term['법령용어명'] for term in terms]
            all_keywords.extend(keywords)
        
        # 중복 제거 및 빈도순 정렬
        keyword_counts = Counter(all_keywords)
        return [keyword for keyword, count in keyword_counts.most_common(20)]
    
    def validate_legal_content(self, content: str) -> bool:
        """내용이 법률 관련인지 검증"""
        try:
            # 텍스트에서 법령용어 추출
            legal_terms = self.extract_legal_terms_from_text(content)
            
            # 법령용어가 1개 이상 발견되면 법률 관련으로 판단
            return len(legal_terms) >= 1
            
        except Exception as e:
            logger.error(f"법률 내용 검증 실패: {e}")
            return False
    
    def get_term_statistics(self) -> Dict[str, Any]:
        """법령용어 통계 조회"""
        try:
            total_count = self.db_manager.get_base_legal_terms_count()
            
            # 도메인별 통계
            domain_stats = {}
            for domain in ["civil_law", "criminal_law", "family_law", "commercial_law", "labor_law", "real_estate"]:
                keywords = self.get_domain_keywords(domain)
                domain_stats[domain] = len(keywords)
            
            return {
                "total_terms": total_count,
                "cached_terms": len(self.term_cache),
                "domain_statistics": domain_stats,
                "cache_size": len(self.cache)
            }
            
        except Exception as e:
            logger.error(f"법령용어 통계 조회 실패: {e}")
            return {}
    
    def clear_cache(self):
        """캐시 초기화"""
        self.cache.clear()
        logger.info("법령용어 검색 캐시 초기화 완료")
    
    def warm_up_cache(self, common_terms: List[str] = None):
        """캐시 워밍업"""
        if common_terms is None:
            common_terms = ["법률", "계약", "손해배상", "이혼", "상속", "범죄", "형법", "민법"]
        
        logger.info("법령용어 검색 캐시 워밍업 시작")
        
        for term in common_terms:
            try:
                self.search_legal_terms(term, limit=5)
                self.get_term_suggestions(term[:3], limit=3)
            except Exception as e:
                logger.warning(f"캐시 워밍업 실패 - 용어: {term}, 오류: {e}")
        
        logger.info("법령용어 검색 캐시 워밍업 완료")
