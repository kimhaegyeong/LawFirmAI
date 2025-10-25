# -*- coding: utf-8 -*-
"""
Dynamic Keyword System
동적 키워드 시스템 - 하드코딩된 키워드 대체
"""

import logging
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
from .legal_term_search_service import LegalTermSearchService

logger = logging.getLogger(__name__)


@dataclass
class DomainKeywords:
    """도메인별 키워드 정보"""
    domain: str
    primary_keywords: List[str]
    secondary_keywords: List[str]
    exclude_keywords: List[str]
    confidence: float
    last_updated: str


class DynamicKeywordSystem:
    """동적 키워드 시스템 - 하드코딩된 키워드 대체"""
    
    def __init__(self, legal_term_service: LegalTermSearchService):
        """동적 키워드 시스템 초기화"""
        self.legal_term_service = legal_term_service
        self.domain_cache = {}  # 도메인별 키워드 캐시
        self.keyword_weights = {
            "primary": 3.0,
            "secondary": 1.0,
            "exclude": -2.0
        }
        self.domain_priority = {
            "civil_law": 1.2,
            "criminal_law": 1.1,
            "family_law": 1.0,
            "commercial_law": 1.0,
            "labor_law": 1.0,
            "real_estate": 1.0,
            "general": 0.8
        }
        
        # 도메인별 시드 키워드 정의
        self.domain_seeds = {
            "civil_law": {
                "primary": ["민법", "계약", "손해배상", "불법행위", "채권", "채무", "소유권", "물권"],
                "secondary": ["계약서", "위약금", "손해", "배상", "채권자", "채무자", "소유자"],
                "exclude": ["형법", "형사", "범죄", "처벌"]
            },
            "criminal_law": {
                "primary": ["형법", "범죄", "처벌", "형량", "구성요건", "고의", "과실"],
                "secondary": ["사기", "절도", "강도", "살인", "상해", "폭행", "협박"],
                "exclude": ["민법", "계약", "손해배상"]
            },
            "family_law": {
                "primary": ["이혼", "상속", "양육권", "친권", "위자료", "재산분할", "유언"],
                "secondary": ["협의이혼", "재판이혼", "상속인", "상속세", "유산", "양육비"],
                "exclude": ["회사", "상법", "주식"]
            },
            "commercial_law": {
                "primary": ["상법", "회사", "주식", "이사", "주주", "회사설립", "합병"],
                "secondary": ["주식회사", "유한회사", "합명회사", "합자회사", "자본금", "정관"],
                "exclude": ["이혼", "상속", "가족"]
            },
            "labor_law": {
                "primary": ["노동법", "근로", "임금", "해고", "근로시간", "휴게시간", "연차"],
                "secondary": ["근로계약서", "임금체불", "부당해고", "노동위원회", "최저임금"],
                "exclude": ["이혼", "상속", "범죄"]
            },
            "real_estate": {
                "primary": ["부동산", "매매", "임대차", "등기", "소유권이전", "전세", "월세"],
                "secondary": ["부동산등기법", "매매계약서", "임대차계약서", "등기부등본"],
                "exclude": ["이혼", "상속", "범죄"]
            },
            "general": {
                "primary": ["법률", "법령", "조문", "법원", "판례", "소송"],
                "secondary": ["법적", "법률적", "법적근거", "법적효력"],
                "exclude": []
            }
        }
        
        logger.info("DynamicKeywordSystem 초기화 완료")
    
    def get_domain_keywords(self, domain: str) -> Dict[str, List[str]]:
        """도메인별 키워드를 데이터베이스에서 동적으로 생성"""
        try:
            # 캐시 확인
            if domain in self.domain_cache:
                return self.domain_cache[domain]
            
            logger.info(f"도메인 '{domain}' 키워드 동적 생성 시작")
            
            # 도메인별 법령용어 검색
            domain_terms = self._search_domain_terms(domain)
            
            # 키워드 분류 (주요/보조/제외)
            keywords = self._classify_keywords(domain_terms, domain)
            
            # 캐시 저장
            self.domain_cache[domain] = keywords
            
            logger.info(f"도메인 '{domain}' 키워드 생성 완료: 주요 {len(keywords['primary'])}개, 보조 {len(keywords['secondary'])}개")
            return keywords
            
        except Exception as e:
            logger.error(f"도메인 키워드 생성 실패: {e}")
            # 폴백: 시드 키워드 반환
            return self.domain_seeds.get(domain, {"primary": [], "secondary": [], "exclude": []})
    
    def _search_domain_terms(self, domain: str) -> List[Dict[str, Any]]:
        """도메인별 법령용어 검색"""
        all_terms = []
        
        # 시드 키워드로 검색
        seeds = self.domain_seeds.get(domain, {})
        primary_seeds = seeds.get("primary", [])
        secondary_seeds = seeds.get("secondary", [])
        
        # 주요 키워드로 검색
        for seed in primary_seeds:
            terms = self.legal_term_service.search_legal_terms(seed, limit=15)
            all_terms.extend(terms)
        
        # 보조 키워드로 검색
        for seed in secondary_seeds:
            terms = self.legal_term_service.search_legal_terms(seed, limit=10)
            all_terms.extend(terms)
        
        # 중복 제거
        return self._deduplicate_terms(all_terms)
    
    def _classify_keywords(self, terms: List[Dict[str, Any]], domain: str) -> Dict[str, List[str]]:
        """키워드 분류 (주요/보조/제외)"""
        # 용어명 추출
        term_names = [term['법령용어명'] for term in terms]
        
        # 시드 키워드와의 유사도 기반으로 분류
        primary_keywords = []
        secondary_keywords = []
        exclude_keywords = []
        
        seeds = self.domain_seeds.get(domain, {})
        primary_seeds = seeds.get("primary", [])
        secondary_seeds = seeds.get("secondary", [])
        exclude_seeds = seeds.get("exclude", [])
        
        for term_name in term_names:
            # 주요 키워드와의 유사도 확인
            primary_score = self._calculate_similarity_score(term_name, primary_seeds)
            secondary_score = self._calculate_similarity_score(term_name, secondary_seeds)
            exclude_score = self._calculate_similarity_score(term_name, exclude_seeds)
            
            # 분류 결정
            if exclude_score > 0.3:
                exclude_keywords.append(term_name)
            elif primary_score > 0.4:
                primary_keywords.append(term_name)
            elif secondary_score > 0.2:
                secondary_keywords.append(term_name)
            else:
                # 기본적으로 보조 키워드로 분류
                secondary_keywords.append(term_name)
        
        # 시드 키워드도 추가
        primary_keywords.extend(primary_seeds)
        secondary_keywords.extend(secondary_seeds)
        exclude_keywords.extend(exclude_seeds)
        
        # 중복 제거 및 정렬
        return {
            "primary": list(set(primary_keywords))[:20],  # 최대 20개
            "secondary": list(set(secondary_keywords))[:30],  # 최대 30개
            "exclude": list(set(exclude_keywords))
        }
    
    def _calculate_similarity_score(self, term: str, seed_list: List[str]) -> float:
        """용어와 시드 키워드 리스트 간의 유사도 계산"""
        if not seed_list:
            return 0.0
        
        max_similarity = 0.0
        for seed in seed_list:
            similarity = self._calculate_string_similarity(term, seed)
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """문자열 유사도 계산 (간단한 구현)"""
        if not str1 or not str2:
            return 0.0
        
        # 완전 일치
        if str1 == str2:
            return 1.0
        
        # 포함 관계
        if str1 in str2 or str2 in str1:
            return 0.8
        
        # 공통 문자 비율
        common_chars = set(str1) & set(str2)
        total_chars = set(str1) | set(str2)
        
        if not total_chars:
            return 0.0
        
        return len(common_chars) / len(total_chars)
    
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
    
    def update_keywords_dynamically(self, user_query: str) -> Dict[str, Any]:
        """사용자 쿼리 기반으로 키워드를 동적으로 업데이트"""
        try:
            logger.info(f"사용자 쿼리 기반 동적 키워드 업데이트: '{user_query}'")
            
            # 쿼리에서 법령용어 추출
            extracted_terms = self.legal_term_service.extract_legal_terms_from_text(user_query)
            
            # 관련 도메인 식별
            related_domains = self._identify_related_domains(extracted_terms, user_query)
            
            # 동적 키워드 생성
            dynamic_keywords = {}
            for domain in related_domains:
                dynamic_keywords[domain] = self.get_domain_keywords(domain)
            
            logger.info(f"동적 키워드 업데이트 완료: {len(related_domains)}개 도메인")
            return dynamic_keywords
            
        except Exception as e:
            logger.error(f"동적 키워드 업데이트 실패: {e}")
            return {}
    
    def _identify_related_domains(self, extracted_terms: List[Dict[str, Any]], query: str) -> List[str]:
        """추출된 용어와 쿼리 기반으로 관련 도메인 식별"""
        related_domains = []
        
        # 추출된 용어로 도메인 식별
        for term in extracted_terms:
            term_name = term['법령용어명']
            for domain in self.domain_seeds.keys():
                domain_keywords = self.get_domain_keywords(domain)
                primary_keywords = domain_keywords.get('primary', [])
                
                # 용어가 도메인의 주요 키워드와 유사한지 확인
                for keyword in primary_keywords:
                    if self._calculate_string_similarity(term_name, keyword) > 0.3:
                        if domain not in related_domains:
                            related_domains.append(domain)
                        break
        
        # 쿼리 텍스트로 도메인 식별
        query_lower = query.lower()
        for domain, seeds in self.domain_seeds.items():
            primary_seeds = seeds.get('primary', [])
            for seed in primary_seeds:
                if seed in query_lower:
                    if domain not in related_domains:
                        related_domains.append(domain)
                    break
        
        # 관련 도메인이 없으면 일반 도메인 추가
        if not related_domains:
            related_domains = ['general']
        
        return related_domains
    
    def get_keyword_weights(self) -> Dict[str, float]:
        """키워드 가중치 반환"""
        return self.keyword_weights.copy()
    
    def get_domain_priority(self) -> Dict[str, float]:
        """도메인 우선순위 반환"""
        return self.domain_priority.copy()
    
    def calculate_domain_score(self, domain: str, query: str) -> float:
        """도메인별 점수 계산"""
        try:
            # 도메인 키워드 가져오기
            domain_keywords = self.get_domain_keywords(domain)
            
            # 쿼리에서 키워드 매칭
            query_lower = query.lower()
            score = 0.0
            
            # 주요 키워드 매칭
            for keyword in domain_keywords.get('primary', []):
                if keyword in query_lower:
                    score += self.keyword_weights['primary']
            
            # 보조 키워드 매칭
            for keyword in domain_keywords.get('secondary', []):
                if keyword in query_lower:
                    score += self.keyword_weights['secondary']
            
            # 제외 키워드 매칭
            for keyword in domain_keywords.get('exclude', []):
                if keyword in query_lower:
                    score += self.keyword_weights['exclude']
            
            # 도메인 우선순위 적용
            priority = self.domain_priority.get(domain, 1.0)
            score *= priority
            
            return score
            
        except Exception as e:
            logger.error(f"도메인 점수 계산 실패: {e}")
            return 0.0
    
    def get_best_domain(self, query: str) -> Optional[str]:
        """쿼리에 가장 적합한 도메인 반환"""
        try:
            domain_scores = {}
            
            for domain in self.domain_seeds.keys():
                score = self.calculate_domain_score(domain, query)
                domain_scores[domain] = score
            
            # 가장 높은 점수의 도메인 반환
            best_domain = max(domain_scores.items(), key=lambda x: x[1])
            
            if best_domain[1] > 0:
                return best_domain[0]
            else:
                return 'general'
                
        except Exception as e:
            logger.error(f"최적 도메인 선택 실패: {e}")
            return 'general'
    
    def clear_cache(self):
        """캐시 초기화"""
        self.domain_cache.clear()
        logger.info("동적 키워드 시스템 캐시 초기화 완료")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """캐시 통계 조회"""
        return {
            "cached_domains": len(self.domain_cache),
            "available_domains": list(self.domain_seeds.keys()),
            "keyword_weights": self.keyword_weights,
            "domain_priority": self.domain_priority
        }
