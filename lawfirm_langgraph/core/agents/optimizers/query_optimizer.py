# -*- coding: utf-8 -*-
"""
검색 쿼리 최적화 클래스
원본 쿼리와 확장 키워드를 조합하여 최적화된 검색 쿼리 생성
"""

import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

try:
    from lawfirm_langgraph.core.utils.korean_stopword_processor import KoreanStopwordProcessor
except ImportError:
    try:
        from core.utils.korean_stopword_processor import KoreanStopwordProcessor
    except ImportError:
        KoreanStopwordProcessor = None

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """검색 쿼리 최적화 클래스"""

    def __init__(self):
        """QueryOptimizer 초기화"""
        try:
            from core.agents.keyword_mapper import LegalKeywordMapper
            self.keyword_mapper = LegalKeywordMapper()
        except ImportError:
            self.keyword_mapper = None
            logger.warning("LegalKeywordMapper not available")

        try:
            from core.services.semantic_domain_classifier import LegalTermsDatabase
            self.terms_db = LegalTermsDatabase()
        except ImportError:
            self.terms_db = None
            logger.warning("LegalTermsDatabase not available")

        # KoreanStopwordProcessor 초기화 (KoNLPy 우선 사용)
        self.stopword_processor = None
        if KoreanStopwordProcessor:
            try:
                self.stopword_processor = KoreanStopwordProcessor()
                logger.debug("KoreanStopwordProcessor initialized successfully")
            except Exception as e:
                logger.warning(f"Error initializing KoreanStopwordProcessor: {e}")

    def optimize_search_query(
        self,
        original_query: str,
        expanded_keywords: List[str],
        query_type: str,
        domain: str = "민사법",
        search_type: str = "semantic"  # "semantic" or "fts5"
    ) -> str:
        """
        원본 쿼리와 확장 키워드를 조합하여 최적화된 검색 쿼리 생성

        Args:
            original_query: 원본 사용자 쿼리
            expanded_keywords: 확장된 키워드 리스트
            query_type: 질문 유형
            domain: 법률 도메인 (기본값: "민사법")
            search_type: 검색 타입 ("semantic" 또는 "fts5")

        Returns:
            최적화된 검색 쿼리 문자열
        """
        try:
            # 1. 원본 쿼리에서 핵심 키워드 추출 (불용어 제거, 빈도 기반)
            core_keywords = self.extract_core_keywords(original_query, query_type, domain)
            logger.debug(f"Extracted core keywords: {core_keywords}")

            if search_type == "fts5":
                # FTS5용: 핵심 키워드만 사용 (최대 3개)
                # FTS5는 키워드 검색이므로 핵심 키워드만으로 충분
                final_keywords = core_keywords[:3]
                search_query = " ".join(final_keywords)
                logger.info(f"Optimized search query for FTS5: '{search_query}' ({len(final_keywords)} core keywords)")
                return search_query

            # Semantic Search용
            # 2. 확장 키워드 중요도 계산 및 필터링
            scored_expanded = self.calculate_keyword_importance(
                expanded_keywords,
                domain,
                original_query
            )
            # 관련도 기준 완화: 0.3 → 0.25로 낮춰 더 많은 키워드 포함
            filtered_expanded = [
                kw for kw, score in scored_expanded
                if score >= 0.25
            ]
            logger.debug(f"Filtered expanded keywords: {len(filtered_expanded)}/{len(expanded_keywords)} (threshold: 0.25)")

            # 3. 최종 쿼리 구성
            # 원본 핵심 키워드 우선 (상위 3-5개)
            core_count = min(len(core_keywords), 5)
            core_selected = core_keywords[:core_count]

            # 확장 키워드 보조 (상위 3-5개, weight >= 0.6으로 완화)
            expanded_selected = []
            for kw, score in scored_expanded:
                if score >= 0.6 and kw not in core_selected:
                    expanded_selected.append(kw)
                    if len(expanded_selected) >= 5:
                        break

            # 최종 키워드 조합 (중복 제거)
            final_keywords = list(dict.fromkeys(core_selected + expanded_selected))  # 순서 유지하며 중복 제거

            # 최소 3개, 최대 7개 키워드 (Semantic Search용 - 개선)
            if len(final_keywords) < 3:
                # 부족하면 확장 키워드 추가 (기준 완화: 0.2 → 0.15)
                for kw, score in scored_expanded:
                    if kw not in final_keywords and score >= 0.15:
                        final_keywords.append(kw)
                        if len(final_keywords) >= 3:
                            break

            # Semantic Search용으로 최대 7개 키워드 사용 (5개 → 7개로 증가)
            final_keywords = final_keywords[:7]

            search_query = " ".join(final_keywords)
            logger.info(f"Optimized search query for semantic search: '{search_query}' (core: {len(core_selected)}, expanded: {len(expanded_selected)})")

            return search_query

        except Exception as e:
            logger.error(f"Error optimizing search query: {e}", exc_info=True)
            # 폴백: FTS5는 핵심 키워드만, Semantic은 확장 키워드 포함
            if search_type == "fts5":
                core_keywords = self.extract_core_keywords(original_query, query_type, domain)
                return " ".join(core_keywords[:3]) if core_keywords else original_query
            else:
                if expanded_keywords:
                    return " ".join(expanded_keywords[:5])
                return original_query

    def extract_core_keywords(
        self,
        query: str,
        query_type: str,
        domain: str
    ) -> List[str]:
        """
        원본 쿼리에서 핵심 키워드 추출

        Args:
            query: 원본 쿼리
            query_type: 질문 유형
            domain: 법률 도메인

        Returns:
            핵심 키워드 리스트 (가중치 순)
        """
        core_keywords = []

        # 1. LegalKeywordMapper 활용 (가중치 기반)
        if self.keyword_mapper:
            try:
                mapper_keywords = self.keyword_mapper.get_keywords_for_question(query, query_type)
                # 가중치 정보 추출 (core > important > supporting)
                weighted_keywords = self._extract_weighted_keywords(mapper_keywords, query_type, domain)
                core_keywords.extend(weighted_keywords)
            except Exception as e:
                logger.debug(f"LegalKeywordMapper extraction failed: {e}")

        # 2. LegalTermsDatabase 활용 (법률 용어 우선)
        if self.terms_db and core_keywords:
            try:
                # 법률 용어 사전에서 weight 확인
                term_weights = {}
                for kw in core_keywords:
                    weight = self._get_term_weight(kw, domain)
                    if weight > 0:
                        term_weights[kw] = weight

                # weight 기준으로 정렬
                core_keywords = sorted(
                    term_weights.keys(),
                    key=lambda x: term_weights[x],
                    reverse=True
                )
            except Exception as e:
                logger.debug(f"LegalTermsDatabase extraction failed: {e}")

        # 3. 기본 추출 (형태소 분석 없이 단순 분리)
        if not core_keywords:
            # 불용어 제거 및 단어 추출 (KoreanStopwordProcessor 사용)
            words = re.findall(r'[가-힣]+', query)
            if self.stopword_processor:
                words = [w for w in words if len(w) >= 2 and not self.stopword_processor.is_stopword(w)]
            else:
                words = [w for w in words if len(w) >= 2]
            core_keywords = list(dict.fromkeys(words))  # 중복 제거

        return core_keywords[:10]  # 최대 10개

    def _extract_weighted_keywords(
        self,
        keywords: List[str],
        query_type: str,
        domain: str
    ) -> List[str]:
        """가중치 기반 키워드 추출 (키워드 리스트 반환)"""
        if not self.keyword_mapper:
            return keywords

        try:
            # LegalKeywordMapper의 가중치 매핑 확인
            mapping = getattr(self.keyword_mapper, 'WEIGHTED_KEYWORD_MAPPING', {})

            keyword_weights = []
            for kw in keywords:
                weight = 0.5  # 기본 가중치

                # query_type과 domain에 맞는 매핑 확인
                category_key = query_type if query_type in mapping else None
                if not category_key:
                    # domain 기반으로 매핑 찾기
                    domain_key = domain.lower().replace('법', '')
                    for key in mapping.keys():
                        if domain_key in key or key in domain_key:
                            category_key = key
                            break

                if category_key and category_key in mapping:
                    category_map = mapping[category_key]
                    for term_key, term_info in category_map.items():
                        if isinstance(term_info, dict):
                            if kw in term_info.get('core', []):
                                weight = 1.0
                                break
                            elif kw in term_info.get('important', []):
                                weight = 0.8
                                break
                            elif kw in term_info.get('supporting', []):
                                weight = 0.6
                                break

                keyword_weights.append((kw, weight))

            # 가중치 기준 정렬
            keyword_weights.sort(key=lambda x: x[1], reverse=True)
            return [kw for kw, _ in keyword_weights]

        except Exception as e:
            logger.debug(f"Error extracting weighted keywords: {e}")
            return keywords

    def _get_term_weight(self, keyword: str, domain: str) -> float:
        """법률 용어 사전에서 키워드 weight 확인"""
        if not self.terms_db:
            return 0.5

        try:
            # LegalTermsDatabase의 terms_by_domain 속성 확인
            terms_by_domain = getattr(self.terms_db, 'terms_by_domain', {})

            # domain 매칭 시도
            domain_key = None
            for key in terms_by_domain.keys():
                if domain in key or key in domain:
                    domain_key = key
                    break

            if not domain_key:
                return 0.3

            domain_terms = terms_by_domain.get(domain_key, {})

            # 직접 매칭
            if keyword in domain_terms:
                term_info = domain_terms[keyword]
                if isinstance(term_info, dict):
                    return term_info.get('weight', 0.5)

            # 동의어 확인
            for term_key, term_info in domain_terms.items():
                if isinstance(term_info, dict):
                    synonyms = term_info.get('synonyms', [])
                    if keyword in synonyms:
                        return term_info.get('weight', 0.5)

            return 0.3  # 기본 weight

        except Exception as e:
            logger.debug(f"Error getting term weight: {e}")
            return 0.5

    def calculate_keyword_importance(
        self,
        keywords: List[str],
        domain: str,
        original_query: str
    ) -> List[Tuple[str, float]]:
        """
        확장 키워드 중요도 계산

        Args:
            keywords: 확장된 키워드 리스트
            domain: 법률 도메인
            original_query: 원본 쿼리

        Returns:
            (키워드, 중요도 점수) 튜플 리스트 (점수 순 정렬)
        """
        scored_keywords = []

        # 원본 쿼리 단어 빈도 계산
        query_words = re.findall(r'[가-힣]+', original_query)
        query_word_freq = Counter(query_words)
        max_freq = max(query_word_freq.values()) if query_word_freq else 1

        for keyword in keywords:
            score = 0.0

            # 1. 법률 용어 사전 weight (45% - 증가)
            term_weight = self._get_term_weight(keyword, domain)
            score += term_weight * 0.45

            # 2. 원본 쿼리와의 유사도 (35% - 증가)
            # 단순 키워드 매칭 기반 (TF-IDF 대신)
            keyword_words = re.findall(r'[가-힣]+', keyword)
            overlap = sum(1 for w in keyword_words if w in query_words)
            if keyword_words:
                overlap_ratio = overlap / len(keyword_words)
                score += overlap_ratio * 0.35
            else:
                # 키워드가 없으면 부분 문자열 매칭 시도
                if keyword in original_query:
                    score += 0.2

            # 3. 키워드 길이 및 구체성 (15% - 감소)
            # 짧은 키워드(2-4자)가 더 구체적일 가능성
            if 2 <= len(keyword) <= 4:
                score += 0.15
            elif 5 <= len(keyword) <= 8:
                score += 0.1
            elif len(keyword) > 10:
                score += 0.05  # 너무 긴 키워드는 감점

            # 4. 불용어 제거 (5% - 감소, KoreanStopwordProcessor 사용)
            if not self.stopword_processor or not self.stopword_processor.is_stopword(keyword):
                score += 0.05

            scored_keywords.append((keyword, min(score, 1.0)))

        # 점수 기준 정렬
        scored_keywords.sort(key=lambda x: x[1], reverse=True)

        return scored_keywords
