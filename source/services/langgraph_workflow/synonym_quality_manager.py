#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
동의어 품질 관리 및 중복 제거 시스템
동의어의 품질을 평가하고 중복을 제거하는 시스템
"""

import re
import difflib
from typing import List, Dict, Set, Tuple, Any, Optional
from dataclasses import dataclass
import logging
from collections import Counter
import json

@dataclass
class QualityMetrics:
    """품질 메트릭"""
    semantic_similarity: float
    context_relevance: float
    domain_relevance: float
    usage_frequency: float
    user_feedback_score: float
    overall_score: float

class SynonymQualityManager:
    """동의어 품질 관리 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.quality_thresholds = {
            "confidence": 0.7,      # 신뢰도 임계값
            "usage_count": 5,        # 최소 사용 횟수
            "user_rating": 3.5,      # 최소 사용자 평점
            "similarity_score": 0.8,  # 유사도 점수
            "overall_score": 0.6     # 전체 점수 임계값
        }
        
        # 법률 도메인별 관련 용어 매핑
        self.domain_keywords = {
            "민사법": ["계약", "손해배상", "위약금", "계약해제", "채무", "채권"],
            "형사법": ["범죄", "형벌", "벌금", "징역", "구속", "기소"],
            "노동법": ["근로", "해고", "임금", "근로시간", "휴가", "부당해고"],
            "가족법": ["이혼", "양육권", "위자료", "재산분할", "친권", "면접교섭권"],
            "부동산법": ["아파트", "매매", "임대", "등기", "전세", "월세"],
            "상법": ["회사", "주식", "이사회", "주주", "합병", "분할"]
        }
        
        # 맥락별 관련 용어 매핑
        self.context_keywords = {
            "계약_관련_맥락": ["계약서", "계약금", "위약금", "계약해제", "계약불이행"],
            "법적_문제_맥락": ["소송", "재판", "판결", "법원", "변호사", "법적구제"],
            "부동산_거래_맥락": ["매매", "임대", "전세", "월세", "등기", "중개"],
            "노동_관계_맥락": ["고용", "해고", "임금", "근로조건", "노동조합"]
        }
    
    def evaluate_synonym_quality(self, synonym: str, keyword: str, 
                               context: str, domain: str, 
                               usage_count: int = 0, user_rating: float = 0.0) -> QualityMetrics:
        """동의어 품질 평가"""
        
        # 1. 의미적 유사도 계산
        semantic_similarity = self._calculate_semantic_similarity(synonym, keyword)
        
        # 2. 맥락 관련성 계산
        context_relevance = self._calculate_context_relevance(synonym, context)
        
        # 3. 도메인 관련성 계산
        domain_relevance = self._calculate_domain_relevance(synonym, domain)
        
        # 4. 사용 빈도 점수 계산
        usage_frequency = self._calculate_usage_frequency_score(usage_count)
        
        # 5. 사용자 피드백 점수 계산
        user_feedback_score = self._calculate_user_feedback_score(user_rating)
        
        # 6. 전체 점수 계산 (가중 평균)
        overall_score = self._calculate_overall_score(
            semantic_similarity, context_relevance, domain_relevance,
            usage_frequency, user_feedback_score
        )
        
        return QualityMetrics(
            semantic_similarity=semantic_similarity,
            context_relevance=context_relevance,
            domain_relevance=domain_relevance,
            usage_frequency=usage_frequency,
            user_feedback_score=user_feedback_score,
            overall_score=overall_score
        )
    
    def _calculate_semantic_similarity(self, synonym: str, keyword: str) -> float:
        """의미적 유사도 계산"""
        # 1. 문자열 유사도 (SequenceMatcher)
        string_similarity = difflib.SequenceMatcher(None, keyword, synonym).ratio()
        
        # 2. 길이 차이 페널티
        length_diff = abs(len(keyword) - len(synonym))
        length_penalty = max(0, 1 - (length_diff / max(len(keyword), len(synonym))))
        
        # 3. 공통 문자 비율
        common_chars = set(keyword) & set(synonym)
        char_ratio = len(common_chars) / max(len(set(keyword)), len(set(synonym)))
        
        # 4. 법률 용어 특수 규칙
        legal_penalty = self._calculate_legal_term_penalty(synonym, keyword)
        
        # 최종 점수 계산
        similarity = (string_similarity * 0.4 + 
                     length_penalty * 0.2 + 
                     char_ratio * 0.3 + 
                     legal_penalty * 0.1)
        
        return min(1.0, similarity)
    
    def _calculate_legal_term_penalty(self, synonym: str, keyword: str) -> float:
        """법률 용어 특수 규칙 적용"""
        # 법률 용어가 아닌 경우 페널티
        legal_indicators = ["법", "규정", "조항", "절", "항", "조", "법원", "판결", "소송"]
        
        keyword_legal = any(indicator in keyword for indicator in legal_indicators)
        synonym_legal = any(indicator in synonym for indicator in legal_indicators)
        
        if keyword_legal and not synonym_legal:
            return 0.7  # 법률 용어가 아닌 동의어에 페널티
        elif not keyword_legal and synonym_legal:
            return 0.8  # 법률 용어로 변환된 경우 보너스
        else:
            return 1.0  # 동일한 성격
    
    def _calculate_context_relevance(self, synonym: str, context: str) -> float:
        """맥락 관련성 계산"""
        if context not in self.context_keywords:
            return 0.5  # 기본값
        
        context_terms = self.context_keywords[context]
        
        # 동의어가 맥락 관련 용어를 포함하는지 확인
        relevance_score = 0.0
        for term in context_terms:
            if term in synonym:
                relevance_score += 0.2
        
        return min(1.0, relevance_score)
    
    def _calculate_domain_relevance(self, synonym: str, domain: str) -> float:
        """도메인 관련성 계산"""
        if domain not in self.domain_keywords:
            return 0.5  # 기본값
        
        domain_terms = self.domain_keywords[domain]
        
        # 동의어가 도메인 관련 용어를 포함하는지 확인
        relevance_score = 0.0
        for term in domain_terms:
            if term in synonym:
                relevance_score += 0.15
        
        return min(1.0, relevance_score)
    
    def _calculate_usage_frequency_score(self, usage_count: int) -> float:
        """사용 빈도 점수 계산"""
        if usage_count == 0:
            return 0.0
        elif usage_count < 5:
            return 0.3
        elif usage_count < 20:
            return 0.6
        elif usage_count < 50:
            return 0.8
        else:
            return 1.0
    
    def _calculate_user_feedback_score(self, user_rating: float) -> float:
        """사용자 피드백 점수 계산"""
        if user_rating == 0:
            return 0.5  # 피드백이 없는 경우 중간값
        elif user_rating < 2.0:
            return 0.2
        elif user_rating < 3.0:
            return 0.4
        elif user_rating < 4.0:
            return 0.7
        else:
            return 1.0
    
    def _calculate_overall_score(self, semantic_similarity: float, 
                               context_relevance: float, domain_relevance: float,
                               usage_frequency: float, user_feedback_score: float) -> float:
        """전체 점수 계산 (가중 평균)"""
        weights = {
            "semantic": 0.4,      # 의미적 유사도
            "context": 0.2,       # 맥락 관련성
            "domain": 0.2,        # 도메인 관련성
            "usage": 0.1,         # 사용 빈도
            "feedback": 0.1       # 사용자 피드백
        }
        
        overall_score = (
            semantic_similarity * weights["semantic"] +
            context_relevance * weights["context"] +
            domain_relevance * weights["domain"] +
            usage_frequency * weights["usage"] +
            user_feedback_score * weights["feedback"]
        )
        
        return round(overall_score, 3)
    
    def should_keep_synonym(self, quality_metrics: QualityMetrics) -> bool:
        """동의어 보관 여부 결정"""
        return (
            quality_metrics.semantic_similarity >= self.quality_thresholds["similarity_score"] and
            quality_metrics.overall_score >= self.quality_thresholds["overall_score"]
        )
    
    def get_quality_recommendations(self, quality_metrics: QualityMetrics) -> List[str]:
        """품질 개선 권장사항 생성"""
        recommendations = []
        
        if quality_metrics.semantic_similarity < self.quality_thresholds["similarity_score"]:
            recommendations.append("의미적 유사도가 낮습니다. 더 유사한 용어를 사용하세요.")
        
        if quality_metrics.context_relevance < 0.5:
            recommendations.append("맥락 관련성이 낮습니다. 질문의 맥락에 맞는 용어를 사용하세요.")
        
        if quality_metrics.domain_relevance < 0.5:
            recommendations.append("도메인 관련성이 낮습니다. 해당 법률 도메인의 전문 용어를 사용하세요.")
        
        if quality_metrics.usage_frequency < 0.3:
            recommendations.append("사용 빈도가 낮습니다. 더 자주 사용되는 용어를 고려하세요.")
        
        if quality_metrics.user_feedback_score < 0.5:
            recommendations.append("사용자 피드백이 부정적입니다. 용어의 적절성을 재검토하세요.")
        
        return recommendations

class SynonymDeduplicator:
    """동의어 중복 제거 및 통합 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.similarity_threshold = 0.85  # 유사도 임계값
        self.quality_manager = SynonymQualityManager()
    
    def deduplicate_synonyms(self, synonyms: List[str], keyword: str = "") -> List[str]:
        """중복 동의어 제거"""
        if len(synonyms) <= 1:
            return synonyms
        
        unique_synonyms = []
        processed = set()
        
        for synonym in synonyms:
            if synonym in processed:
                continue
            
            is_duplicate = False
            for existing in unique_synonyms:
                similarity = self._calculate_similarity(synonym, existing)
                if similarity > self.similarity_threshold:
                    # 더 좋은 품질의 동의어 선택
                    if self._is_better_synonym(synonym, existing, keyword):
                        unique_synonyms.remove(existing)
                        unique_synonyms.append(synonym)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_synonyms.append(synonym)
            
            processed.add(synonym)
        
        return unique_synonyms
    
    def _calculate_similarity(self, synonym1: str, synonym2: str) -> float:
        """두 동의어 간의 유사도 계산"""
        return difflib.SequenceMatcher(None, synonym1, synonym2).ratio()
    
    def _is_better_synonym(self, synonym1: str, synonym2: str, keyword: str) -> bool:
        """더 좋은 동의어 판단"""
        # 길이 기준 (너무 짧거나 긴 것 제외)
        len1, len2 = len(synonym1), len(synonym2)
        if len1 < 2 or len1 > 20:
            return False
        if len2 < 2 or len2 > 20:
            return True
        
        # 키워드와의 유사도 비교
        sim1 = difflib.SequenceMatcher(None, keyword, synonym1).ratio()
        sim2 = difflib.SequenceMatcher(None, keyword, synonym2).ratio()
        
        return sim1 > sim2
    
    def merge_similar_synonyms(self, synonym_groups: List[List[str]], 
                              keyword: str = "") -> List[str]:
        """유사한 동의어 그룹 통합"""
        merged = []
        
        for group in synonym_groups:
            if not group:
                continue
            
            # 그룹 내에서 가장 좋은 동의어 선택
            best_synonym = self._select_best_synonym_from_group(group, keyword)
            merged.append(best_synonym)
        
        return merged
    
    def _select_best_synonym_from_group(self, group: List[str], keyword: str) -> str:
        """그룹에서 가장 좋은 동의어 선택"""
        if len(group) == 1:
            return group[0]
        
        # 점수 기반 선택
        scores = []
        for synonym in group:
            score = self._calculate_synonym_score(synonym, keyword)
            scores.append((synonym, score))
        
        # 점수 기준 내림차순 정렬
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[0][0]
    
    def _calculate_synonym_score(self, synonym: str, keyword: str) -> float:
        """동의어 점수 계산"""
        # 기본 유사도
        similarity = difflib.SequenceMatcher(None, keyword, synonym).ratio()
        
        # 길이 적절성
        length_score = 1.0 - abs(len(synonym) - len(keyword)) / max(len(synonym), len(keyword))
        
        # 법률 용어 특성
        legal_score = 1.0
        legal_indicators = ["법", "규정", "조항", "절", "항", "조"]
        if any(indicator in keyword for indicator in legal_indicators):
            if any(indicator in synonym for indicator in legal_indicators):
                legal_score = 1.2  # 법률 용어 보너스
            else:
                legal_score = 0.8  # 법률 용어가 아닌 경우 페널티
        
        return similarity * 0.5 + length_score * 0.3 + legal_score * 0.2

class SynonymOptimizer:
    """동의어 최적화 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.quality_manager = SynonymQualityManager()
        self.deduplicator = SynonymDeduplicator()
    
    def optimize_synonym_list(self, synonyms: List[str], keyword: str, 
                             context: str = "", domain: str = "") -> List[str]:
        """동의어 목록 최적화"""
        if not synonyms:
            return synonyms
        
        # 1. 중복 제거
        deduplicated = self.deduplicator.deduplicate_synonyms(synonyms, keyword)
        
        # 2. 품질 평가 및 필터링
        quality_filtered = []
        for synonym in deduplicated:
            quality_metrics = self.quality_manager.evaluate_synonym_quality(
                synonym, keyword, context, domain
            )
            
            if self.quality_manager.should_keep_synonym(quality_metrics):
                quality_filtered.append(synonym)
        
        # 3. 품질 점수 기준 정렬
        scored_synonyms = []
        for synonym in quality_filtered:
            quality_metrics = self.quality_manager.evaluate_synonym_quality(
                synonym, keyword, context, domain
            )
            scored_synonyms.append((synonym, quality_metrics.overall_score))
        
        # 점수 기준 내림차순 정렬
        scored_synonyms.sort(key=lambda x: x[1], reverse=True)
        
        return [synonym for synonym, score in scored_synonyms]
    
    def get_optimization_report(self, original_synonyms: List[str], 
                               optimized_synonyms: List[str], keyword: str) -> Dict[str, Any]:
        """최적화 리포트 생성"""
        return {
            "original_count": len(original_synonyms),
            "optimized_count": len(optimized_synonyms),
            "removed_count": len(original_synonyms) - len(optimized_synonyms),
            "reduction_rate": round((len(original_synonyms) - len(optimized_synonyms)) / len(original_synonyms) * 100, 2) if original_synonyms else 0,
            "keyword": keyword,
            "removed_synonyms": list(set(original_synonyms) - set(optimized_synonyms)),
            "kept_synonyms": optimized_synonyms
        }

# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 품질 관리자 초기화
    quality_manager = SynonymQualityManager()
    
    # 테스트 동의어
    test_synonyms = ["계약문서", "계약장", "계약서류", "계약서", "계약서류", "계약문서"]
    keyword = "계약서"
    context = "계약_관련_맥락"
    domain = "민사법"
    
    print(f"원본 동의어: {test_synonyms}")
    
    # 품질 평가
    print("\n=== 품질 평가 ===")
    for synonym in set(test_synonyms):
        metrics = quality_manager.evaluate_synonym_quality(synonym, keyword, context, domain)
        print(f"{synonym}: 전체점수 {metrics.overall_score:.3f}")
        print(f"  - 의미적 유사도: {metrics.semantic_similarity:.3f}")
        print(f"  - 맥락 관련성: {metrics.context_relevance:.3f}")
        print(f"  - 도메인 관련성: {metrics.domain_relevance:.3f}")
    
    # 중복 제거
    deduplicator = SynonymDeduplicator()
    deduplicated = deduplicator.deduplicate_synonyms(test_synonyms, keyword)
    print(f"\n중복 제거 후: {deduplicated}")
    
    # 최적화
    optimizer = SynonymOptimizer()
    optimized = optimizer.optimize_synonym_list(test_synonyms, keyword, context, domain)
    print(f"최적화 후: {optimized}")
    
    # 리포트
    report = optimizer.get_optimization_report(test_synonyms, optimized, keyword)
    print(f"\n최적화 리포트: {report}")

