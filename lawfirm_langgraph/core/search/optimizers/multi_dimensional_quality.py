# -*- coding: utf-8 -*-
"""
Multi-Dimensional Quality Scorer
다차원 품질 점수 시스템
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class QualityScores:
    """품질 점수"""
    relevance: float  # 관련성 (0.0-1.0)
    accuracy: float  # 정확성 (0.0-1.0)
    completeness: float  # 완전성 (0.0-1.0)
    recency: float  # 최신성 (0.0-1.0)
    source_credibility: float  # 출처 신뢰도 (0.0-1.0)
    overall: float  # 종합 점수 (0.0-1.0)


class MultiDimensionalQualityScorer:
    """다차원 품질 점수 계산기"""
    
    # 출처 신뢰도 등급
    SOURCE_CREDIBILITY = {
        "대법원": 1.0,
        "법원": 0.9,
        "법령": 0.95,
        "법률": 0.95,
        "판례": 0.85,
        "법률서": 0.8,
        "해설서": 0.75,
        "기타": 0.5,
    }
    
    def __init__(self):
        """초기화"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("MultiDimensionalQualityScorer initialized")
    
    def calculate_quality(
        self,
        document: Dict[str, Any],
        query: str,
        query_type: str = "general_question",
        extracted_keywords: Optional[List[str]] = None
    ) -> QualityScores:
        """
        다차원 품질 점수 계산
        
        Args:
            document: 문서 딕셔너리
            query: 검색 쿼리
            query_type: 질문 유형
            extracted_keywords: 추출된 키워드
        
        Returns:
            QualityScores: 품질 점수
        """
        try:
            # 1. 관련성 점수
            relevance = self._calculate_relevance(document, query, extracted_keywords)
            
            # 2. 정확성 점수
            accuracy = self._calculate_accuracy(document, query, query_type)
            
            # 3. 완전성 점수
            completeness = self._calculate_completeness(document, query, extracted_keywords)
            
            # 4. 최신성 점수
            recency = self._calculate_recency(document)
            
            # 5. 출처 신뢰도
            source_credibility = self._calculate_source_credibility(document)
            
            # 6. 종합 점수 (가중 평균)
            overall = (
                0.35 * relevance +
                0.25 * accuracy +
                0.20 * completeness +
                0.10 * recency +
                0.10 * source_credibility
            )
            
            return QualityScores(
                relevance=relevance,
                accuracy=accuracy,
                completeness=completeness,
                recency=recency,
                source_credibility=source_credibility,
                overall=overall
            )
        
        except Exception as e:
            self.logger.error(f"Quality calculation failed: {e}")
            return QualityScores(
                relevance=0.5,
                accuracy=0.5,
                completeness=0.5,
                recency=0.5,
                source_credibility=0.5,
                overall=0.5
            )
    
    def _calculate_relevance(
        self,
        document: Dict[str, Any],
        query: str,
        extracted_keywords: Optional[List[str]] = None
    ) -> float:
        """관련성 점수 계산"""
        text = document.get("text", document.get("content", ""))
        if not text:
            return 0.0
        
        text_lower = text.lower()
        query_lower = query.lower()
        
        # 1. 쿼리 키워드 매칭
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())
        keyword_match = len(query_words & text_words) / len(query_words) if query_words else 0.0
        
        # 2. 추출된 키워드 매칭
        extracted_match = 0.0
        if extracted_keywords:
            matched = sum(1 for kw in extracted_keywords if kw.lower() in text_lower)
            extracted_match = matched / len(extracted_keywords) if extracted_keywords else 0.0
        
        # 3. 기존 관련성 점수
        existing_relevance = document.get("relevance_score", document.get("similarity", 0.0))
        
        # 가중 평균
        relevance = (
            0.4 * keyword_match +
            0.3 * extracted_match +
            0.3 * existing_relevance
        )
        
        return min(1.0, relevance)
    
    def _calculate_accuracy(
        self,
        document: Dict[str, Any],
        query: str,
        query_type: str
    ) -> float:
        """정확성 점수 계산"""
        text = document.get("text", document.get("content", ""))
        if not text:
            return 0.0
        
        accuracy = 0.5  # 기본값
        
        # 1. 법률 용어 정확성
        legal_terms = ["법", "조문", "항", "호", "판례", "대법원", "법원"]
        legal_term_count = sum(1 for term in legal_terms if term in text)
        if legal_term_count > 0:
            accuracy += 0.2
        
        # 2. 법령 조문 인용 정확성
        law_pattern = r'[가-힣]+법\s*제?\s*\d+\s*조'
        if re.search(law_pattern, text):
            accuracy += 0.2
        
        # 3. 판례 인용 정확성
        precedent_pattern = r'대법원|법원.*\d{4}[다나마]\d+'
        if re.search(precedent_pattern, text):
            accuracy += 0.1
        
        return min(1.0, accuracy)
    
    def _calculate_completeness(
        self,
        document: Dict[str, Any],
        query: str,
        extracted_keywords: Optional[List[str]] = None
    ) -> float:
        """완전성 점수 계산"""
        text = document.get("text", document.get("content", ""))
        if not text:
            return 0.0
        
        # 1. 텍스트 길이 (너무 짧으면 완전성 낮음)
        text_length = len(text)
        if text_length < 50:
            length_score = 0.3
        elif text_length < 200:
            length_score = 0.6
        else:
            length_score = 1.0
        
        # 2. 키워드 커버리지
        keyword_coverage = 0.5
        if extracted_keywords:
            matched = sum(1 for kw in extracted_keywords if kw.lower() in text.lower())
            keyword_coverage = matched / len(extracted_keywords) if extracted_keywords else 0.5
        
        # 3. 구조적 완전성 (문장 수, 단락 등)
        sentences = text.split('。') + text.split('.')
        sentence_count = len([s for s in sentences if s.strip()])
        structure_score = min(1.0, sentence_count / 5.0)  # 5문장 이상이면 완전
        
        completeness = (
            0.4 * length_score +
            0.4 * keyword_coverage +
            0.2 * structure_score
        )
        
        return min(1.0, completeness)
    
    def _calculate_recency(self, document: Dict[str, Any]) -> float:
        """최신성 점수 계산"""
        metadata = document.get("metadata", {})
        
        # 날짜 정보 추출
        date_str = metadata.get("date") or metadata.get("created_at") or metadata.get("updated_at")
        
        if not date_str:
            return 0.5  # 날짜 정보 없으면 중간 점수
        
        try:
            # 날짜 파싱 (다양한 형식 지원)
            if isinstance(date_str, str):
                # YYYY-MM-DD 형식
                if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                # YYYYMMDD 형식
                elif re.match(r'\d{8}', date_str):
                    date_obj = datetime.strptime(date_str, '%Y%m%d')
                else:
                    return 0.5
            else:
                return 0.5
            
            # 현재 날짜와 비교
            now = datetime.now()
            days_diff = (now - date_obj).days
            
            # 최신성 점수 (1년 이내: 1.0, 5년 이내: 0.7, 10년 이내: 0.5, 그 이상: 0.3)
            if days_diff <= 365:
                recency = 1.0
            elif days_diff <= 1825:  # 5년
                recency = 0.7
            elif days_diff <= 3650:  # 10년
                recency = 0.5
            else:
                recency = 0.3
            
            return recency
        
        except Exception as e:
            self.logger.debug(f"Recency calculation failed: {e}")
            return 0.5
    
    def _calculate_source_credibility(self, document: Dict[str, Any]) -> float:
        """출처 신뢰도 계산 (우선순위 4: 출처 신뢰도 점수 강화)"""
        source = document.get("source", "")
        title = document.get("title", "")
        metadata = document.get("metadata", {})
        doc_type = document.get("type") or document.get("source_type", "")
        
        # 1. 문서 타입별 신뢰도
        type_credibility = {
            "statute_article": 1.0,  # 법령 조문: 최고 신뢰도
            "case_paragraph": 0.9,   # 판례: 높은 신뢰도
            "decision_paragraph": 0.85,  # 결정례: 높은 신뢰도
            "interpretation_paragraph": 0.8  # 해석례: 중상 신뢰도
        }
        credibility = type_credibility.get(doc_type, 0.5)
        
        # 2. 출처 정보 수집
        source_text = f"{source} {title} {metadata.get('source', '')}".lower()
        
        # 3. 출처별 신뢰도 부스팅
        if "대법원" in source_text:
            credibility = min(1.0, credibility + 0.1)
        elif "법원" in source_text:
            credibility = min(1.0, credibility + 0.05)
        elif "법제처" in source_text or "법무부" in source_text:
            credibility = min(1.0, credibility + 0.1)
        
        # 4. 기존 신뢰도 등급 매칭 (타입 신뢰도와 비교하여 더 높은 값 사용)
        for keyword, keyword_credibility in self.SOURCE_CREDIBILITY.items():
            if keyword in source_text:
                credibility = max(credibility, keyword_credibility)
        
        # 5. 직접 매칭 부스팅
        if document.get("direct_match", False):
            credibility = 1.0  # 직접 매칭: 최고 신뢰도
        
        return credibility
    
    def calculate_batch_quality(
        self,
        documents: List[Dict[str, Any]],
        query: str,
        query_type: str = "general_question",
        extracted_keywords: Optional[List[str]] = None
    ) -> List[QualityScores]:
        """배치 품질 점수 계산"""
        return [
            self.calculate_quality(doc, query, query_type, extracted_keywords)
            for doc in documents
        ]

