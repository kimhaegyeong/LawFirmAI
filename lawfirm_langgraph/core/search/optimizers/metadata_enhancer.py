# -*- coding: utf-8 -*-
"""
Metadata Enhancer
메타데이터 활용 강화 모듈
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = get_logger(__name__)


@dataclass
class EnhancedMetadata:
    """강화된 메타데이터"""
    law_name: Optional[str] = None
    article_number: Optional[str] = None
    precedent_case: Optional[str] = None
    court_name: Optional[str] = None
    case_number: Optional[str] = None
    category: Optional[str] = None
    document_type: Optional[str] = None
    date: Optional[str] = None
    source_credibility: float = 0.5


class MetadataEnhancer:
    """메타데이터 강화기"""
    
    # 법령명 패턴
    LAW_NAME_PATTERN = r'([가-힣]+법)'
    
    # 조문 번호 패턴
    ARTICLE_PATTERN = r'제?\s*(\d+)\s*조'
    
    # 판례 패턴
    PRECEDENT_PATTERN = r'(대법원|법원).*?(\d{4}[다나마]\d+)'
    
    # 법원명 패턴
    COURT_PATTERN = r'(대법원|고등법원|지방법원|가정법원|특허법원|행정법원)'
    
    def __init__(self):
        """초기화"""
        self.logger = get_logger(__name__)
        self.logger.info("MetadataEnhancer initialized")
    
    def enhance_metadata(
        self,
        document: Dict[str, Any]
    ) -> EnhancedMetadata:
        """
        메타데이터 강화
        
        Args:
            document: 문서 딕셔너리
        
        Returns:
            EnhancedMetadata: 강화된 메타데이터
        """
        try:
            text = document.get("text", document.get("content", ""))
            source = document.get("source", "")
            title = document.get("title", "")
            metadata = document.get("metadata", {})
            
            # 전체 텍스트에서 추출
            full_text = f"{title} {source} {text}"
            
            # 법령명 추출
            law_name = self._extract_law_name(full_text, metadata)
            
            # 조문 번호 추출
            article_number = self._extract_article_number(full_text, metadata)
            
            # 판례 정보 추출
            precedent_case = self._extract_precedent_case(full_text, metadata)
            
            # 법원명 추출
            court_name = self._extract_court_name(full_text, metadata)
            
            # 사건번호 추출
            case_number = self._extract_case_number(full_text, metadata)
            
            # 카테고리 추출
            category = self._extract_category(document, metadata)
            
            # 문서 유형 추출
            document_type = self._extract_document_type(document, metadata)
            
            # 날짜 추출
            date = self._extract_date(metadata)
            
            # 출처 신뢰도 계산
            source_credibility = self._calculate_source_credibility(
                source, title, metadata
            )
            
            return EnhancedMetadata(
                law_name=law_name,
                article_number=article_number,
                precedent_case=precedent_case,
                court_name=court_name,
                case_number=case_number,
                category=category,
                document_type=document_type,
                date=date,
                source_credibility=source_credibility
            )
        
        except Exception as e:
            self.logger.error(f"Metadata enhancement failed: {e}")
            return EnhancedMetadata()
    
    def _extract_law_name(self, text: str, metadata: Dict[str, Any]) -> Optional[str]:
        """법령명 추출"""
        # 메타데이터에서 먼저 확인
        if "law_name" in metadata:
            return metadata["law_name"]
        
        # 텍스트에서 추출
        matches = re.findall(self.LAW_NAME_PATTERN, text)
        if matches:
            return matches[0]
        
        return None
    
    def _extract_article_number(self, text: str, metadata: Dict[str, Any]) -> Optional[str]:
        """조문 번호 추출"""
        # 메타데이터에서 먼저 확인
        if "article_number" in metadata:
            return str(metadata["article_number"])
        
        # 텍스트에서 추출
        matches = re.findall(self.ARTICLE_PATTERN, text)
        if matches:
            return matches[0]
        
        return None
    
    def _extract_precedent_case(self, text: str, metadata: Dict[str, Any]) -> Optional[str]:
        """판례 정보 추출"""
        # 메타데이터에서 먼저 확인
        if "precedent_case" in metadata:
            return metadata["precedent_case"]
        
        # 텍스트에서 추출
        matches = re.findall(self.PRECEDENT_PATTERN, text)
        if matches:
            return " ".join(matches[0])
        
        return None
    
    def _extract_court_name(self, text: str, metadata: Dict[str, Any]) -> Optional[str]:
        """법원명 추출"""
        # 메타데이터에서 먼저 확인
        if "court_name" in metadata:
            return metadata["court_name"]
        
        # 텍스트에서 추출
        matches = re.findall(self.COURT_PATTERN, text)
        if matches:
            return matches[0]
        
        return None
    
    def _extract_case_number(self, text: str, metadata: Dict[str, Any]) -> Optional[str]:
        """사건번호 추출"""
        # 메타데이터에서 먼저 확인
        if "case_number" in metadata:
            return metadata["case_number"]
        
        # 텍스트에서 추출 (판례 패턴과 유사)
        pattern = r'(\d{4}[다나마]\d+)'
        matches = re.findall(pattern, text)
        if matches:
            return matches[0]
        
        return None
    
    def _extract_category(self, document: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[str]:
        """카테고리 추출"""
        # 메타데이터에서 확인
        if "category" in metadata:
            return metadata["category"]
        
        # 문서에서 추출
        source = document.get("source", "").lower()
        title = document.get("title", "").lower()
        
        # 카테고리 키워드 매칭
        categories = {
            "civil": ["민사", "계약", "손해배상", "불법행위"],
            "criminal": ["형사", "살인", "절도", "사기"],
            "family": ["가족", "이혼", "상속", "양육"],
            "administrative": ["행정", "세무", "건축"],
        }
        
        for category, keywords in categories.items():
            if any(kw in source or kw in title for kw in keywords):
                return category
        
        return None
    
    def _extract_document_type(self, document: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[str]:
        """문서 유형 추출"""
        # 메타데이터에서 확인
        if "document_type" in metadata:
            return metadata["document_type"]
        
        # 텍스트에서 추출
        text = document.get("text", document.get("content", "")).lower()
        source = document.get("source", "").lower()
        
        if "법령" in source or "조문" in text:
            return "law"
        elif "판례" in source or "판결" in text:
            return "precedent"
        elif "계약" in text:
            return "contract"
        else:
            return "general"
    
    def _extract_date(self, metadata: Dict[str, Any]) -> Optional[str]:
        """날짜 추출"""
        date_fields = ["date", "created_at", "updated_at", "published_date"]
        
        for field in date_fields:
            if field in metadata:
                return str(metadata[field])
        
        return None
    
    def _calculate_source_credibility(
        self,
        source: str,
        title: str,
        metadata: Dict[str, Any]
    ) -> float:
        """출처 신뢰도 계산"""
        source_text = f"{source} {title}".lower()
        
        # 신뢰도 등급
        credibility_map = {
            "대법원": 1.0,
            "법원": 0.9,
            "법령": 0.95,
            "법률": 0.95,
            "판례": 0.85,
            "법률서": 0.8,
            "해설서": 0.75,
        }
        
        for keyword, credibility in credibility_map.items():
            if keyword in source_text:
                return credibility
        
        return 0.5
    
    def boost_by_metadata(
        self,
        document: Dict[str, Any],
        query: str,
        query_type: str = "general_question"
    ) -> float:
        """
        메타데이터 기반 부스팅 점수 계산
        
        Args:
            document: 문서 딕셔너리
            query: 검색 쿼리
            query_type: 질문 유형
        
        Returns:
            float: 부스팅 점수 (0.0-1.0)
        """
        try:
            enhanced_metadata = self.enhance_metadata(document)
            boost_score = 0.0
            
            # 1. 법령 조문 매칭 (법령 조문 문의인 경우)
            if query_type == "law_inquiry":
                if enhanced_metadata.law_name and enhanced_metadata.article_number:
                    # 쿼리에서 법령명/조문 추출
                    query_law = self._extract_law_name(query, {})
                    query_article = self._extract_article_number(query, {})
                    
                    if query_law and query_law in enhanced_metadata.law_name:
                        boost_score += 0.3
                    if query_article and query_article == enhanced_metadata.article_number:
                        boost_score += 0.4
            
            # 2. 판례 매칭 (판례 검색인 경우)
            elif query_type == "precedent_search":
                if enhanced_metadata.precedent_case or enhanced_metadata.court_name:
                    boost_score += 0.3
                if enhanced_metadata.case_number:
                    boost_score += 0.2
            
            # 3. 카테고리 매칭
            if enhanced_metadata.category:
                # 쿼리에서 카테고리 키워드 추출
                category_keywords = {
                    "civil": ["민사", "계약", "손해배상"],
                    "criminal": ["형사", "살인", "절도"],
                    "family": ["가족", "이혼", "상속"],
                }
                
                if enhanced_metadata.category in category_keywords:
                    keywords = category_keywords[enhanced_metadata.category]
                    if any(kw in query for kw in keywords):
                        boost_score += 0.2
            
            # 4. 출처 신뢰도
            boost_score += enhanced_metadata.source_credibility * 0.1
            
            return min(1.0, boost_score)
        
        except Exception as e:
            self.logger.error(f"Metadata boost calculation failed: {e}")
            return 0.0

