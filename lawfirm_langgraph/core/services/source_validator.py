# -*- coding: utf-8 -*-
"""
출처 정보 유효성 검증 클래스
"""

import re
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class SourceValidator:
    """출처 정보 유효성 검증 클래스"""
    
    # 법령명 패턴 (한글 법령명)
    LAW_NAME_PATTERN = re.compile(r'^[가-힣]+법(?:\s*시행령|\s*시행규칙)?$')
    
    # 조문 번호 패턴
    ARTICLE_NO_PATTERN = re.compile(r'^제\d+조$')
    
    # 항/호 패턴
    CLAUSE_ITEM_PATTERN = re.compile(r'^제\d+[항호]$')
    
    # 판례 사건번호 패턴
    CASE_NUMBER_PATTERN = re.compile(r'^\d{4}[가-힣]\d+$')
    
    # 법원명 목록
    VALID_COURTS = [
        "대법원", "고등법원", "지방법원", "가정법원", "행정법원",
        "특허법원", "수원지방법원", "서울지방법원", "부산지방법원",
        "대구지방법원", "인천지방법원", "광주지방법원", "대전지방법원",
        "울산지방법원", "춘천지방법원", "청주지방법원", "전주지방법원",
        "창원지방법원", "제주지방법원"
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_source(self, source_type: str, source_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        출처 정보 유효성 검증
        
        Args:
            source_type: 출처 타입
            source_data: 출처 데이터
            
        Returns:
            {
                "is_valid": bool,
                "errors": List[str],
                "warnings": List[str],
                "confidence": float
            }
        """
        if source_type == "statute_article":
            result = self._validate_statute_article(source_data)
        elif source_type == "case_paragraph":
            result = self._validate_case_paragraph(source_data)
        elif source_type == "decision_paragraph":
            result = self._validate_decision_paragraph(source_data)
        elif source_type == "interpretation_paragraph":
            result = self._validate_interpretation_paragraph(source_data)
        else:
            result = {
                "errors": [f"Unknown source_type: {source_type}"],
                "warnings": [],
                "confidence": 0.0
            }
        
        result["is_valid"] = len(result["errors"]) == 0
        return result
    
    def _validate_statute_article(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """법령 조문 검증"""
        errors = []
        warnings = []
        confidence = 1.0
        
        statute_name = data.get("statute_name") or data.get("law_name")
        article_no = data.get("article_no") or data.get("article_number")
        
        if not statute_name:
            errors.append("법령명이 없습니다")
            confidence -= 0.5
        elif not self.LAW_NAME_PATTERN.match(statute_name):
            warnings.append(f"법령명 형식이 올바르지 않을 수 있습니다: {statute_name}")
            confidence -= 0.1
        
        if article_no and not self.ARTICLE_NO_PATTERN.match(article_no):
            warnings.append(f"조문 번호 형식이 올바르지 않을 수 있습니다: {article_no}")
            confidence -= 0.1
        
        clause_no = data.get("clause_no")
        item_no = data.get("item_no")
        
        if clause_no and not self.CLAUSE_ITEM_PATTERN.match(f"제{clause_no}항"):
            warnings.append(f"항 번호 형식이 올바르지 않을 수 있습니다: {clause_no}")
            confidence -= 0.05
        
        if item_no and not self.CLAUSE_ITEM_PATTERN.match(f"제{item_no}호"):
            warnings.append(f"호 번호 형식이 올바르지 않을 수 있습니다: {item_no}")
            confidence -= 0.05
        
        return {
            "errors": errors,
            "warnings": warnings,
            "confidence": max(0.0, confidence)
        }
    
    def _validate_case_paragraph(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """판례 검증"""
        errors = []
        warnings = []
        confidence = 1.0
        
        court = data.get("court")
        doc_id = data.get("doc_id")
        casenames = data.get("casenames")
        
        if court and court not in self.VALID_COURTS:
            if not any(valid_court in court for valid_court in self.VALID_COURTS):
                warnings.append(f"법원명이 표준 형식이 아닐 수 있습니다: {court}")
                confidence -= 0.1
        
        if doc_id and not self.CASE_NUMBER_PATTERN.match(doc_id):
            warnings.append(f"사건번호 형식이 올바르지 않을 수 있습니다: {doc_id}")
            confidence -= 0.1
        
        if not court and not casenames:
            errors.append("법원명 또는 사건명이 필요합니다")
            confidence -= 0.3
        
        return {
            "errors": errors,
            "warnings": warnings,
            "confidence": max(0.0, confidence)
        }
    
    def _validate_decision_paragraph(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """결정례 검증"""
        errors = []
        warnings = []
        confidence = 1.0
        
        org = data.get("org")
        doc_id = data.get("doc_id")
        
        if not org:
            errors.append("기관명이 필요합니다")
            confidence -= 0.3
        
        if not doc_id:
            warnings.append("문서 ID가 없습니다")
            confidence -= 0.1
        
        return {
            "errors": errors,
            "warnings": warnings,
            "confidence": max(0.0, confidence)
        }
    
    def _validate_interpretation_paragraph(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """해석례 검증"""
        errors = []
        warnings = []
        confidence = 1.0
        
        org = data.get("org")
        title = data.get("title")
        doc_id = data.get("doc_id")
        
        if not org and not title:
            errors.append("기관명 또는 제목이 필요합니다")
            confidence -= 0.3
        
        if not doc_id:
            warnings.append("문서 ID가 없습니다")
            confidence -= 0.1
        
        return {
            "errors": errors,
            "warnings": warnings,
            "confidence": max(0.0, confidence)
        }

