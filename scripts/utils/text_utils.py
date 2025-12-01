#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
텍스트 처리 유틸리티

키워드 추출, 텍스트 정규화 등 텍스트 관련 공통 함수
"""

import re
from typing import List, Set


def extract_keywords(query: str, min_length: int = 2) -> List[str]:
    """
    쿼리에서 키워드 추출 (조사 제거)
    
    Args:
        query: 검색 쿼리 문자열
        min_length: 최소 키워드 길이 (기본값: 2)
    
    Returns:
        List[str]: 추출된 키워드 리스트
    """
    # 한국어 조사 및 불필요한 단어 제거
    stopwords = [
        '에', '를', '을', '의', '와', '과', '로', '으로', '에서', '에게', '에게서',
        '에 대해', '에 대해서', '에 관하여', '에 관해서',
        '에 대해 알려주세요', '에 대해 설명해주세요',
        '알려주세요', '설명해주세요', '알려줘', '설명해줘',
        '무엇인가요', '무엇인가', '어떤', '어떻게', '왜', '언제', '어디서', '누가',
        '입니다', '입니다', '이에요', '예요', '입니다', '입니다'
    ]
    
    # 조사 제거
    processed_query = query
    for stopword in stopwords:
        processed_query = processed_query.replace(stopword, ' ')
    
    # 공백으로 분리하고 필터링
    keywords = [
        kw.strip()
        for kw in processed_query.split()
        if kw.strip() and len(kw.strip()) >= min_length
    ]
    
    return keywords


def normalize_text(text: str) -> str:
    """
    텍스트 정규화 (공백 정리, 특수문자 제거)
    
    Args:
        text: 정규화할 텍스트
    
    Returns:
        str: 정규화된 텍스트
    """
    # 연속된 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    # 앞뒤 공백 제거
    text = text.strip()
    return text


def remove_special_chars(text: str, keep_spaces: bool = True) -> str:
    """
    특수문자 제거
    
    Args:
        text: 처리할 텍스트
        keep_spaces: 공백 유지 여부 (기본값: True)
    
    Returns:
        str: 특수문자가 제거된 텍스트
    """
    if keep_spaces:
        # 공백은 유지하고 특수문자만 제거
        pattern = r'[^\w\s가-힣]'
    else:
        # 공백도 제거
        pattern = r'[^\w가-힣]'
    
    return re.sub(pattern, '', text)


def extract_legal_terms(text: str) -> Set[str]:
    """
    법률 용어 추출 (간단한 패턴 매칭)
    
    Args:
        text: 분석할 텍스트
    
    Returns:
        Set[str]: 추출된 법률 용어 집합
    """
    # 법률 용어 패턴 (예시)
    patterns = [
        r'[가-힣]+법',
        r'[가-힣]+조',
        r'[가-힣]+항',
        r'[가-힣]+호',
        r'[가-힣]+규칙',
        r'[가-힣]+령',
    ]
    
    terms = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        terms.update(matches)
    
    return terms


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    두 텍스트 간의 간단한 유사도 계산 (Jaccard 유사도)
    
    Args:
        text1: 첫 번째 텍스트
        text2: 두 번째 텍스트
    
    Returns:
        float: 유사도 점수 (0.0 ~ 1.0)
    """
    words1 = set(extract_keywords(text1))
    words2 = set(extract_keywords(text2))
    
    if not words1 and not words2:
        return 1.0
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union) if union else 0.0

