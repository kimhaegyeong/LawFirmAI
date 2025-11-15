#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
복합 키워드 추출 테스트
"""
import sys
import re
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def extract_keywords_from_query(query: str) -> list:
    """질문에서 핵심 키워드 추출 (개선 버전: 복합 키워드 인식 포함)"""
    if not query:
        return []
    
    stopwords = {
        "에", "를", "을", "의", "와", "과", "은", "는", "이", "가", 
        "도", "만", "조차", "까지", "부터", "에게", "한테", "께", "에서", "에게서",
        "에 대해", "에 대해서", "대해", "대해서", "에 관한", "에 대한", "에 관하여",
        "알려주세요", "알려주시기", "알려", "주세요", "주시기", "부탁", "드립니다", 
        "합니다", "입니다", "인가요", "인지", "인가", "인지요",
        "법률", "규정", "조항", "법령", "법", "법률", "규칙"
    }
    
    query_clean = query
    complex_particles = [
        r'에\s+대해\s*서?', r'에\s+관한', r'에\s+대한', r'에\s+관하여',
        r'에\s+대해', r'에\s+대해서'
    ]
    for pattern in complex_particles:
        query_clean = re.sub(pattern, ' ', query_clean, flags=re.IGNORECASE)
    
    keywords = []
    seen_keywords = set()
    
    # 복합 키워드 패턴 인식
    complex_patterns = [
        r'전세금\s*반환\s*보증',
        r'전세\s*보증금',
        r'보증금\s*반환',
        r'임대차\s*보증금',
        r'계약\s*해지',
        r'손해\s*배상',
        r'법률\s*상담',
    ]
    
    extracted_positions = []
    for pattern in complex_patterns:
        matches = re.finditer(pattern, query_clean, re.IGNORECASE)
        for match in matches:
            complex_keyword = match.group().strip()
            complex_keyword_clean = re.sub(r'\s+', '', complex_keyword)
            if len(complex_keyword_clean) >= 3 and complex_keyword_clean.lower() not in seen_keywords:
                keywords.append(complex_keyword_clean)
                seen_keywords.add(complex_keyword_clean.lower())
                extracted_positions.append((match.start(), match.end()))
    
    if extracted_positions:
        extracted_positions.sort(reverse=True)
        for start, end in extracted_positions:
            query_clean = query_clean[:start] + ' ' + query_clean[end:]
    
    words = re.findall(r'[가-힣]+', query_clean)
    particles = [r'(에|를|을|의|와|과|은|는|이|가|도|만|조차|까지|부터|에게|한테|께|에서|에게서)$']
    
    for word in words:
        if len(word) < 2:
            continue
        word_clean = word
        for particle_pattern in particles:
            word_clean = re.sub(particle_pattern, '', word_clean)
        if len(word_clean) >= 2 and word_clean not in stopwords:
            word_lower = word_clean.lower()
            if word_lower not in seen_keywords:
                keywords.append(word_clean)
                seen_keywords.add(word_lower)
    
    return keywords if keywords else [query.strip()]


if __name__ == '__main__':
    test_queries = [
        "전세금 반환 보증에 대해 알려주세요",
        "전세 보증금 반환에 대해 알려주세요",
        "임대차 보증금에 대해 알려주세요",
    ]
    
    print("=" * 80)
    print("복합 키워드 추출 테스트")
    print("=" * 80)
    
    for query in test_queries:
        keywords = extract_keywords_from_query(query)
        print(f"\n질문: {query}")
        print(f"추출된 키워드: {keywords}")
        print(f"복합 키워드 포함: {any(len(kw) > 4 for kw in keywords)}")

