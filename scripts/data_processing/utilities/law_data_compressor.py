#!/usr/bin/env python3
"""
법률 데이터 압축 유틸리티 모듈
수집 및 전처리 과정에서 데이터를 즉시 압축하는 기능 제공
"""

import json
import re
from typing import Dict, Any, List

def compress_legal_text(text: str) -> str:
    """법률 텍스트 압축"""
    if not text:
        return ""
    
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 반복되는 법률 용어 축약
    replacements = {
        '이 법에 따르면': '이 법에 따라',
        '다음 각 호의 어느 하나에 해당하는': '다음에 해당하는',
        '특별시장·광역시장·특별자치시장·도지사': '시·도지사',
        '특별자치도지사': '특별자치도지사',
        '중앙행정기관의 장': '중앙행정기관장',
        '지방자치단체의 장': '지방자치단체장',
        '국가 또는 지방자치단체': '국가·지방자치단체',
        '이하 "시·도지사"라 한다': '이하 시·도지사라 함',
        '이하 "특례시"라 한다': '이하 특례시라 함',
        '이하 "등록비영리민간단체"라 한다': '이하 등록비영리민간단체라 함',
        '이하 "공익사업"이라 한다': '이하 공익사업이라 함'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text.strip()

# generate_compressed_search_text 함수 제거됨 - 더 이상 사용하지 않음

def compress_law_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """개별 법률 데이터 압축"""
    # 필수 필드만 유지
    compressed = {
        'law_id': data.get('law_id'),
        'law_name': data.get('law_name'),
        'law_type': data.get('law_type'),
        'category': data.get('category'),
        'promulgation_number': data.get('promulgation_number'),
        'promulgation_date': data.get('promulgation_date'),
        'enforcement_date': data.get('enforcement_date'),
        'amendment_type': data.get('amendment_type'),
        'ministry': data.get('ministry'),
        'articles': data.get('articles', [])
    }
    
    # articles 내부 텍스트도 압축
    for article in compressed['articles']:
        if 'article_content' in article:
            article['article_content'] = compress_legal_text(article['article_content'])
        
        # sub_articles도 압축
        for sub_article in article.get('sub_articles', []):
            if 'content' in sub_article:
                sub_article['content'] = compress_legal_text(sub_article['content'])
    
    return compressed

def save_compressed_law_data(data: Dict[str, Any], file_path: str) -> int:
    """압축된 법률 데이터를 파일에 저장하고 파일 크기 반환"""
    compressed_data = compress_law_data(data)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(compressed_data, f, ensure_ascii=False, separators=(',', ':'))
    
    # 파일 크기 반환
    import os
    return os.path.getsize(file_path)

def compress_and_save_page_data(page_data: Dict[str, Any], file_path: str) -> int:
    """페이지 데이터를 압축하여 저장"""
    compressed_page_data = {
        'page_number': page_data.get('page_number'),
        'total_pages': page_data.get('total_pages'),
        'laws_count': page_data.get('laws_count'),
        'collected_at': page_data.get('collected_at'),
        'laws': []
    }
    
    # 각 법률 데이터 압축
    for law in page_data.get('laws', []):
        compressed_law = compress_law_data(law)
        compressed_page_data['laws'].append(compressed_law)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(compressed_page_data, f, ensure_ascii=False, separators=(',', ':'))
    
    # 파일 크기 반환
    import os
    return os.path.getsize(file_path)
