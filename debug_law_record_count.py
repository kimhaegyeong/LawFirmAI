#!/usr/bin/env python3
"""
디버깅용 스크립트: _prepare_law_record가 반환하는 값의 개수 확인
"""

import sys
import os
import json
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.assembly.import_laws_to_db import AssemblyLawImporter

def test_law_record_count():
    """_prepare_law_record가 반환하는 값의 개수 테스트"""
    
    # 테스트용 법률 데이터 생성
    test_law_data = {
        'law_id': 'test_law_001',
        'law_name': '테스트 법률',
        'law_type': '법률',
        'category': '일반',
        'row_number': 1,
        'promulgation_number': '법률 제12345호',
        'promulgation_date': '2025-01-01',
        'enforcement_date': '2025-01-01',
        'amendment_type': '신설',
        'ministry': '법제처',
        'parent_law': None,
        'related_laws': [],
        'full_text': '테스트 법률 내용',
        'summary': '테스트 요약',
        'html_clean_text': '테스트 HTML 정리된 텍스트',
        'content_html': '<div>테스트 HTML</div>',
        'raw_content': '원본 테스트 내용',
        'detail_url': 'http://test.com',
        'cont_id': 'test_cont_id',
        'cont_sid': 'test_cont_sid',
        'collected_at': '2025-01-01T00:00:00',
        'processed_at': '2025-01-01T00:00:00',
        'processing_version': 'v4.0',
        'data_quality': {'parsing_quality_score': 0.95},
        'ml_enhanced': True,
        'parsing_quality_score': 0.95,
        'article_count': 5,
        'supplementary_count': 2,
        'control_characters_removed': True,
        'articles': [
            {
                'article_number': '제1조',
                'article_title': '목적',
                'article_content': '이 법은 테스트를 위함',
                'sub_articles': [],
                'references': [],
                'word_count': 10,
                'char_count': 20,
                'is_supplementary': False,
                'ml_confidence_score': 0.9,
                'parsing_method': 'ml_enhanced',
                'article_type': 'main'
            }
        ]
    }
    
    # AssemblyLawImporter 인스턴스 생성
    importer = AssemblyLawImporter('data/lawfirm.db')
    
    # _prepare_law_record 메서드 호출
    law_record = importer._prepare_law_record(test_law_data)
    
    print(f"반환된 값의 개수: {len(law_record)}")
    print(f"반환된 값들:")
    for i, value in enumerate(law_record):
        print(f"  {i+1}: {repr(value)}")
    
    # INSERT 문의 컬럼 개수 확인
    insert_columns = [
        'law_id', 'source', 'law_name', 'law_type', 'category', 'row_number',
        'promulgation_number', 'promulgation_date', 'enforcement_date', 'amendment_type',
        'ministry', 'parent_law', 'related_laws',
        'full_text', 'searchable_text', 'keywords', 'summary',
        'html_clean_text', 'content_html',
        'raw_content', 'detail_url', 'cont_id', 'cont_sid', 'collected_at',
        'processed_at', 'processing_version', 'data_quality',
        'ml_enhanced', 'parsing_quality_score', 'article_count', 'supplementary_count', 'control_characters_removed'
    ]
    
    print(f"\nINSERT 문의 컬럼 개수: {len(insert_columns)}")
    print(f"INSERT 문의 컬럼들:")
    for i, col in enumerate(insert_columns):
        print(f"  {i+1}: {col}")
    
    print(f"\n차이점: {len(insert_columns) - len(law_record)}")
    
    if len(law_record) != len(insert_columns):
        print("\n[ERROR] 값의 개수가 일치하지 않습니다!")
        print(f"반환된 값: {len(law_record)}개")
        print(f"INSERT 컬럼: {len(insert_columns)}개")
    else:
        print("\n[OK] 값의 개수가 일치합니다!")

if __name__ == "__main__":
    test_law_record_count()
