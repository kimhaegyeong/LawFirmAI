#!/usr/bin/env python3
"""
실제 데이터로 _prepare_law_record 테스트
"""

import sys
import os
import json
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.assembly.import_laws_to_db import AssemblyLawImporter

def test_real_data():
    """실제 데이터로 _prepare_law_record 테스트"""
    
    # 실제 데이터 파일 로드
    data_file = "data/processed/assembly/law/20251013_ml/2025101201/ml_enhanced_law_page_001_112726.json"
    
    with open(data_file, 'r', encoding='utf-8') as f:
        file_data = json.load(f)
    
    # 첫 번째 법률 데이터 추출
    if 'laws' in file_data and len(file_data['laws']) > 0:
        law_data = file_data['laws'][0]
    else:
        print("법률 데이터를 찾을 수 없습니다.")
        return
    
    print("실제 법률 데이터 필드들:")
    for key, value in law_data.items():
        print(f"  {key}: {type(value).__name__} = {repr(value)[:100]}...")
    
    # AssemblyLawImporter 인스턴스 생성
    importer = AssemblyLawImporter('data/lawfirm.db')
    
    try:
        # _prepare_law_record 메서드 호출
        law_record = importer._prepare_law_record(law_data)
        
        print(f"\n반환된 값의 개수: {len(law_record)}")
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
        
        if len(law_record) != len(insert_columns):
            print(f"\n[ERROR] 값의 개수가 일치하지 않습니다!")
            print(f"반환된 값: {len(law_record)}개")
            print(f"INSERT 컬럼: {len(insert_columns)}개")
            print(f"차이: {len(insert_columns) - len(law_record)}개")
        else:
            print("\n[OK] 값의 개수가 일치합니다!")
            
    except Exception as e:
        print(f"\n[ERROR] _prepare_law_record 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_data()

