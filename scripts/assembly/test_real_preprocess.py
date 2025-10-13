#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실제 법률 데이터로 Improved Preprocessor 테스트
"""

import sys
import json
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.assembly.preprocess_laws import LawPreprocessor

def test_with_real_data():
    """실제 법률 데이터로 테스트"""
    print("Testing with Real Law Data")
    print("=" * 40)
    
    # 실제 법률 데이터 파일 로드
    data_file = Path("data/raw/assembly/law/2025101201/law_page_001_112726.json")
    
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        return False
    
    with open(data_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    if 'laws' not in raw_data or not raw_data['laws']:
        print("No laws found in data file")
        return False
    
    # 첫 번째 법률 데이터 사용
    law_data = raw_data['laws'][0]
    print(f"Law name: {law_data.get('law_name', 'N/A')}")
    print(f"Original content length: {len(law_data.get('law_content', ''))}")
    print(f"Contains newlines: {'\\n' in law_data.get('law_content', '')}")
    print()
    
    # 전처리기 초기화
    preprocessor = LawPreprocessor(enable_legal_analysis=False)
    
    try:
        processed_data = preprocessor._process_single_law(law_data)
        
        if not processed_data:
            print("[ERROR] 전처리 결과가 None입니다.")
            return False
        
        print("처리된 데이터:")
        print(f"법률명: {processed_data.get('law_name', 'N/A')}")
        print(f"조문 수: {len(processed_data.get('articles', []))}")
        
        # 첫 번째 조문 확인
        if processed_data.get('articles'):
            first_article = processed_data['articles'][0]
            print(f"\n첫 번째 조문:")
            print(f"조문 번호: {first_article.get('article_number', 'N/A')}")
            print(f"조문 제목: {first_article.get('article_title', 'N/A')}")
            print(f"조문 내용 길이: {len(first_article.get('article_content', ''))}")
            print(f"줄바꿈 포함: {'\\n' in first_article.get('article_content', '')}")
            print(f"조문 내용 미리보기: {first_article.get('article_content', '')[:100]}...")
            
            # 부조문 확인
            sub_articles = first_article.get('sub_articles', [])
            print(f"\n부조문 수: {len(sub_articles)}")
            for i, sub in enumerate(sub_articles):
                print(f"  {i+1}. {sub.get('type', 'N/A')} {sub.get('number', 'N/A')}: {sub.get('content', 'N/A')[:50]}...")
        
        print("\n[OK] 전처리 성공!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 전처리 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_with_real_data()
