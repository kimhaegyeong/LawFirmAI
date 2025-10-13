#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
공중방역수의사 법률로 파싱 테스트
"""

import sys
import json
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.assembly.preprocess_laws import LawPreprocessor

def test_veterinary_law():
    """공중방역수의사 법률로 테스트"""
    print("Testing Veterinary Law Parsing")
    print("=" * 40)
    
    # 공중방역수의사 법률 데이터 로드
    data_file = Path("data/raw/assembly/law/2025101201/law_page_441_112641.json")
    
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        return False
    
    with open(data_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 공중방역수의사 법률 찾기
    veterinary_law = None
    for law in raw_data['laws']:
        if '공중방역수의사' in law.get('law_name', ''):
            veterinary_law = law
            break
    
    if not veterinary_law:
        print("Veterinary law not found")
        return False
    
    print(f"Law name: {veterinary_law.get('law_name', 'N/A')}")
    print(f"Original content length: {len(veterinary_law.get('law_content', ''))}")
    print(f"Contains newlines: {'\\n' in veterinary_law.get('law_content', '')}")
    
    # 제5조 내용 확인
    content = veterinary_law.get('law_content', '')
    if '제5조' in content:
        start = content.find('제5조')
        end = content.find('제6조', start)
        if end == -1:
            end = start + 200
        article_5 = content[start:end]
        print(f"\nOriginal Article 5:")
        print(article_5.encode('utf-8', errors='replace').decode('utf-8'))
    
    print()
    
    # 전처리기 초기화
    preprocessor = LawPreprocessor(enable_legal_analysis=False)
    
    try:
        processed_data = preprocessor._process_single_law(veterinary_law)
        
        if not processed_data:
            print("[ERROR] 전처리 결과가 None입니다.")
            return False
        
        print("처리된 데이터:")
        print(f"법률명: {processed_data.get('law_name', 'N/A')}")
        print(f"조문 수: {len(processed_data.get('articles', []))}")
        
        # 제5조 찾기
        article_5 = None
        for article in processed_data.get('articles', []):
            if article.get('article_number') == '제5조':
                article_5 = article
                break
        
        if article_5:
            print(f"\n제5조 파싱 결과:")
            print(f"조문 번호: {article_5.get('article_number', 'N/A')}")
            print(f"조문 제목: {article_5.get('article_title', 'N/A')}")
            print(f"조문 내용 길이: {len(article_5.get('article_content', ''))}")
            print(f"줄바꿈 포함: {'\\n' in article_5.get('article_content', '')}")
            print(f"조문 내용: {article_5.get('article_content', '')}")
            
            # 부조문 확인
            sub_articles = article_5.get('sub_articles', [])
            print(f"\n부조문 수: {len(sub_articles)}")
            for i, sub in enumerate(sub_articles):
                print(f"  {i+1}. {sub.get('type', 'N/A')} {sub.get('number', 'N/A')}: {sub.get('content', 'N/A')[:100]}...")
        
        print("\n[OK] 전처리 성공!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 전처리 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_veterinary_law()
