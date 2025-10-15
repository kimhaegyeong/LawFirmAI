#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터 구조 확인 스크립트
"""

import json
import sys
from pathlib import Path

# Windows 콘솔에서 UTF-8 인코딩 설정
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

def check_data_structure():
    """데이터 구조 확인"""
    # 에러가 발생하는 파일 중 하나를 확인
    error_file = Path('data/processed/assembly/law/20251013_ml/2025101201/ml_enhanced_law_page_366_080448.json')
    
    try:
        with open(error_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print('파일 구조 확인:')
        print(f'총 법률 수: {len(data.get("laws", []))}')
        
        # 첫 번째 법률의 첫 번째 조문 확인
        if data.get('laws') and len(data['laws']) > 0:
            first_law = data['laws'][0]
            print(f'첫 번째 법률: {first_law.get("law_name", "Unknown")}')
            
            if first_law.get('articles') and len(first_law['articles']) > 0:
                first_article = first_law['articles'][0]
                print(f'첫 번째 조문: {first_article.get("article_number", "Unknown")}')
                
                # 각 필드의 타입 확인
                for field in ['sub_articles', 'references']:
                    value = first_article.get(field)
                    print(f'{field}: {type(value)} = {value}')
                    
                # 모든 조문의 필드 타입 확인
                print('\n모든 조문의 필드 타입 확인:')
                for i, article in enumerate(first_law['articles'][:3]):  # 처음 3개만
                    print(f'조문 {i+1}: {article.get("article_number", "Unknown")}')
                    for field in ['sub_articles', 'references']:
                        value = article.get(field)
                        if not isinstance(value, list):
                            print(f'  {field}: {type(value)} = {value}')
                            
    except Exception as e:
        print(f'에러: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_data_structure()

