#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실제 원본 데이터로 파서 테스트
"""

import json
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.assembly.parsers.improved_article_parser import ImprovedArticleParser

def test_real_data():
    """실제 원본 데이터로 테스트"""
    print("Testing with Real Original Data")
    print("=" * 40)
    
    # 실제 원본 데이터 파일 읽기
    raw_file = Path("data/raw/assembly/law/2025101201/law_page_388_103321.json")
    
    if not raw_file.exists():
        print(f"Raw file not found: {raw_file}")
        return
    
    with open(raw_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 법률 내용 추출
    law_content = raw_data['laws'][0]['law_content']
    
    print(f"Original content length: {len(law_content)}")
    print(f"Contains newlines: {'\\n' in law_content}")
    print(f"Contains tabs: {'\\t' in law_content}")
    print()
    
    # 첫 500자 미리보기
    print("First 500 characters:")
    print(repr(law_content[:500]))
    print()
    
    # 파서 초기화
    parser = ImprovedArticleParser()
    
    try:
        # 파싱 실행
        result = parser.parse_law(law_content)
        
        print("파싱 결과:")
        print(f"총 조문 수: {result.get('total_articles', 0)}")
        print(f"본문 조문 수: {len(result.get('main_articles', []))}")
        print(f"부칙 조문 수: {len(result.get('supplementary_articles', []))}")
        print(f"파싱 상태: {result.get('parsing_status', 'unknown')}")
        
        # 각 조문 확인
        for i, article in enumerate(result.get('all_articles', [])):
            print(f"\n조문 {i+1}:")
            print(f"  조문 번호: {article.get('article_number', 'N/A')}")
            print(f"  조문 제목: '{article.get('article_title', 'N/A')}'")
            print(f"  조문 내용 길이: {len(article.get('article_content', ''))}")
            print(f"  줄바꿈 포함: {'\\n' in article.get('article_content', '')}")
            print(f"  조문 내용 미리보기: {article.get('article_content', '')[:100]}...")
        
        print("\n[OK] 파싱 성공!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 파싱 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_real_data()
