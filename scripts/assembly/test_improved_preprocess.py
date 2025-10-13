#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Preprocessor 테스트
"""

import sys
import json
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.assembly.preprocess_laws import LawPreprocessor

def test_improved_preprocessor():
    """개선된 전처리기 테스트"""
    print("Testing Improved Preprocessor")
    print("=" * 40)
    
    # 전처리기 초기화
    preprocessor = LawPreprocessor(enable_legal_analysis=False)
    
    # 테스트용 법률 데이터 (제어문자 포함)
    test_law_data = {
        "law_id": "test_law_001",
        "law_name": "테스트 법률",
        "law_type": "법률",
        "content_html": """
        <div class="law_content">
            <h3>제1조(목적)</h3>
            <p>이 법은 테스트를 위한 법률이다.\n줄바꿈이 포함된 내용이다.</p>
            
            <h3>제2조(정의)</h3>
            <p>이 법에서 사용하는 용어의 뜻은 다음과 같다.</p>
            <ol>
                <li>가. 첫 번째 항목</li>
                <li>나. 두 번째 항목</li>
                <li>다. 세 번째 항목</li>
            </ol>
        </div>
        """
    }
    
    print("원본 데이터:")
    print(f"Content HTML 길이: {len(test_law_data['content_html'])}")
    print(f"줄바꿈 포함: {'\\n' in test_law_data['content_html']}")
    print()
    
    # 전처리 실행
    try:
        processed_data = preprocessor._process_single_law(test_law_data)
        
        if not processed_data:
            print("❌ 전처리 결과가 None입니다.")
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
    test_improved_preprocessor()
