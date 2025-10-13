#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
공공기관 소방안전관리 규정으로 파서 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.assembly.parsers.improved_article_parser import ImprovedArticleParser

def test_fire_safety_law():
    """공공기관 소방안전관리 규정으로 테스트"""
    print("Testing Fire Safety Law Parsing")
    print("=" * 40)
    
    # 실제 문제가 있는 법률 내용
    test_content = """제1조(목적) 이 영은 「화재의 예방 및 안전관리에 관한 법률」 제39조에 따라 공공기관의 건축물·인공구조물 및 물품 등을 화재로부터 보호하기 위하여 소방안전관리에 필요한 사항을 규정함을 목적으로 한다.
제2조(적용 범위) 이 영은 다음 각 호의 어느 하나에 해당하는 공공기관에 적용한다.
3. 「공공기관의 운영에 관한 법률」 제4조에 따른 공공기관
5. 「사립학교법」 제2조제1항에 따른 사립학교
제3조 삭제
제4조(기관장의 책임) 제2조에 따른 공공기관의 장(이하 "기관장"이라 한다)은 다음 각 호의 사항에 대한 감독책임을 진다.
2. 소방계획의 수립·시행에 관한 사항
제5조(소방안전관리자의 선임) 기관장은 소방안전관리 업무를 원활하게 수행하기 위하여 감독직에 있는 사람으로서 다음 각 호의 구분에 따른 자격을 갖춘 사람을 소방안전관리자로 선임하여야 한다."""
    
    print(f"Test content length: {len(test_content)}")
    print(f"Contains newlines: {'\\n' in test_content}")
    print()
    
    # 파서 초기화
    parser = ImprovedArticleParser()
    
    try:
        # 파싱 실행
        result = parser.parse_law(test_content)
        
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
    test_fire_safety_law()
