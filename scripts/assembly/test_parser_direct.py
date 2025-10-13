#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
직접 파서 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.assembly.parsers.improved_article_parser import ImprovedArticleParser

def test_parser_directly():
    """파서를 직접 테스트"""
    print("Testing Parser Directly")
    print("=" * 30)
    
    # 테스트용 법률 내용
    test_content = """제1조(목적) 이 규칙은 공중방역수의사에 관한 법률 및 같은 법 시행령에서 위임된 사항과 그 시행에 필요한 사항을 규정함을 목적으로 한다.
제2조(명단통보) 공중방역수의사에 관한 법률 제5조제2항에 따른 공중방역수의사에 편입된 수의사 명단의 통보는 별지 제1호서식에 따른다.
제3조(종사명령 등) 농림축산식품부장관은 법 제6조제1항에 따라 공중방역수의사가 근무하는 가축방역기관을 정함에 있어서는 다음 각 호의 순위에 따라 공중방역수의사가 배치되도록 하여야 한다.
제4조(근무기관의 지정) 농림축산식품부장관은 제3조에 따라 근무기관을 정하려는 경우에는 검역본부와 특별시·광역시·도·특별자치도별로 인원을 배정하고 그 명단을 농림축산검역본부장과 시·도지사에게 통보해야 한다.
제5조(직무교육 소집) 농림축산식품부장관·검역본부장 또는 시·도지사는 법 제6조제2항 또는 제3항에 따라 공중방역수의사에 대한 직무교육을 실시하기 위하여 해당공중방역수의사를 소집할 때에는 소집일 5일 전까지 소집대상자의 인적사항·소집일시 및 장소 등 필요한 사항을 명시하여 소집통지를 하여야 한다.
제6조(직무교육 소집연기) 제5조에 따른 소집통지를 받은 자가 다음 각 호의 어느 하나에 해당하는 사유로 소집일시에 그 소집에 응할 수 없는 경우에는 소집일시를 연기할 수 있다."""
    
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
        
        # 제5조 확인
        for article in result.get('all_articles', []):
            if article.get('article_number') == '제5조':
                print(f"\n제5조 파싱 결과:")
                print(f"조문 번호: {article.get('article_number', 'N/A')}")
                print(f"조문 제목: {article.get('article_title', 'N/A')}")
                print(f"조문 내용 길이: {len(article.get('article_content', ''))}")
                print(f"줄바꿈 포함: {'\\n' in article.get('article_content', '')}")
                print(f"조문 내용: {article.get('article_content', '')}")
                
                # 부조문 확인
                sub_articles = article.get('sub_articles', [])
                print(f"\n부조문 수: {len(sub_articles)}")
                for i, sub in enumerate(sub_articles):
                    print(f"  {i+1}. {sub.get('type', 'N/A')} {sub.get('number', 'N/A')}: {sub.get('content', 'N/A')[:50]}...")
                break
        
        print("\n[OK] 파싱 성공!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 파싱 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_parser_directly()
