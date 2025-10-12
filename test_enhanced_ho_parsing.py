#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent / 'scripts' / 'assembly'))

from parsers.article_parser import ArticleParser

def test_enhanced_ho_parsing():
    """개선된 호(1., 2.) 파싱 테스트"""
    
    print("개선된 호(1., 2.) 파싱 테스트 시작...")
    
    # 테스트용 법률 내용
    test_content = """
    제2조(정의) 이 법에서 사용하는 용어의 뜻은 다음과 같다.
    1. "가축"이란 소, 말, 돼지, 양, 염소, 닭, 오리, 칠면조, 거위, 토끼 및 그 밖에 대통령령으로 정하는 동물을 말한다.
    2. "가축전염병"이란 가축에 감염되는 전염병으로서 농림축산식품부장관이 정하여 고시하는 질병을 말한다.
    3. "가축전염병예방"이란 가축전염병의 발생을 방지하고 확산을 차단하기 위한 모든 활동을 말한다.
    4. "가축전염병방역"이란 가축전염병이 발생한 경우 그 확산을 방지하고 근절하기 위한 모든 활동을 말한다.
    5. "가축전염병관리"이란 가축전염병예방 및 가축전염병방역을 말한다.
    6. "가축전염병관리기관"이란 가축전염병관리를 수행하는 기관을 말한다.
    7. "가축전염병관리인"이란 가축전염병관리를 수행하는 사람을 말한다.
    8. "가축전염병관리시설"이란 가축전염병관리에 필요한 시설을 말한다.
    9. "가축전염병관리장비"이란 가축전염병관리에 필요한 장비를 말한다.
    10. "가축전염병관리물품"이란 가축전염병관리에 필요한 물품을 말한다.
    """
    
    # ArticleParser 인스턴스 생성
    parser = ArticleParser()
    
    # 조문 파싱
    articles = parser.parse_articles(test_content, "")
    
    print(f"파싱된 조문 수: {len(articles)}")
    
    for article in articles:
        print(f"\n제{article['article_number']}조:")
        print(f"제목: {article.get('article_title', 'N/A')}")
        
        # 하위 항목 분석
        sub_articles = article.get('sub_articles', [])
        print(f"하위 항목 수: {len(sub_articles)}")
        
        # 호(1., 2.) 항목 확인
        ho_items = [item for item in sub_articles if item.get('type') == '호']
        print(f"호(1., 2.) 항목 수: {len(ho_items)}")
        
        for ho_item in ho_items:
            print(f"  호 {ho_item['number']}: {ho_item['content'][:100]}...")
        
        # 목(가., 나.) 항목 확인
        mok_items = [item for item in sub_articles if item.get('type') == '목']
        print(f"목(가., 나.) 항목 수: {len(mok_items)}")
        
        for mok_item in mok_items:
            print(f"  목 {mok_item['letter']}: {mok_item['content'][:100]}...")
    
    return len(ho_items) > 0

if __name__ == "__main__":
    success = test_enhanced_ho_parsing()
    if success:
        print("\n✅ 호(1., 2.) 파싱 테스트 성공!")
    else:
        print("\n❌ 호(1., 2.) 파싱 테스트 실패!")
