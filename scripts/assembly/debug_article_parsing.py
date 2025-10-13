#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
특정 법률 문서의 조문 파싱 문제 디버깅 스크립트
제2조 2항 누락 문제 분석
"""

import json
import sys
import os
from pathlib import Path
import logging

# Windows 콘솔에서 UTF-8 인코딩 설정
if os.name == 'nt':  # Windows
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# 파서 모듈 경로 추가
sys.path.append(str(Path(__file__).parent / 'parsers'))

from ml_enhanced_parser import MLEnhancedArticleParser
from parsers.improved_article_parser import ImprovedArticleParser

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_article_parsing():
    """조문 파싱 디버깅"""
    
    # 원본 데이터
    original_content = """제1조(목적) 이 규칙은 대한민국 법원이 사법주권을 회복한 날을 기념하기 위하여 『대한민국 법원의 날』을 제정하고, 사법독립과 법치주의의 중요성을 알리며 그 의의를 기념하기 위한 행사 등을 진행함에 있어 필요한 사항을 규정함을 목적으로 한다.
제2조(정의 및 명칭) ① 제1조에서 사법주권을 회복한 날이라 함은, 일제에 사법주권을 빼앗겼다가 대한민국이 1948년 9월 13일 미군정으로부터 사법권을 이양받음으로써 헌법기관인 대한민국 법원이 실질적으로 수립된 날을 의미한다.
② 『대한민국 법원의 날』은 매년 9월 13일로 한다.
제3조(기념식 및 행사) ① 법원은 『대한민국 법원의 날』에 기념식과 그에 부수되는 행사를 실시할 수 있다.
제4조(포상) ① 대법원장은 제2조제1항에 규정된 기념일의 의식에서 사법부의 발전 또는 법률문화의 향상에 공헌한 행적이 뚜렷한 사람에게 포상할 수 있다.
부칙 <제2605호, 2015.6.29.>펼치기접기
이 규칙은 공포한 날부터 시행한다."""
    
    print("=== 원본 내용 ===")
    print(original_content)
    print("\n" + "="*80 + "\n")
    
    # 규칙 기반 파서 테스트
    print("=== 규칙 기반 파서 결과 ===")
    rule_parser = ImprovedArticleParser()
    rule_result = rule_parser.parse_law_document(original_content)
    
    print(f"총 조문 수: {rule_result['total_articles']}")
    for i, article in enumerate(rule_result['all_articles']):
        print(f"\n조문 {i+1}:")
        print(f"  번호: {article['article_number']}")
        print(f"  제목: {article.get('article_title', 'N/A')}")
        print(f"  내용 길이: {len(article['article_content'])}")
        print(f"  내용 미리보기: {article['article_content'][:100]}...")
        
        if article['sub_articles']:
            print(f"  항 수: {len(article['sub_articles'])}")
            for j, sub in enumerate(article['sub_articles']):
                print(f"    항 {j+1}: {sub['content'][:50]}...")
    
    print("\n" + "="*80 + "\n")
    
    # ML 강화 파서 테스트
    print("=== ML 강화 파서 결과 ===")
    ml_parser = MLEnhancedArticleParser()
    ml_result = ml_parser.parse_law_document(original_content)
    
    print(f"총 조문 수: {ml_result['total_articles']}")
    for i, article in enumerate(ml_result['all_articles']):
        print(f"\n조문 {i+1}:")
        print(f"  번호: {article['article_number']}")
        print(f"  제목: {article.get('article_title', 'N/A')}")
        print(f"  내용 길이: {len(article['article_content'])}")
        print(f"  내용 미리보기: {article['article_content'][:100]}...")
        
        if article['sub_articles']:
            print(f"  항 수: {len(article['sub_articles'])}")
            for j, sub in enumerate(article['sub_articles']):
                print(f"    항 {j+1}: {sub['content'][:50]}...")
    
    print("\n" + "="*80 + "\n")
    
    # 문제 분석
    print("=== 문제 분석 ===")
    
    # 제2조 내용 확인
    print("원본 제2조 내용:")
    lines = original_content.split('\n')
    for i, line in enumerate(lines):
        if '제2조' in line:
            print(f"라인 {i+1}: {line}")
            # 다음 라인들도 확인
            for j in range(i+1, min(i+5, len(lines))):
                if lines[j].strip() and not lines[j].startswith('제') and not lines[j].startswith('부칙'):
                    print(f"라인 {j+1}: {lines[j]}")
                else:
                    break
    
    print("\n=== 정규식 패턴 테스트 ===")
    
    # 조문 경계 감지 패턴 테스트
    import re
    
    # 조문 번호 패턴
    article_pattern = r'제(\d+)조'
    matches = list(re.finditer(article_pattern, original_content))
    
    print(f"조문 번호 매치 수: {len(matches)}")
    for match in matches:
        print(f"  위치 {match.start()}-{match.end()}: {match.group()}")
    
    # 항 번호 패턴
    paragraph_pattern = r'[①②③④⑤⑥⑦⑧⑨⑩]'
    paragraph_matches = list(re.finditer(paragraph_pattern, original_content))
    
    print(f"\n항 번호 매치 수: {len(paragraph_matches)}")
    for match in paragraph_matches:
        print(f"  위치 {match.start()}-{match.end()}: {match.group()}")
    
    # 제2조 내의 항들 확인
    print("\n=== 제2조 내 항 분석 ===")
    article2_start = original_content.find('제2조')
    if article2_start != -1:
        # 제3조 시작 찾기
        article3_start = original_content.find('제3조')
        if article3_start != -1:
            article2_content = original_content[article2_start:article3_start]
            print(f"제2조 전체 내용:\n{article2_content}")
            
            # 항 번호 찾기
            paragraphs_in_article2 = list(re.finditer(paragraph_pattern, article2_content))
            print(f"\n제2조 내 항 수: {len(paragraphs_in_article2)}")
            for match in paragraphs_in_article2:
                print(f"  위치 {match.start()}: {match.group()}")

if __name__ == "__main__":
    debug_article_parsing()
