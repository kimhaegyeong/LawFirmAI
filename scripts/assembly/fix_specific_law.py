#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
특정 법률 문서의 조문 파싱 문제 수정 스크립트
제2조 2항 누락 문제 해결
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

from parsers.improved_article_parser import ImprovedArticleParser

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_specific_law_file():
    """특정 법률 파일 수정"""
    
    # 원본 데이터 (완전한 내용)
    original_content = """제1조(목적) 이 규칙은 대한민국 법원이 사법주권을 회복한 날을 기념하기 위하여 『대한민국 법원의 날』을 제정하고, 사법독립과 법치주의의 중요성을 알리며 그 의의를 기념하기 위한 행사 등을 진행함에 있어 필요한 사항을 규정함을 목적으로 한다.
제2조(정의 및 명칭) ① 제1조에서 사법주권을 회복한 날이라 함은, 일제에 사법주권을 빼앗겼다가 대한민국이 1948년 9월 13일 미군정으로부터 사법권을 이양받음으로써 헌법기관인 대한민국 법원이 실질적으로 수립된 날을 의미한다.
② 『대한민국 법원의 날』은 매년 9월 13일로 한다.
제3조(기념식 및 행사) ① 법원은 『대한민국 법원의 날』에 기념식과 그에 부수되는 행사를 실시할 수 있다.
제4조(포상) ① 대법원장은 제2조제1항에 규정된 기념일의 의식에서 사법부의 발전 또는 법률문화의 향상에 공헌한 행적이 뚜렷한 사람에게 포상할 수 있다.
부칙 <제2605호, 2015.6.29.>펼치기접기
이 규칙은 공포한 날부터 시행한다."""
    
    # 규칙 기반 파서로 올바른 파싱
    parser = ImprovedArticleParser()
    result = parser.parse_law_document(original_content)
    
    # 메타데이터 구성
    fixed_data = {
        "law_id": "assembly_law_1951",
        "law_name": "「대한민국 법원의 날」제정에 관한 규칙",
        "law_type": "대법원규칙",
        "category": "제2장 법원행정",
        "promulgation_number": "제2605호",
        "promulgation_date": "2015.6.29",
        "enforcement_date": "2015.6.29",
        "amendment_type": "제정",
        "ministry": "",
        "articles": result['all_articles']
    }
    
    # 수정된 파일 저장
    output_path = "data/processed/assembly/law/ml_enhanced/20251013/_대한민국_법원의_날_제정에_관한_규칙_assembly_law_1951.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, ensure_ascii=False, indent=4)
    
    print(f"수정된 파일 저장: {output_path}")
    print(f"총 조문 수: {len(fixed_data['articles'])}")
    
    # 각 조문 확인
    for i, article in enumerate(fixed_data['articles']):
        print(f"\n조문 {i+1}:")
        print(f"  번호: {article['article_number']}")
        print(f"  제목: {article.get('article_title', 'N/A')}")
        print(f"  항 수: {len(article.get('sub_articles', []))}")
        
        if article.get('sub_articles'):
            for j, sub in enumerate(article['sub_articles']):
                print(f"    항 {j+1}: {sub['content'][:50]}...")
    
    return fixed_data

def validate_fix():
    """수정 결과 검증"""
    
    file_path = "data/processed/assembly/law/ml_enhanced/20251013/_대한민국_법원의_날_제정에_관한_규칙_assembly_law_1951.json"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\n=== 수정 결과 검증 ===")
    print(f"총 조문 수: {len(data['articles'])}")
    
    # 제2조 확인
    article2 = None
    for article in data['articles']:
        if article['article_number'] == '제2조':
            article2 = article
            break
    
    if article2:
        print(f"\n제2조 확인:")
        print(f"  제목: {article2.get('article_title', 'N/A')}")
        print(f"  항 수: {len(article2.get('sub_articles', []))}")
        
        if article2.get('sub_articles'):
            for i, sub in enumerate(article2['sub_articles']):
                print(f"  항 {i+1}: {sub['content']}")
        
        # 제2조 2항이 있는지 확인
        has_paragraph2 = any('매년 9월 13일' in sub['content'] for sub in article2.get('sub_articles', []))
        if has_paragraph2:
            print("\n[OK] 제2조 2항이 올바르게 포함되었습니다!")
        else:
            print("\n[ERROR] 제2조 2항이 여전히 누락되었습니다.")
    else:
        print("\n[ERROR] 제2조를 찾을 수 없습니다.")

if __name__ == "__main__":
    print("특정 법률 문서 조문 파싱 문제 수정")
    print("=" * 50)
    
    # 파일 수정
    fixed_data = fix_specific_law_file()
    
    # 수정 결과 검증
    validate_fix()
    
    print("\n수정 완료!")
