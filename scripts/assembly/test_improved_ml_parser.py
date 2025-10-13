#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개선된 ML 파서 테스트 스크립트
부칙 파싱 로직이 통합된 ML 강화 파서의 성능을 검증
"""

import json
import sys
import os
from pathlib import Path
import logging
from typing import Dict, List, Any

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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_improved_ml_parser():
    """개선된 ML 파서 테스트"""
    
    # 테스트용 법률 내용 (부칙 포함)
    test_content = """
    제1조(목적) 이 법은 공공기관의 소방안전관리에 관한 사항을 규정함을 목적으로 한다.
    
    제2조(적용 범위) 이 법은 다음 각 호의 어느 하나에 해당하는 공공기관에 적용한다.
    1. 국가기관
    2. 지방자치단체
    3. 공공기관의 운영에 관한 법률 제4조에 따른 공공기관
    
    제3조(기관장의 책임) 제2조에 따른 공공기관의 장은 소방안전관리에 대한 책임을 진다.
    
    부칙 <법률 제12345호, 2024. 12. 31.>
    제1조(시행일) 이 법은 공포한 날부터 시행한다.
    제2조(적용례) 이 법은 이 법 시행 이후 최초로 발생하는 위반행위부터 적용한다.
    제3조(경과조치) 이 법 시행 당시 종전의 규정에 따라 인가를 받은 자는 이 법에 따라 허가를 받은 것으로 본다.
    """
    
    print("=== 개선된 ML 파서 테스트 ===")
    
    # 규칙 기반 파서 (비교용)
    rule_parser = ImprovedArticleParser()
    rule_result = rule_parser.parse_law_document(test_content)
    
    # 개선된 ML 파서
    ml_parser = MLEnhancedArticleParser()
    ml_result = ml_parser.parse_law_document(test_content)
    
    print(f"\n규칙 기반 파서 결과:")
    print(f"- 총 조문 수: {rule_result['total_articles']}")
    print(f"- 본칙 조문: {len(rule_result.get('main_articles', []))}")
    print(f"- 부칙 조문: {len(rule_result.get('supplementary_articles', []))}")
    
    print(f"\n개선된 ML 파서 결과:")
    print(f"- 총 조문 수: {ml_result['total_articles']}")
    print(f"- 본칙 조문: {len(ml_result.get('main_articles', []))}")
    print(f"- 부칙 조문: {len(ml_result.get('supplementary_articles', []))}")
    print(f"- ML 모델 사용: {ml_result['ml_enhanced']}")
    
    print(f"\n본칙 조문 비교:")
    rule_main = [a['article_number'] for a in rule_result.get('all_articles', []) if not a.get('is_supplementary', False)]
    ml_main = [a['article_number'] for a in ml_result.get('main_articles', [])]
    
    print(f"- 규칙 기반: {rule_main}")
    print(f"- ML 강화: {ml_main}")
    
    print(f"\n부칙 조문 비교:")
    rule_supp = [a['article_number'] for a in rule_result.get('all_articles', []) if a.get('is_supplementary', False)]
    ml_supp = [a['article_number'] for a in ml_result.get('supplementary_articles', [])]
    
    print(f"- 규칙 기반: {rule_supp}")
    print(f"- ML 강화: {ml_supp}")
    
    # 개선 사항 확인
    print(f"\n=== 개선 사항 확인 ===")
    
    # 부칙 조문 수 비교
    if len(ml_supp) > len(rule_supp):
        print(f"[OK] 부칙 조문 수 증가: {len(rule_supp)} → {len(ml_supp)}")
    elif len(ml_supp) == len(rule_supp):
        print(f"[OK] 부칙 조문 수 동일: {len(ml_supp)}")
    else:
        print(f"[ERROR] 부칙 조문 수 감소: {len(rule_supp)} → {len(ml_supp)}")
    
    # 전체 조문 수 비교
    if ml_result['total_articles'] >= rule_result['total_articles']:
        print(f"[OK] 전체 조문 수 개선: {rule_result['total_articles']} → {ml_result['total_articles']}")
    else:
        print(f"[ERROR] 전체 조문 수 감소: {rule_result['total_articles']} → {ml_result['total_articles']}")
    
    return ml_result

def test_real_law_data():
    """실제 법률 데이터로 테스트"""
    
    # 실제 법률 파일 찾기
    raw_dirs = [
        "data/raw/assembly/law/20251010",
        "data/raw/assembly/law/20251011", 
        "data/raw/assembly/law/20251012",
        "data/raw/assembly/law/2025101201"
    ]
    
    test_file = None
    for raw_dir in raw_dirs:
        if Path(raw_dir).exists():
            files = list(Path(raw_dir).glob("*.json"))
            if files:
                test_file = files[0]
                break
    
    if not test_file:
        print("실제 법률 데이터 파일을 찾을 수 없습니다.")
        return
    
    print(f"\n=== 실제 법률 데이터 테스트: {test_file.name} ===")
    
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'laws' in data and data['laws']:
            law = data['laws'][0]
            law_content = law.get('law_content', '')
            law_name = law.get('law_name', 'Unknown')
            
            if law_content:
                print(f"법률명: {law_name}")
                
                # 개선된 ML 파서로 파싱
                ml_parser = MLEnhancedArticleParser()
                result = ml_parser.parse_law_document(law_content)
                
                print(f"파싱 결과:")
                print(f"- 총 조문 수: {result['total_articles']}")
                print(f"- 본칙 조문: {len(result.get('main_articles', []))}")
                print(f"- 부칙 조문: {len(result.get('supplementary_articles', []))}")
                
                # 부칙 조문 상세 정보
                if result.get('supplementary_articles'):
                    print(f"\n부칙 조문 상세:")
                    for article in result['supplementary_articles']:
                        print(f"- {article['article_number']}: {article['article_title']}")
                
                return result
            else:
                print("법률 내용이 없습니다.")
        else:
            print("법률 데이터가 없습니다.")
            
    except Exception as e:
        print(f"파일 처리 중 오류 발생: {e}")

if __name__ == "__main__":
    # 기본 테스트
    test_improved_ml_parser()
    
    # 실제 데이터 테스트
    test_real_law_data()
