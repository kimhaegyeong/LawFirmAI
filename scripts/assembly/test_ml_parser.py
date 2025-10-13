#!/usr/bin/env python3
"""
ML 강화 파서 테스트 스크립트
실제 법률 데이터로 ML 모델의 성능을 검증
"""

import json
import sys
from pathlib import Path
import logging

# 파서 모듈 경로 추가
sys.path.append(str(Path(__file__).parent / 'parsers'))

from ml_enhanced_parser import MLEnhancedArticleParser
from parsers.improved_article_parser import ImprovedArticleParser

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_with_real_data():
    """실제 법률 데이터로 테스트"""
    
    # 원본 데이터에서 법률 내용 찾기
    law_id = "assembly_law_3727"
    law_content = None
    
    # 원본 데이터 파일들에서 해당 법률 찾기
    raw_files = list(Path("data/raw/assembly/law").glob("**/*.json"))
    
    for raw_file in raw_files:
        try:
            with open(raw_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            if isinstance(raw_data, dict) and 'laws' in raw_data:
                for law in raw_data['laws']:
                    if law.get('law_id') == law_id:
                        law_content = law.get('law_content', '')
                        break
            elif isinstance(raw_data, list):
                for law in raw_data:
                    if law.get('law_id') == law_id:
                        law_content = law.get('law_content', '')
                        break
            
            if law_content:
                break
                
        except Exception as e:
            logger.warning(f"Error reading {raw_file}: {e}")
            continue
    
    if not law_content:
        logger.error(f"Could not find law content for {law_id}")
        return
    
    print("=== ML-Enhanced Parser Test ===")
    print(f"Law ID: {law_id}")
    print(f"Content length: {len(law_content)} characters")
    print()
    
    # 기존 규칙 기반 파서 테스트
    print("1. Rule-based Parser:")
    rule_parser = ImprovedArticleParser()
    rule_result = rule_parser.parse_law_document(law_content)
    
    print(f"   - Total articles: {rule_result['total_articles']}")
    print(f"   - Main articles: {len(rule_result['main_articles'])}")
    print(f"   - Supplementary articles: {len(rule_result['supplementary_articles'])}")
    
    # ML 강화 파서 테스트
    print("\n2. ML-Enhanced Parser:")
    ml_parser = MLEnhancedArticleParser()
    ml_result = ml_parser.parse_law_document(law_content)
    
    print(f"   - Total articles: {ml_result['total_articles']}")
    print(f"   - Main articles: {len(ml_result['main_articles'])}")
    print(f"   - Supplementary articles: {len(ml_result['supplementary_articles'])}")
    print(f"   - ML enhanced: {ml_result['ml_enhanced']}")
    
    # 결과 비교
    print("\n=== Comparison ===")
    print(f"Rule-based articles: {rule_result['total_articles']}")
    print(f"ML-enhanced articles: {ml_result['total_articles']}")
    print(f"Difference: {ml_result['total_articles'] - rule_result['total_articles']}")
    
    # 조문 제목이 있는 조문 수 비교
    rule_with_titles = sum(1 for article in rule_result['all_articles'] if article.get('article_title'))
    ml_with_titles = sum(1 for article in ml_result['all_articles'] if article.get('article_title'))
    
    print(f"\nRule-based articles with titles: {rule_with_titles}")
    print(f"ML-enhanced articles with titles: {ml_with_titles}")
    print(f"Title improvement: {ml_with_titles - rule_with_titles}")
    
    # 상세 조문 정보 출력
    print("\n=== Detailed Article Comparison ===")
    print("Rule-based parser articles:")
    for i, article in enumerate(rule_result['all_articles'][:5]):  # 처음 5개만
        title = article.get('article_title', 'No title')
        print(f"  {i+1}. {article['article_number']}: {title}")
    
    print("\nML-enhanced parser articles:")
    for i, article in enumerate(ml_result['all_articles'][:5]):  # 처음 5개만
        title = article.get('article_title', 'No title')
        print(f"  {i+1}. {article['article_number']}: {title}")

def test_problematic_case():
    """문제가 있던 케이스 테스트"""
    
    # 제39조가 제1조와 제2조 사이에 잘못 파싱되던 케이스
    problematic_content = """
    제1조(목적) 이 영은 「화재의 예방 및 안전관리에 관한 법률」 제39조에 따라 공공기관의 건축물·인공구조물 및 물품 등을 화재로부터 보호하기 위하여 소방안전관리에 필요한 사항을 규정함을 목적으로 한다.
    
    제2조(적용 범위) 이 영은 다음 각 호의 어느 하나에 해당하는 공공기관에 적용한다.
    1. 국가기관
    2. 지방자치단체
    3. 「공공기관의 운영에 관한 법률」 제4조에 따른 공공기관
    5. 「사립학교법」 제2조제1항에 따른 사립학교
    
    제3조 삭제
    
    제4조(기관장의 책임) 제2조에 따른 공공기관의 장(이하 "기관장"이라 한다)은 다음 각 호의 사항에 대한 감독책임을 진다.
    1. 소방안전관리계획의 수립·시행에 관한 사항
    2. 소방계획의 수립·시행에 관한 사항
    """
    
    print("\n=== Problematic Case Test ===")
    print("Testing case where '제39조' was incorrectly parsed between 제1조 and 제2조")
    print()
    
    # 기존 파서 테스트
    print("1. Rule-based Parser:")
    rule_parser = ImprovedArticleParser()
    rule_result = rule_parser.parse_law_document(problematic_content)
    
    print(f"   - Total articles: {rule_result['total_articles']}")
    for article in rule_result['all_articles']:
        print(f"   - {article['article_number']}: {article.get('article_title', 'No title')}")
    
    # ML 강화 파서 테스트
    print("\n2. ML-Enhanced Parser:")
    ml_parser = MLEnhancedArticleParser()
    ml_result = ml_parser.parse_law_document(problematic_content)
    
    print(f"   - Total articles: {ml_result['total_articles']}")
    for article in ml_result['all_articles']:
        print(f"   - {article['article_number']}: {article.get('article_title', 'No title')}")
    
    # 제39조가 잘못 파싱되었는지 확인
    rule_article_numbers = [article['article_number'] for article in rule_result['all_articles']]
    ml_article_numbers = [article['article_number'] for article in ml_result['all_articles']]
    
    print(f"\nRule-based article numbers: {rule_article_numbers}")
    print(f"ML-enhanced article numbers: {ml_article_numbers}")
    
    if '제39조' in rule_article_numbers:
        print("[ERROR] Rule-based parser incorrectly parsed 제39조 as a separate article")
    else:
        print("[OK] Rule-based parser correctly avoided 제39조")
    
    if '제39조' in ml_article_numbers:
        print("[ERROR] ML-enhanced parser incorrectly parsed 제39조 as a separate article")
    else:
        print("[OK] ML-enhanced parser correctly avoided 제39조")

def main():
    """메인 함수"""
    print("ML-Enhanced Article Parser Test")
    print("=" * 50)
    
    # 실제 데이터로 테스트
    test_with_real_data()
    
    # 문제 케이스 테스트
    test_problematic_case()
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    main()
