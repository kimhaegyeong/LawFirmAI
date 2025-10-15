#!/usr/bin/env python3
"""정제된 데이터 확인 스크립트"""

import json

def main():
    # 정제된 데이터 로드
    with open('data/processed/assembly/law/20251011/refined_law_page_001_181503.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"정제된 법령 수: {len(data['laws'])}")
    print(f"처리 통계: {data['processing_stats']}")
    print()
    
    print("각 법령별 조문 수:")
    for i, law in enumerate(data['laws']):
        articles_count = len(law['refined_content']['articles'])
        quality_score = law['data_quality']['quality_score']
        print(f"{i+1}. {law['law_name']}: {articles_count}개 조문, 품질점수: {quality_score:.1f}")
    
    print()
    print("첫 번째 법령 상세 정보:")
    first_law = data['laws'][0]
    print(f"법령명: {first_law['law_name']}")
    print(f"조문 수: {len(first_law['refined_content']['articles'])}")
    print(f"품질 점수: {first_law['data_quality']['quality_score']}")
    print(f"개선 비율: {first_law['data_quality']['improvement_ratio']:.2f}")
    
    print()
    print("첫 5개 조문:")
    for i, article in enumerate(first_law['refined_content']['articles'][:5]):
        print(f"{i+1}. {article['article_number']} {article['article_title']}")
        print(f"   내용 길이: {len(article['article_content'])}")
        print(f"   하위 조문: {len(article['sub_articles'])}개")
        print(f"   내용 미리보기: {article['article_content'][:100]}...")
        print()

if __name__ == "__main__":
    main()



