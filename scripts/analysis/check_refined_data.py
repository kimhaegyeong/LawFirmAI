#!/usr/bin/env python3
"""?�제???�이???�인 ?�크립트"""

import json

def main():
    # ?�제???�이??로드
    with open('data/processed/assembly/law/20251011/refined_law_page_001_181503.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"?�제??법령 ?? {len(data['laws'])}")
    print(f"처리 ?�계: {data['processing_stats']}")
    print()
    
    print("�?법령�?조문 ??")
    for i, law in enumerate(data['laws']):
        articles_count = len(law['refined_content']['articles'])
        quality_score = law['data_quality']['quality_score']
        print(f"{i+1}. {law['law_name']}: {articles_count}�?조문, ?�질?�수: {quality_score:.1f}")
    
    print()
    print("�?번째 법령 ?�세 ?�보:")
    first_law = data['laws'][0]
    print(f"법령�? {first_law['law_name']}")
    print(f"조문 ?? {len(first_law['refined_content']['articles'])}")
    print(f"?�질 ?�수: {first_law['data_quality']['quality_score']}")
    print(f"개선 비율: {first_law['data_quality']['improvement_ratio']:.2f}")
    
    print()
    print("�?5�?조문:")
    for i, article in enumerate(first_law['refined_content']['articles'][:5]):
        print(f"{i+1}. {article['article_number']} {article['article_title']}")
        print(f"   ?�용 길이: {len(article['article_content'])}")
        print(f"   ?�위 조문: {len(article['sub_articles'])}�?)
        print(f"   ?�용 미리보기: {article['article_content'][:100]}...")
        print()

if __name__ == "__main__":
    main()



