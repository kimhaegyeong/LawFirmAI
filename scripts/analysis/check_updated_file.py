#!/usr/bin/env python3
"""업데이트된 파일 확인 스크립트"""

import json

def main():
    # 업데이트된 파일 로드
    with open('data/raw/assembly/law/20251010/law_page_001_181503.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"업데이트 통계: {data['update_stats']}")
    print(f"업데이트 시간: {data['content_updated_at']}")
    print(f"업데이트 버전: {data['content_update_version']}")
    print()
    
    print("각 법령별 업데이트 결과:")
    for i, law in enumerate(data['laws']):
        original_len = law['original_content_length']
        updated_len = law['updated_content_length']
        ratio = law['content_improvement_ratio']
        print(f"{i+1}. {law['law_name']}: {original_len} -> {updated_len} 문자 (개선비율: {ratio:.2f})")
    
    print()
    print("첫 번째 법령 업데이트된 내용 미리보기:")
    first_law = data['laws'][0]
    print(f"법령명: {first_law['law_name']}")
    print(f"업데이트된 내용 길이: {len(first_law['law_content'])}")
    print(f"내용 미리보기:")
    print(first_law['law_content'][:500] + "...")

if __name__ == "__main__":
    main()



