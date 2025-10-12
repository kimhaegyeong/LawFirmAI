#!/usr/bin/env python3
"""
최종 개선된 sub_articles 파싱 결과 검증 스크립트
"""

import json
from pathlib import Path

def verify_final_improved_sub_articles():
    """최종 개선된 sub_articles 파싱 결과를 검증합니다."""
    
    # 최종 개선된 파일과 기존 파일 비교
    final_file = Path("data/processed/assembly/law/2025101201_final_fixed/20251012/_대한민국_법원의_날_제정에_관한_규칙_assembly_law_1951.json")
    original_file = Path("data/processed/assembly/law/2025101201_fixed_sub_articles/20251012/_대한민국_법원의_날_제정에_관한_규칙_assembly_law_1951.json")
    
    if not final_file.exists():
        print(f"최종 개선된 파일을 찾을 수 없습니다: {final_file}")
        return False
    
    if not original_file.exists():
        print(f"기존 파일을 찾을 수 없습니다: {original_file}")
        return False
    
    # 파일 로드
    with open(final_file, 'r', encoding='utf-8') as f:
        final_data = json.load(f)
    
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    print("=== 최종 개선된 sub_articles 파싱 결과 검증 ===")
    print(f"법률명: {final_data.get('law_name', 'Unknown')}")
    
    # 제1조 비교 (목이 없어야 하는 조문)
    final_article_1 = None
    original_article_1 = None
    
    for article in final_data.get('articles', []):
        if article.get('article_number') == '제1조' and article.get('article_title') == '목적':
            final_article_1 = article
            break
    
    for article in original_data.get('articles', []):
        if article.get('article_number') == '제1조' and article.get('article_title') == '목적':
            original_article_1 = article
            break
    
    if final_article_1 and original_article_1:
        print(f"\n=== 제1조 비교 ===")
        print(f"기존 sub_articles 수: {len(original_article_1.get('sub_articles', []))}")
        print(f"최종 sub_articles 수: {len(final_article_1.get('sub_articles', []))}")
        
        # 목(目) 항목 분석
        original_mok_items = [item for item in original_article_1.get('sub_articles', []) if item.get('type') == '목']
        final_mok_items = [item for item in final_article_1.get('sub_articles', []) if item.get('type') == '목']
        
        print(f"기존 목(目) 항목 수: {len(original_mok_items)}")
        print(f"최종 목(目) 항목 수: {len(final_mok_items)}")
        
        if len(final_mok_items) == 0:
            print("✅ 성공: 제1조에서 목(目) 항목이 완전히 제거되었습니다!")
        else:
            print("❌ 실패: 제1조에 여전히 목(目) 항목이 있습니다.")
            for i, item in enumerate(final_mok_items):
                print(f"  {i+1}. {item.get('letter')}: {item.get('content', '')[:50]}...")
        
        # 전체 sub_articles 표시
        print(f"\n=== 최종 제1조 sub_articles ===")
        for i, sub_article in enumerate(final_article_1.get('sub_articles', [])):
            sub_type = sub_article.get('type', 'Unknown')
            number = sub_article.get('number', 'Unknown')
            content = sub_article.get('content', '')
            print(f"{i+1}. {sub_type} {number}: {content[:50]}...")
        
        return len(final_mok_items) == 0
    else:
        print("제1조를 찾을 수 없습니다.")
        return False

if __name__ == "__main__":
    verify_final_improved_sub_articles()
