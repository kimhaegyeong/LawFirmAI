#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
판례 데이터 분석 및 개선 방안 도출
"""

import json
import sys
from pathlib import Path

def analyze_precedent_data():
    """판례 데이터 분석"""
    print("판례 데이터 분석 시작...")
    
    # 메타데이터 로드
    with open('data/embeddings/metadata.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 판례 데이터만 필터링
    precedents = [d for d in data if d['metadata']['data_type'] == 'precedents']
    
    print(f"총 판례 문서 수: {len(precedents)}")
    print("\n판례 제목 샘플 (상위 20개):")
    
    for i, precedent in enumerate(precedents[:20]):
        title = precedent['metadata']['original_document']
        print(f"  {i+1:2d}. {title}")
    
    # 제목 패턴 분석
    print("\n제목 패턴 분석:")
    patterns = {}
    for precedent in precedents:
        title = precedent['metadata']['original_document']
        if not title or title.strip() == "":
            patterns['빈 제목'] = patterns.get('빈 제목', 0) + 1
        elif '대법원' in title:
            patterns['대법원'] = patterns.get('대법원', 0) + 1
        elif '지방법원' in title:
            patterns['지방법원'] = patterns.get('지방법원', 0) + 1
        elif '고등법원' in title:
            patterns['고등법원'] = patterns.get('고등법원', 0) + 1
        elif '판결' in title:
            patterns['판결'] = patterns.get('판결', 0) + 1
        elif '판례' in title:
            patterns['판례'] = patterns.get('판례', 0) + 1
        else:
            patterns['기타'] = patterns.get('기타', 0) + 1
    
    for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count}개 ({count/len(precedents)*100:.1f}%)")
    
    # 빈 제목 문제 분석
    empty_titles = [p for p in precedents if not p['metadata']['original_document'] or p['metadata']['original_document'].strip() == ""]
    print(f"\n빈 제목 문제:")
    print(f"  빈 제목 문서 수: {len(empty_titles)}개")
    
    if empty_titles:
        print("  빈 제목 문서 샘플:")
        for i, precedent in enumerate(empty_titles[:5]):
            print(f"    {i+1}. ID: {precedent['id']}")
            print(f"       내용 미리보기: {precedent['text'][:100]}...")
    
    # 개선 방안 제시
    print("\n개선 방안:")
    print("1. 빈 제목 문제 해결:")
    print("   - 판례 데이터에서 case_name 또는 title 필드 활용")
    print("   - 판례 ID를 기반으로 제목 생성")
    print("   - 판례 내용에서 첫 문장을 제목으로 사용")
    
    print("2. 제목 정규화:")
    print("   - '대법원', '지방법원', '고등법원' 키워드 강화")
    print("   - 판례 관련 키워드 추가 ('판결', '판례', '사건' 등)")
    print("   - 법원명 표준화")
    
    print("3. 임베딩 개선:")
    print("   - 판례 특화 키워드 추가")
    print("   - 법원명과 사건번호 정보 강화")
    print("   - 판례 내용의 법적 맥락 강조")

if __name__ == "__main__":
    analyze_precedent_data()
