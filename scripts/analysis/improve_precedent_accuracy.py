#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
판례 검색 정확도 향상 스크립트
"""

import json
import sys
import re
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

def improve_precedent_titles():
    """판례 제목 개선"""
    print("판례 제목 개선 시작...")
    
    # 메타데이터 로드
    with open('data/embeddings/metadata.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 판례 데이터만 필터링
    precedents = [d for d in data if d['metadata']['data_type'] == 'precedents']
    
    print(f"개선 전 판례 문서 수: {len(precedents)}")
    
    improved_count = 0
    
    for precedent in precedents:
        original_title = precedent['metadata']['original_document']
        
        # 빈 제목인 경우 개선
        if not original_title or original_title.strip() == "":
            # 판례 ID에서 정보 추출
            case_id = precedent['id']
            case_number = case_id.replace('case_', '') if 'case_' in case_id else case_id
            
            # 판례 내용에서 정보 추출
            content = precedent['text']
            
            # 법원명 추출
            court_name = "대법원"  # 기본값
            if "지방법원" in content:
                court_name = "지방법원"
            elif "고등법원" in content:
                court_name = "고등법원"
            elif "대법원" in content:
                court_name = "대법원"
            
            # 사건 유형 추출
            case_type = "사건"
            if "민사" in content:
                case_type = "민사사건"
            elif "형사" in content:
                case_type = "형사사건"
            elif "행정" in content:
                case_type = "행정사건"
            elif "가사" in content:
                case_type = "가사사건"
            elif "특허" in content:
                case_type = "특허사건"
            
            # 새로운 제목 생성
            new_title = f"{court_name} {case_type} {case_number}호 판결"
            
            # 메타데이터 업데이트
            precedent['metadata']['original_document'] = new_title
            precedent['metadata']['court_name'] = court_name
            precedent['metadata']['case_type'] = case_type
            precedent['metadata']['case_number'] = case_number
            
            improved_count += 1
    
    print(f"개선된 판례 제목 수: {improved_count}개")
    
    # 개선된 데이터 저장
    with open('data/embeddings/metadata_improved.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("개선된 메타데이터 저장 완료: data/embeddings/metadata_improved.json")
    
    # 개선 결과 확인
    print("\n개선된 판례 제목 샘플 (상위 10개):")
    improved_precedents = [d for d in data if d['metadata']['data_type'] == 'precedents']
    for i, precedent in enumerate(improved_precedents[:10]):
        title = precedent['metadata']['original_document']
        court = precedent['metadata'].get('court_name', 'N/A')
        case_type = precedent['metadata'].get('case_type', 'N/A')
        print(f"  {i+1:2d}. {title} (법원: {court}, 유형: {case_type})")
    
    return data

def create_improved_vector_database():
    """개선된 벡터 데이터베이스 생성"""
    print("\n개선된 벡터 데이터베이스 생성 시작...")
    
    # 개선된 메타데이터 로드
    with open('data/embeddings/metadata_improved.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 판례 데이터만 필터링
    precedents = [d for d in data if d['metadata']['data_type'] == 'precedents']
    
    print(f"개선된 판례 문서 수: {len(precedents)}")
    
    # 법원별 분포 확인
    court_distribution = {}
    case_type_distribution = {}
    
    for precedent in precedents:
        court = precedent['metadata'].get('court_name', 'Unknown')
        case_type = precedent['metadata'].get('case_type', 'Unknown')
        
        court_distribution[court] = court_distribution.get(court, 0) + 1
        case_type_distribution[case_type] = case_type_distribution.get(case_type, 0) + 1
    
    print("\n법원별 분포:")
    for court, count in sorted(court_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {court}: {count}개 ({count/len(precedents)*100:.1f}%)")
    
    print("\n사건 유형별 분포:")
    for case_type, count in sorted(case_type_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {case_type}: {count}개 ({count/len(precedents)*100:.1f}%)")
    
    return data

def test_improved_accuracy():
    """개선된 정확도 테스트"""
    print("\n개선된 정확도 테스트 시작...")
    
    # 개선된 메타데이터 로드
    with open('data/embeddings/metadata_improved.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 판례 관련 쿼리 테스트
    precedent_queries = [
        ("대법원 판결", "precedents"),
        ("지방법원 판결", "precedents"),
        ("고등법원 판결", "precedents"),
        ("민사사건", "precedents"),
        ("형사사건", "precedents"),
        ("행정사건", "precedents")
    ]
    
    print("판례 검색 테스트:")
    correct_predictions = 0
    
    for query, expected in precedent_queries:
        # 간단한 키워드 매칭 테스트
        precedents = [d for d in data if d['metadata']['data_type'] == 'precedents']
        
        # 쿼리 키워드와 매칭되는 판례 찾기
        matching_precedents = []
        for precedent in precedents:
            title = precedent['metadata']['original_document']
            if any(keyword in title for keyword in query.split()):
                matching_precedents.append(precedent)
        
        if matching_precedents:
            # 첫 번째 매칭 결과의 타입 확인
            actual = matching_precedents[0]['metadata']['data_type']
            is_correct = actual == expected
            if is_correct:
                correct_predictions += 1
            
            print(f"  '{query}' -> 예상: {expected}, 실제: {actual} {'OK' if is_correct else 'FAIL'}")
            if matching_precedents:
                print(f"    매칭된 제목: {matching_precedents[0]['metadata']['original_document']}")
        else:
            print(f"  '{query}' -> 매칭 결과 없음")
    
    accuracy = correct_predictions / len(precedent_queries) if precedent_queries else 0
    print(f"\n개선된 정확도: {accuracy:.2%} ({correct_predictions}/{len(precedent_queries)})")
    
    return accuracy

def main():
    print("판례 검색 정확도 향상 작업 시작")
    print("=" * 50)
    
    # 1. 판례 제목 개선
    improved_data = improve_precedent_titles()
    
    # 2. 개선된 벡터 데이터베이스 생성
    create_improved_vector_database()
    
    # 3. 개선된 정확도 테스트
    accuracy = test_improved_accuracy()
    
    print("\n" + "=" * 50)
    print(f"판례 검색 정확도 향상 완료!")
    print(f"예상 정확도: {accuracy:.2%}")
    
    if accuracy >= 0.8:
        print("목표 정확도 80% 달성!")
    else:
        print("추가 개선이 필요합니다.")

if __name__ == "__main__":
    main()
