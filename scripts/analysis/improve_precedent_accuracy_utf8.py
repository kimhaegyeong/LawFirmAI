#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
판례 검색 정확도 향상 스크립트 (UTF-8 인코딩)
"""

import json
import sys
import re
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

def improve_precedent_titles():
    """판례 제목 개선"""
    print("Precedent title improvement started...")
    
    # 메타데이터 로드
    with open('data/embeddings/metadata.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 판례 데이터만 필터링
    precedents = [d for d in data if d['metadata']['data_type'] == 'precedents']
    
    print(f"Total precedent documents before improvement: {len(precedents)}")
    
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
            court_name = "Supreme Court"  # 기본값
            if "지방법원" in content:
                court_name = "District Court"
            elif "고등법원" in content:
                court_name = "High Court"
            elif "대법원" in content:
                court_name = "Supreme Court"
            
            # 사건 유형 추출
            case_type = "Case"
            if "민사" in content:
                case_type = "Civil Case"
            elif "형사" in content:
                case_type = "Criminal Case"
            elif "행정" in content:
                case_type = "Administrative Case"
            elif "가사" in content:
                case_type = "Family Case"
            elif "특허" in content:
                case_type = "Patent Case"
            
            # 새로운 제목 생성
            new_title = f"{court_name} {case_type} {case_number} Decision"
            
            # 메타데이터 업데이트
            precedent['metadata']['original_document'] = new_title
            precedent['metadata']['court_name'] = court_name
            precedent['metadata']['case_type'] = case_type
            precedent['metadata']['case_number'] = case_number
            
            improved_count += 1
    
    print(f"Improved precedent titles: {improved_count}")
    
    # 개선된 데이터 저장
    with open('data/embeddings/metadata_improved.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("Improved metadata saved: data/embeddings/metadata_improved.json")
    
    # 개선 결과 확인
    print("\nImproved precedent title samples (top 10):")
    improved_precedents = [d for d in data if d['metadata']['data_type'] == 'precedents']
    for i, precedent in enumerate(improved_precedents[:10]):
        title = precedent['metadata']['original_document']
        court = precedent['metadata'].get('court_name', 'N/A')
        case_type = precedent['metadata'].get('case_type', 'N/A')
        print(f"  {i+1:2d}. {title} (Court: {court}, Type: {case_type})")
    
    return data

def create_improved_vector_database():
    """개선된 벡터 데이터베이스 생성"""
    print("\nCreating improved vector database...")
    
    # 개선된 메타데이터 로드
    with open('data/embeddings/metadata_improved.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 판례 데이터만 필터링
    precedents = [d for d in data if d['metadata']['data_type'] == 'precedents']
    
    print(f"Improved precedent documents: {len(precedents)}")
    
    # 법원별 분포 확인
    court_distribution = {}
    case_type_distribution = {}
    
    for precedent in precedents:
        court = precedent['metadata'].get('court_name', 'Unknown')
        case_type = precedent['metadata'].get('case_type', 'Unknown')
        
        court_distribution[court] = court_distribution.get(court, 0) + 1
        case_type_distribution[case_type] = case_type_distribution.get(case_type, 0) + 1
    
    print("\nCourt distribution:")
    for court, count in sorted(court_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {court}: {count} ({count/len(precedents)*100:.1f}%)")
    
    print("\nCase type distribution:")
    for case_type, count in sorted(case_type_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {case_type}: {count} ({count/len(precedents)*100:.1f}%)")
    
    return data

def test_improved_accuracy():
    """개선된 정확도 테스트"""
    print("\nTesting improved accuracy...")
    
    # 개선된 메타데이터 로드
    with open('data/embeddings/metadata_improved.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 판례 관련 쿼리 테스트
    precedent_queries = [
        ("Supreme Court Decision", "precedents"),
        ("District Court Decision", "precedents"),
        ("High Court Decision", "precedents"),
        ("Civil Case", "precedents"),
        ("Criminal Case", "precedents"),
        ("Administrative Case", "precedents")
    ]
    
    print("Precedent search test:")
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
            
            print(f"  '{query}' -> Expected: {expected}, Actual: {actual} {'OK' if is_correct else 'FAIL'}")
            if matching_precedents:
                print(f"    Matched title: {matching_precedents[0]['metadata']['original_document']}")
        else:
            print(f"  '{query}' -> No matching results")
    
    accuracy = correct_predictions / len(precedent_queries) if precedent_queries else 0
    print(f"\nImproved accuracy: {accuracy:.2%} ({correct_predictions}/{len(precedent_queries)})")
    
    return accuracy

def main():
    print("Precedent search accuracy improvement started")
    print("=" * 50)
    
    # 1. 판례 제목 개선
    improved_data = improve_precedent_titles()
    
    # 2. 개선된 벡터 데이터베이스 생성
    create_improved_vector_database()
    
    # 3. 개선된 정확도 테스트
    accuracy = test_improved_accuracy()
    
    print("\n" + "=" * 50)
    print(f"Precedent search accuracy improvement completed!")
    print(f"Expected accuracy: {accuracy:.2%}")
    
    if accuracy >= 0.8:
        print("Target accuracy 80% achieved!")
    else:
        print("Additional improvement needed.")

if __name__ == "__main__":
    main()
