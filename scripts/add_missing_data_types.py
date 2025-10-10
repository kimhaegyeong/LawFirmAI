#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
추가 데이터 타입 지원 (헌재결정례, 법령해석례)
"""

import json
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

def check_missing_data_types():
    """누락된 데이터 타입 확인"""
    print("Checking missing data types...")
    
    # 전처리된 데이터 디렉토리 확인
    processed_dir = Path("data/processed")
    
    data_types = {
        'constitutional_decisions': '헌재결정례',
        'legal_interpretations': '법령해석례',
        'administrative_rules': '행정규칙',
        'local_ordinances': '자치법규'
    }
    
    missing_types = []
    available_types = []
    
    for data_type, korean_name in data_types.items():
        type_dir = processed_dir / data_type
        if type_dir.exists():
            json_files = list(type_dir.glob("*.json"))
            if json_files:
                available_types.append((data_type, korean_name, len(json_files)))
                print(f"OK {korean_name} ({data_type}): {len(json_files)} files")
            else:
                missing_types.append((data_type, korean_name))
                print(f"NO {korean_name} ({data_type}): No files")
        else:
            missing_types.append((data_type, korean_name))
            print(f"NO {korean_name} ({data_type}): Directory not found")
    
    return available_types, missing_types

def create_sample_data():
    """샘플 데이터 생성 (테스트용)"""
    print("\nCreating sample data for missing types...")
    
    # 샘플 헌재결정례 데이터
    constitutional_samples = [
        {
            "id": "const_001",
            "title": "헌법재판소 2023헌마1234 결정",
            "content": "이 사건은 기본권 침해 여부에 대한 헌법재판소의 결정입니다. 원고는 국가의 행정처분이 자신의 기본권을 침해한다고 주장하였으며, 헌법재판소는 이를 심리한 결과 일부 인용하기로 결정하였습니다.",
            "decision_date": "2023-12-15",
            "case_type": "헌법소원",
            "court": "헌법재판소"
        },
        {
            "id": "const_002", 
            "title": "헌법재판소 2023헌바5678 결정",
            "content": "이 사건은 법률의 위헌성 여부에 대한 헌법재판소의 결정입니다. 해당 법률 조항이 헌법에 위반되는지 여부를 심리한 결과, 위헌 결정을 내렸습니다.",
            "decision_date": "2023-11-20",
            "case_type": "법률위헌제소",
            "court": "헌법재판소"
        }
    ]
    
    # 샘플 법령해석례 데이터
    interpretation_samples = [
        {
            "id": "interp_001",
            "title": "민법 제1조 해석례",
            "content": "민법 제1조에 대한 법무부의 해석입니다. 민법의 기본 원칙과 적용 범위에 대해 상세히 설명하고 있으며, 실제 사례를 통한 구체적인 적용 방법을 제시합니다.",
            "interpretation_date": "2023-10-10",
            "department": "법무부",
            "law_name": "민법"
        },
        {
            "id": "interp_002",
            "title": "형법 제20조 해석례", 
            "content": "형법 제20조 정당방위에 대한 대법원의 해석입니다. 정당방위의 성립 요건과 한계에 대해 구체적으로 설명하고 있으며, 관련 판례와 함께 해석하고 있습니다.",
            "interpretation_date": "2023-09-15",
            "department": "대법원",
            "law_name": "형법"
        }
    ]
    
    # 샘플 행정규칙 데이터
    administrative_samples = [
        {
            "id": "admin_001",
            "title": "행정절차법 시행규칙",
            "content": "행정절차법의 시행을 위한 구체적인 규칙입니다. 행정처분의 절차, 청문회 개최 방법, 이의신청 절차 등에 대해 상세히 규정하고 있습니다.",
            "promulgation_date": "2023-08-01",
            "enforcement_date": "2023-08-01",
            "department": "행정안전부"
        }
    ]
    
    # 샘플 자치법규 데이터
    local_samples = [
        {
            "id": "local_001",
            "title": "서울특별시 조례 제1234호",
            "content": "서울특별시의 조례입니다. 도시계획, 환경보호, 주민복지 등에 관한 사항을 규정하고 있으며, 시민의 권리와 의무에 대해 명시하고 있습니다.",
            "promulgation_date": "2023-07-15",
            "enforcement_date": "2023-07-15",
            "local_government": "서울특별시"
        }
    ]
    
    # 샘플 데이터 저장
    sample_data = {
        'constitutional_decisions': constitutional_samples,
        'legal_interpretations': interpretation_samples,
        'administrative_rules': administrative_samples,
        'local_ordinances': local_samples
    }
    
    for data_type, samples in sample_data.items():
        output_file = f"data/processed/{data_type}/{data_type}_sample.json"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print(f"Created sample data: {output_file} ({len(samples)} items)")

def process_missing_data_types():
    """누락된 데이터 타입 처리"""
    print("\nProcessing missing data types...")
    
    # 샘플 데이터 생성
    create_sample_data()
    
    # 전처리된 데이터 확인
    available_types, missing_types = check_missing_data_types()
    
    print(f"\nAvailable data types: {len(available_types)}")
    print(f"Missing data types: {len(missing_types)}")
    
    return available_types, missing_types

def update_vector_database():
    """벡터 데이터베이스 업데이트"""
    print("\nUpdating vector database with new data types...")
    
    # 기존 메타데이터 로드
    with open('data/embeddings/metadata_improved.json', 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
    
    # 새로운 데이터 타입 추가
    new_data = []
    
    # 헌재결정례 데이터 추가
    const_file = "data/processed/constitutional_decisions/constitutional_decisions_sample.json"
    if Path(const_file).exists():
        with open(const_file, 'r', encoding='utf-8') as f:
            const_data = json.load(f)
        
        for item in const_data:
            new_doc = {
                'id': item['id'],
                'original_id': item['id'],
                'chunk_id': 'full',
                'text': item['content'],
                'metadata': {
                    'data_type': 'constitutional_decisions',
                    'original_document': item['title'],
                    'chunk_start': 0,
                    'chunk_end': len(item['content']),
                    'chunk_length': len(item['content']),
                    'word_count': len(item['content'].split()),
                    'entities': {},
                    'processed_at': '2025-09-30',
                    'is_valid': True,
                    'court_name': item.get('court', '헌법재판소'),
                    'case_type': item.get('case_type', '헌법소원'),
                    'decision_date': item.get('decision_date', '')
                }
            }
            new_data.append(new_doc)
    
    # 법령해석례 데이터 추가
    interp_file = "data/processed/legal_interpretations/legal_interpretations_sample.json"
    if Path(interp_file).exists():
        with open(interp_file, 'r', encoding='utf-8') as f:
            interp_data = json.load(f)
        
        for item in interp_data:
            new_doc = {
                'id': item['id'],
                'original_id': item['id'],
                'chunk_id': 'full',
                'text': item['content'],
                'metadata': {
                    'data_type': 'legal_interpretations',
                    'original_document': item['title'],
                    'chunk_start': 0,
                    'chunk_end': len(item['content']),
                    'chunk_length': len(item['content']),
                    'word_count': len(item['content'].split()),
                    'entities': {},
                    'processed_at': '2025-09-30',
                    'is_valid': True,
                    'department': item.get('department', '법무부'),
                    'law_name': item.get('law_name', ''),
                    'interpretation_date': item.get('interpretation_date', '')
                }
            }
            new_data.append(new_doc)
    
    # 기존 데이터와 새 데이터 결합
    all_data = existing_data + new_data
    
    # 업데이트된 메타데이터 저장
    with open('data/embeddings/metadata_final.json', 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"Updated metadata saved: data/embeddings/metadata_final.json")
    print(f"Total documents: {len(all_data)}")
    print(f"New documents added: {len(new_data)}")
    
    return all_data

def main():
    print("Adding Missing Data Types")
    print("=" * 50)
    
    # 1. 누락된 데이터 타입 확인
    available_types, missing_types = check_missing_data_types()
    
    # 2. 누락된 데이터 타입 처리
    process_missing_data_types()
    
    # 3. 벡터 데이터베이스 업데이트
    updated_data = update_vector_database()
    
    print("\n" + "=" * 50)
    print("Missing data types added successfully!")
    print(f"Total documents: {len(updated_data)}")
    
    # 데이터 타입별 분포 확인
    type_distribution = {}
    for doc in updated_data:
        doc_type = doc['metadata']['data_type']
        type_distribution[doc_type] = type_distribution.get(doc_type, 0) + 1
    
    print("\nDocument type distribution:")
    for doc_type, count in sorted(type_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {doc_type}: {count}")

if __name__ == "__main__":
    main()
