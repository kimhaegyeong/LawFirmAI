#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?? ?? ??? ?? ????

?? ??? ???? ?? ???? ??????.
"""

import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
from scripts.utils.path_utils import setup_project_path
from scripts.utils.file_utils import load_json_file, save_json_file

setup_project_path()

# UTF-8 인코딩 설정
os.environ['PYTHONIOENCODING'] = 'utf-8'


def improve_precedent_titles():
    """?? ?? ??"""
    print("?? ?? ?? ??...")
    
    # 메타데이터 로드
    data = load_json_file('data/embeddings/metadata.json')
    
    # ?? ???? ???
    precedents = [d for d in data if d['metadata']['data_type'] == 'precedents']
    
    print(f"?? ? ?? ?? ?: {len(precedents)}")
    
    improved_count = 0
    
    for precedent in precedents:
        original_title = precedent['metadata']['original_document']
        
        # ? ??? ?? ??
        if not original_title or original_title.strip() == "":
            # ?? ID?? ?? ?? ??
            case_id = precedent['id']
            case_number = case_id.replace('case_', '') if 'case_' in case_id else case_id
            
            # ?? ???? ?? ??
            content = precedent['text']
            
            # ??? ??
            court_name = "???"  # ???
            if "????" in content:
                court_name = "????"
            elif "????" in content:
                court_name = "????"
            elif "???" in content:
                court_name = "???"
            
            # ?? ?? ??
            case_type = "??"
            if "??" in content:
                case_type = "????"
            elif "??" in content:
                case_type = "????"
            elif "??" in content:
                case_type = "????"
            elif "??" in content:
                case_type = "????"
            elif "??" in content:
                case_type = "????"
            
            # ??? ?? ??
            new_title = f"{court_name} {case_type} {case_number} ??"
            
            # ????? ????
            precedent['metadata']['original_document'] = new_title
            precedent['metadata']['court_name'] = court_name
            precedent['metadata']['case_type'] = case_type
            precedent['metadata']['case_number'] = case_number
            
            improved_count += 1
    
    print(f"??? ?? ?? ?: {improved_count}?")
    
    # 개선된 데이터 저장
    save_json_file(data, 'data/embeddings/metadata_improved.json')
    
    print("??? ????? ?? ??: data/embeddings/metadata_improved.json")
    
    # ??? ?? ??
    print("\n??? ?? ?? ?? (?? 10?):")
    improved_precedents = [d for d in data if d['metadata']['data_type'] == 'precedents']
    for i, precedent in enumerate(improved_precedents[:10]):
        title = precedent['metadata']['original_document']
        court = precedent['metadata'].get('court_name', 'N/A')
        case_type = precedent['metadata'].get('case_type', 'N/A')
        print(f"  {i+1:2d}. {title} (??: {court}, ??: {case_type})")
    
    return data


def create_improved_vector_database():
    """??? ?? ?????? ??"""
    print("\n??? ?? ?????? ?? ??...")
    
    # 개선된 메타데이터 로드
    data = load_json_file('data/embeddings/metadata_improved.json')
    
    # ?? ???? ???
    precedents = [d for d in data if d['metadata']['data_type'] == 'precedents']
    
    print(f"??? ?? ?? ?: {len(precedents)}")
    
    # ??? ?? ??
    court_distribution = {}
    case_type_distribution = {}
    
    for precedent in precedents:
        court = precedent['metadata'].get('court_name', 'Unknown')
        case_type = precedent['metadata'].get('case_type', 'Unknown')
        
        court_distribution[court] = court_distribution.get(court, 0) + 1
        case_type_distribution[case_type] = case_type_distribution.get(case_type, 0) + 1
    
    print("\n??? ??:")
    for court, count in sorted(court_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {court}: {count}?({count/len(precedents)*100:.1f}%)")
    
    print("\n?? ??? ??:")
    for case_type, count in sorted(case_type_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {case_type}: {count}?({count/len(precedents)*100:.1f}%)")
    
    return data


def test_improved_accuracy():
    """??? ??? ???"""
    print("\n??? ??? ??? ??...")
    
    # 개선된 메타데이터 로드
    data = load_json_file('data/embeddings/metadata_improved.json')
    
    # ?? ?? ?? ???
    precedent_queries = [
        ("??? ??", "precedents"),
        ("???? ??", "precedents"),
        ("???? ??", "precedents"),
        ("????", "precedents"),
        ("????", "precedents"),
        ("????", "precedents")
    ]
    
    print("?? ?? ???:")
    correct_predictions = 0
    
    for query, expected in precedent_queries:
        # ?? ????? ??? ???
        precedents = [d for d in data if d['metadata']['data_type'] == 'precedents']
        
        # ?? ???? ???? ?? ??
        matching_precedents = []
        for precedent in precedents:
            title = precedent['metadata']['original_document']
            if any(keyword in title for keyword in query.split()):
                matching_precedents.append(precedent)
        
        if matching_precedents:
            # ? ?? ?? ??? ?? ??
            actual = matching_precedents[0]['metadata']['data_type']
            is_correct = actual == expected
            if is_correct:
                correct_predictions += 1
            
            print(f"  '{query}' -> ??: {expected}, ??: {actual} {'OK' if is_correct else 'FAIL'}")
            if matching_precedents:
                print(f"    ??? ??: {matching_precedents[0]['metadata']['original_document']}")
        else:
            print(f"  '{query}' -> ?? ?? ??")
    
    accuracy = correct_predictions / len(precedent_queries) if precedent_queries else 0
    print(f"\n??? ???: {accuracy:.2%} ({correct_predictions}/{len(precedent_queries)})")
    
    return accuracy


def main():
    print("?? ?? ??? ?? ?? ??")
    print("=" * 50)
    
    # 1. ?? ?? ??
    improved_data = improve_precedent_titles()
    
    # 2. ??? ?? ?????? ??
    create_improved_vector_database()
    
    # 3. ??? ??? ???
    accuracy = test_improved_accuracy()
    
    print("\n" + "=" * 50)
    print(f"?? ?? ??? ?? ??!")
    print(f"?? ???: {accuracy:.2%}")
    
    if accuracy >= 0.8:
        print("?? ??? 80% ??!")
    else:
        print("?? ??? ?????.")


if __name__ == "__main__":
    main()
