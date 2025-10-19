#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
업데이트된 키워드 데이터베이스 확인 스크립트
"""

import sys
import os
import json
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def check_keyword_database():
    """키워드 데이터베이스 상태 확인"""
    try:
        db_path = "data/legal_terms_database.json"
        
        print("키워드 데이터베이스 상태 확인")
        print("="*50)
        
        if os.path.exists(db_path):
            with open(db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"데이터베이스 파일: {db_path}")
            print(f"파일 크기: {os.path.getsize(db_path):,} bytes")
            
            total_keywords = 0
            print("\n도메인별 키워드 수:")
            for domain, keywords in data.items():
                count = len(keywords)
                total_keywords += count
                print(f"  {domain}: {count:,}개")
            
            print(f"\n총 키워드 수: {total_keywords:,}개")
            
            # 샘플 키워드 확인
            print("\n샘플 키워드 (지적재산권법):")
            if '지적재산권법' in data:
                sample_keywords = list(data['지적재산권법'].keys())[:10]
                for keyword in sample_keywords:
                    print(f"  - {keyword}")
            
            print("\n샘플 키워드 (형사법):")
            if '형사법' in data:
                sample_keywords = list(data['형사법'].keys())[:10]
                for keyword in sample_keywords:
                    print(f"  - {keyword}")
            
        else:
            print(f"데이터베이스 파일이 존재하지 않습니다: {db_path}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_keyword_database()