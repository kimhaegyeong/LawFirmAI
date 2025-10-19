#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KeywordDatabaseLoader 테스트 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.services.keyword_database_loader import KeywordDatabaseLoader

def test_keyword_database_loader():
    """KeywordDatabaseLoader 테스트"""
    try:
        print("KeywordDatabaseLoader 테스트")
        print("="*50)
        
        # KeywordDatabaseLoader 초기화
        loader = KeywordDatabaseLoader("data")
        
        # 모든 키워드 로드
        all_keywords = loader.load_all_keywords()
        
        total_keywords = 0
        for domain, keywords in all_keywords.items():
            count = len(keywords)
            total_keywords += count
            print(f"{domain}: {count:,}개 키워드")
            
            # 샘플 키워드 출력
            if count > 0:
                sample_keywords = keywords[:5]
                print(f"  샘플: {', '.join(sample_keywords)}")
        
        print(f"\n총 키워드 수: {total_keywords:,}개")
        
        # 데이터베이스 파일 상태 확인
        db_path = "data/legal_terms_database.json"
        if os.path.exists(db_path):
            file_size = os.path.getsize(db_path)
            print(f"\n데이터베이스 파일: {db_path}")
            print(f"파일 크기: {file_size:,} bytes")
        else:
            print(f"\n데이터베이스 파일이 존재하지 않습니다: {db_path}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_keyword_database_loader()
