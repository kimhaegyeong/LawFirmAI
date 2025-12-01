#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sources 이벤트에 참조 법령이 포함되는지 테스트
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import sqlite3
from typing import Dict, Any, List

def test_get_sources_by_type_with_reference_statutes():
    """헬퍼 함수 테스트"""
    print("\n" + "=" * 80)
    print("1. 헬퍼 함수 테스트")
    print("=" * 80)
    
    try:
        from api.services.sources_extractor import SourcesExtractor
        from api.services.chat_service import get_chat_service
        
        # ChatService를 통해 SourcesExtractor 가져오기
        chat_service = get_chat_service()
        extractor = chat_service.sources_extractor
        
        # 테스트용 sources_detail 생성 (판례 포함)
        test_sources_detail = [
            {
                "type": "case_paragraph",
                "name": "판례",
                "case_number": "case_2024다209769",
                "doc_id": "case_2024다209769",
                "metadata": {
                    "case_number": "2024다209769"
                }
            }
        ]
        
        # 헬퍼 함수 호출
        result = extractor._get_sources_by_type_with_reference_statutes(test_sources_detail)
        
        print(f"\n✅ 헬퍼 함수 실행 성공")
        print(f"   - statute_article 개수: {len(result.get('statute_article', []))}")
        print(f"   - case_paragraph 개수: {len(result.get('case_paragraph', []))}")
        
        # 참조 법령이 포함되었는지 확인
        statutes = result.get('statute_article', [])
        if statutes:
            print(f"\n📋 추출된 참조 법령:")
            for i, statute in enumerate(statutes[:3], 1):
                print(f"   {i}. {statute.get('statute_name', 'N/A')} 제{statute.get('article_no', 'N/A')}조")
                print(f"      - source_from: {statute.get('source_from', 'N/A')}")
                print(f"      - source_doc_id: {statute.get('source_doc_id', 'N/A')}")
                if statute.get('metadata'):
                    print(f"      - metadata.source_from: {statute.get('metadata', {}).get('source_from', 'N/A')}")
        else:
            print("\n⚠️  참조 법령이 추출되지 않았습니다.")
            print("   데이터베이스에 해당 판례의 reference_statutes가 있는지 확인하세요.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 헬퍼 함수 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_create_sources_event():
    """sources 이벤트 생성 테스트"""
    print("\n" + "=" * 80)
    print("2. sources 이벤트 생성 테스트")
    print("=" * 80)
    
    try:
        from api.routers.chat import _create_sources_event
        
        # 테스트용 metadata 생성
        test_metadata = {
            "sources_detail": [
                {
                    "type": "case_paragraph",
                    "name": "판례",
                    "case_number": "case_2024다209769",
                    "doc_id": "case_2024다209769",
                    "metadata": {
                        "case_number": "2024다209769"
                    }
                }
            ]
        }
        
        # sources 이벤트 생성
        event = _create_sources_event(test_metadata, "test-message-id")
        
        print(f"\n✅ sources 이벤트 생성 성공")
        print(f"   - event type: {event.get('type')}")
        print(f"   - message_id: {event.get('metadata', {}).get('message_id')}")
        
        sources_by_type = event.get('metadata', {}).get('sources_by_type', {})
        print(f"   - sources_by_type keys: {list(sources_by_type.keys())}")
        
        statutes = sources_by_type.get('statute_article', [])
        print(f"   - statute_article 개수: {len(statutes)}")
        
        if statutes:
            print(f"\n📋 sources 이벤트에 포함된 참조 법령:")
            for i, statute in enumerate(statutes[:3], 1):
                print(f"   {i}. {statute.get('statute_name', 'N/A')} 제{statute.get('article_no', 'N/A')}조")
                print(f"      - source_from: {statute.get('source_from', 'N/A')}")
        else:
            print("\n⚠️  sources 이벤트에 참조 법령이 포함되지 않았습니다.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ sources 이벤트 생성 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_reference_statutes():
    """데이터베이스에서 참조 법령 조회 테스트"""
    print("\n" + "=" * 80)
    print("3. 데이터베이스 참조 법령 조회 테스트")
    print("=" * 80)
    
    try:
        db_path = project_root / "data" / "lawfirm_v2.db"
        if not db_path.exists():
            print(f"\n⚠️  데이터베이스 파일을 찾을 수 없습니다: {db_path}")
            return False
        
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 판례 중 reference_statutes가 있는 것 조회
        cursor.execute("""
            SELECT doc_id, reference_statutes 
            FROM cases 
            WHERE reference_statutes IS NOT NULL 
            AND reference_statutes != ''
            LIMIT 5
        """)
        
        rows = cursor.fetchall()
        
        if rows:
            print(f"\n✅ 데이터베이스에서 {len(rows)}개의 판례를 찾았습니다.")
            for i, row in enumerate(rows, 1):
                doc_id = row['doc_id']
                ref_statutes = row['reference_statutes']
                
                print(f"\n   {i}. 판례: {doc_id}")
                try:
                    ref_data = json.loads(ref_statutes) if ref_statutes else []
                    if isinstance(ref_data, list) and ref_data:
                        print(f"      참조 법령 개수: {len(ref_data)}")
                        for j, statute in enumerate(ref_data[:2], 1):
                            print(f"         {j}. {statute.get('statute_name', 'N/A')} 제{statute.get('article_no', 'N/A')}조")
                    else:
                        print(f"      참조 법령: 없음")
                except json.JSONDecodeError:
                    print(f"      참조 법령 파싱 실패")
        else:
            print(f"\n⚠️  데이터베이스에 reference_statutes가 있는 판례를 찾을 수 없습니다.")
            print("   마이그레이션을 실행했는지 확인하세요.")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"\n❌ 데이터베이스 조회 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 80)
    print("sources 이벤트 참조 법령 포함 테스트")
    print("=" * 80)
    
    results = []
    
    # 테스트 실행
    results.append(("데이터베이스 조회", test_database_reference_statutes()))
    results.append(("헬퍼 함수", test_get_sources_by_type_with_reference_statutes()))
    results.append(("sources 이벤트 생성", test_create_sources_event()))
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("테스트 결과 요약")
    print("=" * 80)
    
    for name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✅ 모든 테스트가 통과했습니다!")
    else:
        print("\n⚠️  일부 테스트가 실패했습니다. 위의 오류 메시지를 확인하세요.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

