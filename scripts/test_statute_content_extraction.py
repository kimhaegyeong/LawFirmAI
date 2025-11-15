#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
법령 본문 추출 테스트
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_statute_content_extraction():
    """법령 본문 추출 테스트"""
    print("\n" + "=" * 80)
    print("법령 본문 추출 테스트")
    print("=" * 80)
    
    try:
        from api.services.chat_service import get_chat_service
        
        chat_service = get_chat_service()
        sources_extractor = chat_service.sources_extractor
        
        if not sources_extractor:
            print("❌ SourcesExtractor가 초기화되지 않았습니다.")
            return False
        
        print("✅ SourcesExtractor 확인")
        
        # 테스트용 sources_detail (판례에서 추출된 법령)
        test_sources_detail = [
            {
                "type": "case_paragraph",
                "doc_id": "case_2024다243172",
                "case_number": "2024다243172",
                "metadata": {}
            }
        ]
        
        # _extract_statutes_from_reference_clauses 테스트
        try:
            extracted_statutes = sources_extractor._extract_statutes_from_reference_clauses(test_sources_detail)
            
            print(f"\n✅ 추출된 법령 개수: {len(extracted_statutes)}")
            
            if extracted_statutes:
                print(f"\n📋 추출된 법령 목록:")
                for i, statute in enumerate(extracted_statutes[:5], 1):
                    print(f"\n{i}. {statute.get('statute_name', 'N/A')} 제{statute.get('article_no', 'N/A')}조")
                    if statute.get('clause_no'):
                        print(f"   - 항: {statute.get('clause_no')}")
                    if statute.get('item_no'):
                        print(f"   - 호: {statute.get('item_no')}")
                    
                    # 본문 확인
                    content = statute.get('content')
                    if content:
                        print(f"   ✅ 본문 있음 ({len(content)}자)")
                        print(f"   본문 미리보기: {content[:100]}...")
                    else:
                        print(f"   ⚠️  본문 없음")
                    
                    print(f"   - source_from: {statute.get('source_from', 'N/A')}")
                    print(f"   - source_doc_id: {statute.get('source_doc_id', 'N/A')}")
                
                # 본문이 있는 법령 개수 확인
                statutes_with_content = [s for s in extracted_statutes if s.get('content')]
                print(f"\n📊 통계:")
                print(f"   - 전체 법령: {len(extracted_statutes)}개")
                print(f"   - 본문 있는 법령: {len(statutes_with_content)}개")
                print(f"   - 본문 없는 법령: {len(extracted_statutes) - len(statutes_with_content)}개")
                
                if len(statutes_with_content) > 0:
                    print(f"\n✅ 법령 본문 추출 성공!")
                    return True
                else:
                    print(f"\n⚠️  본문이 추출된 법령이 없습니다.")
                    return False
            else:
                print("\n⚠️  추출된 법령이 없습니다.")
                return False
                
        except Exception as e:
            print(f"❌ 법령 추출 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_statute_content_extraction()
    sys.exit(0 if success else 1)

