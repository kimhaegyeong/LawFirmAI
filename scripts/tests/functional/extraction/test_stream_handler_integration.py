#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StreamHandler 통합 테스트 스크립트
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def test_stream_handler_integration():
    """StreamHandler 통합 테스트"""
    print("\n" + "=" * 80)
    print("StreamHandler 통합 테스트")
    print("=" * 80)
    
    try:
        from api.services.chat_service import get_chat_service
        
        chat_service = get_chat_service()
        stream_handler = chat_service.stream_handler
        
        if not stream_handler:
            print("❌ StreamHandler가 초기화되지 않았습니다.")
            return False
        
        print("✅ StreamHandler 초기화 확인")
        
        # sources_extractor 확인
        if stream_handler.sources_extractor:
            print("✅ SourcesExtractor 확인")
            
            # 테스트용 sources_detail
            test_sources_detail = [
                {
                    "type": "case_paragraph",
                    "doc_id": "case_2024다209769",
                    "case_number": "2024다209769",
                    "metadata": {}
                }
            ]
            
            # _generate_sources_by_type 테스트
            try:
                result = stream_handler._generate_sources_by_type(test_sources_detail)
                
                print(f"✅ _generate_sources_by_type 실행 성공")
                if result:
                    print(f"   - statute_article 개수: {len(result.get('statute_article', []))}")
                    print(f"   - case_paragraph 개수: {len(result.get('case_paragraph', []))}")
                    
                    # 참조 법령 확인
                    statutes = result.get('statute_article', [])
                    if statutes:
                        print(f"\n📋 추출된 참조 법령:")
                        for i, statute in enumerate(statutes[:3], 1):
                            print(f"   {i}. {statute.get('statute_name', 'N/A')} 제{statute.get('article_no', 'N/A')}조")
                            print(f"      - source_from: {statute.get('source_from', 'N/A')}")
                    else:
                        print("\n⚠️  참조 법령이 추출되지 않았습니다.")
                else:
                    print("   - sources_by_type이 None입니다 (정상: sources_detail이 비어있을 수 있음)")
                
                # 예외 처리 테스트
                print("\n🔍 예외 처리 테스트...")
                stream_handler.sources_extractor._get_sources_by_type_with_reference_statutes = lambda x: (_ for _ in ()).throw(Exception("Test error"))
                
                result_with_error = stream_handler._generate_sources_by_type(test_sources_detail)
                if result_with_error:
                    print("✅ 예외 발생 시에도 기본 sources_by_type 반환 확인")
                else:
                    print("⚠️  예외 발생 시 None 반환")
                
                return True
            except Exception as e:
                print(f"❌ _generate_sources_by_type 실행 실패: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("⚠️  SourcesExtractor가 없습니다.")
            return False
            
    except Exception as e:
        print(f"❌ 통합 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_stream_handler_integration())
    sys.exit(0 if success else 1)

