# -*- coding: utf-8 -*-
"""
interpretation_paragraph가 0개인 원인 디버깅 스크립트
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def debug_interpretation_paragraph():
    """interpretation_paragraph가 0개인 원인 디버깅"""
    print("\n" + "=" * 80)
    print("interpretation_paragraph 디버깅 시작")
    print("=" * 80)
    
    try:
        from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        
        config = LangGraphConfig()
        workflow = LangGraphWorkflowService(config)
        
        query = "전세금 반환 보증에 대해 설명해주세요"
        print(f"\n📝 쿼리: {query}")
        
        # 워크플로우 실행
        result = await workflow.process_query(query)
        
        # 검색 결과 확인
        retrieved_docs = result.get("retrieved_documents", [])
        print(f"\n📊 검색된 문서 수: {len(retrieved_docs)}")
        
        # 타입별 분포 확인
        type_distribution = {}
        interpretation_docs = []
        for doc in retrieved_docs:
            if not isinstance(doc, dict):
                continue
            doc_type = (
                doc.get("type") or
                doc.get("source_type") or
                doc.get("metadata", {}).get("type") if isinstance(doc.get("metadata"), dict) else None or
                doc.get("metadata", {}).get("source_type") if isinstance(doc.get("metadata"), dict) else None or
                "unknown"
            )
            type_distribution[doc_type] = type_distribution.get(doc_type, 0) + 1
            
            if doc_type == "interpretation_paragraph":
                interpretation_docs.append(doc)
                print(f"\n✅ interpretation_paragraph 문서 발견:")
                print(f"   - id: {doc.get('id')}")
                print(f"   - source_type: {doc.get('source_type')}")
                print(f"   - type: {doc.get('type')}")
                print(f"   - relevance_score: {doc.get('relevance_score')}")
                print(f"   - is_sample: {doc.get('metadata', {}).get('is_sample', False)}")
                print(f"   - search_type: {doc.get('search_type')}")
        
        print(f"\n📊 타입별 분포:")
        for doc_type, count in sorted(type_distribution.items()):
            print(f"   - {doc_type}: {count}개")
        
        print(f"\n🔍 interpretation_paragraph 문서 수: {len(interpretation_docs)}")
        
        # 프롬프트 확인
        prompt = result.get("prompt", "")
        if "📖 해석례" in prompt:
            print("\n✅ 프롬프트에 해석례 섹션이 포함되어 있습니다")
            # 해석례 섹션 추출
            import re
            pattern = r'### 📖 해석례\n\n(.*?)(?=###|$)'
            match = re.search(pattern, prompt, re.DOTALL)
            if match:
                interpretation_section = match.group(1)
                print(f"   섹션 길이: {len(interpretation_section)}자")
                print(f"   문서 수: {interpretation_section.count('**문서')}")
        else:
            print("\n❌ 프롬프트에 해석례 섹션이 없습니다")
        
        # sources_detail 확인
        sources_detail = result.get("sources_detail", [])
        interpretation_sources = [s for s in sources_detail if s.get("type") == "interpretation_paragraph"]
        print(f"\n📋 sources_detail의 interpretation_paragraph 수: {len(interpretation_sources)}")
        
        # 검색 단계별 확인
        search_results = result.get("search_results", {})
        semantic_results = search_results.get("semantic_results", [])
        interpretation_semantic = [d for d in semantic_results if (
            d.get("type") == "interpretation_paragraph" or
            d.get("source_type") == "interpretation_paragraph"
        )]
        print(f"\n🔍 semantic_results의 interpretation_paragraph 수: {len(interpretation_semantic)}")
        
        # 샘플링 확인
        if interpretation_docs:
            print("\n✅ interpretation_paragraph 문서가 검색 결과에 포함되어 있습니다")
            for doc in interpretation_docs:
                if doc.get("metadata", {}).get("is_sample") or doc.get("search_type") == "type_sample":
                    print(f"   - 샘플링된 문서: {doc.get('id')}")
        else:
            print("\n❌ interpretation_paragraph 문서가 검색 결과에 없습니다")
            print("   원인 분석:")
            print("   1. 샘플링이 실행되지 않았을 수 있음")
            print("   2. 샘플링은 되었지만 검색 결과에 포함되지 않았을 수 있음")
            print("   3. 프롬프트 필터링에서 제외되었을 수 있음")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_interpretation_paragraph())

