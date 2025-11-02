# -*- coding: utf-8 -*-
"""
검색 결과가 generate_answer_enhanced까지 전달되는지 간단한 테스트
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


async def test_search_results_to_answer():
    """검색 결과 전달 테스트"""
    try:
        from core.agents.workflow_service import LangGraphWorkflowService
        from infrastructure.utils.langgraph_config import LangGraphConfig

        print("=" * 80)
        print("검색 결과 전달 테스트 시작")
        print("=" * 80)

        # 설정 로드
        config = LangGraphConfig.from_env()
        print(f"✅ 설정 로드 완료")

        # 워크플로우 서비스 초기화
        workflow_service = LangGraphWorkflowService(config)
        print(f"✅ 워크플로우 서비스 초기화 완료")

        # 테스트 질의
        test_query = "계약 해지 요건"
        print(f"\n테스트 질의: {test_query}")

        # 전체 워크플로우 실행
        print("🔄 전체 워크플로우 실행 중...")
        result = await workflow_service.process_query(test_query, "test_session", enable_checkpoint=False)

        # 검색 결과 확인
        retrieved_docs = result.get("retrieved_docs", [])
        metadata = result.get("metadata", {})
        search_meta = metadata.get("search", {}) if isinstance(metadata, dict) else {}

        semantic_count = search_meta.get("semantic_results_count", 0)
        keyword_count = search_meta.get("keyword_results_count", 0)
        final_count = len(retrieved_docs)

        print(f"\n📊 검색 결과:")
        print(f"   의미적 검색: {semantic_count}개")
        print(f"   키워드 검색: {keyword_count}개")
        print(f"   최종 통합 결과: {final_count}개")

        # 검색 결과 상세
        if final_count > 0:
            print(f"\n📄 검색 결과 샘플:")
            for i, doc in enumerate(retrieved_docs[:3], 1):
                print(f"   {i}. Type: {doc.get('type', 'unknown')}")
                print(f"      Source: {str(doc.get('source', 'unknown'))[:60]}")
                content = str(doc.get('content', '') or doc.get('text', ''))[:100]
                print(f"      Content: {content}...")
        else:
            print("   ⚠️ 검색 결과가 없습니다!")

        # 답변 확인
        answer = result.get("answer", "")
        print(f"\n✍️ generate_answer_enhanced 결과:")
        print(f"   답변 길이: {len(answer)}자")
        print(f"   retrieved_docs 개수: {len(retrieved_docs)}개")

        if answer:
            print(f"   답변 미리보기: {answer[:150]}...")

            # 검색 결과가 답변에 포함되었는지 확인
            if retrieved_docs:
                doc_found = False
                for doc in retrieved_docs[:3]:
                    source = str(doc.get("source", ""))
                    if source and len(source) > 10:
                        # 소스 이름의 일부가 답변에 있는지 확인
                        source_words = source.split()[:3]  # 처음 3단어
                        for word in source_words:
                            if word and len(word) > 5 and word in answer:
                                doc_found = True
                                print(f"   ✅ 검색 결과가 답변에 포함됨: {word}")
                                break
                        if doc_found:
                            break

                if not doc_found:
                    print("   ⚠️ 검색 결과가 답변에 명시적으로 포함되지 않았을 수 있음")

        # 최종 검증
        print(f"\n✅ 최종 검증:")
        print(f"   - 검색 결과 있음: {'✅' if final_count > 0 else '❌'}")
        print(f"   - retrieved_docs 전달됨: {'✅' if len(retrieved_docs) > 0 else '❌'}")
        print(f"   - 답변 생성됨: {'✅' if answer else '❌'}")

        # 검증
        assert final_count > 0, f"검색 결과가 없습니다! (semantic: {semantic_count}, keyword: {keyword_count})"
        assert len(retrieved_docs) > 0, "retrieved_docs가 비어있습니다!"
        assert answer and len(answer) > 0, "답변이 생성되지 않았습니다!"

        print(f"\n✅ 모든 검증 통과! 검색 결과가 generate_answer_enhanced까지 잘 전달되었습니다.")

        return True

    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_search_results_to_answer())
    sys.exit(0 if result else 1)


