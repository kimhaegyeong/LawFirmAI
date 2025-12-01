"""
청킹 전략 테스트 스크립트

각 청킹 전략을 테스트하고 하이브리드 청킹을 검증합니다.
"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.utils.chunking import ChunkingFactory
from scripts.utils.chunking.config import QueryType


def test_standard_chunking():
    """기본 청킹 전략 테스트"""
    print("=" * 60)
    print("기본 청킹 전략 테스트")
    print("=" * 60)
    
    strategy = ChunkingFactory.create_strategy("standard")
    
    # 테스트 데이터
    paragraphs = [
        "첫 번째 단락입니다. 이것은 테스트용 텍스트입니다.",
        "두 번째 단락입니다. 청킹 전략을 테스트하기 위한 내용입니다.",
        "세 번째 단락입니다. 여러 단락을 청크로 나누는 것을 확인합니다.",
        "네 번째 단락입니다. 오버랩이 제대로 작동하는지 확인합니다.",
    ]
    
    results = strategy.chunk(
        content=paragraphs,
        source_type="case_paragraph",
        source_id=1
    )
    
    print(f"생성된 청크 수: {len(results)}")
    for i, result in enumerate(results):
        print(f"\n청크 {i + 1}:")
        print(f"  텍스트 길이: {len(result.text)}자")
        print(f"  청킹 전략: {result.metadata.get('chunking_strategy')}")
        print(f"  크기 카테고리: {result.metadata.get('chunk_size_category')}")
        print(f"  텍스트 미리보기: {result.text[:100]}...")
    
    assert len(results) > 0, "청크가 생성되지 않았습니다."
    print("\n✅ 기본 청킹 전략 테스트 통과")


def test_dynamic_chunking():
    """동적 청킹 전략 테스트"""
    print("\n" + "=" * 60)
    print("동적 청킹 전략 테스트")
    print("=" * 60)
    
    # 질문 유형별 테스트
    query_types = [
        QueryType.LAW_INQUIRY.value,
        QueryType.PRECEDENT_SEARCH.value,
        QueryType.LEGAL_ADVICE.value,
    ]
    
    paragraphs = [
        "법령 조문에 대한 질문입니다. " * 20,
        "판례 검색을 위한 내용입니다. " * 30,
        "법률 상담 관련 텍스트입니다. " * 25,
    ]
    
    for query_type in query_types:
        print(f"\n질문 유형: {query_type}")
        strategy = ChunkingFactory.create_strategy("dynamic", query_type=query_type)
        
        results = strategy.chunk(
            content=paragraphs,
            source_type="case_paragraph",
            source_id=1,
            query_type=query_type
        )
        
        print(f"  생성된 청크 수: {len(results)}")
        if results:
            print(f"  첫 번째 청크 길이: {len(results[0].text)}자")
            print(f"  질문 유형: {results[0].metadata.get('query_type')}")
    
    print("\n✅ 동적 청킹 전략 테스트 통과")


def test_hybrid_chunking():
    """하이브리드 청킹 전략 테스트"""
    print("\n" + "=" * 60)
    print("하이브리드 청킹 전략 테스트")
    print("=" * 60)
    
    strategy = ChunkingFactory.create_strategy("hybrid")
    
    paragraphs = [
        "하이브리드 청킹 테스트를 위한 첫 번째 단락입니다. " * 15,
        "하이브리드 청킹 테스트를 위한 두 번째 단락입니다. " * 15,
        "하이브리드 청킹 테스트를 위한 세 번째 단락입니다. " * 15,
    ]
    
    results = strategy.chunk(
        content=paragraphs,
        source_type="case_paragraph",
        source_id=1
    )
    
    print(f"생성된 총 청크 수: {len(results)}")
    
    # 크기 카테고리별 그룹화
    by_category = {}
    chunk_groups = set()
    
    for result in results:
        category = result.metadata.get('chunk_size_category', 'unknown')
        chunk_group_id = result.metadata.get('chunk_group_id')
        
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(result)
        
        if chunk_group_id:
            chunk_groups.add(chunk_group_id)
    
    print(f"\n청크 그룹 수: {len(chunk_groups)}")
    print(f"크기 카테고리별 청크 수:")
    for category, chunks in by_category.items():
        print(f"  {category}: {len(chunks)}개")
        if chunks:
            print(f"    평균 길이: {sum(len(c.text) for c in chunks) / len(chunks):.0f}자")
    
    # 각 카테고리가 생성되었는지 확인
    expected_categories = ['small', 'medium', 'large']
    for category in expected_categories:
        assert category in by_category, f"{category} 카테고리 청크가 생성되지 않았습니다."
        assert len(by_category[category]) > 0, f"{category} 카테고리 청크가 비어있습니다."
    
    # 모든 청크가 같은 그룹 ID를 가지는지 확인
    assert len(chunk_groups) == 1, "모든 청크가 같은 그룹 ID를 가져야 합니다."
    
    print("\n✅ 하이브리드 청킹 전략 테스트 통과")


def main():
    """메인 함수"""
    print("청킹 전략 테스트 시작\n")
    
    try:
        test_standard_chunking()
        test_dynamic_chunking()
        test_hybrid_chunking()
        
        print("\n" + "=" * 60)
        print("✅ 모든 테스트 통과!")
        print("=" * 60)
    
    except AssertionError as e:
        print(f"\n❌ 테스트 실패: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

