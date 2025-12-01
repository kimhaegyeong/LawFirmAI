# -*- coding: utf-8 -*-
"""
통일된 출처 포맷터 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lawfirm_langgraph.core.generation.formatters.unified_source_formatter import UnifiedSourceFormatter, SourceInfo


def test_unified_source_formatter():
    """통일된 출처 포맷터 테스트"""
    print("\n=== 통일된 출처 포맷터 테스트 ===")
    
    formatter = UnifiedSourceFormatter()
    
    # 1. statute_article 테스트
    print("\n1. 법령 조문 포맷팅 테스트")
    statute_metadata = {
        "statute_name": "민법",
        "article_no": "제1조",
        "clause_no": "1",
        "item_no": "1"
    }
    result = formatter.format_source("statute_article", statute_metadata)
    print(f"  입력: {statute_metadata}")
    print(f"  결과: name={result.name}, type={result.type}, url={result.url}")
    assert result.name == "민법 제1조 제1항 제1호", f"예상: '민법 제1조 제1항 제1호', 실제: '{result.name}'"
    assert result.type == "statute_article"
    assert result.metadata is not None
    print("  ✓ 통과")
    
    # 2. case_paragraph 테스트
    print("\n2. 판례 포맷팅 테스트")
    case_metadata = {
        "court": "대법원",
        "doc_id": "2020다12345",
        "casenames": "손해배상청구 사건",
        "announce_date": "2020.12.25"
    }
    result = formatter.format_source("case_paragraph", case_metadata)
    print(f"  입력: {case_metadata}")
    print(f"  결과: name={result.name}, type={result.type}, url={result.url}")
    assert "대법원" in result.name
    assert "2020다12345" in result.name
    assert result.type == "case_paragraph"
    print("  ✓ 통과")
    
    # 3. decision_paragraph 테스트
    print("\n3. 결정례 포맷팅 테스트")
    decision_metadata = {
        "org": "법제처",
        "doc_id": "2020-123",
        "decision_date": "2020.12.25"
    }
    result = formatter.format_source("decision_paragraph", decision_metadata)
    print(f"  입력: {decision_metadata}")
    print(f"  결과: name={result.name}, type={result.type}, url={result.url}")
    assert "법제처" in result.name
    assert result.type == "decision_paragraph"
    print("  ✓ 통과")
    
    # 4. interpretation_paragraph 테스트
    print("\n4. 해석례 포맷팅 테스트")
    interpretation_metadata = {
        "org": "법제처",
        "title": "법령 해석",
        "doc_id": "2020-456",
        "response_date": "2020.12.25"
    }
    result = formatter.format_source("interpretation_paragraph", interpretation_metadata)
    print(f"  입력: {interpretation_metadata}")
    print(f"  결과: name={result.name}, type={result.type}, url={result.url}")
    assert "법제처" in result.name
    assert result.type == "interpretation_paragraph"
    print("  ✓ 통과")
    
    # 5. URL 생성 테스트
    print("\n5. URL 생성 테스트")
    statute_with_url = {
        "statute_name": "민법",
        "article_no": "제1조",
        "effective_date": "2020-01-01",
        "proclamation_number": "12345"
    }
    result = formatter.format_source("statute_article", statute_with_url)
    print(f"  입력: {statute_with_url}")
    print(f"  URL: {result.url}")
    assert result.url is not None and result.url != "", "URL이 생성되어야 함"
    print("  ✓ 통과")
    
    # 6. format_sources_list 테스트
    print("\n6. 출처 리스트 포맷팅 테스트")
    sources_list = [
        {
            "type": "statute_article",
            "metadata": {
                "statute_name": "형법",
                "article_no": "제250조"
            }
        },
        {
            "type": "case_paragraph",
            "metadata": {
                "court": "서울중앙지법",
                "doc_id": "2020나12345"
            }
        }
    ]
    formatted_list = formatter.format_sources_list(sources_list)
    print(f"  입력: {len(sources_list)}개 출처")
    print(f"  결과: {len(formatted_list)}개 포맷팅된 출처")
    assert len(formatted_list) == 2, f"예상: 2개, 실제: {len(formatted_list)}개"
    assert formatted_list[0].name == "형법 제250조"
    assert formatted_list[1].type == "case_paragraph"
    print("  ✓ 통과")
    
    print("\n=== 모든 테스트 통과 ===")


if __name__ == "__main__":
    test_unified_source_formatter()

