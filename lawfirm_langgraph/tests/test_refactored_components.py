# -*- coding: utf-8 -*-
"""리팩토링된 컴포넌트 테스트"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_source_extractor():
    """SourceExtractor 테스트"""
    print("\n=== SourceExtractor 테스트 ===")
    
    from lawfirm_langgraph.core.agents.handlers.extractors.source_extractor import SourceExtractor
    
    extractor = SourceExtractor()
    
    # 테스트 케이스 1: detail 최상위 레벨에 값이 있는 경우
    detail1 = {
        "type": "statute_article",
        "statute_name": "민법",
        "article_no": "제1조",
        "metadata": {}
    }
    result1 = extractor.extract_statute_info(detail1)
    assert result1 == ("민법", "제1조"), f"Expected ('민법', '제1조'), got {result1}"
    print("✅ 테스트 1 통과: detail 최상위 레벨에서 추출")
    
    # 테스트 케이스 2: metadata에 값이 있는 경우
    detail2 = {
        "type": "statute_article",
        "metadata": {
            "statute_name": "형법",
            "article_no": "제2조"
        }
    }
    result2 = extractor.extract_statute_info(detail2)
    assert result2 == ("형법", "제2조"), f"Expected ('형법', '제2조'), got {result2}"
    print("✅ 테스트 2 통과: metadata에서 추출")
    
    # 테스트 케이스 3: 다른 필드명 사용
    detail3 = {
        "type": "statute_article",
        "law_name": "상법",
        "article_number": "제3조",
        "metadata": {}
    }
    result3 = extractor.extract_statute_info(detail3)
    assert result3 == ("상법", "제3조"), f"Expected ('상법', '제3조'), got {result3}"
    print("✅ 테스트 3 통과: 다른 필드명에서 추출")
    
    # 테스트 케이스 4: legal_references 생성
    sources_detail = [
        {
            "type": "statute_article",
            "statute_name": "민법",
            "article_no": "제1조",
            "clause_no": "1",
            "item_no": "1",
            "metadata": {}
        },
        {
            "type": "statute_article",
            "metadata": {
                "statute_name": "형법",
                "article_no": "제2조"
            }
        }
    ]
    legal_refs = extractor.extract_legal_references_from_sources_detail(sources_detail)
    assert len(legal_refs) == 2, f"Expected 2 legal references, got {len(legal_refs)}"
    assert "민법 제1조 제1항 제1호" in legal_refs, "민법 제1조 제1항 제1호 should be in legal_refs"
    assert "형법 제2조" in legal_refs, "형법 제2조 should be in legal_refs"
    print("✅ 테스트 4 통과: legal_references 생성")
    
    print("\n✅ SourceExtractor 모든 테스트 통과")


def test_confidence_manager():
    """ConfidenceManager 테스트"""
    print("\n=== ConfidenceManager 테스트 ===")
    
    from lawfirm_langgraph.core.agents.handlers.managers.confidence_manager import ConfidenceManager
    
    manager = ConfidenceManager()
    
    # 테스트 케이스 1: 신뢰도 교체
    text = "**신뢰도: 50.0%**\n🟡 **신뢰도: 60.0%**"
    confidence = 0.75
    result = manager.replace_in_text(text, confidence)
    assert "75.0%" in result, "신뢰도 값이 교체되어야 함"
    print("✅ 테스트 1 통과: 신뢰도 값 교체")
    
    # 테스트 케이스 2: 이모지 및 레벨 확인
    emoji = manager.get_emoji(0.75)
    level = manager.get_level(0.75)
    assert emoji == "🟡", f"Expected 🟡, got {emoji}"
    assert level == "medium", f"Expected medium, got {level}"
    print("✅ 테스트 2 통과: 이모지 및 레벨 반환")
    
    # 테스트 케이스 3: 신뢰도 섹션 교체
    text_with_section = "### 💡 신뢰도정보\n🟡 **신뢰도: 60.0%** (medium)\n\n---"
    result = manager.replace_confidence_section(text_with_section, 0.85)
    assert "85.0%" in result, "신뢰도 섹션이 교체되어야 함"
    assert "high" in result, "레벨이 high여야 함"
    print("✅ 테스트 3 통과: 신뢰도 섹션 교체")
    
    print("\n✅ ConfidenceManager 모든 테스트 통과")


def test_answer_cleaner():
    """AnswerCleaner 테스트"""
    print("\n=== AnswerCleaner 테스트 ===")
    
    from lawfirm_langgraph.core.agents.handlers.cleaners.answer_cleaner import AnswerCleaner
    
    cleaner = AnswerCleaner()
    
    # 테스트 케이스 1: 메타데이터 섹션 제거
    text_with_metadata = "답변 내용\n\n### 💡 신뢰도정보\n신뢰도: 75%\n\n### 📚 참고자료\n참고 자료\n\n실제 답변 내용"
    result = cleaner.remove_metadata_sections(text_with_metadata)
    assert "신뢰도정보" not in result, f"신뢰도 정보 섹션이 제거되어야 함. 결과: {result[:200]}"
    assert "참고자료" not in result, f"참고자료 섹션이 제거되어야 함. 결과: {result[:200]}"
    assert "답변 내용" in result or "실제 답변" in result, f"실제 답변은 유지되어야 함. 결과: {result[:200]}"
    print("✅ 테스트 1 통과: 메타데이터 섹션 제거")
    
    # 테스트 케이스 2: 중복 헤더 제거
    text_with_duplicate = "## 답변\n\n### 답변\n\n실제 내용"
    result = cleaner.remove_duplicate_headers(text_with_duplicate)
    assert result.count("답변") <= 1, "중복 헤더가 제거되어야 함"
    print("✅ 테스트 2 통과: 중복 헤더 제거")
    
    # 테스트 케이스 3: 답변 헤더 제거
    text_with_header = "## 답변\n\n실제 답변 내용"
    result = cleaner.remove_answer_header(text_with_header)
    assert "## 답변" not in result, "답변 헤더가 제거되어야 함"
    assert "실제 답변 내용" in result, "실제 내용은 유지되어야 함"
    print("✅ 테스트 3 통과: 답변 헤더 제거")
    
    print("\n✅ AnswerCleaner 모든 테스트 통과")


def test_length_adjuster():
    """AnswerLengthAdjuster 테스트"""
    print("\n=== AnswerLengthAdjuster 테스트 ===")
    
    from lawfirm_langgraph.core.agents.handlers.formatters.length_adjuster import AnswerLengthAdjuster
    
    adjuster = AnswerLengthAdjuster()
    
    # 테스트 케이스 1: 적절한 길이의 답변
    short_answer = "짧은 답변" * 50
    result = adjuster.adjust_length(short_answer, "simple_question", "simple")
    assert len(result) == len(short_answer), "적절한 길이는 그대로 유지되어야 함"
    print("✅ 테스트 1 통과: 적절한 길이 유지")
    
    # 테스트 케이스 2: 너무 긴 답변
    long_answer = "긴 답변 내용입니다. " * 500
    result = adjuster.adjust_length(long_answer, "simple_question", "simple")
    assert len(result) < len(long_answer), "너무 긴 답변은 줄어들어야 함"
    print("✅ 테스트 2 통과: 긴 답변 축소")
    
    print("\n✅ AnswerLengthAdjuster 모든 테스트 통과")


if __name__ == "__main__":
    print("=" * 60)
    print("리팩토링된 컴포넌트 테스트 시작")
    print("=" * 60)
    
    try:
        test_source_extractor()
        test_confidence_manager()
        test_answer_cleaner()
        test_length_adjuster()
        
        print("\n" + "=" * 60)
        print("✅ 모든 테스트 통과!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

