# -*- coding: utf-8 -*-
"""
판례 데이터에 대한 _preprocess_text_for_cross_encoder 단위 테스트 실행 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
_project_root = Path(__file__).parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from lawfirm_langgraph.core.search.processors.result_merger import ResultRanker
except ImportError:
    # 상대 경로로 시도
    from core.search.processors.result_merger import ResultRanker


def test_precedent_marker_normalization():
    """【】 마커 정규화 테스트"""
    ranker = ResultRanker(use_cross_encoder=False)
    
    # 【신 청 인】 -> 신청인
    text = "【신 청 인】이 제출한 서류를 검토한 결과"
    result = ranker._preprocess_text_for_cross_encoder(text)
    assert "신청인" in result, f"Expected '신청인' in result, got: {result}"
    assert "【신 청 인】" not in result, f"Expected no '【신 청 인】' in result, got: {result}"
    assert "신 청 인" not in result, f"Expected no '신 청 인' in result, got: {result}"
    print(f"✅ test_precedent_marker_normalization: PASSED")
    print(f"   Input: {text}")
    print(f"   Output: {result}")
    
    # 【피신청인】 -> 피신청인
    text = "【피신청인】의 주장에 대하여"
    result = ranker._preprocess_text_for_cross_encoder(text)
    assert "피신청인" in result, f"Expected '피신청인' in result, got: {result}"
    assert "【피신청인】" not in result, f"Expected no '【피신청인】' in result, got: {result}"
    print(f"✅ 【피신청인】 테스트: PASSED")
    print(f"   Input: {text}")
    print(f"   Output: {result}")


def test_precedent_multiple_markers():
    """여러 【】 마커 처리 테스트"""
    ranker = ResultRanker(use_cross_encoder=False)
    
    text = "【원고】와 【피고】 사이의 계약 분쟁에 관한 사건입니다."
    result = ranker._preprocess_text_for_cross_encoder(text)
    assert "원고" in result
    assert "피고" in result
    assert "【원고】" not in result
    assert "【피고】" not in result
    print(f"✅ test_precedent_multiple_markers: PASSED")
    print(f"   Input: {text}")
    print(f"   Output: {result}")


def test_precedent_section_markers():
    """판례 섹션 마커 처리 테스트"""
    ranker = ResultRanker(use_cross_encoder=False)
    
    # 【주 문】 -> 주문
    text = "【주 문】원고의 청구를 기각한다."
    result = ranker._preprocess_text_for_cross_encoder(text)
    assert "주문" in result
    assert "【주 문】" not in result
    assert "주 문" not in result
    print(f"✅ 【주 문】 테스트: PASSED")
    print(f"   Input: {text}")
    print(f"   Output: {result}")
    
    # 【이 유】 -> 이유
    text = "【이 유】이 사건의 쟁점은 다음과 같다."
    result = ranker._preprocess_text_for_cross_encoder(text)
    assert "이유" in result
    assert "【이 유】" not in result
    assert "이 유" not in result
    print(f"✅ 【이 유】 테스트: PASSED")
    print(f"   Input: {text}")
    print(f"   Output: {result}")


def test_precedent_complex_markers():
    """복잡한 판례 마커 처리 테스트"""
    ranker = ResultRanker(use_cross_encoder=False)
    
    text = """
    【신 청 인】○○○
    【피신청인】△△△
    【주 문】원고의 청구를 기각한다.
    【이 유】이 사건의 쟁점은 다음과 같다.
    """
    result = ranker._preprocess_text_for_cross_encoder(text)
    assert "신청인" in result
    assert "피신청인" in result
    assert "주문" in result
    assert "이유" in result
    assert "【" not in result
    assert "】" not in result
    print(f"✅ test_precedent_complex_markers: PASSED")
    print(f"   Output contains: 신청인, 피신청인, 주문, 이유")


def test_precedent_with_html_tags():
    """HTML 태그가 포함된 판례 문서 처리 테스트"""
    ranker = ResultRanker(use_cross_encoder=False)
    
    text = "【신 청 인】<p>이 제출한 서류를</p> 검토한 결과"
    result = ranker._preprocess_text_for_cross_encoder(text)
    assert "신청인" in result
    assert "<p>" not in result
    assert "</p>" not in result
    print(f"✅ test_precedent_with_html_tags: PASSED")
    print(f"   Input: {text}")
    print(f"   Output: {result}")


def test_precedent_marker_with_spaces():
    """띄어쓰기가 포함된 【】 마커 처리 테스트"""
    ranker = ResultRanker(use_cross_encoder=False)
    
    test_cases = [
        ("【신 청 인】", "신청인"),
        ("【피 신 청 인】", "피신청인"),
        ("【원 고】", "원고"),
        ("【피 고】", "피고"),
        ("【청 구 인】", "청구인"),
        ("【사 건 본 인】", "사건본인"),
        ("【주 문】", "주문"),
        ("【이 유】", "이유"),
        ("【판 결】", "판결"),
        ("【결 정】", "결정"),
    ]
    
    for input_text, expected in test_cases:
        result = ranker._preprocess_text_for_cross_encoder(input_text)
        assert expected in result, f"입력: {input_text}, 결과: {result}"
        assert " " not in expected or expected.replace(" ", "") in result
    print(f"✅ test_precedent_marker_with_spaces: PASSED ({len(test_cases)} cases)")


def test_precedent_real_world_example():
    """실제 판례 문서 예시 테스트"""
    ranker = ResultRanker(use_cross_encoder=False)
    
    text = """
    【신 청 인】○○○
    【피신청인】△△△
    【주 문】
    1. 원고의 청구를 기각한다.
    2. 소송비용은 원고가 부담한다.
    
    【이 유】
    이 사건의 쟁점은 계약 해지 사유에 관한 것이다.
    원고는 피고가 계약을 위반하였다고 주장하나,
    피고는 계약 위반 사실이 없다고 주장한다.
    """
    result = ranker._preprocess_text_for_cross_encoder(text)
    
    # 마커 정규화 확인
    assert "신청인" in result
    assert "피신청인" in result
    assert "주문" in result
    assert "이유" in result
    
    # 【】 마커 제거 확인
    assert "【" not in result
    assert "】" not in result
    
    # 내용 보존 확인
    assert "원고의 청구를 기각한다" in result or "원고의청구를기각한다" in result
    assert "계약 해지 사유" in result or "계약해지사유" in result
    print(f"✅ test_precedent_real_world_example: PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("판례 데이터에 대한 _preprocess_text_for_cross_encoder 테스트")
    print("=" * 60)
    
    tests = [
        test_precedent_marker_normalization,
        test_precedent_multiple_markers,
        test_precedent_section_markers,
        test_precedent_complex_markers,
        test_precedent_with_html_tags,
        test_precedent_marker_with_spaces,
        test_precedent_real_world_example,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__}: FAILED")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: ERROR")
            print(f"   Error: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"테스트 결과: {passed}개 통과, {failed}개 실패")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)

