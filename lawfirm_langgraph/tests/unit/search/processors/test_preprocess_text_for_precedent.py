# -*- coding: utf-8 -*-
"""
판례 데이터에 대한 _preprocess_text_for_cross_encoder 단위 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가 (conftest.py와 동일한 방식)
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 공통 imports (sys.path 설정 후)
import pytest  # noqa: E402
from lawfirm_langgraph.core.search.processors.result_merger import ResultRanker  # noqa: E402


class TestPreprocessTextForPrecedent:
    """판례 데이터에 대한 _preprocess_text_for_cross_encoder 테스트"""
    
    @pytest.fixture
    def ranker(self):
        """ResultRanker 인스턴스 생성"""
        return ResultRanker(use_cross_encoder=False)
    
    def test_precedent_marker_normalization(self, ranker):
        """【】 마커 정규화 테스트"""
        # 【신 청 인】 -> 신청인
        text = "【신 청 인】이 제출한 서류를 검토한 결과"
        result = ranker._preprocess_text_for_cross_encoder(text)
        assert "신청인" in result
        assert "【신 청 인】" not in result
        assert "신 청 인" not in result
        
        # 【피신청인】 -> 피신청인
        text = "【피신청인】의 주장에 대하여"
        result = ranker._preprocess_text_for_cross_encoder(text)
        assert "피신청인" in result
        assert "【피신청인】" not in result
    
    def test_precedent_multiple_markers(self, ranker):
        """여러 【】 마커 처리 테스트"""
        text = "【원고】와 【피고】 사이의 계약 분쟁에 관한 사건입니다."
        result = ranker._preprocess_text_for_cross_encoder(text)
        assert "원고" in result
        assert "피고" in result
        assert "【원고】" not in result
        assert "【피고】" not in result
    
    def test_precedent_section_markers(self, ranker):
        """판례 섹션 마커 처리 테스트"""
        # 【주 문】 -> 주문
        text = "【주 문】원고의 청구를 기각한다."
        result = ranker._preprocess_text_for_cross_encoder(text)
        assert "주문" in result
        assert "【주 문】" not in result
        assert "주 문" not in result
        
        # 【이 유】 -> 이유
        text = "【이 유】이 사건의 쟁점은 다음과 같다."
        result = ranker._preprocess_text_for_cross_encoder(text)
        assert "이유" in result
        assert "【이 유】" not in result
        assert "이 유" not in result
    
    def test_precedent_complex_markers(self, ranker):
        """복잡한 판례 마커 처리 테스트"""
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
    
    def test_precedent_with_html_tags(self, ranker):
        """HTML 태그가 포함된 판례 문서 처리 테스트"""
        text = "【신 청 인】<p>이 제출한 서류를</p> 검토한 결과"
        result = ranker._preprocess_text_for_cross_encoder(text)
        assert "신청인" in result
        assert "<p>" not in result
        assert "</p>" not in result
    
    def test_precedent_with_html_entities(self, ranker):
        """HTML 엔티티가 포함된 판례 문서 처리 테스트"""
        text = "【원고】&amp; 【피고】의 계약 분쟁"
        result = ranker._preprocess_text_for_cross_encoder(text)
        assert "원고" in result
        assert "피고" in result
        assert "&amp;" not in result or "&" in result
    
    def test_precedent_marker_with_spaces(self, ranker):
        """띄어쓰기가 포함된 【】 마커 처리 테스트"""
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
    
    def test_precedent_real_world_example(self, ranker):
        """실제 판례 문서 예시 테스트"""
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
    
    def test_precedent_max_length(self, ranker):
        """최대 길이 제한 테스트"""
        # 긴 판례 문서
        long_text = "【신 청 인】" + "가" * 1000 + "【피신청인】" + "나" * 1000
        result = ranker._preprocess_text_for_cross_encoder(long_text, max_length=512)
        assert len(result) <= 512
        assert "신청인" in result
    
    def test_precedent_empty_and_none(self, ranker):
        """빈 문자열 및 None 처리 테스트"""
        assert ranker._preprocess_text_for_cross_encoder("") == ""
        assert ranker._preprocess_text_for_cross_encoder(None) == ""
    
    def test_precedent_whitespace_normalization(self, ranker):
        """공백 정리 테스트"""
        text = "【신 청 인】   이   제출한   서류를   검토한   결과"
        result = ranker._preprocess_text_for_cross_encoder(text)
        # 연속된 공백이 하나로 정리되어야 함
        assert "  " not in result  # 연속된 공백이 없어야 함
        assert "신청인" in result
    
    def test_precedent_mixed_content(self, ranker):
        """마커와 일반 텍스트가 혼합된 경우 테스트"""
        text = "이 사건은 【신 청 인】과 【피신청인】 사이의 계약 분쟁에 관한 것으로, 【주 문】에서 원고의 청구를 기각한다고 판단하였다."
        result = ranker._preprocess_text_for_cross_encoder(text)
        
        assert "신청인" in result
        assert "피신청인" in result
        assert "주문" in result
        assert "【" not in result
        assert "】" not in result
        assert "계약 분쟁" in result or "계약분쟁" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

