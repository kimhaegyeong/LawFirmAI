# -*- coding: utf-8 -*-
"""
출처 검증기 테스트 (개선 버전)
추가 엣지 케이스 및 통합 테스트 포함
"""

import sys
import time
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lawfirm_langgraph.core.generation.validators.source_validator import SourceValidator
from lawfirm_langgraph.core.generation.formatters.unified_source_formatter import UnifiedSourceFormatter


class TestStatistics:
    """테스트 통계 추적"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def end(self):
        self.end_time = time.time()
    
    def add_test(self, passed: bool):
        self.total += 1
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def print_summary(self):
        elapsed = self.end_time - self.start_time if self.start_time and self.end_time else 0
        print(f"\n{'='*60}")
        print(f"테스트 결과 요약")
        print(f"{'='*60}")
        print(f"총 테스트: {self.total}개")
        print(f"통과: {self.passed}개")
        print(f"실패: {self.failed}개")
        print(f"성공률: {(self.passed/self.total*100):.1f}%" if self.total > 0 else "0%")
        print(f"실행 시간: {elapsed:.2f}초")
        print(f"{'='*60}")


def test_unknown_source_type():
    """알 수 없는 source_type 테스트"""
    print("\n[엣지 케이스] 알 수 없는 source_type 테스트")
    validator = SourceValidator()
    
    result = validator.validate_source("unknown_type", {"test": "value"})
    print(f"  입력: source_type='unknown_type'")
    print(f"  결과: is_valid={result['is_valid']}, confidence={result['confidence']:.2f}")
    print(f"  errors: {result['errors']}")
    
    assert result['is_valid'] == False, "알 수 없는 타입은 유효하지 않아야 함"
    assert len(result['errors']) > 0, "에러가 있어야 함"
    assert "Unknown source_type" in result['errors'][0], f"에러 메시지에 'Unknown source_type'이 포함되어야 함. 실제: {result['errors']}"
    assert result['confidence'] == 0.0, f"신뢰도가 0.0이어야 함. 실제: {result['confidence']}"
    print("  ✓ 통과")
    return True


def test_none_values():
    """None 값 처리 테스트"""
    print("\n[엣지 케이스] None 값 처리 테스트")
    validator = SourceValidator()
    
    test_cases = [
        ({"statute_name": None, "article_no": "제1조"}, "법령명이 None인 경우"),
        ({"statute_name": "민법", "article_no": None}, "조문 번호가 None인 경우"),
        ({"statute_name": None, "article_no": None}, "법령명과 조문 번호가 모두 None인 경우"),
    ]
    
    for data, description in test_cases:
        result = validator.validate_source("statute_article", data)
        print(f"  {description}: is_valid={result['is_valid']}, confidence={result['confidence']:.2f}")
        
        if data.get("statute_name") is None:
            assert result['is_valid'] == False, f"{description}: 법령명이 None이면 유효하지 않아야 함"
            assert len(result['errors']) > 0, f"{description}: 에러가 있어야 함"
    
    print("  ✓ 통과")
    return True


def test_empty_string_vs_none():
    """빈 문자열 vs None 구분 테스트"""
    print("\n[엣지 케이스] 빈 문자열 vs None 구분 테스트")
    validator = SourceValidator()
    
    result_none = validator.validate_source("statute_article", {"statute_name": None})
    result_empty = validator.validate_source("statute_article", {"statute_name": ""})
    
    print(f"  None 값: is_valid={result_none['is_valid']}, confidence={result_none['confidence']:.2f}")
    print(f"  빈 문자열: is_valid={result_empty['is_valid']}, confidence={result_empty['confidence']:.2f}")
    
    # 둘 다 에러가 발생해야 함
    assert result_none['is_valid'] == False, "None 값은 유효하지 않아야 함"
    assert result_empty['is_valid'] == False, "빈 문자열도 유효하지 않아야 함"
    
    print("  ✓ 통과")
    return True


def test_enforcement_regulations():
    """시행령/시행규칙 패턴 테스트"""
    print("\n[패턴 검증] 시행령/시행규칙 패턴 테스트")
    validator = SourceValidator()
    
    test_cases = [
        {"statute_name": "민법 시행령", "article_no": "제1조"},
        {"statute_name": "형법 시행규칙", "article_no": "제1조"},
        {"statute_name": "상법시행령", "article_no": "제1조"},  # 공백 없음
    ]
    
    for data in test_cases:
        result = validator.validate_source("statute_article", data)
        print(f"  {data['statute_name']}: is_valid={result['is_valid']}, confidence={result['confidence']:.2f}")
        
        # 시행령/시행규칙은 유효한 패턴이어야 함
        assert result['is_valid'] == True, f"{data['statute_name']}은 유효해야 함"
        assert len(result['warnings']) == 0, f"{data['statute_name']}은 경고가 없어야 함"
    
    print("  ✓ 통과")
    return True


def test_court_name_partial_matching():
    """법원명 부분 매칭 테스트"""
    print("\n[패턴 검증] 법원명 부분 매칭 테스트")
    validator = SourceValidator()
    
    test_cases = [
        {"court": "서울중앙지방법원", "doc_id": "2020다12345", "casenames": "손해배상청구"},
        {"court": "부산지방법원 서부지원", "doc_id": "2020다12345", "casenames": "손해배상청구"},
        {"court": "서울고등법원", "doc_id": "2020다12345", "casenames": "손해배상청구"},
    ]
    
    for data in test_cases:
        result = validator.validate_source("case_paragraph", data)
        print(f"  {data['court']}: is_valid={result['is_valid']}, confidence={result['confidence']:.2f}")
        print(f"    warnings: {result['warnings']}")
        
        # 부분 매칭이 되면 경고가 없어야 함
        assert result['is_valid'] == True, f"{data['court']}은 유효해야 함"
        # 부분 매칭이 되면 경고가 없거나 적어야 함
        court_warnings = [w for w in result['warnings'] if "법원명" in w]
        assert len(court_warnings) == 0, f"{data['court']}은 법원명 경고가 없어야 함 (부분 매칭)"
    
    print("  ✓ 통과")
    return True


def test_confidence_boundary():
    """신뢰도 경계값 테스트"""
    print("\n[경계값 검증] 신뢰도 경계값 테스트")
    validator = SourceValidator()
    
    # 여러 문제가 동시에 발생하여 신뢰도가 0.0 이하로 떨어질 수 있는 경우
    complex_data = {
        "statute_name": "invalid_law",  # -0.1
        "article_no": "invalid",  # -0.1
        "clause_no": "invalid",  # -0.05
        "item_no": "invalid",  # -0.05
    }
    
    result = validator.validate_source("statute_article", complex_data)
    print(f"  입력: {complex_data}")
    print(f"  결과: confidence={result['confidence']:.2f}")
    print(f"  warnings: {len(result['warnings'])}개")
    
    # 신뢰도가 0.0 이하로 떨어지지 않아야 함
    assert result['confidence'] >= 0.0, "신뢰도가 0.0 이하로 떨어지면 안 됨"
    assert result['confidence'] == 0.7, f"신뢰도가 0.7이어야 함 (1.0 - 0.1 - 0.1 - 0.05 - 0.05). 실제: {result['confidence']}"
    
    # 모든 필드가 누락된 경우
    empty_data = {}
    result_empty = validator.validate_source("statute_article", empty_data)
    print(f"  모든 필드 누락: confidence={result_empty['confidence']:.2f}")
    assert result_empty['confidence'] >= 0.0, "신뢰도가 0.0 이하로 떨어지면 안 됨"
    assert result_empty['confidence'] == 0.5, f"신뢰도가 0.5여야 함 (1.0 - 0.5). 실제: {result_empty['confidence']}"
    
    print("  ✓ 통과")
    return True


def test_special_characters():
    """특수 문자 포함 케이스 테스트"""
    print("\n[엣지 케이스] 특수 문자 포함 케이스 테스트")
    validator = SourceValidator()
    
    test_cases = [
        ({"statute_name": "민법(개정)", "article_no": "제1조"}, "괄호 포함"),
        ({"statute_name": "형법-특별법", "article_no": "제1조"}, "하이픈 포함"),
        ({"doc_id": "2020-다-12345", "court": "대법원", "casenames": "손해배상청구"}, "사건번호에 하이픈 포함"),
    ]
    
    for data, description in test_cases:
        source_type = "statute_article" if "statute_name" in data else "case_paragraph"
        result = validator.validate_source(source_type, data)
        print(f"  {description}: is_valid={result['is_valid']}, confidence={result['confidence']:.2f}")
        print(f"    warnings: {result['warnings']}")
        
        # 특수 문자가 포함되면 경고가 발생할 수 있음
        # 하지만 유효성은 유지될 수 있음
    
    print("  ✓ 통과")
    return True


def test_integration_formatter_validator():
    """UnifiedSourceFormatter와 SourceValidator 통합 테스트"""
    print("\n[통합 테스트] UnifiedSourceFormatter와 SourceValidator 통합 테스트")
    
    formatter = UnifiedSourceFormatter()
    validator = SourceValidator()
    
    test_data = {
        "statute_name": "민법",
        "article_no": "제1조",
        "clause_no": "1",
        "item_no": "1"
    }
    
    # 포맷터로 출처 정보 생성
    source_info = formatter.format_source("statute_article", test_data)
    print(f"  포맷터 결과: name={source_info.name}, type={source_info.type}")
    
    # 검증기로 검증
    validation_result = validator.validate_source("statute_article", test_data)
    print(f"  검증기 결과: is_valid={validation_result['is_valid']}, confidence={validation_result['confidence']:.2f}")
    
    # 포맷팅된 결과와 검증 결과 일치 확인
    assert source_info.name is not None, "포맷터가 출처명을 생성해야 함"
    assert source_info.type == "statute_article", "출처 타입이 일치해야 함"
    assert validation_result['is_valid'] == True, "검증 결과가 유효해야 함"
    assert validation_result['confidence'] > 0.8, "신뢰도가 높아야 함"
    
    # 검증 결과를 source_info에 추가
    source_info.validation = validation_result
    assert source_info.validation is not None, "검증 결과가 추가되어야 함"
    
    print("  ✓ 통과")
    return True


def test_performance():
    """성능 테스트"""
    print("\n[성능 테스트] 대량 데이터 처리 성능 테스트")
    validator = SourceValidator()
    
    # 테스트 데이터 생성
    test_sources = []
    for i in range(100):
        test_sources.append({
            "type": "statute_article",
            "data": {
                "statute_name": f"법{i}",
                "article_no": f"제{i}조"
            }
        })
    
    # 성능 측정
    start_time = time.time()
    for source in test_sources:
        validator.validate_source(source["type"], source["data"])
    elapsed_time = time.time() - start_time
    
    print(f"  100개 검증 시간: {elapsed_time:.3f}초")
    print(f"  평균 검증 시간: {(elapsed_time/100)*1000:.2f}ms")
    
    # 성능 기준: 100개 검증이 1초 이내에 완료되어야 함
    assert elapsed_time < 1.0, f"100개 검증이 1초 이내에 완료되어야 함. 실제: {elapsed_time:.3f}초"
    
    print("  ✓ 통과")
    return True


def run_all_tests():
    """모든 테스트 실행"""
    print("=" * 60)
    print("출처 검증기 개선 테스트 시작")
    print("=" * 60)
    
    stats = TestStatistics()
    stats.start()
    
    test_functions = [
        test_unknown_source_type,
        test_none_values,
        test_empty_string_vs_none,
        test_enforcement_regulations,
        test_court_name_partial_matching,
        test_confidence_boundary,
        test_special_characters,
        test_integration_formatter_validator,
        test_performance,
    ]
    
    try:
        for test_func in test_functions:
            try:
                result = test_func()
                stats.add_test(result)
            except AssertionError as e:
                print(f"  ❌ 실패: {e}")
                stats.add_test(False)
            except Exception as e:
                print(f"  ❌ 오류: {e}")
                import traceback
                traceback.print_exc()
                stats.add_test(False)
        
        stats.end()
        stats.print_summary()
        
        if stats.failed == 0:
            print("\n✅ 모든 테스트 통과!")
            return True
        else:
            print(f"\n❌ {stats.failed}개 테스트 실패")
            return False
            
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

