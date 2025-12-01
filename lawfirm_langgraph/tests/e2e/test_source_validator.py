# -*- coding: utf-8 -*-
"""
출처 검증기 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lawfirm_langgraph.core.generation.validators.source_validator import SourceValidator


def test_source_validator():
    """출처 검증기 테스트"""
    print("\n=== 출처 검증기 테스트 ===")
    
    validator = SourceValidator()
    
    # 1. statute_article 검증 테스트
    print("\n1. 법령 조문 검증 테스트")
    statute_data = {
        "statute_name": "민법",
        "article_no": "제1조",
        "clause_no": "1",
        "item_no": "1"
    }
    result = validator.validate_source("statute_article", statute_data)
    print(f"  입력: {statute_data}")
    print(f"  결과: is_valid={result['is_valid']}, confidence={result['confidence']:.2f}")
    print(f"  errors: {result['errors']}")
    print(f"  warnings: {result['warnings']}")
    assert result['is_valid'] == True, "유효한 법령 조문이어야 함"
    assert result['confidence'] > 0.8, "신뢰도가 높아야 함"
    print("  ✓ 통과")
    
    # 2. case_paragraph 검증 테스트
    print("\n2. 판례 검증 테스트")
    case_data = {
        "court": "대법원",
        "doc_id": "2020다12345",
        "casenames": "손해배상청구 사건"
    }
    result = validator.validate_source("case_paragraph", case_data)
    print(f"  입력: {case_data}")
    print(f"  결과: is_valid={result['is_valid']}, confidence={result['confidence']:.2f}")
    print(f"  errors: {result['errors']}")
    print(f"  warnings: {result['warnings']}")
    assert result['is_valid'] == True, "유효한 판례여야 함"
    print("  ✓ 통과")
    
    # 3. 잘못된 법령명 검증 테스트
    print("\n3. 잘못된 법령명 검증 테스트")
    invalid_statute_data = {
        "statute_name": "invalid_law",
        "article_no": "제1조"
    }
    result = validator.validate_source("statute_article", invalid_statute_data)
    print(f"  입력: {invalid_statute_data}")
    print(f"  결과: is_valid={result['is_valid']}, confidence={result['confidence']:.2f}")
    print(f"  errors: {result['errors']}")
    print(f"  warnings: {result['warnings']}")
    
    # 경고가 발생해야 함
    assert len(result['warnings']) > 0, "경고가 있어야 함"
    # 경고 메시지에 법령명이 포함되어야 함
    warning_found = any("법령명 형식" in warning or "invalid_law" in warning for warning in result['warnings'])
    assert warning_found, f"법령명 형식 경고가 포함되어야 함. 실제 경고: {result['warnings']}"
    # 신뢰도가 감소해야 함 (1.0 - 0.1 = 0.9)
    assert result['confidence'] < 1.0, f"신뢰도가 감소해야 함. 실제: {result['confidence']}"
    assert result['confidence'] == 0.9, f"신뢰도가 0.9여야 함. 실제: {result['confidence']}"
    # 에러는 없어야 함 (경고만 발생)
    assert len(result['errors']) == 0, "에러는 없어야 함 (경고만 발생)"
    # is_valid는 True여야 함 (경고만 있고 에러는 없으므로)
    assert result['is_valid'] == True, "경고만 있으면 유효해야 함"
    print("  ✓ 통과")
    
    # 3-1. 잘못된 조문 번호 형식 테스트
    print("\n3-1. 잘못된 조문 번호 형식 테스트")
    invalid_article_data = {
        "statute_name": "민법",
        "article_no": "1조"  # "제" 없음
    }
    result = validator.validate_source("statute_article", invalid_article_data)
    print(f"  입력: {invalid_article_data}")
    print(f"  결과: is_valid={result['is_valid']}, confidence={result['confidence']:.2f}")
    print(f"  warnings: {result['warnings']}")
    
    assert len(result['warnings']) > 0, "조문 번호 형식 경고가 있어야 함"
    warning_found = any("조문 번호 형식" in warning for warning in result['warnings'])
    assert warning_found, f"조문 번호 형식 경고가 포함되어야 함. 실제 경고: {result['warnings']}"
    assert result['confidence'] < 1.0, "신뢰도가 감소해야 함"
    print("  ✓ 통과")
    
    # 3-2. 잘못된 항/호 번호 형식 테스트
    print("\n3-2. 잘못된 항/호 번호 형식 테스트")
    invalid_clause_data = {
        "statute_name": "민법",
        "article_no": "제1조",
        "clause_no": "invalid",  # 잘못된 형식
        "item_no": "invalid"  # 잘못된 형식
    }
    result = validator.validate_source("statute_article", invalid_clause_data)
    print(f"  입력: {invalid_clause_data}")
    print(f"  결과: is_valid={result['is_valid']}, confidence={result['confidence']:.2f}")
    print(f"  warnings: {result['warnings']}")
    
    assert len(result['warnings']) >= 2, "항/호 번호 형식 경고가 2개 이상 있어야 함"
    clause_warning = any("항 번호 형식" in warning for warning in result['warnings'])
    item_warning = any("호 번호 형식" in warning for warning in result['warnings'])
    assert clause_warning, "항 번호 형식 경고가 있어야 함"
    assert item_warning, "호 번호 형식 경고가 있어야 함"
    # 신뢰도 감소: 1.0 - 0.05 (항) - 0.05 (호) = 0.9
    assert result['confidence'] <= 0.9, "신뢰도가 감소해야 함"
    print("  ✓ 통과")
    
    # 4. 필수 필드 누락 검증 테스트
    print("\n4. 필수 필드 누락 검증 테스트")
    missing_data = {
        "court": "",
        "casenames": ""
    }
    result = validator.validate_source("case_paragraph", missing_data)
    print(f"  입력: {missing_data}")
    print(f"  결과: is_valid={result['is_valid']}, confidence={result['confidence']:.2f}")
    print(f"  errors: {result['errors']}")
    print(f"  warnings: {result['warnings']}")
    
    # 에러가 발생해야 함
    assert result['is_valid'] == False, "유효하지 않아야 함"
    assert len(result['errors']) > 0, "에러가 있어야 함"
    # 에러 메시지에 필수 필드 정보가 포함되어야 함
    error_found = any("법원명" in error or "사건명" in error for error in result['errors'])
    assert error_found, f"필수 필드 누락 에러가 포함되어야 함. 실제 에러: {result['errors']}"
    # 신뢰도가 감소해야 함 (1.0 - 0.3 = 0.7)
    assert result['confidence'] < 1.0, "신뢰도가 감소해야 함"
    assert result['confidence'] == 0.7, f"신뢰도가 0.7이어야 함. 실제: {result['confidence']}"
    print("  ✓ 통과")
    
    # 4-1. 법령명 누락 테스트
    print("\n4-1. 법령명 누락 테스트")
    missing_statute_data = {
        "article_no": "제1조"
    }
    result = validator.validate_source("statute_article", missing_statute_data)
    print(f"  입력: {missing_statute_data}")
    print(f"  결과: is_valid={result['is_valid']}, confidence={result['confidence']:.2f}")
    print(f"  errors: {result['errors']}")
    
    assert result['is_valid'] == False, "법령명이 없으면 유효하지 않아야 함"
    assert len(result['errors']) > 0, "에러가 있어야 함"
    error_found = any("법령명" in error for error in result['errors'])
    assert error_found, f"법령명 누락 에러가 포함되어야 함. 실제 에러: {result['errors']}"
    assert result['confidence'] == 0.5, f"신뢰도가 0.5여야 함. 실제: {result['confidence']}"
    print("  ✓ 통과")
    
    # 4-2. 기관명 누락 테스트 (결정례)
    print("\n4-2. 기관명 누락 테스트 (결정례)")
    missing_org_data = {
        "doc_id": "2020-123"
    }
    result = validator.validate_source("decision_paragraph", missing_org_data)
    print(f"  입력: {missing_org_data}")
    print(f"  결과: is_valid={result['is_valid']}, confidence={result['confidence']:.2f}")
    print(f"  errors: {result['errors']}")
    print(f"  warnings: {result['warnings']}")
    
    assert result['is_valid'] == False, "기관명이 없으면 유효하지 않아야 함"
    assert len(result['errors']) > 0, "에러가 있어야 함"
    error_found = any("기관명" in error for error in result['errors'])
    assert error_found, f"기관명 누락 에러가 포함되어야 함. 실제 에러: {result['errors']}"
    # doc_id가 있으므로 경고는 없어야 함
    assert len(result['warnings']) == 0, "doc_id가 있으면 경고가 없어야 함"
    # 신뢰도: 1.0 - 0.3 (기관명 누락) = 0.7
    assert result['confidence'] == 0.7, f"신뢰도가 0.7이어야 함 (1.0 - 0.3). 실제: {result['confidence']}"
    print("  ✓ 통과")
    
    # 4-2-1. 기관명과 문서 ID 모두 누락 테스트 (결정례)
    print("\n4-2-1. 기관명과 문서 ID 모두 누락 테스트 (결정례)")
    missing_all_decision_data = {}
    result = validator.validate_source("decision_paragraph", missing_all_decision_data)
    print(f"  입력: {missing_all_decision_data}")
    print(f"  결과: is_valid={result['is_valid']}, confidence={result['confidence']:.2f}")
    print(f"  errors: {result['errors']}")
    print(f"  warnings: {result['warnings']}")
    
    assert result['is_valid'] == False, "기관명과 문서 ID가 모두 없으면 유효하지 않아야 함"
    assert len(result['errors']) > 0, "에러가 있어야 함"
    assert len(result['warnings']) > 0, "문서 ID 누락 경고가 있어야 함"
    error_found = any("기관명" in error for error in result['errors'])
    assert error_found, f"기관명 누락 에러가 포함되어야 함. 실제 에러: {result['errors']}"
    warning_found = any("문서 ID" in warning for warning in result['warnings'])
    assert warning_found, f"문서 ID 누락 경고가 포함되어야 함. 실제 경고: {result['warnings']}"
    # 신뢰도: 1.0 - 0.3 (기관명 누락) - 0.1 (문서 ID 누락) = 0.6
    assert result['confidence'] == 0.6, f"신뢰도가 0.6이어야 함 (1.0 - 0.3 - 0.1). 실제: {result['confidence']}"
    print("  ✓ 통과")
    
    # 4-3. 기관명과 제목 모두 누락 테스트 (해석례)
    print("\n4-3. 기관명과 제목 모두 누락 테스트 (해석례)")
    missing_all_data = {
        "doc_id": "2020-456"
    }
    result = validator.validate_source("interpretation_paragraph", missing_all_data)
    print(f"  입력: {missing_all_data}")
    print(f"  결과: is_valid={result['is_valid']}, confidence={result['confidence']:.2f}")
    print(f"  errors: {result['errors']}")
    print(f"  warnings: {result['warnings']}")
    
    assert result['is_valid'] == False, "기관명과 제목이 모두 없으면 유효하지 않아야 함"
    assert len(result['errors']) > 0, "에러가 있어야 함"
    error_found = any("기관명" in error or "제목" in error for error in result['errors'])
    assert error_found, f"기관명 또는 제목 누락 에러가 포함되어야 함. 실제 에러: {result['errors']}"
    # doc_id가 있으므로 경고는 없어야 함
    assert len(result['warnings']) == 0, "doc_id가 있으면 경고가 없어야 함"
    # 신뢰도: 1.0 - 0.3 (기관명/제목 누락) = 0.7
    assert result['confidence'] == 0.7, f"신뢰도가 0.7이어야 함 (1.0 - 0.3). 실제: {result['confidence']}"
    print("  ✓ 통과")
    
    # 4-3-1. 기관명, 제목, 문서 ID 모두 누락 테스트 (해석례)
    print("\n4-3-1. 기관명, 제목, 문서 ID 모두 누락 테스트 (해석례)")
    missing_all_interpretation_data = {}
    result = validator.validate_source("interpretation_paragraph", missing_all_interpretation_data)
    print(f"  입력: {missing_all_interpretation_data}")
    print(f"  결과: is_valid={result['is_valid']}, confidence={result['confidence']:.2f}")
    print(f"  errors: {result['errors']}")
    print(f"  warnings: {result['warnings']}")
    
    assert result['is_valid'] == False, "기관명, 제목, 문서 ID가 모두 없으면 유효하지 않아야 함"
    assert len(result['errors']) > 0, "에러가 있어야 함"
    assert len(result['warnings']) > 0, "문서 ID 누락 경고가 있어야 함"
    error_found = any("기관명" in error or "제목" in error for error in result['errors'])
    assert error_found, f"기관명 또는 제목 누락 에러가 포함되어야 함. 실제 에러: {result['errors']}"
    warning_found = any("문서 ID" in warning for warning in result['warnings'])
    assert warning_found, f"문서 ID 누락 경고가 포함되어야 함. 실제 경고: {result['warnings']}"
    # 신뢰도: 1.0 - 0.3 (기관명/제목 누락) - 0.1 (문서 ID 누락) = 0.6
    assert result['confidence'] == 0.6, f"신뢰도가 0.6이어야 함 (1.0 - 0.3 - 0.1). 실제: {result['confidence']}"
    print("  ✓ 통과")
    
    # 5. 잘못된 사건번호 형식 테스트
    print("\n5. 잘못된 사건번호 형식 테스트")
    invalid_case_number_data = {
        "court": "대법원",
        "doc_id": "invalid123",  # 잘못된 형식
        "casenames": "손해배상청구 사건"
    }
    result = validator.validate_source("case_paragraph", invalid_case_number_data)
    print(f"  입력: {invalid_case_number_data}")
    print(f"  결과: is_valid={result['is_valid']}, confidence={result['confidence']:.2f}")
    print(f"  warnings: {result['warnings']}")
    
    assert result['is_valid'] == True, "경고만 있으면 유효해야 함"
    assert len(result['warnings']) > 0, "사건번호 형식 경고가 있어야 함"
    warning_found = any("사건번호 형식" in warning for warning in result['warnings'])
    assert warning_found, f"사건번호 형식 경고가 포함되어야 함. 실제 경고: {result['warnings']}"
    assert result['confidence'] < 1.0, "신뢰도가 감소해야 함"
    print("  ✓ 통과")
    
    # 6. 잘못된 법원명 테스트
    print("\n6. 잘못된 법원명 테스트")
    invalid_court_data = {
        "court": "잘못된법원명",
        "doc_id": "2020다12345",
        "casenames": "손해배상청구 사건"
    }
    result = validator.validate_source("case_paragraph", invalid_court_data)
    print(f"  입력: {invalid_court_data}")
    print(f"  결과: is_valid={result['is_valid']}, confidence={result['confidence']:.2f}")
    print(f"  warnings: {result['warnings']}")
    
    assert result['is_valid'] == True, "경고만 있으면 유효해야 함"
    assert len(result['warnings']) > 0, "법원명 형식 경고가 있어야 함"
    warning_found = any("법원명" in warning or "표준 형식" in warning for warning in result['warnings'])
    assert warning_found, f"법원명 형식 경고가 포함되어야 함. 실제 경고: {result['warnings']}"
    print("  ✓ 통과")
    
    # 7. decision_paragraph 검증 테스트
    print("\n7. 결정례 검증 테스트")
    decision_data = {
        "org": "법제처",
        "doc_id": "2020-123"
    }
    result = validator.validate_source("decision_paragraph", decision_data)
    print(f"  입력: {decision_data}")
    print(f"  결과: is_valid={result['is_valid']}, confidence={result['confidence']:.2f}")
    assert result['is_valid'] == True, "유효한 결정례여야 함"
    assert result['confidence'] == 1.0, f"완벽한 데이터는 신뢰도 1.0이어야 함. 실제: {result['confidence']}"
    assert len(result['errors']) == 0, "에러가 없어야 함"
    print("  ✓ 통과")
    
    # 8. interpretation_paragraph 검증 테스트
    print("\n8. 해석례 검증 테스트")
    interpretation_data = {
        "org": "법제처",
        "title": "법령 해석",
        "doc_id": "2020-456"
    }
    result = validator.validate_source("interpretation_paragraph", interpretation_data)
    print(f"  입력: {interpretation_data}")
    print(f"  결과: is_valid={result['is_valid']}, confidence={result['confidence']:.2f}")
    assert result['is_valid'] == True, "유효한 해석례여야 함"
    assert result['confidence'] == 1.0, f"완벽한 데이터는 신뢰도 1.0이어야 함. 실제: {result['confidence']}"
    assert len(result['errors']) == 0, "에러가 없어야 함"
    print("  ✓ 통과")
    
    # 9. 복합 검증 테스트 (여러 문제가 동시에 있는 경우)
    print("\n9. 복합 검증 테스트 (여러 문제가 동시에 있는 경우)")
    complex_data = {
        "statute_name": "invalid_law",  # 잘못된 법령명
        "article_no": "1조",  # 잘못된 조문 번호 형식
        "clause_no": "invalid"  # 잘못된 항 번호 형식
    }
    result = validator.validate_source("statute_article", complex_data)
    print(f"  입력: {complex_data}")
    print(f"  결과: is_valid={result['is_valid']}, confidence={result['confidence']:.2f}")
    print(f"  errors: {result['errors']}")
    print(f"  warnings: {result['warnings']}")
    
    # 여러 경고가 발생해야 함
    assert len(result['warnings']) >= 2, f"여러 경고가 발생해야 함. 실제: {len(result['warnings'])}개"
    # 신뢰도가 여러 번 감소해야 함
    assert result['confidence'] < 0.9, "신뢰도가 여러 번 감소해야 함"
    print("  ✓ 통과")
    
    print("\n=== 모든 테스트 통과 ===")


if __name__ == "__main__":
    test_source_validator()

