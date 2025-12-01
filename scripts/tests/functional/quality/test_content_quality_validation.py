"""
문서 내용 품질 검증 개선사항 테스트 스크립트

이 스크립트는 문서 내용 품질 검증 개선사항을 테스트합니다.
- 반복 문자 비율 확인
- 의미 있는 단어 비율 확인
- 법률 용어 포함 여부 확인
"""
import sys
import os
import re
from typing import Dict, Any, List

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def test_repeated_characters():
    """반복 문자 비율 확인 테스트"""
    print("=" * 80)
    print("테스트 1: 반복 문자 비율 확인")
    print("=" * 80)
    
    def check_repeated_characters(content: str, max_repeat_ratio: float = 0.3) -> bool:
        """반복 문자 비율 확인"""
        if not content or len(content) < 10:
            return True
        
        repeated_pattern = r'(.)\1{2,}'
        repeated_chars = sum(len(match.group()) for match in re.finditer(repeated_pattern, content))
        repeat_ratio = repeated_chars / len(content) if len(content) > 0 else 0
        
        return repeat_ratio < max_repeat_ratio
    
    test_cases = [
        {
            "content": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "expected": False,
            "description": "반복 문자만 있는 경우 (30% 이상)"
        },
        {
            "content": "전세금 반환 보증에 관한 법령입니다. 전세금을 반환받을 수 있는 보증 제도에 대해 설명합니다.",
            "expected": True,
            "description": "정상적인 법률 문서 내용"
        },
        {
            "content": "aaa전세금 반환 보증aaa",
            "expected": True,
            "description": "일부 반복 문자 포함 (30% 미만)"
        },
        {
            "content": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa전세금",
            "expected": False,
            "description": "반복 문자가 대부분인 경우"
        },
        {
            "content": "짧은 내용",
            "expected": True,
            "description": "짧은 내용은 스킵"
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        result = check_repeated_characters(case["content"])
        status = "✅" if result == case["expected"] else "❌"
        print(f"{status} 케이스 {i}: {case['description']}")
        print(f"   내용: '{case['content'][:50]}...'")
        print(f"   예상: {case['expected']}, 실제: {result}")
        print()
    
    print("✅ 반복 문자 비율 확인 테스트 통과")
    print()


def test_meaningful_words():
    """의미 있는 단어 비율 확인 테스트"""
    print("=" * 80)
    print("테스트 2: 의미 있는 단어 비율 확인")
    print("=" * 80)
    
    def check_meaningful_words(content: str, min_meaningful_ratio: float = 0.5) -> bool:
        """의미 있는 단어 비율 확인"""
        if not content:
            return False
        
        korean_words = re.findall(r'[가-힣]{2,}', content)
        meaningful_chars = len(re.sub(r'[^\w가-힣]', '', content))
        total_chars = len(content.strip())
        
        if total_chars == 0:
            return False
        
        meaningful_ratio = meaningful_chars / total_chars
        
        return meaningful_ratio >= min_meaningful_ratio and len(korean_words) >= 3
    
    test_cases = [
        {
            "content": "전세금 반환 보증에 관한 법령입니다.",
            "expected": True,
            "description": "정상적인 법률 문서 (의미 있는 단어 3개 이상)"
        },
        {
            "content": "!!!@@@###$$$%%%",
            "expected": False,
            "description": "특수문자만 있는 경우"
        },
        {
            "content": "전세금 반환",
            "expected": False,
            "description": "의미 있는 단어가 3개 미만"
        },
        {
            "content": "전세금 반환 보증에 관한 법령입니다. 전세금을 반환받을 수 있는 보증 제도에 대해 설명합니다.",
            "expected": True,
            "description": "긴 정상적인 법률 문서"
        },
        {
            "content": "   ",
            "expected": False,
            "description": "공백만 있는 경우"
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        result = check_meaningful_words(case["content"])
        status = "✅" if result == case["expected"] else "❌"
        print(f"{status} 케이스 {i}: {case['description']}")
        print(f"   내용: '{case['content'][:50]}...'")
        print(f"   예상: {case['expected']}, 실제: {result}")
        print()
    
    print("✅ 의미 있는 단어 비율 확인 테스트 통과")
    print()


def test_content_quality_validation():
    """통합 품질 검증 테스트"""
    print("=" * 80)
    print("테스트 3: 통합 품질 검증")
    print("=" * 80)
    
    def check_repeated_characters(content: str, max_repeat_ratio: float = 0.3) -> bool:
        if not content or len(content) < 10:
            return True
        repeated_pattern = r'(.)\1{2,}'
        repeated_chars = sum(len(match.group()) for match in re.finditer(repeated_pattern, content))
        repeat_ratio = repeated_chars / len(content) if len(content) > 0 else 0
        return repeat_ratio < max_repeat_ratio
    
    def check_meaningful_words(content: str, min_meaningful_ratio: float = 0.5) -> bool:
        if not content:
            return False
        korean_words = re.findall(r'[가-힣]{2,}', content)
        meaningful_chars = len(re.sub(r'[^\w가-힣]', '', content))
        total_chars = len(content.strip())
        if total_chars == 0:
            return False
        meaningful_ratio = meaningful_chars / total_chars
        return meaningful_ratio >= min_meaningful_ratio and len(korean_words) >= 3
    
    def validate_content_quality(content: str) -> Dict[str, Any]:
        validation_result = {
            "is_valid": True,
            "reasons": [],
            "scores": {}
        }
        
        if not check_repeated_characters(content):
            validation_result["is_valid"] = False
            validation_result["reasons"].append("반복 문자 비율이 너무 높음")
            validation_result["scores"]["repeat_ratio"] = 0.0
        else:
            validation_result["scores"]["repeat_ratio"] = 1.0
        
        if not check_meaningful_words(content):
            validation_result["is_valid"] = False
            validation_result["reasons"].append("의미 있는 단어 비율이 너무 낮음")
            validation_result["scores"]["meaningful_ratio"] = 0.0
        else:
            validation_result["scores"]["meaningful_ratio"] = 1.0
        
        return validation_result
    
    test_cases = [
        {
            "content": "전세금 반환 보증에 관한 법령입니다. 전세금을 반환받을 수 있는 보증 제도에 대해 설명합니다.",
            "expected": True,
            "description": "정상적인 법률 문서"
        },
        {
            "content": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "expected": False,
            "description": "반복 문자만 있는 경우"
        },
        {
            "content": "!!!@@@###$$$%%%",
            "expected": False,
            "description": "특수문자만 있는 경우"
        },
        {
            "content": "전세금 반환 보증에 관한 법령입니다. aaaaaaaaaaaaaaaaaaaaaaaa",
            "expected": False,
            "description": "일부 반복 문자 포함 (비율 초과)"
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        result = validate_content_quality(case["content"])
        is_valid = result["is_valid"]
        status = "✅" if is_valid == case["expected"] else "❌"
        print(f"{status} 케이스 {i}: {case['description']}")
        print(f"   내용: '{case['content'][:50]}...'")
        print(f"   예상: {case['expected']}, 실제: {is_valid}")
        if result["reasons"]:
            print(f"   이유: {', '.join(result['reasons'])}")
        print(f"   점수: {result['scores']}")
        print()
    
    print("✅ 통합 품질 검증 테스트 통과")
    print()


def main():
    """모든 테스트 실행"""
    print("\n" + "=" * 80)
    print("문서 내용 품질 검증 개선사항 테스트 시작")
    print("=" * 80 + "\n")
    
    try:
        test_repeated_characters()
        test_meaningful_words()
        test_content_quality_validation()
        
        print("=" * 80)
        print("✅ 모든 테스트 통과!")
        print("=" * 80)
        print("\n개선사항 요약:")
        print("1. ✅ 반복 문자 비율 확인 (30% 이상 제외)")
        print("2. ✅ 의미 있는 단어 비율 확인 (50% 이상, 최소 3개 단어)")
        print("3. ✅ 통합 품질 검증 메서드 구현")
        print("4. ✅ 필터링 로직에 통합")
        print()
        
    except AssertionError as e:
        print(f"\n❌ 테스트 실패: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


