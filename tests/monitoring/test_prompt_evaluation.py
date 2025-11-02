"""
프롬프트 평가 및 개선 테스트
generate_answer_enhanced에서 생성된 프롬프트를 평가하고 개선점을 찾습니다.
"""
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Any

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def evaluate_prompt(prompt_text: str) -> Dict[str, Any]:
    """프롬프트를 평가하고 개선점을 찾습니다."""
    issues = []
    suggestions = []

    # 1. 중복 문서 섹션 확인
    doc_section_patterns = [
        r"##\s*검색된 법률 문서",
        r"##\s*제공된 법률 문서",
        r"##\s*검색된 판례 문서",
        r"##\s*검색된 법률 문서 및 정보",
        r"##\s*검색된 법률 문서 및 판례"
    ]

    found_sections = []
    for pattern in doc_section_patterns:
        matches = re.findall(pattern, prompt_text, re.IGNORECASE)
        if matches:
            found_sections.extend(matches)

    if len(found_sections) > 1:
        unique_sections = set(found_sections)
        if len(unique_sections) > 1:
            issues.append({
                "type": "중복 문서 섹션",
                "severity": "high",
                "description": f"문서 섹션이 {len(unique_sections)}번 나타남",
                "sections": list(unique_sections)
            })
            suggestions.append("문서 섹션을 한 번만 포함하도록 수정 필요")

    # 2. 지침 문구 중복 확인
    instruction_phrases = [
        "반드시 참고하여 답변하세요",
        "반드시 이 문서들을 참고",
        "반드시 활용",
        "최소 2개 이상 인용",
        "절대 금지"
    ]

    phrase_counts = {}
    for phrase in instruction_phrases:
        count = len(re.findall(re.escape(phrase), prompt_text, re.IGNORECASE))
        if count > 1:
            phrase_counts[phrase] = count

    if phrase_counts:
        issues.append({
            "type": "지침 문구 중복",
            "severity": "medium",
            "description": "같은 지침이 여러 번 반복됨",
            "phrases": phrase_counts
        })
        suggestions.append("지침 문구를 통합하여 한 번만 표시")

    # 3. 문서 목록 중복 확인
    # "문서 1:", "문서 2:" 같은 패턴 찾기
    doc_number_pattern = r"문서\s*\d+\s*:"
    doc_numbers = re.findall(doc_number_pattern, prompt_text)

    if len(doc_numbers) > len(set(doc_numbers)):
        issues.append({
            "type": "문서 번호 중복",
            "severity": "high",
            "description": "같은 문서 번호가 여러 번 나타남",
            "count": len(doc_numbers)
        })
        suggestions.append("문서 목록 중복 제거 필요")

    # 4. 프롬프트 길이 확인
    prompt_length = len(prompt_text)
    token_estimate = prompt_length // 3  # 대략적인 토큰 수 추정

    if prompt_length > 8000:
        issues.append({
            "type": "프롬프트 길이",
            "severity": "medium",
            "description": f"프롬프트가 너무 깁니다 ({prompt_length}자, 약 {token_estimate} 토큰)",
            "length": prompt_length,
            "estimated_tokens": token_estimate
        })
        suggestions.append("프롬프트 길이 최적화 필요 (중복 제거, 불필요한 섹션 제거)")

    # 5. 문서 없음 메시지 오류 확인
    if "현재 관련 법률 문서를 찾지 못했습니다" in prompt_text:
        # 문서 섹션이 있는지 확인
        if "검색된 법률 문서" in prompt_text or "## 🔍" in prompt_text:
            issues.append({
                "type": "로직 오류",
                "severity": "critical",
                "description": "문서가 있는데도 '문서를 찾지 못했습니다' 메시지가 표시됨"
            })
            suggestions.append("final_instruction_section 로직 수정 필요")

    # 6. 필수 준수 사항 섹션 확인
    if "## ⚠️ 필수 준수 사항" in prompt_text or "## ⚠️ 핵심 지침" in prompt_text:
        section_match = re.search(r"##\s*⚠️\s*(필수 준수 사항|핵심 지침)\s*\n\n(.*?)(?=\n##|\Z)", prompt_text, re.DOTALL)
        if section_match:
            section_content = section_match.group(2).strip()
            if len(section_content) < 50:
                issues.append({
                    "type": "빈 필수 준수 사항",
                    "severity": "medium",
                    "description": "필수 준수 사항 섹션이 거의 비어있음",
                    "content_length": len(section_content)
                })
                suggestions.append("필수 준수 사항 섹션에 적절한 내용 추가 필요")

    return {
        "issues": issues,
        "suggestions": suggestions,
        "metrics": {
            "prompt_length": prompt_length,
            "estimated_tokens": token_estimate,
            "document_sections": len(found_sections),
            "total_issues": len(issues),
            "critical_issues": len([i for i in issues if i.get("severity") == "critical"]),
            "high_issues": len([i for i in issues if i.get("severity") == "high"]),
            "medium_issues": len([i for i in issues if i.get("severity") == "medium"])
        }
    }

def find_latest_prompt_file() -> str:
    """가장 최근 프롬프트 파일 찾기"""
    debug_dir = Path("debug/prompts")
    if not debug_dir.exists():
        return None

    prompt_files = list(debug_dir.glob("prompt_*.txt"))
    if not prompt_files:
        return None

    # 수정 시간 기준으로 정렬
    prompt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(prompt_files[0])

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("프롬프트 평가 및 개선 테스트")
    print("=" * 80)

    # 최신 프롬프트 파일 찾기
    prompt_file = find_latest_prompt_file()

    if not prompt_file or not os.path.exists(prompt_file):
        print("❌ 프롬프트 파일을 찾을 수 없습니다.")
        print("   먼저 LangGraph 테스트를 실행하여 프롬프트를 생성하세요.")
        return

    print(f"\n📄 프롬프트 파일: {prompt_file}")

    # 프롬프트 읽기
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    print(f"프롬프트 길이: {len(prompt_text):,}자\n")

    # 프롬프트 평가
    evaluation = evaluate_prompt(prompt_text)

    # 결과 출력
    print("=" * 80)
    print("📊 평가 결과")
    print("=" * 80)

    metrics = evaluation["metrics"]
    print(f"\n메트릭:")
    print(f"  - 프롬프트 길이: {metrics['prompt_length']:,}자")
    print(f"  - 예상 토큰 수: {metrics['estimated_tokens']:,}")
    print(f"  - 문서 섹션 수: {metrics['document_sections']}")
    print(f"  - 총 이슈 수: {metrics['total_issues']}")
    print(f"  - 심각 이슈: {metrics['critical_issues']}")
    print(f"  - 높은 우선순위 이슈: {metrics['high_issues']}")
    print(f"  - 중간 우선순위 이슈: {metrics['medium_issues']}")

    # 이슈 출력
    if evaluation["issues"]:
        print(f"\n⚠️ 발견된 이슈 ({len(evaluation['issues'])}개):")
        for idx, issue in enumerate(evaluation["issues"], 1):
            severity_icon = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🔵"
            }.get(issue["severity"], "⚪")

            print(f"\n{idx}. {severity_icon} [{issue['severity'].upper()}] {issue['type']}")
            print(f"   설명: {issue['description']}")
            if "details" in issue:
                print(f"   상세: {issue['details']}")

    # 개선 제안 출력
    if evaluation["suggestions"]:
        print(f"\n💡 개선 제안 ({len(evaluation['suggestions'])}개):")
        for idx, suggestion in enumerate(evaluation["suggestions"], 1):
            print(f"  {idx}. {suggestion}")

    if not evaluation["issues"]:
        print("\n✅ 프롬프트에 심각한 문제가 없습니다!")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
