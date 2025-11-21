#!/usr/bin/env python3
"""
답변 품질 확인 스크립트
"""

import sys
import asyncio
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
lawfirm_langgraph_path = project_root / "lawfirm_langgraph"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_path) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_path))

from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
from lawfirm_langgraph.config.langgraph_config import LangGraphConfig

async def test_query():
    """질의 테스트 및 답변 품질 확인"""
    print("="*80)
    print("답변 품질 확인 테스트")
    print("="*80)
    
    config = LangGraphConfig.from_env()
    service = LangGraphWorkflowService(config)
    
    query = "임대차 계약서 작성 시 임대인과 임차인의 권리와 의무는 무엇인가요?"
    print(f"\n[질의]")
    print(f"  {query}")
    
    result = await service.process_query(query)
    
    print(f"\n[결과]")
    answer = result.get("answer", "")
    # answer가 dict인 경우 처리
    if isinstance(answer, dict):
        answer = answer.get("answer", str(answer))
    if not isinstance(answer, str):
        answer = str(answer)
    quality_score = result.get("metadata", {}).get("quality_score", 0.0)
    quality_check_passed = result.get("metadata", {}).get("quality_check_passed", False)
    sources = result.get("sources", [])
    confidence = result.get("confidence", 0.0)
    
    print(f"  답변 길이: {len(answer)}자")
    print(f"  품질 점수: {quality_score:.2f}")
    print(f"  검증 통과: {quality_check_passed}")
    print(f"  신뢰도: {confidence:.2f}")
    print(f"  소스 개수: {len(sources)}")
    
    print(f"\n[답변 내용 (처음 500자)]")
    if isinstance(answer, str):
        print(f"  {answer[:500]}...")
    else:
        print(f"  {str(answer)[:500]}...")
    
    print(f"\n[소스 정보]")
    for i, source in enumerate(sources[:10], 1):  # 개선: 5개 → 10개로 증가
        source_type = source.get("type", "unknown")
        source_name = source.get("source", "N/A")
        print(f"  [{i}] {source_type}: {source_name[:60]}...")
    
    print(f"\n[품질 분석]")
    issues = []
    
    if not answer or len(answer) < 100:
        issues.append(f"❌ 답변 길이 부족: {len(answer)}자 (최소 100자 필요)")
    else:
        print(f"  ✅ 답변 길이: {len(answer)}자 (기준 통과)")
    
    if quality_score < 0.75:
        issues.append(f"❌ 품질 점수 미달: {quality_score:.2f} (기준: 0.75)")
    else:
        print(f"  ✅ 품질 점수: {quality_score:.2f} (기준 통과)")
    
    if not quality_check_passed:
        issues.append("❌ 검증 실패")
    else:
        print(f"  ✅ 검증 통과")
    
    if not sources or len(sources) == 0:
        issues.append("❌ 법률 소스 부재")
    else:
        print(f"  ✅ 소스 개수: {len(sources)}개")
    
    # 형식 오류 확인
    format_errors = []
    if "STEP" in answer or "STEP 0" in answer:
        format_errors.append("STEP 패턴 포함")
    if "원본 품질 평가" in answer:
        format_errors.append("평가 템플릿 포함")
    if "• [ ]" in answer:
        format_errors.append("체크리스트 형식 포함")
    
    if format_errors:
        issues.append(f"❌ 형식 오류: {', '.join(format_errors)}")
    else:
        print(f"  ✅ 형식 오류 없음")
    
    if issues:
        print(f"\n[발견된 문제점]")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"\n  ✅ 모든 품질 기준을 통과했습니다!")
    
    return result

if __name__ == "__main__":
    result = asyncio.run(test_query())

