#!/usr/bin/env python3
"""
LangGraph 답변 문제점 분석 스크립트
"""

import sys
from pathlib import Path
lawfirm_langgraph_path = Path('.')
sys.path.insert(0, str(lawfirm_langgraph_path))

from core.utils.langsmith_analyzer import LangGraphQueryAnalyzer
from core.agents.workflow_constants import QualityThresholds, WorkflowConstants

analyzer = LangGraphQueryAnalyzer()
run = analyzer.client.read_run('6dc1e8e1-658f-4494-aee5-749037b02ec7')

print("="*80)
print("LangGraph 답변 문제점 분석")
print("="*80)

print(f"\n[Run 정보]")
print(f"  - Run ID: {run.id}")
print(f"  - 질의: {analyzer._extract_query(run)}")
print(f"  - 상태: {run.status}")
print(f"  - 실행 시간: {(run.end_time - run.start_time).total_seconds():.2f}초" if run.start_time and run.end_time else "N/A")

print(f"\n[품질 검증 기준]")
print(f"  - 최소 답변 길이: {WorkflowConstants.MIN_ANSWER_LENGTH_VALIDATION}자")
print(f"  - 품질 통과 기준: {QualityThresholds.QUALITY_PASS_THRESHOLD}")
print(f"  - 중간 품질 기준: {QualityThresholds.MEDIUM_QUALITY_THRESHOLD}")

if run.outputs:
    outputs = run.outputs if isinstance(run.outputs, dict) else {}
    answer = outputs.get("answer", "")
    quality_score = outputs.get("quality_score", 0.0)
    quality_check_passed = outputs.get("quality_check_passed", False)
    sources = outputs.get("sources", [])
    
    print(f"\n[답변 정보]")
    print(f"  - 답변 길이: {len(answer) if answer else 0}자")
    print(f"  - 품질 점수: {quality_score:.2f}")
    print(f"  - 검증 통과: {quality_check_passed}")
    print(f"  - 소스 개수: {len(sources) if isinstance(sources, list) else 0}")
    
    if answer:
        print(f"\n[답변 내용 (처음 500자)]")
        print(f"  {answer[:500]}...")
    
    print(f"\n[문제점 분석]")
    issues = []
    
    if not answer or len(answer) < WorkflowConstants.MIN_ANSWER_LENGTH_VALIDATION:
        issues.append(f"1. 답변 길이 부족: {len(answer) if answer else 0}자 (최소 {WorkflowConstants.MIN_ANSWER_LENGTH_VALIDATION}자 필요)")
    
    if quality_score < QualityThresholds.QUALITY_PASS_THRESHOLD:
        issues.append(f"2. 품질 점수 미달: {quality_score:.2f} (기준: {QualityThresholds.QUALITY_PASS_THRESHOLD})")
    
    if not sources or len(sources) == 0:
        issues.append("3. 법률 소스 부재: 답변에 인용된 법률 소스가 없음")
    
    if quality_check_passed and quality_score < 0.6:
        issues.append("4. 검증 기준 과소: 품질 점수가 낮음에도 검증 통과")
    
    if not issues:
        issues.append("기본 검증은 통과했으나, 상세 검증 필요")
    
    for i, issue in enumerate(issues, 1):
        print(f"  {issue}")

child_runs = analyzer._get_child_runs(run.id)
if child_runs:
    print(f"\n[자식 노드 실행 정보]")
    for i, child in enumerate(child_runs[:5], 1):
        duration = (child.end_time - child.start_time).total_seconds() if child.start_time and child.end_time else 0
        print(f"  [{i}] {child.name} ({child.run_type}): {duration:.2f}초")

