#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""개선된 답변 품질 테스트"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

print("=" * 70)
print("개선된 답변 품질 테스트")
print("=" * 70)

from source.services.langgraph_workflow.legal_workflow_enhanced import (
    EnhancedLegalQuestionWorkflow,
)
from source.services.langgraph_workflow.state_definitions import create_initial_state
from source.utils.langgraph_config import LangGraphConfig

config = LangGraphConfig()
workflow = EnhancedLegalQuestionWorkflow(config)

test_query = "이혼 절차에 대해 알려주세요"
print(f"\n질문: {test_query}")

state = create_initial_state(test_query, "test_quality", "user_1")
state["user_query"] = test_query

# 입력 검증
state = workflow.validate_input(state)
state = workflow.detect_special_queries(state)
state["query"] = test_query
state = workflow.classify_query(state)
state = workflow.analyze_query_hybrid(state)
state = workflow.validate_legal_restrictions(state)
state = workflow.retrieve_documents(state)
state = workflow.enrich_conversation_context(state)
state = workflow.personalize_response(state)
state = workflow.manage_memory_quality(state)
state = workflow.generate_answer_enhanced(state)
state = workflow.enhance_completion(state)
state = workflow.add_disclaimer(state)

answer = state["answer"]
print(f"\n답변 (전체 {len(answer)}자):")
print(answer)
print("\n" + "=" * 70)

# 품질 검증
issues = []
if "metadata:" in answer or "law_id:" in answer:
    issues.append("❌ 메타데이터가 답변에 포함됨")
if len(answer) < 200:
    issues.append("❌ 답변이 너무 짧음")
if "죄송합니다. 해당 질문에 대한 관련 법률 정보를 찾을 수 없었습니다" in answer:
    issues.append("❌ 컨텍스트를 인식하지 못함")

if issues:
    print("⚠️  품질 문제:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("✅ 답변 품질 양호")
