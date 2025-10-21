#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2-stage LLM 심판기 (옵션)
 - 환경변수 USE_LLM_REFEREE=1 이면 활성화
 - 현재는 플러그형 인터페이스: 외부 LLM 호출부는 사용자 환경에 맞게 구현 필요
 - 기본 동작: 민감 카테고리 경계 샘플만 제한 편향으로 보정
"""

import os
from typing import Dict, Any


class LLMReferee:
    def __init__(self):
        self.enabled = os.getenv("USE_LLM_REFEREE", "0") == "1"
        # 실제 LLM 호출 설정이 있다면 여기에 초기화
        try:
            self.min_conf = float(os.getenv("LLM_REFEREE_MIN", "0.40"))
            self.max_conf = float(os.getenv("LLM_REFEREE_MAX", "0.70"))
        except Exception:
            self.min_conf = 0.40
            self.max_conf = 0.70

    def _should_review(self, result: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        category = result.get("category", "")
        conf = float(result.get("confidence", 0.0))
        if category not in ("criminal_case_advice", "medical_legal_advice", "illegal_activity_assistance"):
            return False
        return self.min_conf <= conf <= self.max_conf

    def _call_llm(self, query: str) -> str:
        # TODO: 실제 LLM 호출 구현. 현재는 보수 기본값으로 'restricted' 반환
        return "restricted"

    def judge(self, query: str, result: Dict[str, Any]) -> Dict[str, Any]:
        if not self._should_review(result):
            return result
        try:
            llm_decision = self._call_llm(query)
            if llm_decision in ("restricted", "allowed"):
                result["final_decision"] = llm_decision
                result.setdefault("reasoning", []).append("LLM 심판기 보정 적용")
            return result
        except Exception:
            return result


