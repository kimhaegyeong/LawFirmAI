#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
경계 샘플 심판기 (두 번째 판단자)
 - 입력: 1차 판정(traditional/improved) 결과와 질의
 - 역할: 경계 구간(신뢰도 0.45~0.55 등)에 한해 보수적으로 제한 쪽으로 보정
   또는 일반정보성 표현이 강하면 허용 유지
환경 변수:
 - BOUNDARY_MIN=0.45, BOUNDARY_MAX=0.55
 - REFEREE_STRICT=1 이면 제한 편향, 0이면 중립
"""

import os
from typing import Dict, Any


class BoundaryReferee:
    def __init__(self):
        self.min_bound = float(os.getenv("BOUNDARY_MIN", "0.45"))
        self.max_bound = float(os.getenv("BOUNDARY_MAX", "0.55"))
        self.strict = os.getenv("REFEREE_STRICT", "1") == "1"
        # 카테고리별 덮어쓰기
        self.sensitive_min = float(os.getenv("BOUNDARY_MIN_SENSITIVE", str(self.min_bound)))
        self.sensitive_max = float(os.getenv("BOUNDARY_MAX_SENSITIVE", str(self.max_bound)))
        self.general_min = float(os.getenv("BOUNDARY_MIN_GENERAL", str(self.min_bound)))
        self.general_max = float(os.getenv("BOUNDARY_MAX_GENERAL", str(self.max_bound)))

        self.general_markers = ["일반적으로", "보통", "정의", "절차", "요건", "설명"]
        self.harmful_markers = ["방법을 알려주세요", "수법", "빠져나가", "회피", "피하는", "은닉", "변조", "조작", "위조", "탈세", "회피"]

    def is_boundary(self, confidence: float, category: str = "") -> bool:
        min_b, max_b = self.min_bound, self.max_bound
        if category in ("criminal_case_advice", "medical_legal_advice", "illegal_activity_assistance"):
            min_b, max_b = self.sensitive_min, self.sensitive_max
        elif category in ("general_legal_information", "edge_cases"):
            min_b, max_b = self.general_min, self.general_max
        return min_b <= confidence <= max_b

    def judge(self, query: str, result: Dict[str, Any]) -> Dict[str, Any]:
        conf = float(result.get("confidence", 0.0))
        category = result.get("category", "")
        if not self.is_boundary(conf, category):
            return result

        final = result.get("final_decision", "restricted")
        # 일반 정보성이 강하고 유해 마커가 없으면 허용 유지
        is_general = any(tok in query for tok in self.general_markers)
        is_harmful = any(tok in query for tok in self.harmful_markers)

        # 중립 정책으로 변경: REFEREE_STRICT=0일 때는 더 관대하게 처리
        if not self.strict:
            # 중립 모드에서는 일반 정보성이 있으면 허용 유지
            if is_general and not is_harmful:
                result["final_decision"] = "allowed"
                result.setdefault("reasoning", []).append("경계 심판기: 중립 모드 - 일반 정보성으로 허용")
                return result
            # 중립 모드에서는 유해 마커가 없으면 허용 고려
            elif not is_harmful:
                result["final_decision"] = "allowed"
                result.setdefault("reasoning", []).append("경계 심판기: 중립 모드 - 유해 마커 없음으로 허용")
                return result

        # 의료법 카테고리는 더 관대한 기준 적용
        if category == "medical_legal_advice":
            # 의료법에서는 일반 정보성이 있으면 허용 유지
            if is_general and not is_harmful:
                result["final_decision"] = "allowed"
                result.setdefault("reasoning", []).append("경계 심판기: 의료법 일반 정보성으로 허용 유지")
                return result
            # 의료법에서는 제한 편향을 완화
            if self.strict and final == "allowed":
                result["final_decision"] = "restricted"
                result.setdefault("reasoning", []).append("경계 심판기: 의료법 보수적 제한 (완화)")
            return result

        # 형사법과 불법행위 카테고리는 더 엄격한 기준 적용
        if category in ("criminal_case_advice", "illegal_activity_assistance"):
            # 형사법은 적당한 엄격함 적용 (완화)
            if category == "criminal_case_advice":
                # 형사법에서는 일반 정보성이 있으면 허용 고려
                if is_general and not is_harmful:
                    result["final_decision"] = "allowed"
                    result.setdefault("reasoning", []).append("경계 심판기: 형사법 일반 정보성으로 허용")
                    return result
                # 형사법에서는 제한 편향을 적당히 적용
                if self.strict and final == "allowed":
                    result["final_decision"] = "restricted"
                    result.setdefault("reasoning", []).append("경계 심판기: 형사법 보수적 제한 (적당함)")
            # 불법행위는 여전히 엄격한 기준 적용
            elif category == "illegal_activity_assistance":
                # 불법행위에서는 제한 편향을 강화
                if self.strict and final == "allowed":
                    result["final_decision"] = "restricted"
                    result.setdefault("reasoning", []).append(f"경계 심판기: {category} 엄격한 제한 보정")
                # 일반 정보성이 있어도 유해 마커가 있으면 제한
                elif is_harmful:
                    result["final_decision"] = "restricted"
                    result.setdefault("reasoning", []).append(f"경계 심판기: {category} 유해 마커로 제한")
            return result

        # 보수 정책: 경계에서는 제한 편향 (strict 모드에서만)
        if self.strict and not is_general and is_harmful:
            if final == "allowed":
                result["final_decision"] = "restricted"
                result.setdefault("reasoning", []).append("경계구간: 유해 마커 감지로 제한 보정")
            return result

        # 중립 또는 일반정보성 강한 경우 허용 유지
        if final == "restricted" and is_general and not is_harmful:
            result.setdefault("reasoning", []).append("경계구간: 일반정보 마커로 허용 유지")
        return result

    def re_evaluate(self, result: Dict[str, Any], category: str = "") -> Dict[str, Any]:
        """ML 통합 검증 시스템에서 사용하는 re_evaluate 메서드"""
        query = result.get("query", "")
        return self.judge(query, result)


