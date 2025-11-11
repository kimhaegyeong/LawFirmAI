# -*- coding: utf-8 -*-
"""
키워드 포함도 향상 시스템
답변 품질 향상을 위한 키워드 포함도 개선 (0.390 → 0.7+ 목표)
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict, Counter


class KeywordCoverageEnhancer:
    """키워드 포함도 향상 시스템"""
    
    def __init__(self):
        """초기화"""
        self.target_coverage = 0.7  # 목표 포함도
        self.current_coverage = 0.0  # 현재 포함도
        
        # 법률 도메인별 핵심 키워드 매핑
        self.domain_keywords = {
            "민사법": {
                "core": ["계약", "불법행위", "소유권", "채권", "채무", "손해배상"],
                "important": ["계약서", "당사자", "조건", "기간", "효력", "무효", "취소"],
                "supporting": ["체결", "이행", "해지", "위약금", "담보", "보증"]
            },
            "형사법": {
                "core": ["범죄", "형량", "처벌", "구성요건", "고의", "과실"],
                "important": ["수사", "재판", "기소", "공소", "증거", "변호인"],
                "supporting": ["구속", "보석", "선고", "집행", "가석방", "형의 집행"]
            },
            "가족법": {
                "core": ["이혼", "상속", "양육", "재산분할", "위자료", "면접교섭권"],
                "important": ["협의이혼", "조정이혼", "재판이혼", "상속인", "상속분", "유언"],
                "supporting": ["가정법원", "조정", "재판", "확정", "유류분", "한정승인"]
            },
            "상사법": {
                "core": ["회사", "주식", "이사", "주주", "상행위", "어음"],
                "important": ["주식회사", "유한회사", "합명회사", "합자회사", "상장", "비상장"],
                "supporting": ["정관", "이사회", "주주총회", "감사", "회계", "재무제표"]
            },
            "노동법": {
                "core": ["근로계약", "임금", "근로시간", "휴가", "해고", "퇴직금"],
                "important": ["근로기준법", "노동조합", "단체협약", "노동위원회", "임금체불"],
                "supporting": ["연장근로", "야간근로", "휴일근로", "연차유급휴가", "산전후휴가"]
            }
        }
        
        # 질문 유형별 키워드 패턴
        self.question_patterns = {
            "계약서_검토": ["계약서", "당사자", "조건", "기간", "효력", "무효", "취소", "해지"],
            "이혼_절차": ["이혼", "협의", "조정", "재판", "위자료", "재산분할", "양육비"],
            "상속_절차": ["상속", "상속인", "상속분", "유언", "유류분", "한정승인", "포기"],
            "소송_절차": ["소송", "소장", "답변서", "증거", "증인", "재판", "판결"],
            "범죄_처벌": ["범죄", "형량", "처벌", "수사", "기소", "재판", "변호인"]
        }
    
    def analyze_keyword_coverage(self, answer: str, query_type: str, question: str = "") -> Dict[str, Any]:
        """키워드 포함도 분석"""
        try:
            # 도메인 추출
            domain = self._extract_domain(query_type, question)
            
            # 관련 키워드 추출
            relevant_keywords = self._get_relevant_keywords(domain, query_type, question)
            
            # 포함도 계산
            coverage_results = self._calculate_coverage(answer, relevant_keywords)
            
            # 누락된 키워드 분석
            missing_keywords = self._find_missing_keywords(answer, relevant_keywords)
            
            # 개선 제안 생성
            improvements = self._generate_improvements(coverage_results, missing_keywords, domain)
            
            return {
                "domain": domain,
                "query_type": query_type,
                "question": question,
                "coverage_results": coverage_results,
                "missing_keywords": missing_keywords,
                "improvements": improvements,
                "target_coverage": self.target_coverage,
                "current_coverage": coverage_results.get("overall_coverage", 0.0),
                "needs_improvement": coverage_results.get("overall_coverage", 0.0) < self.target_coverage,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"키워드 포함도 분석 실패: {e}")
            return {"error": str(e)}
    
    def enhance_keyword_coverage(self, answer: str, query_type: str, question: str = "") -> Dict[str, Any]:
        """키워드 포함도 향상 제안"""
        try:
            # 현재 포함도 분석
            analysis = self.analyze_keyword_coverage(answer, query_type, question)
            
            if analysis.get("error"):
                return analysis
            
            current_coverage = analysis.get("current_coverage", 0.0)
            
            # 목표 달성 여부 확인
            if current_coverage >= self.target_coverage:
                return {
                    "status": "achieved",
                    "current_coverage": current_coverage,
                    "target_coverage": self.target_coverage,
                    "message": "목표 포함도를 달성했습니다.",
                    "improvements": []
                }
            
            # 개선 제안 생성
            improvements = analysis.get("improvements", [])
            missing_keywords = analysis.get("missing_keywords", {})
            
            # 우선순위별 개선 제안
            priority_improvements = self._categorize_improvements(improvements, missing_keywords)
            
            # 구체적인 행동 계획 생성
            action_plan = self._create_action_plan(priority_improvements, current_coverage)
            
            # 예상 개선 효과 계산
            potential_improvement = self._calculate_potential_improvement(current_coverage, priority_improvements)
            
            return {
                "status": "needs_improvement",
                "current_coverage": current_coverage,
                "target_coverage": self.target_coverage,
                "gap": self.target_coverage - current_coverage,
                "potential_improvement": potential_improvement,
                "priority_improvements": priority_improvements,
                "action_plan": action_plan,
                "recommended_keywords": self._get_recommended_keywords(missing_keywords),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"키워드 포함도 향상 제안 생성 실패: {e}")
            return {"error": str(e)}
    
    def _extract_domain(self, query_type: str, question: str) -> str:
        """도메인 추출"""
        # 질문 내용 기반 도메인 추출
        if "계약" in question or "contract" in query_type:
            return "민사법"
        elif "이혼" in question or "상속" in question or "family" in query_type:
            return "가족법"
        elif "범죄" in question or "형사" in question or "criminal" in query_type:
            return "형사법"
        elif "회사" in question or "주식" in question or "commercial" in query_type:
            return "상사법"
        elif "근로" in question or "노동" in question or "labor" in query_type:
            return "노동법"
        else:
            return "민사법"  # 기본값
    
    def _get_relevant_keywords(self, domain: str, query_type: str, question: str) -> Dict[str, List[str]]:
        """관련 키워드 추출"""
        keywords = {}
        
        # 도메인별 키워드
        if domain in self.domain_keywords:
            keywords.update(self.domain_keywords[domain])
        
        # 질문 유형별 키워드 추가
        for pattern, pattern_keywords in self.question_patterns.items():
            if pattern in query_type or any(keyword in question for keyword in pattern_keywords):
                # 기존 키워드에 추가
                for level in ["core", "important", "supporting"]:
                    if level not in keywords:
                        keywords[level] = []
                    keywords[level].extend(pattern_keywords)
        
        # 중복 제거
        for level in keywords:
            keywords[level] = list(set(keywords[level]))
        
        return keywords
    
    def _calculate_coverage(self, answer: str, keywords: Dict[str, List[str]]) -> Dict[str, Any]:
        """포함도 계산"""
        answer_lower = answer.lower()
        coverage_results = {}
        
        total_keywords = 0
        matched_keywords = 0
        
        for level, level_keywords in keywords.items():
            level_total = len(level_keywords)
            level_matched = sum(1 for keyword in level_keywords if keyword.lower() in answer_lower)
            
            coverage_results[f"{level}_total"] = level_total
            coverage_results[f"{level}_matched"] = level_matched
            coverage_results[f"{level}_coverage"] = level_matched / level_total if level_total > 0 else 0.0
            
            total_keywords += level_total
            matched_keywords += level_matched
        
        # 전체 포함도 계산 (가중 평균)
        weights = {"core": 1.0, "important": 0.8, "supporting": 0.6}
        weighted_coverage = 0.0
        total_weight = 0.0
        
        for level in weights:
            if level in keywords and keywords[level]:
                level_coverage = coverage_results.get(f"{level}_coverage", 0.0)
                weighted_coverage += level_coverage * weights[level]
                total_weight += weights[level]
        
        coverage_results["overall_coverage"] = weighted_coverage / total_weight if total_weight > 0 else 0.0
        coverage_results["total_keywords"] = total_keywords
        coverage_results["matched_keywords"] = matched_keywords
        
        return coverage_results
    
    def _find_missing_keywords(self, answer: str, keywords: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """누락된 키워드 찾기"""
        answer_lower = answer.lower()
        missing_keywords = {}
        
        for level, level_keywords in keywords.items():
            missing = [keyword for keyword in level_keywords if keyword.lower() not in answer_lower]
            missing_keywords[level] = missing
        
        return missing_keywords
    
    def _generate_improvements(self, coverage_results: Dict[str, Any], 
                             missing_keywords: Dict[str, List[str]], domain: str) -> List[Dict[str, Any]]:
        """개선 제안 생성"""
        improvements = []
        
        # 핵심 키워드 개선
        if missing_keywords.get("core"):
            improvements.append({
                "type": "core_keywords",
                "priority": "high",
                "missing_count": len(missing_keywords["core"]),
                "missing_keywords": missing_keywords["core"][:5],  # 상위 5개만
                "suggestion": f"핵심 키워드 {len(missing_keywords['core'])}개를 답변에 포함하세요",
                "impact": "높음",
                "expected_improvement": 0.2
            })
        
        # 중요 키워드 개선
        if missing_keywords.get("important"):
            improvements.append({
                "type": "important_keywords",
                "priority": "medium",
                "missing_count": len(missing_keywords["important"]),
                "missing_keywords": missing_keywords["important"][:5],
                "suggestion": f"중요 키워드 {len(missing_keywords['important'])}개를 답변에 포함하세요",
                "impact": "중간",
                "expected_improvement": 0.15
            })
        
        # 보조 키워드 개선
        if missing_keywords.get("supporting"):
            improvements.append({
                "type": "supporting_keywords",
                "priority": "low",
                "missing_count": len(missing_keywords["supporting"]),
                "missing_keywords": missing_keywords["supporting"][:5],
                "suggestion": f"보조 키워드 {len(missing_keywords['supporting'])}개를 답변에 포함하세요",
                "impact": "낮음",
                "expected_improvement": 0.1
            })
        
        # 답변 구조 개선
        overall_coverage = coverage_results.get("overall_coverage", 0.0)
        if overall_coverage < 0.5:
            improvements.append({
                "type": "structure_improvement",
                "priority": "high",
                "suggestion": "답변을 체계적으로 구조화하여 키워드 포함도를 높이세요",
                "specific_actions": [
                    "상황 정리 섹션에 핵심 키워드 포함",
                    "법적 분석 섹션에 중요 키워드 포함",
                    "실무 조언 섹션에 보조 키워드 포함"
                ],
                "impact": "높음",
                "expected_improvement": 0.25
            })
        
        return improvements
    
    def _categorize_improvements(self, improvements: List[Dict[str, Any]], 
                               missing_keywords: Dict[str, List[str]]) -> Dict[str, List[Dict[str, Any]]]:
        """개선 제안을 우선순위별로 분류"""
        categorized = {
            "high_priority": [],
            "medium_priority": [],
            "low_priority": []
        }
        
        for improvement in improvements:
            priority = improvement.get("priority", "low")
            categorized[f"{priority}_priority"].append(improvement)
        
        return categorized
    
    def _create_action_plan(self, priority_improvements: Dict[str, List[Dict[str, Any]]], 
                          current_coverage: float) -> List[str]:
        """구체적인 행동 계획 생성"""
        action_plan = []
        
        # 고우선순위 행동
        for improvement in priority_improvements.get("high_priority", []):
            if improvement["type"] == "core_keywords":
                action_plan.append(f"🔥 핵심 키워드 포함: {', '.join(improvement['missing_keywords'][:3])}")
            elif improvement["type"] == "structure_improvement":
                action_plan.append("🔥 답변 구조 개선: 체계적인 섹션별 구성")
        
        # 중우선순위 행동
        for improvement in priority_improvements.get("medium_priority", []):
            if improvement["type"] == "important_keywords":
                action_plan.append(f"⚡ 중요 키워드 포함: {', '.join(improvement['missing_keywords'][:3])}")
        
        # 저우선순위 행동
        for improvement in priority_improvements.get("low_priority", []):
            if improvement["type"] == "supporting_keywords":
                action_plan.append(f"💡 보조 키워드 포함: {', '.join(improvement['missing_keywords'][:3])}")
        
        return action_plan
    
    def _get_recommended_keywords(self, missing_keywords: Dict[str, List[str]]) -> List[str]:
        """권장 키워드 목록 생성"""
        recommended = []
        
        # 우선순위별로 키워드 추가
        for level in ["core", "important", "supporting"]:
            if level in missing_keywords:
                recommended.extend(missing_keywords[level][:3])  # 상위 3개만
        
        return recommended[:10]  # 최대 10개
    
    def _calculate_potential_improvement(self, current_coverage: float, 
                                       priority_improvements: Dict[str, List[Dict[str, Any]]]) -> float:
        """예상 개선 효과 계산"""
        potential = current_coverage
        
        # 각 개선 제안의 예상 효과 누적
        for improvements in priority_improvements.values():
            for improvement in improvements:
                expected = improvement.get("expected_improvement", 0.0)
                potential += expected
        
        return min(1.0, potential)  # 최대 1.0으로 제한
    
    def get_coverage_metrics(self) -> Dict[str, Any]:
        """포함도 메트릭 반환"""
        return {
            "target_coverage": self.target_coverage,
            "current_coverage": self.current_coverage,
            "gap": self.target_coverage - self.current_coverage,
            "improvement_needed": self.current_coverage < self.target_coverage,
            "metrics_description": {
                "target_coverage": "목표 포함도 (0.7)",
                "current_coverage": "현재 포함도",
                "gap": "목표와 현재의 차이",
                "improvement_needed": "개선 필요 여부"
            }
        }


# 전역 인스턴스
keyword_coverage_enhancer = KeywordCoverageEnhancer()
