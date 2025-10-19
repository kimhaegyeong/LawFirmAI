# -*- coding: utf-8 -*-
"""
법률 용어 정확성 검증 시스템
답변 품질 향상을 위한 법률 용어 정확성 강화
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict, Counter


class LegalTermValidator:
    """법률 용어 정확성 검증 시스템"""
    
    def __init__(self):
        """초기화"""
        self.legal_terms_db = self._load_legal_terms_database()
        self.term_patterns = self._load_term_patterns()
        self.accuracy_rules = self._load_accuracy_rules()
        self.domain_terms = self._load_domain_terms()
    
    def _load_legal_terms_database(self) -> Dict[str, Dict[str, Any]]:
        """법률 용어 데이터베이스 로드"""
        return {
            # 민사법 용어
            "계약": {
                "definition": "당사자 상호간의 의사표시의 합치에 의하여 법률관계를 발생시키는 법률행위",
                "legal_basis": "민법 제105조",
                "related_terms": ["체결", "효력", "무효", "취소", "해지"],
                "domain": "민사법",
                "accuracy_level": "high"
            },
            "불법행위": {
                "definition": "고의 또는 과실로 인한 위법한 행위로 타인에게 손해를 가하는 행위",
                "legal_basis": "민법 제750조",
                "related_terms": ["손해배상", "과실", "고의", "위법성"],
                "domain": "민사법",
                "accuracy_level": "high"
            },
            "소유권": {
                "definition": "물건을 자유롭게 사용, 수익, 처분할 수 있는 물권",
                "legal_basis": "민법 제211조",
                "related_terms": ["물권", "점유", "등기", "이전"],
                "domain": "민사법",
                "accuracy_level": "high"
            },
            "채권": {
                "definition": "특정인이 특정인에 대하여 일정한 행위를 요구할 수 있는 권리",
                "legal_basis": "민법 제387조",
                "related_terms": ["채무", "이행", "담보", "보증"],
                "domain": "민사법",
                "accuracy_level": "high"
            },
            "손해배상": {
                "definition": "불법행위나 채무불이행으로 인한 손해를 배상하는 의무",
                "legal_basis": "민법 제750조, 제390조",
                "related_terms": ["손해", "배상", "과실", "고의"],
                "domain": "민사법",
                "accuracy_level": "high"
            },
            
            # 형사법 용어
            "범죄": {
                "definition": "형법에 의해 처벌받는 위법하고 유책한 행위",
                "legal_basis": "형법 제1조",
                "related_terms": ["구성요건", "위법성", "책임", "형량"],
                "domain": "형사법",
                "accuracy_level": "high"
            },
            "고의": {
                "definition": "범죄사실을 인식하고 용인하는 심리상태",
                "legal_basis": "형법 제13조",
                "related_terms": ["과실", "인식", "의욕", "목적"],
                "domain": "형사법",
                "accuracy_level": "high"
            },
            "과실": {
                "definition": "주의의무를 위반하여 범죄사실을 인식하지 못한 심리상태",
                "legal_basis": "형법 제14조",
                "related_terms": ["고의", "주의의무", "예견가능성", "회피의무"],
                "domain": "형사법",
                "accuracy_level": "high"
            },
            "정당방위": {
                "definition": "현재의 부당한 침해에 대하여 자신 또는 타인의 법익을 방위하기 위한 행위",
                "legal_basis": "형법 제21조",
                "related_terms": ["긴급피난", "방위", "침해", "법익"],
                "domain": "형사법",
                "accuracy_level": "high"
            },
            "미수범": {
                "definition": "범죄의 실행에 착수하였으나 기수에 이르지 못한 경우",
                "legal_basis": "형법 제25조",
                "related_terms": ["기수범", "실행의 착수", "미완성", "형의 감경"],
                "domain": "형사법",
                "accuracy_level": "high"
            },
            
            # 가족법 용어
            "이혼": {
                "definition": "혼인관계를 해소하는 법률행위",
                "legal_basis": "민법 제840조",
                "related_terms": ["협의이혼", "조정이혼", "재판이혼", "위자료"],
                "domain": "가족법",
                "accuracy_level": "high"
            },
            "상속": {
                "definition": "사망한 사람의 재산상의 권리와 의무를 특정인이 승계하는 것",
                "legal_basis": "민법 제997조",
                "related_terms": ["상속인", "상속분", "유언", "유류분"],
                "domain": "가족법",
                "accuracy_level": "high"
            },
            "양육권": {
                "definition": "미성년 자녀를 양육할 권리",
                "legal_basis": "민법 제837조",
                "related_terms": ["면접교섭권", "양육비", "친권", "후견"],
                "domain": "가족법",
                "accuracy_level": "high"
            },
            "위자료": {
                "definition": "이혼 시 정신적 피해에 대한 금전적 보상",
                "legal_basis": "민법 제843조",
                "related_terms": ["재산분할", "이혼", "정신적 피해", "보상"],
                "domain": "가족법",
                "accuracy_level": "high"
            },
            "재산분할": {
                "definition": "이혼 시 부부 공동재산을 분할하는 것",
                "legal_basis": "민법 제839조",
                "related_terms": ["위자료", "이혼", "공동재산", "분할"],
                "domain": "가족법",
                "accuracy_level": "high"
            },
            
            # 상사법 용어
            "주식회사": {
                "definition": "자본을 주식으로 분할하여 주주가 출자한 금액을 한도로 책임을 지는 회사",
                "legal_basis": "상법 제169조",
                "related_terms": ["주식", "주주", "이사", "이사회"],
                "domain": "상사법",
                "accuracy_level": "high"
            },
            "주식": {
                "definition": "주식회사의 자본을 구성하는 단위",
                "legal_basis": "상법 제334조",
                "related_terms": ["주주", "주주권", "배당", "자본"],
                "domain": "상사법",
                "accuracy_level": "high"
            },
            "이사": {
                "definition": "주식회사의 업무집행기관",
                "legal_basis": "상법 제382조",
                "related_terms": ["이사회", "대표이사", "업무집행", "책임"],
                "domain": "상사법",
                "accuracy_level": "high"
            },
            "상행위": {
                "definition": "상인이 영업으로 행하는 행위",
                "legal_basis": "상법 제46조",
                "related_terms": ["상인", "영업", "상법", "특칙"],
                "domain": "상사법",
                "accuracy_level": "high"
            },
            "어음": {
                "definition": "일정한 금액의 지급을 목적으로 발행하는 유가증권",
                "legal_basis": "어음법 제1조",
                "related_terms": ["수표", "어음법", "지급", "유가증권"],
                "domain": "상사법",
                "accuracy_level": "high"
            },
            
            # 노동법 용어
            "근로계약": {
                "definition": "근로자가 사용자에게 근로를 제공하고 사용자가 이에 대하여 임금을 지급하는 계약",
                "legal_basis": "근로기준법 제15조",
                "related_terms": ["근로자", "사용자", "임금", "근로시간"],
                "domain": "노동법",
                "accuracy_level": "high"
            },
            "임금": {
                "definition": "근로의 대가로 사용자가 근로자에게 임금, 봉급, 그 밖에 어떠한 명칭으로든지 지급하는 일체의 금품",
                "legal_basis": "근로기준법 제2조",
                "related_terms": ["근로계약", "근로자", "지급", "체불"],
                "domain": "노동법",
                "accuracy_level": "high"
            },
            "근로시간": {
                "definition": "근로자가 사용자의 지휘·감독 아래 근로에 종사하는 시간",
                "legal_basis": "근로기준법 제50조",
                "related_terms": ["휴게시간", "연장근로", "야간근로", "휴일근로"],
                "domain": "노동법",
                "accuracy_level": "high"
            },
            "해고": {
                "definition": "사용자가 근로계약을 일방적으로 해지하는 것",
                "legal_basis": "근로기준법 제23조",
                "related_terms": ["해지", "퇴직", "정당한 사유", "제한"],
                "domain": "노동법",
                "accuracy_level": "high"
            },
            "노동조합": {
                "definition": "근로자가 주체가 되어 자주적으로 단결하여 근로조건의 유지·개선 기타 근로자의 경제적·사회적 지위의 향상을 도모함을 목적으로 조직하는 단체",
                "legal_basis": "노동조합법 제2조",
                "related_terms": ["단체교섭", "단체협약", "파업", "쟁의행위"],
                "domain": "노동법",
                "accuracy_level": "high"
            }
        }
    
    def _load_term_patterns(self) -> Dict[str, List[str]]:
        """용어 패턴 로드"""
        return {
            "legal_article_patterns": [
                r'제\s*\d+\s*조',
                r'제\s*\d+\s*조\s*제\s*\d+\s*항',
                r'제\s*\d+\s*조\s*제\s*\d+\s*항\s*제\s*\d+\s*호',
                r'민법\s*제\s*\d+\s*조',
                r'형법\s*제\s*\d+\s*조',
                r'상법\s*제\s*\d+\s*조',
                r'근로기준법\s*제\s*\d+\s*조'
            ],
            "court_case_patterns": [
                r'대법원\s*\d{4}\.\s*\d{1,2}\.\s*\d{1,2}',
                r'대법원\s*\d{4}다\d+',
                r'서울고법\s*\d{4}나\d+',
                r'서울중앙지법\s*\d{4}고합\d+',
                r'판례\s*\d{4}다\d+'
            ],
            "legal_procedure_patterns": [
                r'소송',
                r'고소',
                r'고발',
                r'조정',
                r'중재',
                r'화해',
                r'집행'
            ]
        }
    
    def _load_accuracy_rules(self) -> Dict[str, Any]:
        """정확성 규칙 로드"""
        return {
            "mandatory_elements": {
                "legal_basis": "법적 근거 제시 필수",
                "definition": "정확한 정의 포함",
                "related_terms": "관련 용어 언급"
            },
            "accuracy_thresholds": {
                "high": 0.9,    # 90% 이상 정확성
                "medium": 0.7,  # 70% 이상 정확성
                "low": 0.5      # 50% 이상 정확성
            },
            "validation_criteria": {
                "term_usage": "용어 사용의 정확성",
                "legal_basis": "법적 근거의 정확성",
                "context_appropriateness": "맥락의 적절성",
                "completeness": "완성도"
            }
        }
    
    def _load_domain_terms(self) -> Dict[str, List[str]]:
        """도메인별 용어 목록 로드"""
        domain_terms = defaultdict(list)
        
        for term, info in self.legal_terms_db.items():
            domain = info.get("domain", "기타")
            domain_terms[domain].append(term)
        
        return dict(domain_terms)
    
    def validate_legal_terms(self, answer: str, domain: str = "general") -> Dict[str, Any]:
        """법률 용어 정확성 검증"""
        try:
            # 답변에서 법률 용어 추출
            extracted_terms = self._extract_legal_terms(answer)
            
            # 용어별 정확성 검증
            validation_results = {}
            for term in extracted_terms:
                validation_results[term] = self._validate_single_term(term, answer, domain)
            
            # 전체 정확성 점수 계산
            overall_accuracy = self._calculate_overall_accuracy(validation_results)
            
            # 개선 제안 생성
            improvements = self._generate_term_improvements(validation_results, domain)
            
            # 품질 메트릭 계산
            quality_metrics = self._calculate_term_quality_metrics(answer, validation_results)
            
            return {
                "extracted_terms": extracted_terms,
                "validation_results": validation_results,
                "overall_accuracy": overall_accuracy,
                "improvements": improvements,
                "quality_metrics": quality_metrics,
                "domain": domain,
                "validation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"법률 용어 검증 실패: {e}")
            return {"error": str(e)}
    
    def _extract_legal_terms(self, answer: str) -> List[str]:
        """답변에서 법률 용어 추출"""
        extracted_terms = []
        
        # 데이터베이스에 있는 용어들 검색
        for term in self.legal_terms_db.keys():
            if term in answer:
                extracted_terms.append(term)
        
        # 관련 용어도 검색
        for term_info in self.legal_terms_db.values():
            related_terms = term_info.get("related_terms", [])
            for related_term in related_terms:
                if related_term in answer and related_term not in extracted_terms:
                    extracted_terms.append(related_term)
        
        return extracted_terms
    
    def _validate_single_term(self, term: str, answer: str, domain: str) -> Dict[str, Any]:
        """개별 용어 정확성 검증"""
        validation_result = {
            "term": term,
            "is_accurate": False,
            "accuracy_score": 0.0,
            "issues": [],
            "suggestions": [],
            "legal_basis_provided": False,
            "definition_provided": False,
            "context_appropriate": False
        }
        
        # 용어가 데이터베이스에 있는지 확인
        if term in self.legal_terms_db:
            term_info = self.legal_terms_db[term]
            
            # 법적 근거 제공 여부 확인
            legal_basis = term_info.get("legal_basis", "")
            if legal_basis and legal_basis in answer:
                validation_result["legal_basis_provided"] = True
            
            # 정의 제공 여부 확인
            definition = term_info.get("definition", "")
            if definition and any(word in answer for word in definition.split()[:3]):
                validation_result["definition_provided"] = True
            
            # 맥락 적절성 확인
            validation_result["context_appropriate"] = self._check_context_appropriateness(term, answer, domain)
            
            # 정확성 점수 계산
            accuracy_score = 0.0
            if validation_result["legal_basis_provided"]:
                accuracy_score += 0.4
            if validation_result["definition_provided"]:
                accuracy_score += 0.3
            if validation_result["context_appropriate"]:
                accuracy_score += 0.3
            
            validation_result["accuracy_score"] = accuracy_score
            validation_result["is_accurate"] = accuracy_score >= 0.7
            
            # 개선 제안 생성
            if not validation_result["legal_basis_provided"]:
                validation_result["suggestions"].append(f"'{term}'에 대한 법적 근거를 추가하세요: {legal_basis}")
            
            if not validation_result["definition_provided"]:
                validation_result["suggestions"].append(f"'{term}'의 정의를 명확히 하세요")
            
            if not validation_result["context_appropriate"]:
                validation_result["suggestions"].append(f"'{term}'의 사용 맥락을 개선하세요")
        
        return validation_result
    
    def _check_context_appropriateness(self, term: str, answer: str, domain: str) -> bool:
        """맥락 적절성 확인"""
        if term in self.legal_terms_db:
            term_info = self.legal_terms_db[term]
            term_domain = term_info.get("domain", "기타")
            
            # 도메인 일치 여부 확인
            if domain != "general" and term_domain != domain:
                return False
            
            # 관련 용어와 함께 사용되는지 확인
            related_terms = term_info.get("related_terms", [])
            if related_terms:
                context_score = sum(1 for related_term in related_terms if related_term in answer)
                return context_score >= len(related_terms) * 0.3  # 30% 이상 관련 용어 포함
        
        return True
    
    def _calculate_overall_accuracy(self, validation_results: Dict[str, Dict[str, Any]]) -> float:
        """전체 정확성 점수 계산"""
        if not validation_results:
            return 0.0
        
        total_score = sum(result["accuracy_score"] for result in validation_results.values())
        return total_score / len(validation_results)
    
    def _generate_term_improvements(self, validation_results: Dict[str, Dict[str, Any]], 
                                  domain: str) -> List[Dict[str, Any]]:
        """용어 개선 제안 생성"""
        improvements = []
        
        for term, result in validation_results.items():
            if not result["is_accurate"]:
                improvement = {
                    "term": term,
                    "priority": "high" if result["accuracy_score"] < 0.5 else "medium",
                    "current_accuracy": result["accuracy_score"],
                    "target_accuracy": 0.8,
                    "suggestions": result["suggestions"],
                    "impact": "높음" if result["accuracy_score"] < 0.5 else "중간"
                }
                improvements.append(improvement)
        
        # 도메인별 특화 개선 제안
        if domain in self.domain_terms:
            domain_specific_terms = self.domain_terms[domain]
            missing_terms = [term for term in domain_specific_terms if term not in validation_results]
            
            if missing_terms:
                improvements.append({
                    "type": "missing_domain_terms",
                    "priority": "medium",
                    "suggestion": f"{domain} 분야의 핵심 용어를 추가하세요",
                    "missing_terms": missing_terms[:5],  # 상위 5개만
                    "impact": "중간"
                })
        
        return improvements
    
    def _calculate_term_quality_metrics(self, answer: str, 
                                      validation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """용어 품질 메트릭 계산"""
        metrics = {
            "total_terms": len(validation_results),
            "accurate_terms": len([r for r in validation_results.values() if r["is_accurate"]]),
            "accuracy_rate": 0.0,
            "legal_basis_coverage": 0.0,
            "definition_coverage": 0.0,
            "context_appropriateness": 0.0,
            "overall_quality": 0.0
        }
        
        if validation_results:
            metrics["accuracy_rate"] = metrics["accurate_terms"] / metrics["total_terms"]
            metrics["legal_basis_coverage"] = len([r for r in validation_results.values() if r["legal_basis_provided"]]) / metrics["total_terms"]
            metrics["definition_coverage"] = len([r for r in validation_results.values() if r["definition_provided"]]) / metrics["total_terms"]
            metrics["context_appropriateness"] = len([r for r in validation_results.values() if r["context_appropriate"]]) / metrics["total_terms"]
            
            # 전체 품질 점수 (가중 평균)
            metrics["overall_quality"] = (
                metrics["accuracy_rate"] * 0.4 +
                metrics["legal_basis_coverage"] * 0.3 +
                metrics["definition_coverage"] * 0.2 +
                metrics["context_appropriateness"] * 0.1
            )
        
        return metrics
    
    def enhance_term_accuracy(self, answer: str, domain: str = "general") -> Dict[str, Any]:
        """용어 정확성 향상 제안"""
        try:
            # 현재 용어 정확성 검증
            validation_result = self.validate_legal_terms(answer, domain)
            
            if validation_result.get("error"):
                return validation_result
            
            overall_accuracy = validation_result.get("overall_accuracy", 0.0)
            
            # 목표 정확성 달성 여부 확인
            target_accuracy = 0.8
            if overall_accuracy >= target_accuracy:
                return {
                    "status": "achieved",
                    "current_accuracy": overall_accuracy,
                    "target_accuracy": target_accuracy,
                    "message": "목표 정확성을 달성했습니다.",
                    "improvements": []
                }
            
            # 개선 제안 생성
            improvements = validation_result.get("improvements", [])
            
            # 우선순위별 개선 제안 분류
            priority_improvements = {
                "high_priority": [imp for imp in improvements if imp.get("priority") == "high"],
                "medium_priority": [imp for imp in improvements if imp.get("priority") == "medium"],
                "low_priority": [imp for imp in improvements if imp.get("priority") == "low"]
            }
            
            # 구체적인 행동 계획 생성
            action_plan = self._create_term_accuracy_action_plan(priority_improvements, overall_accuracy)
            
            # 예상 개선 효과 계산
            potential_improvement = self._calculate_potential_term_improvement(overall_accuracy, improvements)
            
            return {
                "status": "needs_improvement",
                "current_accuracy": overall_accuracy,
                "target_accuracy": target_accuracy,
                "gap": target_accuracy - overall_accuracy,
                "potential_improvement": potential_improvement,
                "priority_improvements": priority_improvements,
                "action_plan": action_plan,
                "quality_metrics": validation_result.get("quality_metrics", {}),
                "enhancement_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"용어 정확성 향상 제안 생성 실패: {e}")
            return {"error": str(e)}
    
    def _create_term_accuracy_action_plan(self, priority_improvements: Dict[str, List[Dict[str, Any]]], 
                                        current_accuracy: float) -> List[str]:
        """용어 정확성 향상 행동 계획 생성"""
        action_plan = []
        
        # 고우선순위 행동
        for improvement in priority_improvements.get("high_priority", []):
            if "term" in improvement:
                term = improvement["term"]
                suggestions = improvement.get("suggestions", [])
                action_plan.append(f"🔥 '{term}' 용어 정확성 개선: {suggestions[0] if suggestions else '정확한 정의와 법적 근거 추가'}")
        
        # 중우선순위 행동
        for improvement in priority_improvements.get("medium_priority", []):
            if "term" in improvement:
                term = improvement["term"]
                action_plan.append(f"⚡ '{term}' 용어 사용 맥락 개선")
            elif improvement.get("type") == "missing_domain_terms":
                missing_terms = improvement.get("missing_terms", [])
                action_plan.append(f"⚡ 도메인 특화 용어 추가: {', '.join(missing_terms[:3])}")
        
        # 저우선순위 행동
        for improvement in priority_improvements.get("low_priority", []):
            if "term" in improvement:
                term = improvement["term"]
                action_plan.append(f"💡 '{term}' 용어 관련성 강화")
        
        return action_plan
    
    def _calculate_potential_term_improvement(self, current_accuracy: float, 
                                            improvements: List[Dict[str, Any]]) -> float:
        """예상 용어 정확성 개선 효과 계산"""
        potential = current_accuracy
        
        # 각 개선 제안의 예상 효과 누적
        for improvement in improvements:
            if improvement.get("priority") == "high":
                potential += 0.2  # 20% 개선
            elif improvement.get("priority") == "medium":
                potential += 0.1  # 10% 개선
            else:
                potential += 0.05  # 5% 개선
        
        return min(1.0, potential)  # 최대 1.0으로 제한
    
    def get_term_suggestions(self, domain: str, limit: int = 10) -> List[str]:
        """도메인별 용어 제안"""
        if domain in self.domain_terms:
            return self.domain_terms[domain][:limit]
        else:
            # 모든 도메인에서 상위 용어 반환
            all_terms = []
            for terms in self.domain_terms.values():
                all_terms.extend(terms)
            return all_terms[:limit]


# 전역 인스턴스
legal_term_validator = LegalTermValidator()
