# -*- coding: utf-8 -*-
"""
답변 품질 향상 통합 시스템
모든 품질 향상 기능을 통합하여 답변 품질을 종합적으로 개선
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# 개별 품질 향상 모듈들 import
from .unified_prompt_manager import UnifiedPromptManager, LegalDomain, QuestionType
from .confidence_calculator import ConfidenceCalculator
from .keyword_coverage_enhancer import KeywordCoverageEnhancer
from .answer_structure_enhancer import AnswerStructureEnhancer, QuestionType as StructureQuestionType
from .legal_term_validator import LegalTermValidator
from .context_quality_enhancer import ContextQualityEnhancer


class AnswerQualityEnhancer:
    """답변 품질 향상 통합 시스템"""
    
    def __init__(self):
        """초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 개별 품질 향상 모듈들 초기화
        self.prompt_manager = UnifiedPromptManager()
        self.confidence_calculator = ConfidenceCalculator()
        self.keyword_enhancer = KeywordCoverageEnhancer()
        self.structure_enhancer = AnswerStructureEnhancer()
        self.term_validator = LegalTermValidator()
        self.context_enhancer = ContextQualityEnhancer()
        
        # 품질 향상 설정
        self.quality_targets = {
            "keyword_coverage": 0.7,      # 키워드 포함도 목표
            "confidence_score": 0.8,      # 신뢰도 목표
            "structure_quality": 0.8,     # 구조화 품질 목표
            "term_accuracy": 0.8,         # 용어 정확성 목표
            "context_quality": 0.7        # 컨텍스트 품질 목표
        }
        
        self.enhancement_weights = {
            "prompt_optimization": 0.25,   # 프롬프트 최적화
            "keyword_coverage": 0.20,     # 키워드 포함도
            "structure_quality": 0.20,    # 구조화 품질
            "term_accuracy": 0.15,        # 용어 정확성
            "context_quality": 0.10,      # 컨텍스트 품질
            "confidence_calculation": 0.10 # 신뢰도 계산
        }
    
    def enhance_answer_quality(self, 
                             answer: str,
                             query: str,
                             question_type: str = "general",
                             domain: str = "general",
                             search_results: Optional[List[Dict[str, Any]]] = None,
                             sources: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """답변 품질 종합 향상"""
        try:
            self.logger.info(f"Starting answer quality enhancement for query: {query[:100]}...")
            
            # 1. 현재 답변 품질 분석
            quality_analysis = self._analyze_current_quality(
                answer, query, question_type, domain, search_results, sources
            )
            
            # 2. 개별 품질 향상 실행
            enhancement_results = self._run_individual_enhancements(
                answer, query, question_type, domain, search_results, sources
            )
            
            # 3. 통합 품질 향상 적용
            enhanced_answer = self._apply_integrated_enhancements(
                answer, query, question_type, domain, enhancement_results
            )
            
            # 4. 향상된 답변 품질 재평가
            final_quality_analysis = self._analyze_current_quality(
                enhanced_answer, query, question_type, domain, search_results, sources
            )
            
            # 5. 품질 향상 효과 계산
            improvement_metrics = self._calculate_improvement_metrics(
                quality_analysis, final_quality_analysis
            )
            
            # 6. 최종 결과 구성
            result = {
                "original_answer": answer,
                "enhanced_answer": enhanced_answer,
                "quality_analysis": quality_analysis,
                "enhancement_results": enhancement_results,
                "final_quality_analysis": final_quality_analysis,
                "improvement_metrics": improvement_metrics,
                "overall_improvement": improvement_metrics.get("overall_improvement", 0.0),
                "enhancement_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Answer quality enhancement completed. Overall improvement: {improvement_metrics.get('overall_improvement', 0.0):.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Answer quality enhancement failed: {e}")
            return {"error": str(e)}
    
    def _analyze_current_quality(self, answer: str, query: str, question_type: str, 
                               domain: str, search_results: Optional[List[Dict[str, Any]]], 
                               sources: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """현재 답변 품질 분석"""
        analysis = {
            "answer_length": len(answer),
            "keyword_coverage": 0.0,
            "confidence_score": 0.0,
            "structure_quality": 0.0,
            "term_accuracy": 0.0,
            "context_quality": 0.0,
            "overall_quality": 0.0
        }
        
        try:
            # 키워드 포함도 분석
            keyword_analysis = self.keyword_enhancer.analyze_keyword_coverage(answer, question_type, query)
            analysis["keyword_coverage"] = keyword_analysis.get("current_coverage", 0.0)
            
            # 신뢰도 계산
            if sources:
                confidence_info = self.confidence_calculator.calculate_enhanced_confidence(
                    answer, sources, question_type, domain
                )
                analysis["confidence_score"] = confidence_info.confidence
            
            # 구조화 품질 분석
            structure_analysis = self.structure_enhancer.enhance_answer_structure(answer, question_type, query, domain)
            analysis["structure_quality"] = structure_analysis.get("quality_metrics", {}).get("overall_score", 0.0)
            
            # 용어 정확성 분석
            term_analysis = self.term_validator.validate_legal_terms(answer, domain)
            analysis["term_accuracy"] = term_analysis.get("overall_accuracy", 0.0)
            
            # 컨텍스트 품질 분석
            if search_results:
                context_analysis = self.context_enhancer.enhance_context_quality(search_results, query, question_type, domain)
                analysis["context_quality"] = context_analysis.get("quality_metrics", {}).get("overall_quality", 0.0)
            
            # 전체 품질 점수 계산 (가중 평균)
            analysis["overall_quality"] = (
                analysis["keyword_coverage"] * self.enhancement_weights["keyword_coverage"] +
                analysis["confidence_score"] * self.enhancement_weights["confidence_calculation"] +
                analysis["structure_quality"] * self.enhancement_weights["structure_quality"] +
                analysis["term_accuracy"] * self.enhancement_weights["term_accuracy"] +
                analysis["context_quality"] * self.enhancement_weights["context_quality"]
            )
            
        except Exception as e:
            self.logger.error(f"Quality analysis failed: {e}")
        
        return analysis
    
    def _run_individual_enhancements(self, answer: str, query: str, question_type: str, 
                                   domain: str, search_results: Optional[List[Dict[str, Any]]], 
                                   sources: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """개별 품질 향상 실행"""
        enhancements = {}
        
        try:
            # 1. 프롬프트 최적화
            if hasattr(self.prompt_manager, 'enhance_answer_quality'):
                prompt_enhancement = self.prompt_manager.enhance_answer_quality(query, question_type, domain)
                enhancements["prompt_optimization"] = prompt_enhancement
            
            # 2. 키워드 포함도 향상
            keyword_enhancement = self.keyword_enhancer.enhance_keyword_coverage(answer, question_type, query)
            enhancements["keyword_coverage"] = keyword_enhancement
            
            # 3. 답변 구조화 향상
            structure_enhancement = self.structure_enhancer.enhance_answer_structure(answer, question_type, query, domain)
            enhancements["structure_quality"] = structure_enhancement
            
            # 4. 용어 정확성 향상
            term_enhancement = self.term_validator.enhance_term_accuracy(answer, domain)
            enhancements["term_accuracy"] = term_enhancement
            
            # 5. 컨텍스트 품질 향상
            if search_results:
                context_enhancement = self.context_enhancer.enhance_context_quality(search_results, query, question_type, domain)
                enhancements["context_quality"] = context_enhancement
            
        except Exception as e:
            self.logger.error(f"Individual enhancements failed: {e}")
        
        return enhancements
    
    def _apply_integrated_enhancements(self, answer: str, query: str, question_type: str, 
                                     domain: str, enhancement_results: Dict[str, Any]) -> str:
        """통합 품질 향상 적용"""
        enhanced_answer = answer
        
        try:
            # 1. 구조화 향상 적용
            if "structure_quality" in enhancement_results:
                structure_result = enhancement_results["structure_quality"]
                if "structured_answer" in structure_result:
                    enhanced_answer = structure_result["structured_answer"]
            
            # 2. 키워드 포함도 향상 적용
            if "keyword_coverage" in enhancement_results:
                keyword_result = enhancement_results["keyword_coverage"]
                if keyword_result.get("status") == "needs_improvement":
                    # 누락된 키워드 추가
                    missing_keywords = keyword_result.get("recommended_keywords", [])
                    if missing_keywords:
                        enhanced_answer = self._add_missing_keywords(enhanced_answer, missing_keywords)
            
            # 3. 용어 정확성 향상 적용
            if "term_accuracy" in enhancement_results:
                term_result = enhancement_results["term_accuracy"]
                if term_result.get("status") == "needs_improvement":
                    # 용어 정확성 개선
                    improvements = term_result.get("priority_improvements", {})
                    enhanced_answer = self._apply_term_improvements(enhanced_answer, improvements)
            
            # 4. 프롬프트 최적화 적용
            if "prompt_optimization" in enhancement_results:
                prompt_guidance = enhancement_results["prompt_optimization"]
                if prompt_guidance:
                    enhanced_answer = self._apply_prompt_guidance(enhanced_answer, prompt_guidance)
            
        except Exception as e:
            self.logger.error(f"Integrated enhancements application failed: {e}")
        
        return enhanced_answer
    
    def _add_missing_keywords(self, answer: str, missing_keywords: List[str]) -> str:
        """누락된 키워드 추가"""
        if not missing_keywords:
            return answer
        
        # 답변에 자연스럽게 키워드 추가
        enhanced_answer = answer
        
        for keyword in missing_keywords[:3]:  # 상위 3개만 추가
            if keyword not in enhanced_answer:
                # 키워드를 자연스럽게 추가할 위치 찾기
                if "결론적으로" in enhanced_answer:
                    enhanced_answer = enhanced_answer.replace("결론적으로", f"결론적으로, {keyword}와 관련하여")
                elif "따라서" in enhanced_answer:
                    enhanced_answer = enhanced_answer.replace("따라서", f"따라서, {keyword}를 고려할 때")
                else:
                    # 답변 끝에 추가
                    enhanced_answer += f"\n\n{keyword}에 대한 추가 고려사항도 중요합니다."
        
        return enhanced_answer
    
    def _apply_term_improvements(self, answer: str, improvements: Dict[str, List[Dict[str, Any]]]) -> str:
        """용어 정확성 개선 적용"""
        enhanced_answer = answer
        
        # 고우선순위 개선사항 적용
        for improvement in improvements.get("high_priority", []):
            if "term" in improvement:
                term = improvement["term"]
                suggestions = improvement.get("suggestions", [])
                if suggestions:
                    # 용어 정의 추가
                    enhanced_answer = self._add_term_definition(enhanced_answer, term, suggestions[0])
        
        return enhanced_answer
    
    def _add_term_definition(self, answer: str, term: str, suggestion: str) -> str:
        """용어 정의 추가"""
        if f"{term}의 정의" in answer or f"{term}는" in answer:
            return answer  # 이미 정의가 있는 경우
        
        # 용어 정의를 자연스럽게 추가
        if "결론적으로" in answer:
            enhanced_answer = answer.replace("결론적으로", f"결론적으로, {term}에 대해 {suggestion}")
        else:
            enhanced_answer = answer + f"\n\n{term}에 대해서는 {suggestion}"
        
        return enhanced_answer
    
    def _apply_prompt_guidance(self, answer: str, prompt_guidance: str) -> str:
        """프롬프트 가이드 적용"""
        # 프롬프트 가이드에 따라 답변 개선
        if "법적 근거" in prompt_guidance and "법령" not in answer:
            answer += "\n\n관련 법령을 참조하시기 바랍니다."
        
        if "판례 인용" in prompt_guidance and "판례" not in answer:
            answer += "\n\n관련 판례를 확인하시기 바랍니다."
        
        return answer
    
    def _calculate_improvement_metrics(self, original_analysis: Dict[str, Any], 
                                     final_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """품질 향상 효과 계산"""
        metrics = {}
        
        for key in original_analysis:
            if key in final_analysis:
                original_value = original_analysis[key]
                final_value = final_analysis[key]
                
                if original_value > 0:
                    improvement = ((final_value - original_value) / original_value) * 100
                else:
                    improvement = final_value * 100
                
                metrics[f"{key}_improvement"] = improvement
        
        # 전체 개선 효과 계산
        original_overall = original_analysis.get("overall_quality", 0.0)
        final_overall = final_analysis.get("overall_quality", 0.0)
        
        if original_overall > 0:
            overall_improvement = ((final_overall - original_overall) / original_overall) * 100
        else:
            overall_improvement = final_overall * 100
        
        metrics["overall_improvement"] = overall_improvement
        
        return metrics
    
    def get_quality_report(self, enhancement_result: Dict[str, Any]) -> Dict[str, Any]:
        """품질 향상 보고서 생성"""
        if "error" in enhancement_result:
            return {"error": enhancement_result["error"]}
        
        original_analysis = enhancement_result.get("quality_analysis", {})
        final_analysis = enhancement_result.get("final_quality_analysis", {})
        improvement_metrics = enhancement_result.get("improvement_metrics", {})
        
        report = {
            "summary": {
                "original_quality": original_analysis.get("overall_quality", 0.0),
                "final_quality": final_analysis.get("overall_quality", 0.0),
                "improvement": improvement_metrics.get("overall_improvement", 0.0)
            },
            "detailed_metrics": {
                "keyword_coverage": {
                    "original": original_analysis.get("keyword_coverage", 0.0),
                    "final": final_analysis.get("keyword_coverage", 0.0),
                    "improvement": improvement_metrics.get("keyword_coverage_improvement", 0.0)
                },
                "confidence_score": {
                    "original": original_analysis.get("confidence_score", 0.0),
                    "final": final_analysis.get("confidence_score", 0.0),
                    "improvement": improvement_metrics.get("confidence_score_improvement", 0.0)
                },
                "structure_quality": {
                    "original": original_analysis.get("structure_quality", 0.0),
                    "final": final_analysis.get("structure_quality", 0.0),
                    "improvement": improvement_metrics.get("structure_quality_improvement", 0.0)
                },
                "term_accuracy": {
                    "original": original_analysis.get("term_accuracy", 0.0),
                    "final": final_analysis.get("term_accuracy", 0.0),
                    "improvement": improvement_metrics.get("term_accuracy_improvement", 0.0)
                },
                "context_quality": {
                    "original": original_analysis.get("context_quality", 0.0),
                    "final": final_analysis.get("context_quality", 0.0),
                    "improvement": improvement_metrics.get("context_quality_improvement", 0.0)
                }
            },
            "recommendations": self._generate_quality_recommendations(final_analysis),
            "target_achievement": self._check_target_achievement(final_analysis),
            "report_timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def _generate_quality_recommendations(self, final_analysis: Dict[str, Any]) -> List[str]:
        """품질 권장사항 생성"""
        recommendations = []
        
        # 각 품질 지표별 권장사항
        if final_analysis.get("keyword_coverage", 0.0) < self.quality_targets["keyword_coverage"]:
            recommendations.append("키워드 포함도를 더 높이세요")
        
        if final_analysis.get("confidence_score", 0.0) < self.quality_targets["confidence_score"]:
            recommendations.append("신뢰도를 높이기 위해 더 많은 근거를 제시하세요")
        
        if final_analysis.get("structure_quality", 0.0) < self.quality_targets["structure_quality"]:
            recommendations.append("답변 구조를 더 체계적으로 개선하세요")
        
        if final_analysis.get("term_accuracy", 0.0) < self.quality_targets["term_accuracy"]:
            recommendations.append("법률 용어의 정확성을 더 높이세요")
        
        if final_analysis.get("context_quality", 0.0) < self.quality_targets["context_quality"]:
            recommendations.append("컨텍스트 품질을 개선하세요")
        
        return recommendations
    
    def _check_target_achievement(self, final_analysis: Dict[str, Any]) -> Dict[str, bool]:
        """목표 달성 여부 확인"""
        achievement = {}
        
        for metric, target in self.quality_targets.items():
            current_value = final_analysis.get(metric, 0.0)
            achievement[metric] = current_value >= target
        
        return achievement


# 전역 인스턴스
answer_quality_enhancer = AnswerQualityEnhancer()
