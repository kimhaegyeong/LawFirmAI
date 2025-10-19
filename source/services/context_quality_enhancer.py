# -*- coding: utf-8 -*-
"""
컨텍스트 품질 향상 시스템
검색 결과 품질 개선을 통한 답변 품질 향상
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict, Counter
import math


class ContextQualityEnhancer:
    """컨텍스트 품질 향상 시스템"""
    
    def __init__(self):
        """초기화"""
        self.quality_criteria = self._load_quality_criteria()
        self.relevance_weights = self._load_relevance_weights()
        self.context_filters = self._load_context_filters()
        self.optimization_rules = self._load_optimization_rules()
    
    def _load_quality_criteria(self) -> Dict[str, Dict[str, Any]]:
        """품질 기준 로드"""
        return {
            "relevance": {
                "weight": 0.3,
                "description": "질문과의 관련성",
                "criteria": ["키워드 매칭", "의미적 유사성", "도메인 일치"]
            },
            "completeness": {
                "weight": 0.25,
                "description": "정보의 완성도",
                "criteria": ["필수 정보 포함", "상세한 설명", "구체적 사례"]
            },
            "accuracy": {
                "weight": 0.25,
                "description": "정보의 정확성",
                "criteria": ["법적 정확성", "최신성", "신뢰할 수 있는 출처"]
            },
            "usability": {
                "weight": 0.2,
                "description": "실용성",
                "criteria": ["실행 가능성", "구체적 조언", "단계별 안내"]
            }
        }
    
    def _load_relevance_weights(self) -> Dict[str, float]:
        """관련성 가중치 로드"""
        return {
            "exact_match": 1.0,      # 정확한 매칭
            "partial_match": 0.8,     # 부분 매칭
            "semantic_match": 0.6,    # 의미적 매칭
            "domain_match": 0.4,      # 도메인 매칭
            "context_match": 0.2      # 컨텍스트 매칭
        }
    
    def _load_context_filters(self) -> Dict[str, List[str]]:
        """컨텍스트 필터 로드"""
        return {
            "high_quality_indicators": [
                "대법원", "판례", "법령", "조문", "조항",
                "구체적", "실행", "단계별", "절차", "방법"
            ],
            "low_quality_indicators": [
                "불확실", "추정", "가능성", "아마도", "일반적으로",
                "단순", "기본", "일반", "보통", "평균"
            ],
            "legal_authority_indicators": [
                "민법", "형법", "상법", "근로기준법", "노동조합법",
                "대법원", "헌법재판소", "법원", "검찰", "법무부"
            ]
        }
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """최적화 규칙 로드"""
        return {
            "max_context_length": 4000,      # 최대 컨텍스트 길이
            "min_context_length": 500,       # 최소 컨텍스트 길이
            "max_sources": 10,               # 최대 소스 수
            "min_sources": 3,                # 최소 소스 수
            "relevance_threshold": 0.6,      # 관련성 임계값
            "quality_threshold": 0.7,        # 품질 임계값
            "diversity_weight": 0.3,         # 다양성 가중치
            "recency_weight": 0.2            # 최신성 가중치
        }
    
    def enhance_context_quality(self, search_results: List[Dict[str, Any]], 
                              query: str, question_type: str = "general", 
                              domain: str = "general") -> Dict[str, Any]:
        """컨텍스트 품질 향상"""
        try:
            # 검색 결과 품질 분석
            quality_analysis = self._analyze_search_results_quality(search_results, query, question_type, domain)
            
            # 품질 기반 필터링
            filtered_results = self._filter_by_quality(search_results, quality_analysis)
            
            # 관련성 기반 재순위화
            reranked_results = self._rerank_by_relevance(filtered_results, query, question_type, domain)
            
            # 컨텍스트 최적화
            optimized_context = self._optimize_context(reranked_results, query, question_type, domain)
            
            # 품질 메트릭 계산
            quality_metrics = self._calculate_context_quality_metrics(optimized_context, quality_analysis)
            
            # 개선 제안 생성
            improvements = self._generate_context_improvements(quality_analysis, optimized_context)
            
            return {
                "original_results": search_results,
                "filtered_results": filtered_results,
                "reranked_results": reranked_results,
                "optimized_context": optimized_context,
                "quality_analysis": quality_analysis,
                "quality_metrics": quality_metrics,
                "improvements": improvements,
                "enhancement_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"컨텍스트 품질 향상 실패: {e}")
            return {"error": str(e)}
    
    def _analyze_search_results_quality(self, search_results: List[Dict[str, Any]], 
                                      query: str, question_type: str, domain: str) -> Dict[str, Any]:
        """검색 결과 품질 분석"""
        analysis = {
            "total_results": len(search_results),
            "quality_scores": [],
            "relevance_scores": [],
            "completeness_scores": [],
            "accuracy_scores": [],
            "usability_scores": [],
            "overall_quality": 0.0,
            "issues": [],
            "recommendations": []
        }
        
        for i, result in enumerate(search_results):
            # 개별 결과 품질 분석
            result_analysis = self._analyze_single_result(result, query, question_type, domain)
            
            analysis["quality_scores"].append(result_analysis["quality_score"])
            analysis["relevance_scores"].append(result_analysis["relevance_score"])
            analysis["completeness_scores"].append(result_analysis["completeness_score"])
            analysis["accuracy_scores"].append(result_analysis["accuracy_score"])
            analysis["usability_scores"].append(result_analysis["usability_score"])
        
        # 전체 품질 점수 계산
        if analysis["quality_scores"]:
            analysis["overall_quality"] = sum(analysis["quality_scores"]) / len(analysis["quality_scores"])
        
        # 품질 이슈 식별
        analysis["issues"] = self._identify_quality_issues(analysis)
        
        # 개선 권장사항 생성
        analysis["recommendations"] = self._generate_quality_recommendations(analysis)
        
        return analysis
    
    def _analyze_single_result(self, result: Dict[str, Any], query: str, 
                             question_type: str, domain: str) -> Dict[str, Any]:
        """개별 검색 결과 분석"""
        content = result.get("content", "")
        title = result.get("title", "")
        source = result.get("source", "")
        
        # 관련성 점수 계산
        relevance_score = self._calculate_relevance_score(content, title, query, question_type, domain)
        
        # 완성도 점수 계산
        completeness_score = self._calculate_completeness_score(content, question_type)
        
        # 정확성 점수 계산
        accuracy_score = self._calculate_accuracy_score(content, source, domain)
        
        # 실용성 점수 계산
        usability_score = self._calculate_usability_score(content, question_type)
        
        # 전체 품질 점수 계산 (가중 평균)
        quality_score = (
            relevance_score * self.quality_criteria["relevance"]["weight"] +
            completeness_score * self.quality_criteria["completeness"]["weight"] +
            accuracy_score * self.quality_criteria["accuracy"]["weight"] +
            usability_score * self.quality_criteria["usability"]["weight"]
        )
        
        return {
            "quality_score": quality_score,
            "relevance_score": relevance_score,
            "completeness_score": completeness_score,
            "accuracy_score": accuracy_score,
            "usability_score": usability_score,
            "content_length": len(content),
            "has_legal_authority": self._check_legal_authority(content, source),
            "is_recent": self._check_recency(result),
            "is_detailed": self._check_detail_level(content)
        }
    
    def _calculate_relevance_score(self, content: str, title: str, query: str, 
                                 question_type: str, domain: str) -> float:
        """관련성 점수 계산"""
        score = 0.0
        
        # 키워드 매칭 점수
        query_keywords = query.lower().split()
        content_lower = content.lower()
        title_lower = title.lower()
        
        # 정확한 매칭
        exact_matches = sum(1 for keyword in query_keywords if keyword in content_lower)
        if exact_matches > 0:
            score += (exact_matches / len(query_keywords)) * self.relevance_weights["exact_match"]
        
        # 제목 매칭 (가중치 높음)
        title_matches = sum(1 for keyword in query_keywords if keyword in title_lower)
        if title_matches > 0:
            score += (title_matches / len(query_keywords)) * self.relevance_weights["exact_match"] * 1.2
        
        # 의미적 매칭 (간단한 유사어 매칭)
        semantic_matches = self._calculate_semantic_matches(content, query)
        score += semantic_matches * self.relevance_weights["semantic_match"]
        
        # 도메인 매칭
        if domain != "general":
            domain_matches = self._calculate_domain_matches(content, domain)
            score += domain_matches * self.relevance_weights["domain_match"]
        
        return min(1.0, score)
    
    def _calculate_semantic_matches(self, content: str, query: str) -> float:
        """의미적 매칭 점수 계산"""
        # 간단한 유사어 매핑
        synonyms = {
            "계약": ["계약서", "계약관계", "계약체결"],
            "이혼": ["이혼절차", "이혼소송", "이혼조정"],
            "상속": ["상속절차", "상속분", "상속인"],
            "소송": ["소송절차", "소송제기", "소송진행"],
            "범죄": ["범죄행위", "범죄사실", "범죄구성요건"]
        }
        
        query_lower = query.lower()
        content_lower = content.lower()
        
        semantic_score = 0.0
        for main_term, synonyms_list in synonyms.items():
            if main_term in query_lower:
                matches = sum(1 for synonym in synonyms_list if synonym in content_lower)
                semantic_score += matches / len(synonyms_list)
        
        return min(1.0, semantic_score)
    
    def _calculate_domain_matches(self, content: str, domain: str) -> float:
        """도메인 매칭 점수 계산"""
        domain_keywords = {
            "민사법": ["계약", "불법행위", "소유권", "채권", "채무", "손해배상"],
            "형사법": ["범죄", "형량", "처벌", "고의", "과실", "정당방위"],
            "가족법": ["이혼", "상속", "양육권", "위자료", "재산분할"],
            "상사법": ["회사", "주식", "이사", "주주", "상행위"],
            "노동법": ["근로계약", "임금", "근로시간", "해고", "노동조합"]
        }
        
        if domain not in domain_keywords:
            return 0.0
        
        content_lower = content.lower()
        domain_terms = domain_keywords[domain]
        matches = sum(1 for term in domain_terms if term in content_lower)
        
        return matches / len(domain_terms)
    
    def _calculate_completeness_score(self, content: str, question_type: str) -> float:
        """완성도 점수 계산"""
        score = 0.0
        
        # 길이 기반 점수 (너무 짧거나 너무 길면 감점)
        length = len(content)
        if 200 <= length <= 1000:
            score += 0.3
        elif 100 <= length < 200 or 1000 < length <= 2000:
            score += 0.2
        else:
            score += 0.1
        
        # 질문 유형별 완성도 요구사항
        completeness_requirements = {
            "precedent_search": ["판례", "사건번호", "판결요지", "법원"],
            "law_inquiry": ["법령", "조문", "해설", "적용"],
            "legal_advice": ["상황", "분석", "조언", "방안"],
            "procedure_guide": ["절차", "단계", "서류", "기간"],
            "term_explanation": ["정의", "근거", "사례", "관련"]
        }
        
        if question_type in completeness_requirements:
            requirements = completeness_requirements[question_type]
            content_lower = content.lower()
            matches = sum(1 for req in requirements if req in content_lower)
            score += (matches / len(requirements)) * 0.4
        
        # 구체적 설명 포함 여부
        detail_indicators = ["구체적", "상세", "단계별", "실행", "방법", "절차"]
        content_lower = content.lower()
        detail_matches = sum(1 for indicator in detail_indicators if indicator in content_lower)
        score += (detail_matches / len(detail_indicators)) * 0.3
        
        return min(1.0, score)
    
    def _calculate_accuracy_score(self, content: str, source: str, domain: str) -> float:
        """정확성 점수 계산"""
        score = 0.0
        
        # 법적 권위성 확인
        if self._check_legal_authority(content, source):
            score += 0.4
        
        # 법적 용어 사용 정확성
        legal_terms = ["법령", "조문", "조항", "판례", "법원", "대법원"]
        content_lower = content.lower()
        legal_term_matches = sum(1 for term in legal_terms if term in content_lower)
        score += (legal_term_matches / len(legal_terms)) * 0.3
        
        # 구체적 법적 근거 제시
        legal_basis_patterns = [
            r'제\s*\d+\s*조',
            r'민법\s*제\s*\d+\s*조',
            r'형법\s*제\s*\d+\s*조',
            r'상법\s*제\s*\d+\s*조'
        ]
        basis_matches = sum(1 for pattern in legal_basis_patterns if re.search(pattern, content))
        score += (basis_matches / len(legal_basis_patterns)) * 0.3
        
        return min(1.0, score)
    
    def _calculate_usability_score(self, content: str, question_type: str) -> float:
        """실용성 점수 계산"""
        score = 0.0
        
        # 실행 가능한 조언 포함 여부
        practical_indicators = ["구체적", "실행", "단계별", "방법", "절차", "조치", "권장"]
        content_lower = content.lower()
        practical_matches = sum(1 for indicator in practical_indicators if indicator in content_lower)
        score += (practical_matches / len(practical_indicators)) * 0.4
        
        # 예시나 사례 포함 여부
        example_indicators = ["예를 들어", "예시", "사례", "실제", "구체적"]
        example_matches = sum(1 for indicator in example_indicators if indicator in content_lower)
        score += (example_matches / len(example_indicators)) * 0.3
        
        # 주의사항이나 제한사항 언급
        caution_indicators = ["주의", "주의사항", "제한", "한계", "고려"]
        caution_matches = sum(1 for indicator in caution_indicators if indicator in content_lower)
        score += (caution_matches / len(caution_indicators)) * 0.3
        
        return min(1.0, score)
    
    def _check_legal_authority(self, content: str, source: str) -> bool:
        """법적 권위성 확인"""
        authority_indicators = self.context_filters["legal_authority_indicators"]
        content_lower = content.lower()
        source_lower = source.lower()
        
        # 내용에서 법적 권위 지표 확인
        content_authority = any(indicator in content_lower for indicator in authority_indicators)
        
        # 출처에서 법적 권위 지표 확인
        source_authority = any(indicator in source_lower for indicator in authority_indicators)
        
        return content_authority or source_authority
    
    def _check_recency(self, result: Dict[str, Any]) -> bool:
        """최신성 확인"""
        # 간단한 최신성 확인 (실제로는 날짜 필드가 있어야 함)
        return True  # 임시로 항상 True 반환
    
    def _check_detail_level(self, content: str) -> bool:
        """상세 수준 확인"""
        # 상세한 설명인지 확인
        detail_indicators = ["구체적", "상세", "단계별", "실행", "방법"]
        content_lower = content.lower()
        detail_matches = sum(1 for indicator in detail_indicators if indicator in content_lower)
        
        return detail_matches >= 2  # 2개 이상의 상세 지표가 있으면 상세한 것으로 간주
    
    def _filter_by_quality(self, search_results: List[Dict[str, Any]], 
                          quality_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """품질 기반 필터링"""
        filtered_results = []
        quality_threshold = self.optimization_rules["quality_threshold"]
        
        for i, result in enumerate(search_results):
            if i < len(quality_analysis["quality_scores"]):
                quality_score = quality_analysis["quality_scores"][i]
                if quality_score >= quality_threshold:
                    filtered_results.append(result)
        
        return filtered_results
    
    def _rerank_by_relevance(self, filtered_results: List[Dict[str, Any]], 
                           query: str, question_type: str, domain: str) -> List[Dict[str, Any]]:
        """관련성 기반 재순위화"""
        # 관련성 점수 계산
        scored_results = []
        for result in filtered_results:
            relevance_score = self._calculate_relevance_score(
                result.get("content", ""),
                result.get("title", ""),
                query,
                question_type,
                domain
            )
            scored_results.append((result, relevance_score))
        
        # 관련성 점수 기준으로 정렬
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return [result for result, score in scored_results]
    
    def _optimize_context(self, reranked_results: List[Dict[str, Any]], 
                        query: str, question_type: str, domain: str) -> Dict[str, Any]:
        """컨텍스트 최적화"""
        max_sources = self.optimization_rules["max_sources"]
        max_length = self.optimization_rules["max_context_length"]
        
        # 상위 결과 선택
        selected_results = reranked_results[:max_sources]
        
        # 컨텍스트 구성
        context_parts = []
        total_length = 0
        
        for i, result in enumerate(selected_results):
            content = result.get("content", "")
            title = result.get("title", "")
            source = result.get("source", "")
            
            # 개별 결과 컨텍스트 구성
            result_context = f"### {title}\n{content}\n*출처: {source}*"
            
            # 길이 제한 확인
            if total_length + len(result_context) <= max_length:
                context_parts.append(result_context)
                total_length += len(result_context)
            else:
                # 남은 공간에 맞게 잘라서 추가
                remaining_length = max_length - total_length
                if remaining_length > 100:  # 최소 100자 이상은 남겨야 함
                    truncated_content = content[:remaining_length - len(title) - len(source) - 50]
                    result_context = f"### {title}\n{truncated_content}...\n*출처: {source}*"
                    context_parts.append(result_context)
                break
        
        return {
            "context": "\n\n".join(context_parts),
            "total_length": total_length,
            "source_count": len(context_parts),
            "optimization_applied": True
        }
    
    def _calculate_context_quality_metrics(self, optimized_context: Dict[str, Any], 
                                         quality_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """컨텍스트 품질 메트릭 계산"""
        return {
            "context_length": optimized_context.get("total_length", 0),
            "source_count": optimized_context.get("source_count", 0),
            "overall_quality": quality_analysis.get("overall_quality", 0.0),
            "quality_improvement": 0.0,  # 이전 버전과 비교하여 계산
            "relevance_score": sum(quality_analysis.get("relevance_scores", [])) / len(quality_analysis.get("relevance_scores", [1])),
            "completeness_score": sum(quality_analysis.get("completeness_scores", [])) / len(quality_analysis.get("completeness_scores", [1])),
            "accuracy_score": sum(quality_analysis.get("accuracy_scores", [])) / len(quality_analysis.get("accuracy_scores", [1])),
            "usability_score": sum(quality_analysis.get("usability_scores", [])) / len(quality_analysis.get("usability_scores", [1]))
        }
    
    def _identify_quality_issues(self, analysis: Dict[str, Any]) -> List[str]:
        """품질 이슈 식별"""
        issues = []
        
        # 전체 품질이 낮은 경우
        if analysis["overall_quality"] < 0.6:
            issues.append("전체적인 검색 결과 품질이 낮습니다")
        
        # 관련성이 낮은 경우
        if analysis["relevance_scores"] and sum(analysis["relevance_scores"]) / len(analysis["relevance_scores"]) < 0.5:
            issues.append("검색 결과의 관련성이 낮습니다")
        
        # 정확성이 낮은 경우
        if analysis["accuracy_scores"] and sum(analysis["accuracy_scores"]) / len(analysis["accuracy_scores"]) < 0.5:
            issues.append("검색 결과의 정확성이 낮습니다")
        
        # 실용성이 낮은 경우
        if analysis["usability_scores"] and sum(analysis["usability_scores"]) / len(analysis["usability_scores"]) < 0.5:
            issues.append("검색 결과의 실용성이 낮습니다")
        
        return issues
    
    def _generate_quality_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """품질 권장사항 생성"""
        recommendations = []
        
        # 품질이 낮은 경우 개선 권장
        if analysis["overall_quality"] < 0.7:
            recommendations.append("더 신뢰할 수 있는 출처에서 검색하세요")
            recommendations.append("검색 쿼리를 더 구체적으로 작성하세요")
        
        # 관련성이 낮은 경우
        if analysis["relevance_scores"] and sum(analysis["relevance_scores"]) / len(analysis["relevance_scores"]) < 0.6:
            recommendations.append("질문과 더 관련성 높은 키워드로 검색하세요")
        
        # 정확성이 낮은 경우
        if analysis["accuracy_scores"] and sum(analysis["accuracy_scores"]) / len(analysis["accuracy_scores"]) < 0.6:
            recommendations.append("법적 권위가 있는 출처를 우선적으로 참조하세요")
        
        return recommendations
    
    def _generate_context_improvements(self, quality_analysis: Dict[str, Any], 
                                     optimized_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """컨텍스트 개선 제안 생성"""
        improvements = []
        
        # 품질 기반 개선 제안
        if quality_analysis["overall_quality"] < 0.7:
            improvements.append({
                "type": "quality_improvement",
                "priority": "high",
                "suggestion": "검색 결과의 품질을 개선하세요",
                "specific_actions": [
                    "더 신뢰할 수 있는 출처 사용",
                    "검색 쿼리 구체화",
                    "도메인별 특화 검색"
                ],
                "impact": "높음"
            })
        
        # 관련성 기반 개선 제안
        if quality_analysis["relevance_scores"] and sum(quality_analysis["relevance_scores"]) / len(quality_analysis["relevance_scores"]) < 0.6:
            improvements.append({
                "type": "relevance_improvement",
                "priority": "medium",
                "suggestion": "검색 결과의 관련성을 높이세요",
                "specific_actions": [
                    "키워드 매칭 개선",
                    "의미적 유사성 고려",
                    "도메인 특화 검색"
                ],
                "impact": "중간"
            })
        
        # 컨텍스트 길이 기반 개선 제안
        context_length = optimized_context.get("total_length", 0)
        if context_length < 500:
            improvements.append({
                "type": "context_expansion",
                "priority": "medium",
                "suggestion": "컨텍스트를 더 풍부하게 구성하세요",
                "specific_actions": [
                    "더 많은 관련 소스 추가",
                    "상세한 설명 포함",
                    "구체적 사례 추가"
                ],
                "impact": "중간"
            })
        elif context_length > 3000:
            improvements.append({
                "type": "context_compression",
                "priority": "low",
                "suggestion": "컨텍스트를 더 간결하게 구성하세요",
                "specific_actions": [
                    "핵심 내용만 선별",
                    "중복 정보 제거",
                    "요약 정보 활용"
                ],
                "impact": "낮음"
            })
        
        return improvements


# 전역 인스턴스
context_quality_enhancer = ContextQualityEnhancer()
