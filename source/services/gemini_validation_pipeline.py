# -*- coding: utf-8 -*-
import logging
from typing import List, Dict, Any, Tuple, Optional
import json
import re
from datetime import datetime
from .gemini_client import GeminiClient, GeminiResponse

logger = logging.getLogger(__name__)

class GeminiValidationPipeline:
    """Gemini 기반 다단계 검증 파이프라인"""
    
    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client
        self.validation_stages = [
            "basic_validation",      # 기본 검증
            "domain_classification", # 도메인 분류
            "quality_assessment",    # 품질 평가
            "context_analysis"       # 문맥 분석
        ]
        self.logger = logging.getLogger(__name__)
    
    def validate_term(self, term: str, context: Optional[str] = None) -> Dict[str, Any]:
        """용어 다단계 검증"""
        validation_results = {}
        
        for stage in self.validation_stages:
            try:
                result = getattr(self, stage)(term, context)
                validation_results[stage] = result
                self.logger.info(f"{stage} 완료: {term}")
            except Exception as e:
                self.logger.error(f"{stage} 중 오류 발생: {e}")
                validation_results[stage] = {
                    "error": str(e),
                    "valid": False,
                    "confidence": 0.0
                }
        
        # 최종 결과 집계
        final_result = self.aggregate_results(validation_results, term, context)
        return final_result
    
    def basic_validation(self, term: str, context: Optional[str] = None) -> Dict[str, Any]:
        """기본 용어 검증"""
        prompt = f"""
        다음이 유효한 한국 법률 용어인지 검증해주세요:
        
        용어: "{term}"
        문맥: "{context or '없음'}"
        
        검증 기준:
        1. 법률 용어로서의 정확성
        2. 한국어 맞춤법 정확성
        3. 법률 분야에서 사용되는 용어인지
        
        응답 형식:
        - 유효성: [유효/무효/불확실]
        - 신뢰도: [0-100점]
        - 이유: [구체적 설명]
        """
        
        try:
            response = self.gemini_client.generate(prompt)
            return self.parse_validation_response(response)
        except Exception as e:
            self.logger.error(f"기본 검증 중 오류: {e}")
            return {"valid": False, "confidence": 0.0, "reason": str(e)}
    
    def domain_classification(self, term: str, context: Optional[str] = None) -> Dict[str, Any]:
        """도메인 분류 검증"""
        prompt = f"""
        다음 법률 용어가 어느 법률 도메인에 속하는지 분류해주세요:
        
        용어: "{term}"
        문맥: "{context or '없음'}"
        
        도메인 옵션:
        - 민사법 (계약, 불법행위, 소유권 등)
        - 형사법 (살인, 절도, 사기 등)
        - 가족법 (이혼, 혼인, 친자 등)
        - 상사법 (회사, 주식, 상행위 등)
        - 노동법 (근로, 해고, 임금 등)
        - 부동산법 (부동산, 등기, 매매 등)
        - 지적재산권법 (특허, 상표, 저작권 등)
        - 세법 (소득세, 법인세 등)
        - 민사소송법 (소송, 재판, 증거 등)
        - 형사소송법 (수사, 기소, 변호인 등)
        - 기타/일반
        
        응답 형식:
        - 도메인: [도메인명]
        - 신뢰도: [0-100점]
        - 근거: [분류 근거]
        """
        
        try:
            response = self.gemini_client.generate(prompt)
            return self.parse_domain_response(response)
        except Exception as e:
            self.logger.error(f"도메인 분류 중 오류: {e}")
            return {"domain": "기타/일반", "confidence": 0.0, "reason": str(e)}
    
    def quality_assessment(self, term: str, context: Optional[str] = None) -> Dict[str, Any]:
        """용어 품질 평가"""
        prompt = f"""
        다음 법률 용어의 품질을 평가해주세요:
        
        용어: "{term}"
        문맥: "{context or '없음'}"
        
        평가 기준:
        1. 법률적 정확성 (0-25점)
        2. 실무 사용 빈도 (0-25점)
        3. 용어의 중요도 (0-25점)
        4. 명확성 및 이해도 (0-25점)
        
        총점: [0-100점]
        세부 점수: [각 기준별 점수]
        개선 제안: [구체적 제안사항]
        """
        
        try:
            response = self.gemini_client.generate(prompt)
            return self.parse_quality_response(response)
        except Exception as e:
            self.logger.error(f"품질 평가 중 오류: {e}")
            return {"total_score": 0, "details": {}, "suggestions": str(e)}
    
    def context_analysis(self, term: str, context: Optional[str] = None) -> Dict[str, Any]:
        """문맥 분석"""
        prompt = f"""
        다음 법률 용어의 문맥적 사용을 분석해주세요:
        
        용어: "{term}"
        문맥: "{context or '없음'}"
        
        분석 항목:
        1. 용어의 의미와 정의
        2. 관련 용어 및 동의어
        3. 사용되는 법적 상황
        4. 주의사항 및 특이점
        
        응답 형식:
        - 정의: [용어 정의]
        - 동의어: [동의어 목록]
        - 관련용어: [관련 용어 목록]
        - 문맥키워드: [문맥 키워드 목록]
        - 가중치: [0.0-1.0]
        """
        
        try:
            response = self.gemini_client.generate(prompt)
            return self.parse_context_response(response)
        except Exception as e:
            self.logger.error(f"문맥 분석 중 오류: {e}")
            return {
                "definition": "",
                "synonyms": [],
                "related_terms": [],
                "context_keywords": [],
                "weight": 0.0
            }
    
    def parse_validation_response(self, response: GeminiResponse) -> Dict[str, Any]:
        """검증 응답 파싱"""
        try:
            text = response.response.strip()
            
            # 유효성 추출
            validity_match = re.search(r'유효성:\s*([가-힣]+)', text)
            validity = validity_match.group(1) if validity_match else "불확실"
            
            # 신뢰도 추출
            confidence_match = re.search(r'신뢰도:\s*(\d+)', text)
            confidence = int(confidence_match.group(1)) if confidence_match else 0
            
            # 이유 추출
            reason_match = re.search(r'이유:\s*(.+)', text)
            reason = reason_match.group(1).strip() if reason_match else ""
            
            return {
                "valid": validity == "유효",
                "confidence": confidence / 100.0,
                "reason": reason
            }
        except Exception as e:
            self.logger.error(f"검증 응답 파싱 중 오류: {e}")
            return {"valid": False, "confidence": 0.0, "reason": str(e)}
    
    def parse_domain_response(self, response: GeminiResponse) -> Dict[str, Any]:
        """도메인 분류 응답 파싱"""
        try:
            text = response.response.strip()
            
            # 도메인 추출
            domain_match = re.search(r'도메인:\s*([가-힣/]+)', text)
            domain = domain_match.group(1).strip() if domain_match else "기타/일반"
            
            # 신뢰도 추출
            confidence_match = re.search(r'신뢰도:\s*(\d+)', text)
            confidence = int(confidence_match.group(1)) if confidence_match else 0
            
            # 근거 추출
            reason_match = re.search(r'근거:\s*(.+)', text)
            reason = reason_match.group(1).strip() if reason_match else ""
            
            return {
                "domain": domain,
                "confidence": confidence / 100.0,
                "reason": reason
            }
        except Exception as e:
            self.logger.error(f"도메인 응답 파싱 중 오류: {e}")
            return {"domain": "기타/일반", "confidence": 0.0, "reason": str(e)}
    
    def parse_quality_response(self, response: GeminiResponse) -> Dict[str, Any]:
        """품질 평가 응답 파싱"""
        try:
            text = response.response.strip()
            
            # 총점 추출
            total_score_match = re.search(r'총점:\s*(\d+)', text)
            total_score = int(total_score_match.group(1)) if total_score_match else 0
            
            # 세부 점수 추출
            details = {}
            criteria = ["법률적 정확성", "실무 사용 빈도", "용어의 중요도", "명확성 및 이해도"]
            for criterion in criteria:
                pattern = f"{criterion}.*?(\\d+)"
                match = re.search(pattern, text)
                if match:
                    details[criterion] = int(match.group(1))
                else:
                    details[criterion] = 0
            
            # 개선 제안 추출
            suggestions_match = re.search(r'개선 제안:\s*(.+)', text)
            suggestions = suggestions_match.group(1).strip() if suggestions_match else ""
            
            return {
                "total_score": total_score,
                "details": details,
                "suggestions": suggestions
            }
        except Exception as e:
            self.logger.error(f"품질 응답 파싱 중 오류: {e}")
            return {"total_score": 0, "details": {}, "suggestions": str(e)}
    
    def parse_context_response(self, response: GeminiResponse) -> Dict[str, Any]:
        """문맥 분석 응답 파싱"""
        try:
            text = response.response.strip()
            
            # 정의 추출
            definition_match = re.search(r'정의:\s*(.+)', text)
            definition = definition_match.group(1).strip() if definition_match else ""
            
            # 동의어 추출
            synonyms_match = re.search(r'동의어:\s*(.+)', text)
            synonyms_text = synonyms_match.group(1).strip() if synonyms_match else ""
            synonyms = [s.strip() for s in synonyms_text.split(',') if s.strip()]
            
            # 관련 용어 추출
            related_match = re.search(r'관련용어:\s*(.+)', text)
            related_text = related_match.group(1).strip() if related_match else ""
            related_terms = [r.strip() for r in related_text.split(',') if r.strip()]
            
            # 문맥 키워드 추출
            context_match = re.search(r'문맥키워드:\s*(.+)', text)
            context_text = context_match.group(1).strip() if context_match else ""
            context_keywords = [c.strip() for c in context_text.split(',') if c.strip()]
            
            # 가중치 추출
            weight_match = re.search(r'가중치:\s*([\d.]+)', text)
            weight = float(weight_match.group(1)) if weight_match else 0.5
            
            return {
                "definition": definition,
                "synonyms": synonyms,
                "related_terms": related_terms,
                "context_keywords": context_keywords,
                "weight": weight
            }
        except Exception as e:
            self.logger.error(f"문맥 응답 파싱 중 오류: {e}")
            return {
                "definition": "",
                "synonyms": [],
                "related_terms": [],
                "context_keywords": [],
                "weight": 0.5
            }
    
    def aggregate_results(self, validation_results: Dict[str, Any], term: str, context: Optional[str]) -> Dict[str, Any]:
        """검증 결과 집계"""
        try:
            # 기본 검증 결과
            basic = validation_results.get("basic_validation", {})
            is_valid = basic.get("valid", False)
            basic_confidence = basic.get("confidence", 0.0)
            
            # 도메인 분류 결과
            domain = validation_results.get("domain_classification", {})
            domain_name = domain.get("domain", "기타/일반")
            domain_confidence = domain.get("confidence", 0.0)
            
            # 품질 평가 결과
            quality = validation_results.get("quality_assessment", {})
            quality_score = quality.get("total_score", 0)
            quality_details = quality.get("details", {})
            suggestions = quality.get("suggestions", "")
            
            # 문맥 분석 결과
            context_analysis = validation_results.get("context_analysis", {})
            definition = context_analysis.get("definition", "")
            synonyms = context_analysis.get("synonyms", [])
            related_terms = context_analysis.get("related_terms", [])
            context_keywords = context_analysis.get("context_keywords", [])
            weight = context_analysis.get("weight", 0.5)
            
            # 최종 신뢰도 계산
            final_confidence = (basic_confidence + domain_confidence) / 2
            
            # 품질 점수 기반 필터링
            quality_threshold = 60
            is_high_quality = quality_score >= quality_threshold
            
            # 최종 결과
            result = {
                "term": term,
                "context": context,
                "is_valid": is_valid,
                "is_high_quality": is_high_quality,
                "final_confidence": final_confidence,
                "domain": domain_name,
                "domain_confidence": domain_confidence,
                "quality_score": quality_score,
                "quality_details": quality_details,
                "definition": definition,
                "synonyms": synonyms,
                "related_terms": related_terms,
                "context_keywords": context_keywords,
                "weight": weight,
                "suggestions": suggestions,
                "validation_timestamp": datetime.now().isoformat(),
                "all_stages": validation_results
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"결과 집계 중 오류: {e}")
            return {
                "term": term,
                "context": context,
                "is_valid": False,
                "is_high_quality": False,
                "final_confidence": 0.0,
                "error": str(e)
            }
    
    def batch_validate(self, terms: List[Tuple[str, Optional[str]]]) -> List[Dict[str, Any]]:
        """배치 용어 검증"""
        results = []
        
        for i, (term, context) in enumerate(terms):
            self.logger.info(f"용어 검증 진행: {i+1}/{len(terms)} - {term}")
            
            try:
                result = self.validate_term(term, context)
                results.append(result)
            except Exception as e:
                self.logger.error(f"용어 검증 실패: {term} - {e}")
                results.append({
                    "term": term,
                    "context": context,
                    "is_valid": False,
                    "error": str(e)
                })
        
        return results
