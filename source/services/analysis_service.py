# -*- coding: utf-8 -*-
"""
Analysis Service
문서 분석 서비스
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from ..data.data_processor import LegalDataProcessor as DataProcessor
from ..models.model_manager import LegalModelManager
from ..utils.config import Config

logger = logging.getLogger(__name__)


class AnalysisService:
    """분석 서비스 클래스"""
    
    def __init__(self, config: Config, model_manager: LegalModelManager):
        """분석 서비스 초기화"""
        self.config = config
        self.model_manager = model_manager
        self.data_processor = DataProcessor()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("AnalysisService initialized")
    
    def analyze_contract(self, contract_text: str) -> Dict[str, Any]:
        """계약서 분석"""
        try:
            # 텍스트 전처리
            processed_doc = self.data_processor.process_legal_document({
                "title": "계약서",
                "content": contract_text
            })
            
            # 계약서 특화 분석
            analysis_result = {
                "summary": self._generate_contract_summary(contract_text),
                "risk_factors": self._identify_risk_factors(contract_text),
                "key_clauses": self._extract_key_clauses(contract_text),
                "missing_elements": self._check_missing_elements(contract_text),
                "recommendations": self._generate_recommendations(contract_text),
                "entities": processed_doc.get("entities", {}),
                "word_count": processed_doc.get("word_count", 0),
                "confidence": 0.8  # Placeholder
            }
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing contract: {e}")
            return {
                "error": str(e),
                "summary": "계약서 분석 중 오류가 발생했습니다.",
                "risk_factors": [],
                "key_clauses": [],
                "missing_elements": [],
                "recommendations": []
            }
    
    def analyze_legal_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """법률 문서 분석"""
        try:
            # 문서 전처리
            processed_doc = self.data_processor.process_legal_document(document)
            
            # 문서 유형별 분석
            doc_type = document.get("document_type", "general")
            
            if doc_type == "contract":
                return self.analyze_contract(document.get("content", ""))
            elif doc_type == "case":
                return self._analyze_legal_case(document)
            elif doc_type == "law":
                return self._analyze_law_document(document)
            else:
                return self._analyze_general_document(document)
                
        except Exception as e:
            self.logger.error(f"Error analyzing legal document: {e}")
            return {
                "error": str(e),
                "summary": "문서 분석 중 오류가 발생했습니다."
            }
    
    def _generate_contract_summary(self, contract_text: str) -> str:
        """계약서 요약 생성"""
        try:
            # 간단한 요약 로직 (실제로는 AI 모델 사용)
            sentences = contract_text.split('.')
            key_sentences = []
            
            # 중요한 문장 키워드
            important_keywords = ['계약', '의무', '권리', '책임', '손해', '배상', '해지', '기간', '금액', '조건']
            
            for sentence in sentences:
                if any(keyword in sentence for keyword in important_keywords):
                    key_sentences.append(sentence.strip())
            
            # 상위 5개 문장만 선택
            summary = '. '.join(key_sentences[:5])
            return summary if summary else "계약서 내용을 요약할 수 없습니다."
            
        except Exception as e:
            self.logger.error(f"Error generating contract summary: {e}")
            return "계약서 요약 생성 중 오류가 발생했습니다."
    
    def _identify_risk_factors(self, contract_text: str) -> List[Dict[str, Any]]:
        """위험 요소 식별"""
        try:
            risk_factors = []
            
            # 위험 키워드 패턴
            risk_patterns = {
                "불명확한 조항": [r'적절한', r'합리적인', r'필요한', r'충분한'],
                "일방적 의무": [r'갑은\s+반드시', r'을은\s+반드시', r'일방적으로'],
                "손해배상 제한": [r'손해배상.*제한', r'배상.*한도', r'책임.*제한'],
                "해지 조건": [r'해지.*조건', r'계약.*해지', r'즉시.*해지'],
                "기간 관련": [r'기간.*명시', r'만료.*조건', r'연장.*조건']
            }
            
            for risk_type, patterns in risk_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, contract_text, re.IGNORECASE)
                    if matches:
                        risk_factors.append({
                            "type": risk_type,
                            "description": f"{risk_type} 관련 조항이 발견되었습니다.",
                            "severity": "medium",
                            "matches": matches[:3]  # 상위 3개만
                        })
            
            return risk_factors
            
        except Exception as e:
            self.logger.error(f"Error identifying risk factors: {e}")
            return []
    
    def _extract_key_clauses(self, contract_text: str) -> List[Dict[str, Any]]:
        """핵심 조항 추출"""
        try:
            key_clauses = []
            
            # 조항 패턴
            clause_patterns = {
                "계약 목적": [r'계약.*목적', r'목적.*계약'],
                "의무 사항": [r'의무.*사항', r'갑.*의무', r'을.*의무'],
                "권리 사항": [r'권리.*사항', r'갑.*권리', r'을.*권리'],
                "손해배상": [r'손해배상', r'배상.*책임'],
                "해지 조건": [r'해지.*조건', r'계약.*해지'],
                "기간": [r'계약.*기간', r'유효.*기간']
            }
            
            for clause_type, patterns in clause_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, contract_text, re.IGNORECASE)
                    for match in matches:
                        start = max(0, match.start() - 50)
                        end = min(len(contract_text), match.end() + 50)
                        context = contract_text[start:end].strip()
                        
                        key_clauses.append({
                            "type": clause_type,
                            "context": context,
                            "position": match.start()
                        })
            
            return key_clauses[:10]  # 상위 10개만
            
        except Exception as e:
            self.logger.error(f"Error extracting key clauses: {e}")
            return []
    
    def _check_missing_elements(self, contract_text: str) -> List[str]:
        """누락된 요소 확인"""
        try:
            missing_elements = []
            
            # 필수 요소 체크
            required_elements = {
                "계약 당사자": [r'갑.*을', r'당사자', r'계약자'],
                "계약 목적": [r'목적', r'계약.*목적'],
                "계약 기간": [r'기간', r'유효.*기간', r'만료'],
                "계약 금액": [r'금액', r'가격', r'비용', r'대가'],
                "해지 조건": [r'해지', r'계약.*해지'],
                "손해배상": [r'손해배상', r'배상'],
                "분쟁 해결": [r'분쟁', r'중재', r'소송', r'관할']
            }
            
            for element, patterns in required_elements.items():
                if not any(re.search(pattern, contract_text, re.IGNORECASE) for pattern in patterns):
                    missing_elements.append(element)
            
            return missing_elements
            
        except Exception as e:
            self.logger.error(f"Error checking missing elements: {e}")
            return []
    
    def _generate_recommendations(self, contract_text: str) -> List[str]:
        """권장사항 생성"""
        try:
            recommendations = []
            
            # 기본 권장사항
            recommendations.append("계약서 전문가의 검토를 받으시기 바랍니다.")
            recommendations.append("모든 조항을 명확하게 정의하시기 바랍니다.")
            recommendations.append("분쟁 해결 방법을 명시하시기 바랍니다.")
            
            # 위험 요소 기반 권장사항
            if "손해배상" not in contract_text:
                recommendations.append("손해배상 조항을 추가하시기 바랍니다.")
            
            if "해지" not in contract_text:
                recommendations.append("계약 해지 조건을 명시하시기 바랍니다.")
            
            if "기간" not in contract_text:
                recommendations.append("계약 기간을 명확히 하시기 바랍니다.")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["분석 중 오류가 발생했습니다."]
    
    def _analyze_legal_case(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """판례 분석"""
        try:
            content = document.get("content", "")
            
            return {
                "summary": "판례 분석 결과",
                "case_type": "민사" if "민사" in content else "형사" if "형사" in content else "기타",
                "key_issues": self._extract_case_issues(content),
                "legal_principles": self._extract_legal_principles(content),
                "outcome": self._extract_case_outcome(content),
                "confidence": 0.7
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing legal case: {e}")
            return {"error": str(e)}
    
    def _analyze_law_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """법령 문서 분석"""
        try:
            content = document.get("content", "")
            
            return {
                "summary": "법령 분석 결과",
                "law_type": "형법" if "형법" in content else "민법" if "민법" in content else "기타",
                "articles": self._extract_articles(content),
                "key_concepts": self._extract_legal_concepts(content),
                "confidence": 0.8
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing law document: {e}")
            return {"error": str(e)}
    
    def _analyze_general_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """일반 문서 분석"""
        try:
            processed_doc = self.data_processor.process_legal_document(document)
            
            return {
                "summary": "문서 분석 결과",
                "entities": processed_doc.get("entities", {}),
                "word_count": processed_doc.get("word_count", 0),
                "chunk_count": processed_doc.get("chunk_count", 0),
                "confidence": 0.6
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing general document: {e}")
            return {"error": str(e)}
    
    def _extract_case_issues(self, content: str) -> List[str]:
        """판례 쟁점 추출"""
        # 간단한 패턴 매칭
        issues = []
        if "손해" in content:
            issues.append("손해배상")
        if "계약" in content:
            issues.append("계약 관련")
        if "불법행위" in content:
            issues.append("불법행위")
        return issues
    
    def _extract_legal_principles(self, content: str) -> List[str]:
        """법적 원칙 추출"""
        principles = []
        if "신의성실" in content:
            principles.append("신의성실의 원칙")
        if "자유" in content:
            principles.append("계약자유의 원칙")
        return principles
    
    def _extract_case_outcome(self, content: str) -> str:
        """판례 결과 추출"""
        if "기각" in content:
            return "기각"
        elif "인용" in content:
            return "인용"
        else:
            return "기타"
    
    def _extract_articles(self, content: str) -> List[str]:
        """조문 추출"""
        pattern = r'제(\d+)조'
        matches = re.findall(pattern, content)
        return [f"제{match}조" for match in matches]
    
    def _extract_legal_concepts(self, content: str) -> List[str]:
        """법적 개념 추출"""
        concepts = []
        if "의무" in content:
            concepts.append("의무")
        if "권리" in content:
            concepts.append("권리")
        if "책임" in content:
            concepts.append("책임")
        return concepts
