# -*- coding: utf-8 -*-
"""
법적 근거 통합 서비스
기존 서비스들과 법적 근거 제시 기능을 통합하여 제공
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Dict, List, Any, Optional
from datetime import datetime

from .answer_structure_enhancer import AnswerStructureEnhancer, QuestionType
from .legal_citation_enhancer import LegalCitationEnhancer
from .legal_basis_validator import LegalBasisValidator
from ...search.connectors.legal_data_connector_v2 import LegalDataConnectorV2

logger = get_logger(__name__)


class LegalBasisIntegrationService:
    """법적 근거 통합 서비스"""
    
    def __init__(self, db_manager: Optional[LegalDataConnectorV2] = None):
        """초기화"""
        self.db_manager = db_manager or LegalDataConnectorV2()
        self.structure_enhancer = AnswerStructureEnhancer()
        self.citation_enhancer = LegalCitationEnhancer()
        self.basis_validator = LegalBasisValidator(self.db_manager)
        self.logger = get_logger(__name__)
    
    def process_query_with_legal_basis(self, query: str, answer: str, 
                                     question_type: Optional[QuestionType] = None) -> Dict[str, Any]:
        """법적 근거를 포함한 쿼리 처리"""
        try:
            self.logger.info(f"Processing query with legal basis: {query[:100]}...")
            
            # 1. 질문 유형 자동 분류 (제공되지 않은 경우)
            if not question_type:
                question_type = self._classify_question_type(query)
            
            # 2. 법적 근거 강화된 답변 생성
            enhanced_result = self.structure_enhancer.enhance_answer_with_legal_basis(
                answer, question_type, query
            )
            
            # 3. 추가 분석 정보
            analysis_info = self._generate_analysis_info(query, answer, enhanced_result)
            
            # 4. 통합 결과 생성
            integrated_result = {
                "success": True,
                "query": query,
                "question_type": question_type.value if question_type else "unknown",
                "original_answer": answer,
                "enhanced_answer": enhanced_result.get("enhanced_answer", answer),
                "structured_answer": enhanced_result.get("structured_answer", answer),
                "legal_basis": {
                    "citations": enhanced_result.get("citations", {}),
                    "validation": enhanced_result.get("validation", {}),
                    "summary": enhanced_result.get("legal_basis_summary", {})
                },
                "confidence": enhanced_result.get("confidence", 0.0),
                "is_legally_sound": enhanced_result.get("is_legally_sound", False),
                "analysis": analysis_info,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            # 5. 결과 로깅
            self._log_processing_result(integrated_result)
            
            return integrated_result
            
        except Exception as e:
            self.logger.error(f"Error processing query with legal basis: {e}")
            return {
                "success": False,
                "query": query,
                "question_type": "unknown",
                "original_answer": answer,
                "enhanced_answer": answer,
                "structured_answer": answer,
                "legal_basis": {
                    "citations": {"citations": [], "citation_count": 0},
                    "validation": {"is_valid": False, "confidence": 0.0},
                    "summary": {}
                },
                "confidence": 0.0,
                "is_legally_sound": False,
                "analysis": {"error": str(e)},
                "processing_timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _classify_question_type(self, query: str) -> QuestionType:
        """질문 유형 자동 분류"""
        query_lower = query.lower()
        
        # 판례 검색 관련 키워드
        precedent_keywords = ['판례', '사건', '법원', '판결', '대법원', '고등법원']
        if any(keyword in query_lower for keyword in precedent_keywords):
            return QuestionType.PRECEDENT_SEARCH
        
        # 법령 문의 관련 키워드
        law_keywords = ['법령', '조문', '법률', '항', '호', '목']
        if any(keyword in query_lower for keyword in law_keywords):
            return QuestionType.LAW_INQUIRY
        
        # 법률 상담 관련 키워드
        advice_keywords = ['상담', '조언', '도움', '어떻게', '방법', '해결']
        if any(keyword in query_lower for keyword in advice_keywords):
            return QuestionType.LEGAL_ADVICE
        
        # 절차 안내 관련 키워드
        procedure_keywords = ['절차', '신청', '제출', '처리', '기간', '비용']
        if any(keyword in query_lower for keyword in procedure_keywords):
            return QuestionType.PROCEDURE_GUIDE
        
        # 용어 해설 관련 키워드
        term_keywords = ['의미', '정의', '뜻', '해석', '개념']
        if any(keyword in query_lower for keyword in term_keywords):
            return QuestionType.TERM_EXPLANATION
        
        # 계약 검토 관련 키워드
        contract_keywords = ['계약', '계약서', '검토', '조항']
        if any(keyword in query_lower for keyword in contract_keywords):
            return QuestionType.CONTRACT_REVIEW
        
        # 이혼 관련 키워드
        divorce_keywords = ['이혼', '협의이혼', '재판이혼']
        if any(keyword in query_lower for keyword in divorce_keywords):
            return QuestionType.DIVORCE_PROCEDURE
        
        # 상속 관련 키워드
        inheritance_keywords = ['상속', '유산', '상속인', '유언']
        if any(keyword in query_lower for keyword in inheritance_keywords):
            return QuestionType.INHERITANCE_PROCEDURE
        
        # 형사 관련 키워드
        criminal_keywords = ['형사', '범죄', '고소', '고발', '벌금', '징역']
        if any(keyword in query_lower for keyword in criminal_keywords):
            return QuestionType.CRIMINAL_CASE
        
        # 노동 관련 키워드
        labor_keywords = ['노동', '근로', '임금', '해고', '퇴직']
        if any(keyword in query_lower for keyword in labor_keywords):
            return QuestionType.LABOR_DISPUTE
        
        return QuestionType.GENERAL_QUESTION
    
    def _generate_analysis_info(self, query: str, answer: str, enhanced_result: Dict[str, Any]) -> Dict[str, Any]:
        """분석 정보 생성"""
        try:
            analysis_info = {
                "query_analysis": {
                    "length": len(query),
                    "complexity": self._assess_query_complexity(query),
                    "legal_terms_count": self._count_legal_terms(query)
                },
                "answer_analysis": {
                    "length": len(answer),
                    "structure_quality": self._assess_answer_structure(answer),
                    "legal_citations_count": enhanced_result.get("citations", {}).get("citation_count", 0)
                },
                "legal_basis_analysis": {
                    "has_law_references": len(enhanced_result.get("legal_basis_summary", {}).get("laws_referenced", [])) > 0,
                    "has_precedent_references": len(enhanced_result.get("legal_basis_summary", {}).get("precedents_referenced", [])) > 0,
                    "validation_passed": enhanced_result.get("is_legally_sound", False),
                    "confidence_level": self._get_confidence_level(enhanced_result.get("confidence", 0.0))
                }
            }
            
            return analysis_info
            
        except Exception as e:
            self.logger.error(f"Error generating analysis info: {e}")
            return {"error": str(e)}
    
    def _assess_query_complexity(self, query: str) -> str:
        """쿼리 복잡도 평가"""
        complexity_indicators = {
            'simple': ['무엇', '언제', '어디서', '누가'],
            'medium': ['어떻게', '왜', '방법', '절차'],
            'complex': ['분석', '비교', '검토', '평가', '판단']
        }
        
        query_lower = query.lower()
        
        for level, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                return level
        
        return 'simple'
    
    def _count_legal_terms(self, text: str) -> int:
        """법률 용어 개수 계산"""
        legal_terms = [
            '법령', '조문', '항', '호', '목', '판례', '법원', '판결',
            '계약', '손해배상', '위약금', '해지', '무효', '취소',
            '소송', '고소', '고발', '형사', '민사', '행정',
            '권리', '의무', '책임', '면책', '시효', '기간'
        ]
        
        text_lower = text.lower()
        return sum(1 for term in legal_terms if term in text_lower)
    
    def _assess_answer_structure(self, answer: str) -> str:
        """답변 구조 품질 평가"""
        structure_indicators = {
            'excellent': ['###', '**', '- ', '1.', '2.', '3.'],
            'good': ['##', '*', '•'],
            'fair': ['#', '>'],
            'poor': []
        }
        
        for level, indicators in structure_indicators.items():
            if any(indicator in answer for indicator in indicators):
                return level
        
        return 'poor'
    
    def _get_confidence_level(self, confidence: float) -> str:
        """신뢰도 레벨 반환"""
        if confidence >= 0.9:
            return 'very_high'
        elif confidence >= 0.8:
            return 'high'
        elif confidence >= 0.7:
            return 'medium'
        elif confidence >= 0.6:
            return 'low'
        else:
            return 'very_low'
    
    def _log_processing_result(self, result: Dict[str, Any]):
        """처리 결과 로깅"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO legal_basis_processing_log 
                    (query_text, question_type, confidence_score, is_legally_sound, 
                     citations_count, processing_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    result["query"][:500],
                    result["question_type"],
                    result["confidence"],
                    result["is_legally_sound"],
                    result["legal_basis"]["citations"].get("citation_count", 0),
                    result["processing_timestamp"]
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to log processing result: {e}")
    
    def get_legal_basis_statistics(self, days: int = 30) -> Dict[str, Any]:
        """법적 근거 통계 조회"""
        try:
            # 검증 통계
            validation_stats = self.basis_validator.get_validation_statistics(days)
            
            # 처리 통계 (가상의 테이블에서 조회)
            processing_stats = self._get_processing_statistics(days)
            
            return {
                "validation_statistics": validation_stats,
                "processing_statistics": processing_stats,
                "period_days": days,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting legal basis statistics: {e}")
            return {"error": str(e)}
    
    def _get_processing_statistics(self, days: int) -> Dict[str, Any]:
        """처리 통계 조회 (가상 구현)"""
        # 실제로는 데이터베이스에서 조회
        return {
            "total_queries_processed": 0,
            "average_confidence": 0.0,
            "legally_sound_answers": 0,
            "question_type_distribution": {},
            "average_citations_per_answer": 0.0
        }
    
    def enhance_existing_answer(self, answer: str, query: str = "") -> Dict[str, Any]:
        """기존 답변을 법적 근거로 강화"""
        try:
            # 질문 유형 분류
            question_type = self._classify_question_type(query)
            
            # 법적 근거 강화
            enhanced_result = self.structure_enhancer.enhance_answer_with_legal_basis(
                answer, question_type, query
            )
            
            return {
                "success": True,
                "original_answer": answer,
                "enhanced_answer": enhanced_result.get("enhanced_answer", answer),
                "structured_answer": enhanced_result.get("structured_answer", answer),
                "legal_citations": enhanced_result.get("citations", {}),
                "validation": enhanced_result.get("validation", {}),
                "confidence": enhanced_result.get("confidence", 0.0),
                "is_legally_sound": enhanced_result.get("is_legally_sound", False),
                "processing_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error enhancing existing answer: {e}")
            return {
                "success": False,
                "original_answer": answer,
                "enhanced_answer": answer,
                "structured_answer": answer,
                "error": str(e),
                "processing_timestamp": datetime.now().isoformat()
            }
    
    def validate_legal_citations(self, text: str) -> Dict[str, Any]:
        """텍스트의 법적 인용 검증"""
        try:
            # 인용 추출
            citation_result = self.citation_enhancer.enhance_text_with_citations(text)
            
            # 인용 검증
            validation_result = self.citation_enhancer.validate_citations(citation_result["citations"])
            
            return {
                "success": True,
                "citations_found": citation_result["citation_count"],
                "valid_citations": validation_result["valid_citations"],
                "invalid_citations": validation_result["invalid_citations"],
                "validation_details": validation_result["validation_details"],
                "enhanced_text": citation_result["enhanced_text"]
            }
            
        except Exception as e:
            self.logger.error(f"Error validating legal citations: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# 전역 인스턴스
legal_basis_integration_service = LegalBasisIntegrationService()
