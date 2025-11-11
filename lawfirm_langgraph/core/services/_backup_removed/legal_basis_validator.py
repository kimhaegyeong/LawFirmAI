# -*- coding: utf-8 -*-
"""
법적 근거 검증 시스템
법령 인용의 정확성과 신뢰성을 검증하고 데이터베이스와 연동하여 법적 근거를 검증
"""

import sqlite3
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from .legal_citation_enhancer import LegalCitationEnhancer, LegalCitation
from ..data.database import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class LegalBasisValidation:
    """법적 근거 검증 결과"""
    is_valid: bool
    confidence: float
    validation_details: List[Dict[str, Any]]
    legal_sources: List[Dict[str, Any]]
    issues: List[str]
    recommendations: List[str]


@dataclass
class LegalSource:
    """법적 근거 소스"""
    type: str  # law, precedent, court_decision
    identifier: str  # 조문 번호, 판례 번호 등
    title: str
    content: str
    source_url: Optional[str]
    validity_status: str  # valid, invalid, unknown
    last_updated: Optional[str]


class LegalBasisValidator:
    """법적 근거 검증 시스템"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """초기화"""
        self.db_manager = db_manager or DatabaseManager()
        self.citation_enhancer = LegalCitationEnhancer()
        self.logger = logging.getLogger(__name__)
        
        # 검증 규칙 로드
        self.validation_rules = self._load_validation_rules()
        
        # 법령 데이터베이스 초기화 확인
        self._ensure_legal_database()
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """검증 규칙 로드"""
        return {
            'min_confidence': 0.7,
            'min_citation_count': 1,
            'max_citation_age_years': 10,  # 판례의 경우
            'required_citation_types': ['law_articles', 'precedent_numbers'],
            'penalty_factors': {
                'missing_law_reference': 0.2,
                'missing_precedent_reference': 0.1,
                'low_confidence_citation': 0.15,
                'outdated_precedent': 0.1,
                'invalid_format': 0.25
            }
        }
    
    def _ensure_legal_database(self):
        """법령 데이터베이스 초기화 확인"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # 법령 메타데이터 테이블 확인
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS law_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id TEXT NOT NULL,
                        law_name TEXT,
                        article_number INTEGER,
                        promulgation_date TEXT,
                        enforcement_date TEXT,
                        department TEXT,
                        is_active BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (document_id) REFERENCES documents (id)
                    )
                """)
                
                # 판례 메타데이터 테이블 확인
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS precedent_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id TEXT NOT NULL,
                        precedent_number TEXT,
                        case_name TEXT,
                        court_name TEXT,
                        decision_date TEXT,
                        case_type TEXT,
                        is_active BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (document_id) REFERENCES documents (id)
                    )
                """)
                
                # 법적 근거 검증 로그 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS legal_basis_validation_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_text TEXT NOT NULL,
                        answer_text TEXT NOT NULL,
                        validation_result TEXT NOT NULL,
                        confidence_score REAL,
                        validation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        citations_found INTEGER DEFAULT 0,
                        valid_citations INTEGER DEFAULT 0
                    )
                """)
                
                conn.commit()
                self.logger.info("Legal database tables ensured")
                
        except Exception as e:
            self.logger.error(f"Failed to ensure legal database: {e}")
            raise
    
    def validate_legal_basis(self, query: str, answer: str) -> LegalBasisValidation:
        """법적 근거 검증"""
        try:
            self.logger.info(f"Validating legal basis for query: {query[:100]}...")
            
            # 1. 답변에서 법적 인용 추출
            citation_result = self.citation_enhancer.enhance_text_with_citations(answer)
            citations = citation_result['citations']
            
            # 2. 각 인용의 유효성 검증
            validation_details = []
            legal_sources = []
            issues = []
            recommendations = []
            
            for citation in citations:
                detail = self._validate_single_citation(citation)
                validation_details.append(detail)
                
                if detail['is_valid']:
                    # 데이터베이스에서 법적 소스 검색
                    legal_source = self._find_legal_source(citation)
                    if legal_source:
                        legal_sources.append(legal_source)
                else:
                    issues.extend(detail['issues'])
            
            # 3. 전체 검증 점수 계산
            confidence = self._calculate_overall_confidence(
                citations, validation_details, legal_sources
            )
            
            # 4. 권장사항 생성
            recommendations = self._generate_recommendations(
                citations, validation_details, legal_sources, issues
            )
            
            # 5. 검증 결과 생성
            validation_result = LegalBasisValidation(
                is_valid=confidence >= self.validation_rules['min_confidence'],
                confidence=confidence,
                validation_details=validation_details,
                legal_sources=legal_sources,
                issues=issues,
                recommendations=recommendations
            )
            
            # 6. 검증 로그 저장
            self._log_validation(query, answer, validation_result)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating legal basis: {e}")
            return LegalBasisValidation(
                is_valid=False,
                confidence=0.0,
                validation_details=[],
                legal_sources=[],
                issues=[f"검증 중 오류 발생: {str(e)}"],
                recommendations=["시스템 관리자에게 문의하세요"]
            )
    
    def _validate_single_citation(self, citation: LegalCitation) -> Dict[str, Any]:
        """개별 인용 검증"""
        validation_detail = {
            'citation': citation.text,
            'type': citation.type,
            'is_valid': True,
            'confidence': citation.confidence,
            'issues': [],
            'suggestions': []
        }
        
        # 기본 검증
        if citation.confidence < self.validation_rules['min_confidence']:
            validation_detail['is_valid'] = False
            validation_detail['issues'].append("신뢰도가 낮음")
        
        # 패턴별 특별 검증
        if citation.type == 'law_articles':
            article_num = citation.metadata.get('article_number')
            if article_num and (article_num < 1 or article_num > 10000):
                validation_detail['is_valid'] = False
                validation_detail['issues'].append("조문 번호가 비정상적임")
        
        elif citation.type == 'precedent_numbers':
            precedent_num = citation.metadata.get('precedent_number', '')
            if len(precedent_num) < 8:
                validation_detail['is_valid'] = False
                validation_detail['issues'].append("판례 번호 형식이 올바르지 않음")
            
            # 판례 연도 확인
            year = citation.metadata.get('year')
            if year:
                current_year = datetime.now().year
                if int(year) < current_year - self.validation_rules['max_citation_age_years']:
                    validation_detail['issues'].append("오래된 판례")
                    validation_detail['suggestions'].append("최신 판례 확인 필요")
        
        elif citation.type == 'law_names':
            law_name = citation.metadata.get('law_name', '')
            if not self._is_valid_law_name(law_name):
                validation_detail['is_valid'] = False
                validation_detail['issues'].append("법률명이 올바르지 않음")
        
        return validation_detail
    
    def _find_legal_source(self, citation: LegalCitation) -> Optional[LegalSource]:
        """데이터베이스에서 법적 소스 검색"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                if citation.type == 'law_articles':
                    # 법령 조문 검색
                    article_num = citation.metadata.get('article_number')
                    if article_num:
                        cursor.execute("""
                            SELECT d.id, d.title, d.content, d.source_url, lm.law_name, lm.article_number
                            FROM documents d
                            JOIN law_metadata lm ON d.id = lm.document_id
                            WHERE lm.article_number = ? AND lm.is_active = 1
                            ORDER BY lm.updated_at DESC
                            LIMIT 1
                        """, (article_num,))
                        
                        result = cursor.fetchone()
                        if result:
                            return LegalSource(
                                type='law',
                                identifier=f"제{article_num}조",
                                title=result[1],
                                content=result[2],
                                source_url=result[3],
                                validity_status='valid',
                                last_updated=result[5] if len(result) > 5 else None
                            )
                
                elif citation.type == 'precedent_numbers':
                    # 판례 검색
                    precedent_num = citation.metadata.get('precedent_number')
                    if precedent_num:
                        cursor.execute("""
                            SELECT d.id, d.title, d.content, d.source_url, pm.precedent_number, pm.case_name, pm.court_name
                            FROM documents d
                            JOIN precedent_metadata pm ON d.id = pm.document_id
                            WHERE pm.precedent_number = ? AND pm.is_active = 1
                            ORDER BY pm.updated_at DESC
                            LIMIT 1
                        """, (precedent_num,))
                        
                        result = cursor.fetchone()
                        if result:
                            return LegalSource(
                                type='precedent',
                                identifier=precedent_num,
                                title=result[1],
                                content=result[2],
                                source_url=result[3],
                                validity_status='valid',
                                last_updated=None
                            )
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error finding legal source: {e}")
            return None
    
    def _is_valid_law_name(self, law_name: str) -> bool:
        """법률명 유효성 검증"""
        # 기본적인 법률명 패턴 확인
        valid_patterns = [
            r'[가-힣]+법$',
            r'[가-힣]+법\s*시행령$',
            r'[가-힣]+법\s*시행규칙$',
            r'헌법$',
            r'민법$',
            r'형법$',
            r'상법$',
            r'행정법$'
        ]
        
        import re
        for pattern in valid_patterns:
            if re.match(pattern, law_name.strip()):
                return True
        
        return False
    
    def _calculate_overall_confidence(self, citations: List[LegalCitation], 
                                    validation_details: List[Dict[str, Any]], 
                                    legal_sources: List[LegalSource]) -> float:
        """전체 신뢰도 계산"""
        if not citations:
            return 0.0
        
        # 기본 점수
        base_score = 0.5
        
        # 인용 개수 보너스
        citation_bonus = min(0.3, len(citations) * 0.05)
        
        # 유효한 인용 비율
        valid_citations = sum(1 for detail in validation_details if detail['is_valid'])
        validity_ratio = valid_citations / len(citations) if citations else 0
        validity_bonus = validity_ratio * 0.2
        
        # 법적 소스 확인 보너스
        source_bonus = min(0.2, len(legal_sources) * 0.1)
        
        # 평균 신뢰도
        avg_confidence = sum(c.confidence for c in citations) / len(citations) if citations else 0
        confidence_bonus = avg_confidence * 0.3
        
        # 패널티 적용
        penalties = 0
        penalty_factors = self.validation_rules['penalty_factors']
        
        # 필수 인용 유형 확인
        citation_types = {c.type for c in citations}
        required_types = set(self.validation_rules['required_citation_types'])
        if not required_types.intersection(citation_types):
            penalties += penalty_factors['missing_law_reference']
        
        # 낮은 신뢰도 인용 패널티
        low_confidence_count = sum(1 for c in citations if c.confidence < 0.7)
        if low_confidence_count > 0:
            penalties += penalty_factors['low_confidence_citation'] * (low_confidence_count / len(citations))
        
        # 최종 점수 계산
        final_score = base_score + citation_bonus + validity_bonus + source_bonus + confidence_bonus - penalties
        
        return max(0.0, min(1.0, round(final_score, 3)))
    
    def _generate_recommendations(self, citations: List[LegalCitation], 
                                validation_details: List[Dict[str, Any]], 
                                legal_sources: List[LegalSource], 
                                issues: List[str]) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        # 인용이 없는 경우
        if not citations:
            recommendations.append("법령 조문이나 판례를 인용하여 답변의 신뢰성을 높이세요")
            return recommendations
        
        # 낮은 신뢰도 인용이 있는 경우
        low_confidence_citations = [c for c in citations if c.confidence < 0.7]
        if low_confidence_citations:
            recommendations.append("신뢰도가 낮은 인용들을 더 정확한 형식으로 수정하세요")
        
        # 법적 소스가 부족한 경우
        if len(legal_sources) < len(citations) * 0.5:
            recommendations.append("데이터베이스에서 확인되지 않은 인용들을 검증하세요")
        
        # 특정 문제별 권장사항
        if any("조문 번호가 비정상적임" in issue for issue in issues):
            recommendations.append("법령 조문 번호를 정확히 확인하세요")
        
        if any("판례 번호 형식이 올바르지 않음" in issue for issue in issues):
            recommendations.append("판례 번호 형식을 표준 형식에 맞게 수정하세요")
        
        if any("오래된 판례" in issue for issue in issues):
            recommendations.append("최신 판례를 확인하여 답변을 업데이트하세요")
        
        # 긍정적 권장사항
        if len(citations) >= 3:
            recommendations.append("충분한 법적 근거를 제시하고 있어 좋습니다")
        
        if len(legal_sources) >= len(citations) * 0.8:
            recommendations.append("대부분의 인용이 데이터베이스에서 확인되어 신뢰할 수 있습니다")
        
        return recommendations
    
    def _log_validation(self, query: str, answer: str, validation_result: LegalBasisValidation):
        """검증 로그 저장"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO legal_basis_validation_log 
                    (query_text, answer_text, validation_result, confidence_score, citations_found, valid_citations)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    query[:500],  # 길이 제한
                    answer[:1000],  # 길이 제한
                    str(validation_result.is_valid),
                    validation_result.confidence,
                    len(validation_result.validation_details),
                    sum(1 for detail in validation_result.validation_details if detail['is_valid'])
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to log validation: {e}")
    
    def get_validation_statistics(self, days: int = 30) -> Dict[str, Any]:
        """검증 통계 조회"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_validations,
                        AVG(confidence_score) as avg_confidence,
                        COUNT(CASE WHEN validation_result = 'True' THEN 1 END) as valid_count,
                        COUNT(CASE WHEN validation_result = 'False' THEN 1 END) as invalid_count,
                        AVG(citations_found) as avg_citations,
                        AVG(valid_citations) as avg_valid_citations
                    FROM legal_basis_validation_log
                    WHERE validation_timestamp >= datetime('now', '-{} days')
                """.format(days))
                
                result = cursor.fetchone()
                
                if result:
                    return {
                        'total_validations': result[0],
                        'average_confidence': round(result[1], 3) if result[1] else 0,
                        'valid_count': result[2],
                        'invalid_count': result[3],
                        'average_citations': round(result[4], 2) if result[4] else 0,
                        'average_valid_citations': round(result[5], 2) if result[5] else 0,
                        'validation_period_days': days
                    }
                
                return {}
                
        except Exception as e:
            self.logger.error(f"Failed to get validation statistics: {e}")
            return {}
    
    def add_legal_source(self, legal_source: LegalSource) -> bool:
        """법적 소스 추가"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # 문서 먼저 추가
                cursor.execute("""
                    INSERT OR REPLACE INTO documents 
                    (id, document_type, title, content, source_url, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    f"{legal_source.type}_{legal_source.identifier}",
                    legal_source.type,
                    legal_source.title,
                    legal_source.content,
                    legal_source.source_url
                ))
                
                # 메타데이터 추가
                if legal_source.type == 'law':
                    cursor.execute("""
                        INSERT OR REPLACE INTO law_metadata 
                        (document_id, law_name, article_number, is_active, updated_at)
                        VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
                    """, (
                        f"{legal_source.type}_{legal_source.identifier}",
                        legal_source.title,
                        int(legal_source.identifier.replace('제', '').replace('조', ''))
                    ))
                
                elif legal_source.type == 'precedent':
                    cursor.execute("""
                        INSERT OR REPLACE INTO precedent_metadata 
                        (document_id, precedent_number, case_name, is_active, updated_at)
                        VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
                    """, (
                        f"{legal_source.type}_{legal_source.identifier}",
                        legal_source.identifier,
                        legal_source.title
                    ))
                
                conn.commit()
                self.logger.info(f"Legal source added: {legal_source.type}_{legal_source.identifier}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to add legal source: {e}")
            return False
