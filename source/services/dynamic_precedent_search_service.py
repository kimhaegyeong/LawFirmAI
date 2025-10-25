# -*- coding: utf-8 -*-
"""
동적 판례 검색 서비스
조문에 따라 관련 판례를 동적으로 검색하고 제공하는 서비스
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PrecedentResult:
    """판례 검색 결과 데이터 클래스"""
    case_id: str
    case_name: str
    case_number: str
    decision_date: str
    summary: str
    field: str
    court: str
    detail_url: str
    relevance_score: float = 0.0
    key_point: str = ""


class DynamicPrecedentSearchService:
    """동적 판례 검색 서비스"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.logger.info("DynamicPrecedentSearchService 초기화 완료")
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """데이터베이스 쿼리 실행 래퍼 메서드"""
        try:
            return self.db_manager.execute_query(query, params)
        except Exception as e:
            self.logger.error(f"쿼리 실행 실패: {e}")
            return []

    def get_related_precedents(self, law_name: str, article_number: str, article_content: str, limit: int = 3) -> List[PrecedentResult]:
        """
        주어진 법령명, 조문번호, 조문 내용에 기반하여 관련 판례를 동적으로 검색합니다.
        하드코딩된 판례 대신 데이터베이스에서 관련성 높은 판례를 조회합니다.
        """
        self.logger.info(f"Searching related precedents for {law_name} 제{article_number}조")
        
        try:
            # 다양한 검색 패턴으로 관련 판례 찾기
            search_patterns = self._generate_search_patterns(law_name, article_number, article_content)
            
            all_precedents = []
            
            for pattern_info in search_patterns:
                pattern_precedents = self._search_with_pattern(pattern_info, limit)
                all_precedents.extend(pattern_precedents)
            
            # 중복 제거 및 관련성 점수로 정렬
            unique_precedents = self._deduplicate_and_rank_precedents(all_precedents)
            
            # 상위 N개 반환
            final_precedents = unique_precedents[:limit]
            
            self.logger.info(f"Found {len(final_precedents)} related precedents for {law_name} 제{article_number}조")
            return final_precedents

        except Exception as e:
            self.logger.error(f"Error searching precedents: {e}")
            return []

    def _generate_search_patterns(self, law_name: str, article_number: str, article_content: str) -> List[Dict[str, Any]]:
        """다양한 검색 패턴 생성"""
        patterns = []
        
        # 패턴 1: 정확한 법령 조문 매칭
        patterns.append({
            'type': 'exact_law_article',
            'query': f"{law_name} 제{article_number}조",
            'weight': 3.0,
            'description': f"정확한 {law_name} 제{article_number}조 매칭"
        })
        
        # 패턴 2: 법령명만 매칭
        patterns.append({
            'type': 'law_name_only',
            'query': law_name,
            'weight': 2.0,
            'description': f"{law_name} 관련 판례"
        })
        
        # 패턴 3: 조문번호만 매칭
        patterns.append({
            'type': 'article_number_only',
            'query': f"제{article_number}조",
            'weight': 2.5,
            'description': f"제{article_number}조 관련 판례"
        })
        
        # 패턴 4: 조문 내용에서 추출한 키워드
        keywords = self._extract_keywords_from_content(article_content)
        for keyword in keywords:
            patterns.append({
                'type': 'keyword_match',
                'query': keyword,
                'weight': 1.5,
                'description': f"키워드 '{keyword}' 매칭"
            })
        
        # 패턴 5: 법령 분야별 검색
        field_keywords = self._get_field_keywords(law_name, article_number)
        for field_keyword in field_keywords:
            patterns.append({
                'type': 'field_match',
                'query': field_keyword,
                'weight': 1.0,
                'description': f"분야 키워드 '{field_keyword}' 매칭"
            })
        
        return patterns

    def _search_with_pattern(self, pattern_info: Dict[str, Any], limit: int) -> List[PrecedentResult]:
        """특정 패턴으로 판례 검색"""
        try:
            query_text = pattern_info['query']
            weight = pattern_info['weight']
            
            # 검색 쿼리 구성
            search_query = """
                SELECT 
                    case_id, case_name, case_number, decision_date, field, court, detail_url,
                    ? as pattern_weight,
                    ? as pattern_type
                FROM precedent_cases
                WHERE searchable_text LIKE ?
                ORDER BY decision_date DESC
                LIMIT ?
            """
            
            results = self.db_manager.execute_query(
                search_query, 
                (weight, pattern_info['type'], f"%{query_text}%", limit)
            )
            
            precedents = []
            for row in results:
                # 판례 섹션에서 요약 정보 가져오기
                summary_sections = self.db_manager.execute_query(
                    "SELECT section_content FROM precedent_sections WHERE case_id = ? AND (section_type = 'summary' OR section_type = 'gist') LIMIT 1",
                    (row['case_id'],)
                )
                summary = summary_sections[0]['section_content'] if summary_sections else "요약 정보 없음"

                precedents.append(PrecedentResult(
                    case_id=row['case_id'],
                    case_name=row['case_name'],
                    case_number=row['case_number'],
                    decision_date=row['decision_date'],
                    summary=summary,
                    field=row['field'],
                    court=row['court'],
                    detail_url=row['detail_url'],
                    relevance_score=row['pattern_weight'],
                    key_point=summary[:100] + "..." if len(summary) > 100 else summary
                ))
            
            self.logger.debug(f"Pattern '{pattern_info['description']}' found {len(precedents)} precedents")
            return precedents
            
        except Exception as e:
            self.logger.error(f"Error searching with pattern {pattern_info['type']}: {e}")
            return []

    def _deduplicate_and_rank_precedents(self, precedents: List[PrecedentResult]) -> List[PrecedentResult]:
        """중복 제거 및 관련성 점수로 정렬"""
        # case_id로 중복 제거하면서 최고 점수 유지
        unique_precedents = {}
        for precedent in precedents:
            if precedent.case_id not in unique_precedents:
                unique_precedents[precedent.case_id] = precedent
            else:
                # 더 높은 점수로 업데이트
                if precedent.relevance_score > unique_precedents[precedent.case_id].relevance_score:
                    unique_precedents[precedent.case_id] = precedent
        
        # 점수 순으로 정렬
        sorted_precedents = sorted(
            unique_precedents.values(), 
            key=lambda x: x.relevance_score, 
            reverse=True
        )
        
        return sorted_precedents

    def _extract_keywords_from_content(self, content: str) -> List[str]:
        """조문 내용에서 핵심 키워드 추출 (간단한 예시)"""
        # 실제 구현에서는 더 정교한 NLP 기반 키워드 추출 필요
        keywords = []
        if "고의" in content or "과실" in content:
            keywords.append("고의 또는 과실")
        if "손해" in content or "배상" in content:
            keywords.append("손해배상")
        if "위법행위" in content or "불법행위" in content:
            keywords.append("불법행위")
        if "책임" in content:
            keywords.append("책임")
        
        # 중복 제거 및 상위 3개 반환
        return list(dict.fromkeys(keywords))[:3]

    def _get_field_keywords(self, law_name: str, article_number: str) -> List[str]:
        """법령 분야별 키워드 생성"""
        field_keywords = []
        
        # 법령별 분야 키워드 매핑
        field_mapping = {
            "민법": ["불법행위", "손해배상", "계약", "채권", "채무", "소유권", "물권"],
            "형법": ["범죄", "처벌", "형량", "구성요건", "고의", "과실"],
            "상법": ["회사", "주식", "이사", "주주", "회사설립", "합병"],
            "노동법": ["근로", "임금", "해고", "근로시간", "휴게시간", "연차"],
            "가족법": ["이혼", "상속", "양육권", "친권", "위자료", "재산분할"]
        }
        
        if law_name in field_mapping:
            field_keywords.extend(field_mapping[law_name])
        
        return field_keywords[:3]  # 상위 3개만 반환

    def get_precedent_statistics(self) -> Dict[str, Any]:
        """판례 통계 조회"""
        try:
            stats = {}
            
            # 총 판례 수
            count_query = "SELECT COUNT(*) as count FROM precedent_cases"
            result = self.db_manager.execute_query(count_query)
            stats['total_count'] = result[0]['count'] if result else 0
            
            # 분야별 통계
            field_query = """
                SELECT field, COUNT(*) as count
                FROM precedent_cases 
                WHERE field IS NOT NULL
                GROUP BY field
                ORDER BY count DESC
            """
            stats['by_field'] = self.db_manager.execute_query(field_query)
            
            # 법원별 통계
            court_query = """
                SELECT court, COUNT(*) as count
                FROM precedent_cases 
                WHERE court IS NOT NULL
                GROUP BY court
                ORDER BY count DESC
            """
            stats['by_court'] = self.db_manager.execute_query(court_query)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"판례 통계 조회 실패: {e}")
            return {}
    
    def get_recent_precedents(self, days: int = 30, limit: int = 5) -> List[PrecedentResult]:
        """최근 판례 조회"""
        try:
            query = """
                SELECT case_id, case_name, case_number, decision_date, court, field, detail_url
                FROM precedent_cases 
                WHERE decision_date >= date('now', '-{} days')
                ORDER BY decision_date DESC
                LIMIT ?
            """.format(days)
            
            results = self.db_manager.execute_query(query, (limit,))
            
            precedents = []
            for row in results:
                # 판례 요약 조회
                summary_sections = self.db_manager.execute_query(
                    "SELECT section_content FROM precedent_sections WHERE case_id = ? AND (section_type = 'summary' OR section_type = 'gist') LIMIT 1",
                    (row['case_id'],)
                )
                summary = summary_sections[0]['section_content'] if summary_sections else "요약 정보 없음"
                
                precedent = PrecedentResult(
                    case_id=row['case_id'],
                    case_name=row['case_name'],
                    case_number=row['case_number'],
                    decision_date=row['decision_date'],
                    summary=summary,
                    field=row['field'],
                    court=row['court'],
                    detail_url=row['detail_url'],
                    relevance_score=0.8,
                    key_point=summary[:100] + "..." if len(summary) > 100 else summary
                )
                precedents.append(precedent)
            
            return precedents
            
        except Exception as e:
            self.logger.error(f"최근 판례 조회 실패: {e}")
            return []