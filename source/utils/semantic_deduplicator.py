# -*- coding: utf-8 -*-
"""
의미 기반 중복 감지 및 제거 시스템
Semantic Deduplication System
"""

import re
import logging
from typing import List, Dict, Any, Set, Tuple
from difflib import SequenceMatcher
import hashlib

class SemanticDeduplicator:
    """의미 기반 중복 감지 및 제거 클래스"""
    
    def __init__(self):
        """초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 유사도 임계값 설정
        self.similarity_threshold = 0.8
        self.min_length_threshold = 20
        
        # 불필요한 패턴들
        self.unwanted_patterns = [
            # 섹션 제목 패턴들
            r'###\s*[^\n]+\s*\n*',
            r'##\s*법률\s*문의\s*답변\s*\n*',
            
            # 플레이스홀더 패턴들
            r'\*[^*]+\*\s*\n*',
            r'정확한\s*조문\s*번호와\s*내용\s*\n*',
            r'쉬운\s*말로\s*풀어서\s*설명\s*\n*',
            r'구체적\s*예시와\s*설명\s*\n*',
            r'법적\s*리스크와\s*제한사항\s*\n*',
            
            # 면책 조항 패턴들
            r'본\s*답변은.*?바랍니다\.\s*\n*',
            r'구체적인\s*법률\s*문제는.*?바랍니다\.\s*\n*',
            r'변호사와\s*상담.*?바랍니다\.\s*\n*',
            r'법률\s*전문가와\s*상담.*?바랍니다\.\s*\n*',
            
            # 서론 패턴들
            r'(문의하신|질문하신)\s*내용에\s*대해\s*',
            r'관련해서\s*말씀드리면\s*',
            r'질문하신\s*[^에]*에\s*대해\s*',
            r'문의하신\s*[^에]*에\s*대해\s*',
            r'궁금하시군요\.\s*',
            r'궁금하시네요\.\s*',
            r'에\s*대해\s*궁금하시군요\.\s*',
            r'에\s*대해\s*궁금하시네요\.\s*',
            r'질문해\s*주신\s*내용에\s*대해\s*',
            r'문의해\s*주신\s*내용에\s*대해\s*',
            r'말씀드리면\s*',
            r'설명드리면\s*',
        ]
        
        # 의미적 유사성을 위한 키워드 매핑
        self.semantic_keywords = {
            '법령': ['법률', '조문', '규정', '법규'],
            '손해': ['피해', '손실', '배상', '보상'],
            '계약': ['계약서', '합의', '약정', '계약서'],
            '소송': ['재판', '법원', '소장', '판결'],
            '이혼': ['혼인', '부부', '가족', '친권'],
            '부동산': ['토지', '건물', '매매', '임대'],
        }
    
    def normalize_text(self, text: str) -> str:
        """텍스트 정규화"""
        # 공백 정리
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 특수문자 제거 (한글, 영문, 숫자만 유지)
        text = re.sub(r'[^\w가-힣\s]', '', text)
        
        # 소문자 변환
        text = text.lower()
        
        return text
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """의미적 유사도 계산"""
        # 기본 텍스트 유사도
        basic_similarity = SequenceMatcher(None, text1, text2).ratio()
        
        # 키워드 기반 유사도 계산
        keywords1 = self._extract_keywords(text1)
        keywords2 = self._extract_keywords(text2)
        
        keyword_similarity = self._calculate_keyword_similarity(keywords1, keywords2)
        
        # 가중 평균으로 최종 유사도 계산
        final_similarity = (basic_similarity * 0.7) + (keyword_similarity * 0.3)
        
        return final_similarity
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """키워드 추출"""
        keywords = set()
        
        # 의미적 키워드 매핑 적용
        for main_keyword, synonyms in self.semantic_keywords.items():
            if main_keyword in text:
                keywords.add(main_keyword)
                keywords.update(synonyms)
        
        # 일반적인 법률 키워드들
        legal_keywords = [
            '민법', '형법', '상법', '행정법', '헌법',
            '계약', '손해배상', '소송', '재판', '판결',
            '이혼', '부동산', '임대차', '매매', '상속',
            '친권', '양육권', '위자료', '재산분할'
        ]
        
        for keyword in legal_keywords:
            if keyword in text:
                keywords.add(keyword)
        
        return keywords
    
    def _calculate_keyword_similarity(self, keywords1: Set[str], keywords2: Set[str]) -> float:
        """키워드 기반 유사도 계산"""
        if not keywords1 and not keywords2:
            return 1.0
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        return intersection / union if union > 0 else 0.0
    
    def remove_unwanted_patterns(self, text: str) -> str:
        """불필요한 패턴 제거"""
        for pattern in self.unwanted_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # 연속된 빈 줄 정리
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def deduplicate_content(self, text: str) -> str:
        """의미 기반 중복 제거"""
        try:
            # 불필요한 패턴 먼저 제거
            text = self.remove_unwanted_patterns(text)
            
            # 줄 단위로 분할
            lines = text.split('\n')
            cleaned_lines = []
            seen_content = set()
            
            for line in lines:
                line_stripped = line.strip()
                
                # 빈 줄이나 너무 짧은 줄은 그대로 유지
                if not line_stripped or len(line_stripped) < self.min_length_threshold:
                    cleaned_lines.append(line)
                    continue
                
                # 정규화된 텍스트로 중복 체크
                normalized_line = self.normalize_text(line_stripped)
                
                # 기존 내용과의 유사도 체크
                is_duplicate = False
                for seen_text in seen_content:
                    similarity = self.calculate_semantic_similarity(normalized_line, seen_text)
                    if similarity >= self.similarity_threshold:
                        is_duplicate = True
                        self.logger.debug(f"중복 내용 제거 (유사도: {similarity:.2f}): {line_stripped[:50]}...")
                        break
                
                if not is_duplicate:
                    seen_content.add(normalized_line)
                    cleaned_lines.append(line)
            
            result = '\n'.join(cleaned_lines)
            
            # 최종 정리
            result = re.sub(r'\n{3,}', '\n\n', result)
            
            return result.strip()
            
        except Exception as e:
            self.logger.error(f"중복 제거 중 오류: {e}")
            return text
    
    def analyze_content_quality(self, text: str) -> Dict[str, Any]:
        """내용 품질 분석"""
        try:
            lines = text.split('\n')
            
            # 통계 계산
            total_lines = len(lines)
            non_empty_lines = len([line for line in lines if line.strip()])
            
            # 섹션 제목 개수
            section_titles = len([line for line in lines if re.match(r'^#+\s+', line.strip())])
            
            # 플레이스홀더 개수
            placeholders = len([line for line in lines if re.search(r'\*[^*]+\*', line)])
            
            # 면책 조항 개수
            disclaimers = len([line for line in lines if '바랍니다' in line and ('변호사' in line or '전문가' in line)])
            
            # 중복 가능성 체크
            normalized_lines = [self.normalize_text(line.strip()) for line in lines if line.strip()]
            unique_lines = len(set(normalized_lines))
            duplication_rate = 1 - (unique_lines / len(normalized_lines)) if normalized_lines else 0
            
            return {
                'total_lines': total_lines,
                'non_empty_lines': non_empty_lines,
                'section_titles': section_titles,
                'placeholders': placeholders,
                'disclaimers': disclaimers,
                'duplication_rate': duplication_rate,
                'quality_score': self._calculate_quality_score(
                    non_empty_lines, section_titles, placeholders, 
                    disclaimers, duplication_rate
                )
            }
            
        except Exception as e:
            self.logger.error(f"품질 분석 중 오류: {e}")
            return {'quality_score': 0.0}
    
    def _calculate_quality_score(self, non_empty_lines: int, section_titles: int, 
                                placeholders: int, disclaimers: int, duplication_rate: float) -> float:
        """품질 점수 계산"""
        try:
            # 기본 점수
            base_score = 1.0
            
            # 섹션 제목이 많을수록 감점
            if section_titles > 0:
                base_score -= min(section_titles * 0.1, 0.5)
            
            # 플레이스홀더가 많을수록 감점
            if placeholders > 0:
                base_score -= min(placeholders * 0.15, 0.3)
            
            # 면책 조항이 많을수록 감점
            if disclaimers > 0:
                base_score -= min(disclaimers * 0.1, 0.2)
            
            # 중복률이 높을수록 감점
            base_score -= duplication_rate * 0.5
            
            # 최소 점수 보장
            return max(base_score, 0.0)
            
        except Exception as e:
            self.logger.error(f"품질 점수 계산 중 오류: {e}")
            return 0.0


# 전역 인스턴스
semantic_deduplicator = SemanticDeduplicator()
