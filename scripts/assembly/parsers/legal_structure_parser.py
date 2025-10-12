"""
Legal Structure Parser for Korean Law Data

This module parses the structural elements of Korean laws including
articles, paragraphs, subparagraphs, enforcement clauses, and amendment history.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class LegalStructureParser:
    """법률 구조 분석기"""
    
    def __init__(self):
        """Initialize structure parser with Korean legal structure patterns"""
        self.structure_patterns = {
            'articles': re.compile(r'제(\d+)조\s*\(([^)]+)\)'),
            'paragraphs': re.compile(r'제(\d+)항'),
            'subparagraphs': re.compile(r'제(\d+)호'),
            'items': re.compile(r'제(\d+)목'),
            'numbered_items': re.compile(r'(\d+)\.'),
            'lettered_items': re.compile(r'([가-힣])\.'),
            'enforcement_clause': re.compile(r'\[시행\s+([^\]]+)\]'),
            'amendment_clause': re.compile(r'<개정\s+([^>]+)>'),
            'supplementary_provisions': re.compile(r'부칙\s*<([^>]+)>'),
            'purpose_clause': re.compile(r'제1조\s*\(목적\)'),
            'definition_clause': re.compile(r'제2조\s*\(정의\)'),
            'scope_clause': re.compile(r'제3조\s*\(적용범위\)')
        }
        
        # 특별한 구조 패턴들
        self.special_patterns = {
            'penalty_clause': re.compile(r'제\d+조\s*\(벌칙\)'),
            'transitional_clause': re.compile(r'제\d+조\s*\(경과조치\)'),
            'delegation_clause': re.compile(r'제\d+조\s*\(위임\)'),
            'enforcement_clause': re.compile(r'제\d+조\s*\(시행\)')
        }
    
    def parse_legal_structure(self, law_content: str) -> Dict[str, Any]:
        """
        법률 구조 분석
        
        Args:
            law_content (str): Law content text
            
        Returns:
            Dict[str, Any]: Structure analysis results
        """
        try:
            structure_info = {
                'total_articles': 0,
                'total_paragraphs': 0,
                'total_subparagraphs': 0,
                'total_items': 0,
                'articles': [],
                'enforcement_info': {},
                'amendment_history': [],
                'supplementary_provisions': [],
                'special_clauses': [],
                'structure_complexity': 0.0,
                'structure_type': 'unknown',
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # 조문 분석
            articles = self._parse_articles(law_content)
            structure_info['articles'] = articles
            structure_info['total_articles'] = len(articles)
            
            # 항, 호, 목 분석
            total_paragraphs = 0
            total_subparagraphs = 0
            total_items = 0
            
            for article in articles:
                article_content = article.get('content', '')
                if article_content:
                    paragraphs = self._parse_paragraphs(article_content)
                    total_paragraphs += len(paragraphs)
                    
                    for paragraph in paragraphs:
                        paragraph_content = paragraph.get('content', '')
                        if paragraph_content:
                            subparagraphs = self._parse_subparagraphs(paragraph_content)
                            total_subparagraphs += len(subparagraphs)
                            
                            for subparagraph in subparagraphs:
                                subparagraph_content = subparagraph.get('content', '')
                                if subparagraph_content:
                                    items = self._parse_items(subparagraph_content)
                                    total_items += len(items)
            
            structure_info['total_paragraphs'] = total_paragraphs
            structure_info['total_subparagraphs'] = total_subparagraphs
            structure_info['total_items'] = total_items
            
            # 시행 조항 분석
            structure_info['enforcement_info'] = self._parse_enforcement_clause(law_content)
            
            # 개정 이력 분석
            structure_info['amendment_history'] = self._parse_amendment_history(law_content)
            
            # 부칙 분석
            structure_info['supplementary_provisions'] = self._parse_supplementary_provisions(law_content)
            
            # 특별 조항 분석
            structure_info['special_clauses'] = self._parse_special_clauses(law_content)
            
            # 구조 복잡도 계산
            structure_info['structure_complexity'] = self._calculate_complexity(structure_info)
            
            # 구조 유형 결정
            structure_info['structure_type'] = self._determine_structure_type(structure_info)
            
            return structure_info
            
        except Exception as e:
            logger.error(f"Error parsing legal structure: {e}")
            return {
                'total_articles': 0,
                'structure_complexity': 0.0,
                'error': str(e)
            }
    
    def _parse_articles(self, content: str) -> List[Dict[str, Any]]:
        """조문 파싱"""
        articles = []
        article_matches = self.structure_patterns['articles'].findall(content)
        
        for match in article_matches:
            article_num = match[0]
            article_title = match[1]
            
            # 조문 내용 추출
            article_pattern = f'제{article_num}조\\s*\\([^)]+\\)'
            article_match = re.search(article_pattern, content)
            
            if article_match:
                start_pos = article_match.end()
                # 다음 조문까지의 내용 추출
                next_article_pattern = f'제{int(article_num)+1}조'
                next_match = re.search(next_article_pattern, content)
                
                if next_match:
                    end_pos = next_match.start()
                else:
                    end_pos = len(content)
                
                article_content = content[start_pos:end_pos].strip()
                
                # 조문 유형 분석
                article_type = self._analyze_article_type(article_title, article_content)
                
                # Safe parsing with error handling
                try:
                    paragraphs = self._parse_paragraphs(article_content) if article_content else []
                    
                    articles.append({
                        'article_number': f'제{article_num}조',
                        'article_title': article_title,
                        'content': article_content,  # Use 'content' key for consistency
                        'article_type': article_type,
                        'paragraphs': paragraphs
                    })
                except Exception as e:
                    logger.warning(f"Error parsing article {article_num}: {e}")
                    # Add minimal article info
                    articles.append({
                        'article_number': f'제{article_num}조',
                        'article_title': article_title,
                        'content': article_content,
                        'article_type': 'unknown',
                        'paragraphs': []
                    })
        
        return articles
    
    def _parse_paragraphs(self, content: str) -> List[Dict[str, Any]]:
        """항 파싱"""
        paragraphs = []
        paragraph_matches = self.structure_patterns['paragraphs'].findall(content)
        
        for match in paragraph_matches:
            para_num = match
            paragraph_pattern = f'제{para_num}항'
            para_match = re.search(paragraph_pattern, content)
            
            if para_match:
                start_pos = para_match.end()
                # 다음 항까지의 내용 추출
                next_para_pattern = f'제{int(para_num)+1}항'
                next_match = re.search(next_para_pattern, content)
                
                if next_match:
                    end_pos = next_match.start()
                else:
                    end_pos = len(content)
                
                paragraph_content = content[start_pos:end_pos].strip()
                
                # Safe parsing with error handling
                try:
                    subparagraphs = self._parse_subparagraphs(paragraph_content) if paragraph_content else []
                    
                    paragraphs.append({
                        'paragraph_number': f'제{para_num}항',
                        'content': paragraph_content,  # Use 'content' key for consistency
                        'subparagraphs': subparagraphs
                    })
                except Exception as e:
                    logger.warning(f"Error parsing paragraph {para_num}: {e}")
                    paragraphs.append({
                        'paragraph_number': f'제{para_num}항',
                        'content': paragraph_content,
                        'subparagraphs': []
                    })
        
        return paragraphs
    
    def _parse_subparagraphs(self, content: str) -> List[Dict[str, Any]]:
        """호 파싱"""
        subparagraphs = []
        subpara_matches = self.structure_patterns['subparagraphs'].findall(content)
        
        for match in subpara_matches:
            subpara_num = match
            subpara_pattern = f'제{subpara_num}호'
            subpara_match = re.search(subpara_pattern, content)
            
            if subpara_match:
                start_pos = subpara_match.end()
                # 다음 호까지의 내용 추출
                next_subpara_pattern = f'제{int(subpara_num)+1}호'
                next_match = re.search(next_subpara_pattern, content)
                
                if next_match:
                    end_pos = next_match.start()
                else:
                    end_pos = len(content)
                
                subpara_content = content[start_pos:end_pos].strip()
                
                # Safe parsing with error handling
                try:
                    items = self._parse_items(subpara_content) if subpara_content else []
                    
                    subparagraphs.append({
                        'subparagraph_number': f'제{subpara_num}호',
                        'content': subpara_content,  # Use 'content' key for consistency
                        'items': items
                    })
                except Exception as e:
                    logger.warning(f"Error parsing subparagraph {subpara_num}: {e}")
                    subparagraphs.append({
                        'subparagraph_number': f'제{subpara_num}호',
                        'content': subpara_content,
                        'items': []
                    })
        
        return subparagraphs
    
    def _parse_items(self, content: str) -> List[Dict[str, Any]]:
        """목 파싱"""
        items = []
        
        # 숫자 목 (1., 2., 3.)
        numbered_matches = self.structure_patterns['numbered_items'].findall(content)
        for match in numbered_matches:
            item_num = match
            item_pattern = f'{item_num}\\.'
            item_match = re.search(item_pattern, content)
            
            if item_match:
                start_pos = item_match.end()
                # 다음 목까지의 내용 추출
                next_item_pattern = f'{int(item_num)+1}\\.'
                next_match = re.search(next_item_pattern, content)
                
                if next_match:
                    end_pos = next_match.start()
                else:
                    end_pos = len(content)
                
                item_content = content[start_pos:end_pos].strip()
                
                items.append({
                    'item_number': f'{item_num}.',
                    'item_content': item_content,
                    'item_type': 'numbered'
                })
        
        # 문자 목 (가., 나., 다.)
        lettered_matches = self.structure_patterns['lettered_items'].findall(content)
        for match in lettered_matches:
            item_letter = match
            item_pattern = f'{item_letter}\\.'
            item_match = re.search(item_pattern, content)
            
            if item_match:
                start_pos = item_match.end()
                # 다음 목까지의 내용 추출
                next_item_pattern = f'{chr(ord(item_letter)+1)}\\.'
                next_match = re.search(next_item_pattern, content)
                
                if next_match:
                    end_pos = next_match.start()
                else:
                    end_pos = len(content)
                
                item_content = content[start_pos:end_pos].strip()
                
                # Validate item content to avoid empty content errors
                if item_content and len(item_content) > 0:
                    items.append({
                        'item_number': f'{item_letter}.',
                        'item_content': item_content,
                        'item_type': 'lettered'
                    })
        
        return items
    
    def _parse_enforcement_clause(self, content: str) -> Dict[str, Any]:
        """시행 조항 파싱"""
        enforcement_match = self.structure_patterns['enforcement_clause'].search(content)
        
        if enforcement_match:
            enforcement_text = enforcement_match.group(1)
            return {
                'enforcement_date': enforcement_text,
                'enforcement_text': f'[시행 {enforcement_text}]',
                'parsed_date': self._parse_date(enforcement_text),
                'enforcement_type': 'standard'
            }
        
        return {}
    
    def _parse_amendment_history(self, content: str) -> List[Dict[str, Any]]:
        """개정 이력 파싱"""
        amendments = []
        amendment_matches = self.structure_patterns['amendment_clause'].findall(content)
        
        for match in amendment_matches:
            amendment_text = match
            amendments.append({
                'amendment_text': f'<개정 {amendment_text}>',
                'amendment_info': amendment_text,
                'parsed_date': self._parse_date(amendment_text),
                'amendment_type': self._classify_amendment_type(amendment_text)
            })
        
        return amendments
    
    def _parse_supplementary_provisions(self, content: str) -> List[Dict[str, Any]]:
        """부칙 파싱"""
        provisions = []
        provision_matches = self.structure_patterns['supplementary_provisions'].findall(content)
        
        for match in provision_matches:
            provision_text = match
            provisions.append({
                'provision_text': f'부칙 <{provision_text}>',
                'provision_info': provision_text,
                'provision_type': self._classify_provision_type(provision_text)
            })
        
        return provisions
    
    def _parse_special_clauses(self, content: str) -> List[Dict[str, Any]]:
        """특별 조항 파싱"""
        special_clauses = []
        
        for clause_type, pattern in self.special_patterns.items():
            matches = pattern.findall(content)
            for match in matches:
                special_clauses.append({
                    'clause_type': clause_type,
                    'clause_text': match,
                    'clause_number': self._extract_clause_number(match)
                })
        
        return special_clauses
    
    def _analyze_article_type(self, title: str, content: str) -> str:
        """조문 유형 분석"""
        title_lower = title.lower()
        
        if '목적' in title:
            return 'purpose'
        elif '정의' in title:
            return 'definition'
        elif '적용범위' in title:
            return 'scope'
        elif '벌칙' in title:
            return 'penalty'
        elif '경과조치' in title:
            return 'transitional'
        elif '위임' in title:
            return 'delegation'
        elif '시행' in title:
            return 'enforcement'
        else:
            return 'general'
    
    def _classify_amendment_type(self, amendment_text: str) -> str:
        """개정 유형 분류"""
        if '일부개정' in amendment_text:
            return 'partial_amendment'
        elif '전부개정' in amendment_text:
            return 'full_amendment'
        elif '신설' in amendment_text:
            return 'new_establishment'
        elif '폐지' in amendment_text:
            return 'abolition'
        else:
            return 'unknown'
    
    def _classify_provision_type(self, provision_text: str) -> str:
        """부칙 유형 분류"""
        if '시행' in provision_text:
            return 'enforcement'
        elif '경과' in provision_text:
            return 'transitional'
        elif '위임' in provision_text:
            return 'delegation'
        else:
            return 'general'
    
    def _extract_clause_number(self, clause_text: str) -> str:
        """조항 번호 추출"""
        number_match = re.search(r'제(\d+)조', clause_text)
        return number_match.group(1) if number_match else ''
    
    def _parse_date(self, date_text: str) -> Optional[str]:
        """날짜 파싱"""
        # 다양한 날짜 형식 지원
        date_patterns = [
            r'(\d{4})\.(\d{1,2})\.(\d{1,2})\.?',  # 2025.1.1. 또는 2025.1.1
            r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',  # 2025년 1월 1일
            r'(\d{4})-(\d{2})-(\d{2})',  # 2025-01-01
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_text)
            if match:
                year, month, day = match.groups()
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        return None
    
    def _calculate_complexity(self, structure_info: Dict[str, Any]) -> float:
        """구조 복잡도 계산"""
        total_elements = (
            structure_info['total_articles'] +
            structure_info['total_paragraphs'] +
            structure_info['total_subparagraphs'] +
            structure_info['total_items']
        )
        
        # 복잡도 점수 (0-1)
        if total_elements == 0:
            return 0.0
        elif total_elements < 20:
            return 0.2
        elif total_elements < 50:
            return 0.4
        elif total_elements < 100:
            return 0.6
        elif total_elements < 200:
            return 0.8
        else:
            return 1.0
    
    def _determine_structure_type(self, structure_info: Dict[str, Any]) -> str:
        """구조 유형 결정"""
        total_articles = structure_info['total_articles']
        total_paragraphs = structure_info['total_paragraphs']
        total_subparagraphs = structure_info['total_subparagraphs']
        
        if total_articles == 0:
            return 'unstructured'
        elif total_articles < 10:
            return 'simple'
        elif total_articles < 30:
            if total_paragraphs > total_articles * 2:
                return 'detailed'
            else:
                return 'moderate'
        else:
            if total_subparagraphs > total_articles * 3:
                return 'complex'
            else:
                return 'comprehensive'
    
    def get_structure_statistics(self, structure_info: Dict[str, Any]) -> Dict[str, Any]:
        """구조 통계 생성"""
        return {
            'total_elements': (
                structure_info['total_articles'] +
                structure_info['total_paragraphs'] +
                structure_info['total_subparagraphs'] +
                structure_info['total_items']
            ),
            'average_paragraphs_per_article': (
                structure_info['total_paragraphs'] / structure_info['total_articles']
                if structure_info['total_articles'] > 0 else 0
            ),
            'average_subparagraphs_per_paragraph': (
                structure_info['total_subparagraphs'] / structure_info['total_paragraphs']
                if structure_info['total_paragraphs'] > 0 else 0
            ),
            'structure_density': structure_info['structure_complexity'],
            'structure_type': structure_info['structure_type']
        }
