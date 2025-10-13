"""
Improved Article Parser for Assembly Law Data

This module provides an improved parser that correctly handles Korean legal structure
including proper separation of main articles and supplementary provisions.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ImprovedArticleParser:
    """Improved parser for Korean legal documents with proper structure handling"""
    
    def __init__(self):
        """Initialize the improved parser with enhanced patterns"""
        # Article patterns (조)
        self.article_pattern = re.compile(r'제(\d+)조\s*\(([^)]+)\)')
        self.article_pattern_no_title = re.compile(r'제(\d+)조')
        
        # Paragraph patterns (항) - Korean legal format
        self.paragraph_patterns = {
            'circle_numbers': re.compile(r'[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]'),
            'numbered': re.compile(r'(\d+)\s*항'),
            'numbered_alt': re.compile(r'제(\d+)\s*항')
        }
        
        # Sub-paragraph patterns (호) - Korean legal format
        self.subparagraph_patterns = {
            'numbered': re.compile(r'(\d+)\s*\.'),
            'numbered_alt': re.compile(r'제(\d+)\s*호')
        }
        
        # Item patterns (목) - Korean legal format
        self.item_patterns = {
            'lettered': re.compile(r'([가-힣])\s*\.'),
            'lettered_alt': re.compile(r'제([가-힣])\s*목')
        }
        
        # Supplementary provisions patterns
        self.supplementary_patterns = {
            'supplementary_start': re.compile(r'부칙\s*<([^>]+)>'),
            'supplementary_article': re.compile(r'제(\d+)조\s*\(([^)]+)\)'),
            'enforcement_date': re.compile(r'이\s*(?:규칙|영|법)\s*은\s*([^부터]+부터)?\s*시행한다'),
            'amendment_info': re.compile(r'<([^>]+)>')
        }
        
        # Content cleaning patterns
        self.cleanup_patterns = {
            'multiple_spaces': re.compile(r'\s+'),
            'html_tags': re.compile(r'<[^>]+>'),
            'amendment_markers': re.compile(r'<개정\s+[^>]+>'),
            'ui_elements': re.compile(r'\[[^\]]*\]')
        }
        
    def parse_law(self, law_content: str) -> Dict[str, Any]:
        """Alias for parse_law_document for compatibility"""
        return self.parse_law_document(law_content)
    
    def parse_law_document(self, law_content: str) -> Dict[str, Any]:
        """
        Parse a complete law document with proper structure separation
        
        Args:
            law_content (str): Raw law content
            
        Returns:
            Dict[str, Any]: Parsed law document with separated main and supplementary content
        """
        try:
            # Clean the content
            cleaned_content = self._clean_content(law_content)
            
            # Separate main content and supplementary provisions
            main_content, supplementary_content = self._separate_main_and_supplementary(cleaned_content)
            
            # Parse main articles
            main_articles = self._parse_articles_from_text(main_content)
            
            # Parse supplementary provisions
            supplementary_articles = self._parse_supplementary_provisions(supplementary_content)
            
            # Combine and validate
            all_articles = main_articles + supplementary_articles
            
            return {
                'main_articles': main_articles,
                'supplementary_articles': supplementary_articles,
                'all_articles': all_articles,
                'total_articles': len(all_articles),
                'parsing_status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error parsing law document: {e}")
            return {
                'main_articles': [],
                'supplementary_articles': [],
                'all_articles': [],
                'total_articles': 0,
                'parsing_status': 'failed',
                'error': str(e)
            }
    
    def _clean_content(self, content: str) -> str:
        """Clean content by removing unwanted elements"""
        cleaned = content
        
        # Remove HTML tags
        cleaned = re.sub(self.cleanup_patterns['html_tags'], '', cleaned)
        
        # Remove amendment markers
        cleaned = re.sub(self.cleanup_patterns['amendment_markers'], '', cleaned)
        
        # Remove UI elements
        cleaned = re.sub(self.cleanup_patterns['ui_elements'], '', cleaned)
        
        # Remove control characters (both actual and escaped)
        # Actual control characters
        cleaned = cleaned.replace('\n', ' ')  # Replace actual newline with space
        cleaned = cleaned.replace('\t', ' ')  # Replace actual tab with space
        cleaned = cleaned.replace('\r', ' ')  # Replace actual carriage return with space
        cleaned = cleaned.replace('\f', ' ')  # Replace form feed with space
        cleaned = cleaned.replace('\v', ' ')  # Replace vertical tab with space
        
        # Escaped control characters
        cleaned = cleaned.replace('\\n', ' ')  # Replace escaped newline with space
        cleaned = cleaned.replace('\\t', ' ')  # Replace escaped tab with space
        cleaned = cleaned.replace('\\r', ' ')  # Replace escaped carriage return with space
        cleaned = cleaned.replace('\\"', '"')  # Replace escaped quotes
        cleaned = cleaned.replace("\\'", "'")  # Replace escaped single quotes
        cleaned = cleaned.replace('\\\\', '\\')  # Replace escaped backslashes
        
        # Remove other control characters (ASCII 0-31 except space)
        import string
        control_chars = ''.join(chr(i) for i in range(32) if chr(i) not in string.whitespace)
        for char in control_chars:
            cleaned = cleaned.replace(char, ' ')
        
        # Normalize whitespace
        cleaned = re.sub(self.cleanup_patterns['multiple_spaces'], ' ', cleaned)
        
        return cleaned.strip()
    
    def _separate_main_and_supplementary(self, content: str) -> Tuple[str, str]:
        """
        Separate main content from supplementary provisions
        
        Args:
            content (str): Full law content
            
        Returns:
            Tuple[str, str]: (main_content, supplementary_content)
        """
        # Find supplementary provisions start
        supplementary_match = re.search(r'부칙', content)
        
        if supplementary_match:
            main_content = content[:supplementary_match.start()].strip()
            supplementary_content = content[supplementary_match.start():].strip()
        else:
            main_content = content
            supplementary_content = ""
        
        return main_content, supplementary_content
    
    def _parse_articles_from_text(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse articles from main content text with improved boundary detection
        
        Args:
            content (str): Main law content
            
        Returns:
            List[Dict[str, Any]]: List of parsed articles
        """
        articles = []
        
        # First, identify article boundaries BEFORE cleaning control characters
        # This preserves the natural structure of the law text
        article_pattern = re.compile(r'제(\d+(?:의\d+)?)조(?:\s*\(([^)]+)\))?')
        
        # Find all article positions in the original content
        matches = list(article_pattern.finditer(content))
        
        # Enhanced heuristic filtering for article boundaries
        valid_matches = self._enhanced_heuristic_filtering(content, matches)
        
        # Context-based filtering to distinguish article references
        valid_matches = self._context_based_filtering(content, valid_matches)
        
        # Sequence validation for logical article order
        valid_matches = self._sequence_validation(valid_matches)
        
        # Final hybrid validation combining all methods
        valid_matches = self._hybrid_final_validation(content, valid_matches)
        
        # Process valid matches
        for i, match in enumerate(valid_matches):
            article_number = f"제{match.group(1)}조"
            article_title = match.group(2) if match.group(2) else ""
            
            # Find the end of this article
            if i + 1 < len(valid_matches):
                next_match = valid_matches[i + 1]
                article_content = content[match.start():next_match.start()].strip()
            else:
                article_content = content[match.start():].strip()
            
            # Remove the article header from content to get pure content
            article_header = match.group(0)
            if article_content.startswith(article_header):
                article_content = article_content[len(article_header):].strip()
            
            # Parse the article
            parsed_article = self._parse_single_article(
                article_number, article_title, article_content
            )
            
            if parsed_article:
                articles.append(parsed_article)
        
        return articles
    
    def _enhanced_heuristic_filtering(self, content: str, matches: List) -> List:
        """Enhanced heuristic filtering for article boundaries"""
        valid_matches = []
        
        for i, match in enumerate(matches):
            article_start = match.start()
            article_number = int(match.group(1)) if match.group(1).isdigit() else 0
            
            # 1. 위치 기반 필터링
            if not self._is_at_article_boundary(content, article_start):
                continue
                
            # 2. 문맥 기반 필터링
            if not self._has_proper_context(content, article_start):
                continue
                
            # 3. 순서 기반 필터링
            if not self._follows_article_sequence(article_number, valid_matches):
                continue
                
            # 4. 길이 기반 필터링
            if not self._has_reasonable_length(content, match):
                continue
                
            valid_matches.append(match)
        
        return valid_matches
    
    def _is_at_article_boundary(self, content: str, position: int) -> bool:
        """조문 경계에 위치하는지 확인"""
        
        if position == 0:
            return True
            
        # 이전 문자 확인
        prev_char = content[position - 1]
        
        # 조문 경계 패턴 (완화된 버전)
        boundary_patterns = [
            '\n',           # 줄바꿈 후
            '.',            # 마침표 후
            '>',            # 개정 표시 후
            ']',            # 각주 후
            ' ',            # 공백 후 (완화)
        ]
        
        # 추가: 마침표와 공백 조합도 허용
        if prev_char == ' ' and position > 1:
            prev_prev_char = content[position - 2]
            if prev_prev_char in '.>]':
                return True
        
        return prev_char in boundary_patterns
    
    def _has_proper_context(self, content: str, position: int) -> bool:
        """적절한 문맥을 가지고 있는지 확인 (완화된 버전)"""
        
        context_start = max(0, position - 100)  # 문맥 범위 축소
        context = content[context_start:position]
        
        # 문장 끝 패턴 확인
        sentence_endings = re.findall(r'[.!?]\s*$', context)
        
        # 조문 참조 패턴 확인 (더 엄격한 패턴만)
        strict_reference_patterns = [
            r'제\d+조에\s*따라.*?제\d+조',  # 연속된 조문 참조
            r'제\d+조제\d+항.*?제\d+조',    # 항 번호와 조문 참조
        ]
        
        for pattern in strict_reference_patterns:
            if re.search(pattern, context):
                return False
        
        # 문장 끝이 있거나 첫 번째 조문이면 허용
        return len(sentence_endings) > 0 or position < 200
    
    def _follows_article_sequence(self, article_number: int, valid_matches: List) -> bool:
        """조문 번호 순서를 따르는지 확인 (완화된 버전)"""
        
        if not valid_matches:
            return True  # 첫 번째 조문은 항상 유효
        
        # 마지막 유효한 조문 번호 확인
        last_match = valid_matches[-1]
        last_number = int(last_match.group(1)) if last_match.group(1).isdigit() else 0
        
        # 순서 검증 규칙 (완화)
        if article_number == last_number + 1:
            return True  # 연속된 번호
        elif article_number > last_number + 5:  # 임계값 완화
            return True  # 큰 점프 (부칙 등)
        elif article_number == 1 and last_number > 5:  # 임계값 완화
            return True  # 부칙에서 다시 제1조
        elif article_number <= last_number + 3:  # 작은 점프도 허용
            return True
        else:
            return False  # 순서에 맞지 않음
    
    def _has_reasonable_length(self, content: str, match) -> bool:
        """조문이 합리적인 길이를 가지는지 확인"""
        
        article_start = match.start()
        article_end = match.end()
        
        # 조문 헤더 길이 확인
        header_length = article_end - article_start
        
        # 너무 짧거나 긴 헤더는 제외
        if header_length < 5 or header_length > 50:
            return False
        
        return True
    
    def _context_based_filtering(self, content: str, matches: List) -> List:
        """문맥 기반 필터링으로 조문 참조와 실제 조문 구분"""
        valid_matches = []
        
        for i, match in enumerate(matches):
            article_start = match.start()
            article_number = match.group(1)
            
            # 문맥 분석
            context_score = self._analyze_context(content, article_start, article_number)
            
            # 점수가 임계값 이상이면 유효한 조문으로 판단 (조정된 임계값)
            if context_score >= 0.6:
                valid_matches.append(match)
        
        return valid_matches
    
    def _analyze_context(self, content: str, position: int, article_number: str) -> float:
        """조문의 문맥을 분석하여 실제 조문인지 점수로 판단"""
        score = 0.0
        
        # 1. 이전 문맥 분석
        context_before = content[max(0, position - 150):position]
        
        # 2. 이후 문맥 분석
        context_after = content[position:min(len(content), position + 100)]
        
        # 3. 문장 끝 패턴 확인 (가중치: 0.4)
        sentence_endings = re.findall(r'[.!?]\s*$', context_before)
        if sentence_endings:
            score += 0.4
        
        # 4. 조문 제목 패턴 확인 (가중치: 0.3)
        title_pattern = r'제\d+조\s*\([^)]+\)'
        if re.search(title_pattern, context_after):
            score += 0.3
        
        # 5. 조문 참조 패턴 확인 (가중치: -0.5)
        reference_patterns = [
            r'제\d+조에\s*따라',
            r'제\d+조제\d+항',
            r'제\d+조의\d+',
            r'제\d+조.*?에\s*의하여',
            r'제\d+조.*?에\s*따라',
        ]
        
        for pattern in reference_patterns:
            if re.search(pattern, context_before):
                score -= 0.5
                break
        
        # 6. 위치 기반 점수 (가중치: 0.2)
        if position < 200:  # 문서 시작 부분
            score += 0.2
        elif position > len(content) * 0.8:  # 문서 끝 부분 (부칙)
            score += 0.1
        
        # 7. 조문 내용 길이 확인 (가중치: 0.1)
        next_article_pos = self._find_next_article_position(content, position)
        if next_article_pos:
            article_length = next_article_pos - position
            if 50 <= article_length <= 2000:  # 합리적인 조문 길이
                score += 0.1
        
        return max(0.0, min(1.0, score))  # 0-1 범위로 제한
    
    def _find_next_article_position(self, content: str, current_pos: int) -> int:
        """다음 조문의 위치를 찾기"""
        remaining_content = content[current_pos + 1:]
        next_match = re.search(r'제\d+조', remaining_content)
        if next_match:
            return current_pos + 1 + next_match.start()
        return None
    
    def _sequence_validation(self, matches: List) -> List:
        """조문 번호 순서를 검증하여 논리적 순서 확인"""
        if not matches:
            return matches
        
        valid_matches = []
        expected_number = 1
        
        for i, match in enumerate(matches):
            article_number = int(match.group(1)) if match.group(1).isdigit() else 0
            
            # 순서 검증 로직
            if self._is_valid_sequence(article_number, expected_number, valid_matches):
                valid_matches.append(match)
                expected_number = article_number + 1
            else:
                # 순서에 맞지 않는 경우, 특별한 경우인지 확인
                if self._is_special_case(article_number, valid_matches):
                    valid_matches.append(match)
                    expected_number = article_number + 1
        
        return valid_matches
    
    def _is_valid_sequence(self, current_number: int, expected_number: int, valid_matches: List) -> bool:
        """현재 조문 번호가 예상 순서에 맞는지 확인"""
        
        # 첫 번째 조문
        if not valid_matches:
            return current_number == 1
        
        # 연속된 번호
        if current_number == expected_number:
            return True
        
        # 작은 점프 (1-3 범위)
        if 1 <= current_number - expected_number <= 3:
            return True
        
        # 큰 점프 (부칙 등)
        if current_number - expected_number > 10:
            return True
        
        # 부칙에서 다시 제1조
        if current_number == 1 and expected_number > 10:
            return True
        
        return False
    
    def _is_special_case(self, article_number: int, valid_matches: List) -> bool:
        """특별한 경우인지 확인 (예: 부칙, 삭제된 조문 등)"""
        
        if not valid_matches:
            return False
        
        # 마지막 유효한 조문 번호 확인
        last_match = valid_matches[-1]
        last_number = int(last_match.group(1)) if last_match.group(1).isdigit() else 0
        
        # 부칙 패턴 확인
        if article_number == 1 and last_number > 10:
            return True
        
        # 큰 점프 (부칙 등)
        if article_number - last_number > 10:
            return True
        
        # 조문의2, 조문의3 등
        if '의' in str(article_number):
            return True
        
        return False
    
    def _hybrid_final_validation(self, content: str, matches: List) -> List:
        """하이브리드 최종 검증 - 모든 방법을 통합한 최종 필터링"""
        if not matches:
            return matches
        
        valid_matches = []
        
        for i, match in enumerate(matches):
            article_start = match.start()
            article_number = int(match.group(1)) if match.group(1).isdigit() else 0
            
            # 종합 점수 계산
            total_score = self._calculate_comprehensive_score(content, match, valid_matches)
            
            # 임계값 이상이면 유효한 조문으로 판단 (조정된 임계값)
            if total_score >= 0.7:  # 조정된 임계값
                valid_matches.append(match)
        
        return valid_matches
    
    def _calculate_comprehensive_score(self, content: str, match, valid_matches: List) -> float:
        """종합 점수 계산 - 모든 검증 방법의 점수를 종합"""
        score = 0.0
        
        article_start = match.start()
        article_number = int(match.group(1)) if match.group(1).isdigit() else 0
        
        # 1. 위치 기반 점수 (가중치: 0.2)
        if self._is_at_article_boundary(content, article_start):
            score += 0.2
        
        # 2. 문맥 기반 점수 (가중치: 0.3)
        context_score = self._analyze_context(content, article_start, str(article_number))
        score += context_score * 0.3
        
        # 3. 순서 기반 점수 (가중치: 0.2)
        if self._follows_article_sequence(article_number, valid_matches):
            score += 0.2
        
        # 4. 길이 기반 점수 (가중치: 0.1)
        if self._has_reasonable_length(content, match):
            score += 0.1
        
        # 5. 조문 제목 유무 점수 (가중치: 0.1)
        if match.group(2):  # 제목이 있으면
            score += 0.1
        
        # 6. 조문 내용 품질 점수 (가중치: 0.1)
        content_quality = self._assess_content_quality(content, match)
        score += content_quality * 0.1
        
        return max(0.0, min(1.0, score))
    
    def _assess_content_quality(self, content: str, match) -> float:
        """조문 내용의 품질을 평가"""
        article_start = match.start()
        article_end = match.end()
        
        # 다음 조문까지의 내용 길이 확인
        next_article_pos = self._find_next_article_position(content, article_start)
        if next_article_pos:
            article_length = next_article_pos - article_start
            
            # 합리적인 조문 길이 (100-1500자)
            if 100 <= article_length <= 1500:
                return 1.0
            elif 50 <= article_length < 100 or 1500 < article_length <= 2000:
                return 0.5
            else:
                return 0.0
        
        return 0.5  # 다음 조문을 찾을 수 없는 경우
    
    def _parse_supplementary_provisions(self, supplementary_content: str) -> List[Dict[str, Any]]:
        """
        Parse supplementary provisions with proper numbering
        
        Args:
            supplementary_content (str): Supplementary provisions content
            
        Returns:
            List[Dict[str, Any]]: List of parsed supplementary articles
        """
        if not supplementary_content:
            return []
        
        articles = []
        
        # Find all article positions in supplementary content
        article_matches = list(self.article_pattern.finditer(supplementary_content))
        
        for i, match in enumerate(article_matches):
            # Use sequential numbering for supplementary articles to avoid conflicts
            article_number = f"부칙제{i+1}조"
            article_title = match.group(2)
            
            # Find the end of this article
            if i + 1 < len(article_matches):
                next_match = article_matches[i + 1]
                article_content = supplementary_content[match.end():next_match.start()].strip()
            else:
                article_content = supplementary_content[match.end():].strip()
            
            # Parse the article with supplementary context
            parsed_article = self._parse_single_article(
                article_number, article_title, article_content, is_supplementary=True
            )
            
            if parsed_article:
                articles.append(parsed_article)
        
        return articles
    
    def _parse_single_article(self, article_number: str, article_title: str, 
                            article_content: str, is_supplementary: bool = False) -> Optional[Dict[str, Any]]:
        """
        Parse a single article with proper structure analysis
        
        Args:
            article_number (str): Article number (e.g., "제1조")
            article_title (str): Article title
            article_content (str): Article content
            is_supplementary (bool): Whether this is a supplementary provision
            
        Returns:
            Optional[Dict[str, Any]]: Parsed article data
        """
        try:
            # Parse sub-articles (paragraphs, sub-paragraphs, items)
            sub_articles = self._parse_sub_articles(article_content)
            
            # Extract references
            references = self._extract_references(article_content)
            
            # Calculate accurate word and character counts
            word_count = len(article_content.split())
            char_count = len(article_content)
            
            # Create complete article content (without control characters)
            complete_content = f"{article_number}({article_title}) {article_content}"
            
            return {
                'article_number': article_number,
                'article_title': article_title,
                'article_content': complete_content,
                'sub_articles': sub_articles,
                'references': references,
                'word_count': word_count,
                'char_count': char_count,
                'is_supplementary': is_supplementary
            }
            
        except Exception as e:
            logger.error(f"Error parsing article {article_number}: {e}")
            return None
    
    def _parse_sub_articles(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse sub-articles (paragraphs, sub-paragraphs, items) with proper structure
        
        Args:
            content (str): Article content
            
        Returns:
            List[Dict[str, Any]]: List of parsed sub-articles
        """
        sub_articles = []
        
        # Enhanced pattern to catch paragraphs that start immediately after article title
        # This handles cases like "제3조의2 ① 내용..." where paragraph starts right after article title
        enhanced_paragraph_pattern = re.compile(r'([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])\s*([^①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]*)')
        
        # Parse paragraphs (항) - circle numbers with enhanced pattern
        paragraph_matches = list(enhanced_paragraph_pattern.finditer(content))
        
        for i, match in enumerate(paragraph_matches):
            paragraph_num = self._get_circle_number_value(match.group(1))
            paragraph_content = match.group(2).strip()
            start_pos = match.start()
            
            # Find the end of this paragraph
            if i + 1 < len(paragraph_matches):
                end_pos = paragraph_matches[i + 1].start()
            else:
                end_pos = len(content)
            
            paragraph_content = content[start_pos:end_pos].strip()
            
            # Parse sub-paragraphs within this paragraph
            sub_paragraphs = self._parse_sub_paragraphs(paragraph_content)
            
            sub_articles.append({
                'type': '항',
                'number': paragraph_num,
                'content': paragraph_content,
                'position': start_pos,
                'sub_paragraphs': sub_paragraphs
            })
        
        # If no paragraphs found, look for numbered items
        if not sub_articles:
            sub_articles = self._parse_numbered_items(content)
        
        return sub_articles
    
    def _parse_sub_paragraphs(self, paragraph_content: str) -> List[Dict[str, Any]]:
        """
        Parse sub-paragraphs (호) within a paragraph
        
        Args:
            paragraph_content (str): Paragraph content
            
        Returns:
            List[Dict[str, Any]]: List of parsed sub-paragraphs
        """
        sub_paragraphs = []
        
        # Look for numbered sub-paragraphs (1., 2., 3., etc.)
        numbered_matches = list(self.subparagraph_patterns['numbered'].finditer(paragraph_content))
        
        for i, match in enumerate(numbered_matches):
            sub_paragraph_num = match.group(1)
            start_pos = match.start()
            
            # Find the end of this sub-paragraph
            if i + 1 < len(numbered_matches):
                end_pos = numbered_matches[i + 1].start()
            else:
                end_pos = len(paragraph_content)
            
            sub_paragraph_content = paragraph_content[start_pos:end_pos].strip()
            
            # Parse items within this sub-paragraph
            items = self._parse_items(sub_paragraph_content)
            
            sub_paragraphs.append({
                'type': '호',
                'number': sub_paragraph_num,
                'content': sub_paragraph_content,
                'position': start_pos,
                'items': items
            })
        
        return sub_paragraphs
    
    def _parse_items(self, sub_paragraph_content: str) -> List[Dict[str, Any]]:
        """
        Parse items (목) within a sub-paragraph
        
        Args:
            sub_paragraph_content (str): Sub-paragraph content
            
        Returns:
            List[Dict[str, Any]]: List of parsed items
        """
        items = []
        
        # Look for lettered items (가., 나., 다., etc.)
        lettered_matches = list(self.item_patterns['lettered'].finditer(sub_paragraph_content))
        
        for i, match in enumerate(lettered_matches):
            item_letter = match.group(1)
            start_pos = match.start()
            
            # Find the end of this item
            if i + 1 < len(lettered_matches):
                end_pos = lettered_matches[i + 1].start()
            else:
                end_pos = len(sub_paragraph_content)
            
            item_content = sub_paragraph_content[start_pos:end_pos].strip()
            
            items.append({
                'type': '목',
                'number': item_letter,
                'content': item_content,
                'position': start_pos
            })
        
        return items
    
    def _parse_numbered_items(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse numbered items when no paragraphs are found
        
        Args:
            content (str): Article content
            
        Returns:
            List[Dict[str, Any]]: List of parsed items
        """
        items = []
        
        # Look for numbered items (1., 2., 3., etc.)
        numbered_matches = list(self.subparagraph_patterns['numbered'].finditer(content))
        
        for i, match in enumerate(numbered_matches):
            item_num = match.group(1)
            start_pos = match.start()
            
            # Find the end of this item
            if i + 1 < len(numbered_matches):
                end_pos = numbered_matches[i + 1].start()
            else:
                end_pos = len(content)
            
            item_content = content[start_pos:end_pos].strip()
            
            items.append({
                'type': '호',
                'number': item_num,
                'content': item_content,
                'position': start_pos
            })
        
        return items
    
    def _get_circle_number_value(self, circle_char: str) -> str:
        """Convert circle number character to numeric value"""
        circle_to_number = {
            '①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5',
            '⑥': '6', '⑦': '7', '⑧': '8', '⑨': '9', '⑩': '10',
            '⑪': '11', '⑫': '12', '⑬': '13', '⑭': '14', '⑮': '15',
            '⑯': '16', '⑰': '17', '⑱': '18', '⑲': '19', '⑳': '20'
        }
        return circle_to_number.get(circle_char, '1')
    
    def _extract_references(self, content: str) -> List[str]:
        """
        Extract law references from content
        
        Args:
            content (str): Article content
            
        Returns:
            List[str]: List of referenced laws
        """
        references = []
        
        # Pattern for law references (「법률명」)
        law_pattern = re.compile(r'「([^」]+)」')
        matches = law_pattern.findall(content)
        
        for match in matches:
            if '법' in match or '규칙' in match or '령' in match:
                references.append(match)
        
        return list(set(references))  # Remove duplicates
    
    def validate_parsed_data(self, parsed_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate parsed data for quality and consistency
        
        Args:
            parsed_data (Dict[str, Any]): Parsed law document
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []
        
        # Check if parsing was successful
        if parsed_data.get('parsing_status') != 'success':
            errors.append("Parsing failed")
            return False, errors
        
        # Check article count
        total_articles = parsed_data.get('total_articles', 0)
        if total_articles == 0:
            errors.append("No articles found")
        
        # Check for duplicate article numbers (excluding supplementary articles)
        all_articles = parsed_data.get('all_articles', [])
        main_articles = parsed_data.get('main_articles', [])
        
        # Check main articles for duplicates
        main_numbers = [article.get('article_number') for article in main_articles]
        if len(main_numbers) != len(set(main_numbers)):
            errors.append("Duplicate article numbers found in main articles")
        
        # Check supplementary articles for duplicates
        supplementary_articles = parsed_data.get('supplementary_articles', [])
        supp_numbers = [article.get('article_number') for article in supplementary_articles]
        if len(supp_numbers) != len(set(supp_numbers)):
            errors.append("Duplicate article numbers found in supplementary articles")
        
        # Check article content quality
        for article in all_articles:
            content = article.get('article_content', '')
            if len(content.strip()) < 10:
                errors.append(f"Article {article.get('article_number')} has insufficient content")
        
        return len(errors) == 0, errors
