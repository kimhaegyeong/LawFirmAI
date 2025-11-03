"""
Article Parser for Assembly Law Data

This module parses article structure from law content text to extract
article numbers, titles, sub-articles, and content.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ArticleParser:
    """Parser for extracting structured article information from law text"""
    
    def __init__(self):
        """Initialize the article parser with improved patterns for Korean legal structure"""
        # Article patterns (Ï°?
        self.article_pattern = re.compile(r'??\d+)Ï°?s*\(([^)]+)\)')
        self.article_pattern_no_title = re.compile(r'??\d+)Ï°?)
        
        # Paragraph patterns (?? - Korean legal format
        self.paragraph_patterns = {
            'numbered': re.compile(r'????????????????????????????????????????),
            'numbered_alt': re.compile(r'(\d+)\s*??),
            'numbered_alt2': re.compile(r'??\d+)\s*??)
        }
        
        # Sub-paragraph patterns (?? - Korean legal format
        self.subparagraph_patterns = {
            'numbered': re.compile(r'(\d+)\s*\.'),
            'numbered_alt': re.compile(r'??\d+)\s*??)
        }
        
        # Item patterns (Î™? - Korean legal format
        self.item_patterns = {
            'lettered': re.compile(r'([Í∞Ä-??)\s*\.'),
            'lettered_alt': re.compile(r'??[Í∞Ä-??)\s*Î™?)
        }
        
        # Amendment patterns
        self.amendment_pattern = re.compile(r'<Í∞úÏ†ï\s+([^>]+)>')
        
        # Validation patterns to avoid false matches
        self.invalid_patterns = [
            re.compile(r'^\d{4,}'),  # Avoid years like 2004, 2006
            re.compile(r'^\d+$'),   # Avoid pure numbers
            re.compile(r'^[,\.\s]+$'),  # Avoid punctuation only
            re.compile(r'^\d{1,2}\.\d{1,2}\.\d{1,2}'),  # Avoid dates
        ]
        
        # Content validation patterns
        self.min_content_length = 1  # Minimum meaningful content length (reduced for better parsing)
        
        # Disable some expensive validations for speed
        self.enable_expensive_validation = False
        
    def parse_articles(self, law_content: str, html_content: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Parse articles from law content text, with optional HTML parsing
        
        Args:
            law_content (str): Raw law content text
            html_content (Optional[str]): HTML content for enhanced parsing
            
        Returns:
            List[Dict[str, Any]]: List of parsed articles
        """
        try:
            # Try HTML parsing first if HTML content is available
            if html_content and html_content.strip():
                html_articles = self._parse_articles_from_html(html_content)
                if html_articles:
                    logger.debug(f"Successfully parsed {len(html_articles)} articles from HTML")
                    return html_articles
            
            # Fall back to text parsing
            return self._parse_articles_from_text(law_content)
            
        except Exception as e:
            logger.error(f"Error parsing articles: {e}")
            return []
    
    def _parse_articles_from_html(self, html_content: str) -> List[Dict[str, Any]]:
        """
        Parse articles from HTML content using BeautifulSoup
        
        Args:
            html_content (str): HTML content
            
        Returns:
            List[Dict[str, Any]]: List of parsed articles
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            articles = []
            
            # Remove UI elements and unwanted HTML elements before parsing
            self._remove_ui_elements_from_html(soup)
            
            # Enhanced HTML parsing for Korean legal documents
            # Look for various HTML structures that contain legal content
            
            article_texts = []
            
            # Method 1: Find articles by looking for numbered articles in text
            for text_node in soup.find_all(text=True):
                text = text_node.strip()
                if re.search(r'??d+Ï°?, text) and len(text) > 20:
                    # Get the parent element that contains this text
                    parent = text_node.parent
                    if parent:
                        # Extract the full article text from the parent element
                        article_text = parent.get_text(strip=True)
                        # Clean UI elements from the extracted text
                        article_text = self._clean_ui_elements_from_text(article_text)
                        if len(article_text) > 50:  # Ensure it's substantial content
                            article_texts.append(article_text)
            
            # Method 2: Look for specific HTML classes or IDs that might contain articles
            article_selectors = [
                'article', 'Ï°?, 'article-content', 'law-article', 'legal-article',
                '.article', '.Ï°?, '.article-content', '.law-article', '.legal-article',
                '[class*="article"]', '[class*="Ï°?]', '[id*="article"]', '[id*="Ï°?]'
            ]
            
            for selector in article_selectors:
                try:
                    elements = soup.select(selector)
                    for element in elements:
                        text = element.get_text(strip=True)
                        # Clean UI elements from the extracted text
                        text = self._clean_ui_elements_from_text(text)
                        if re.search(r'??d+Ï°?, text) and len(text) > 50:
                            article_texts.append(text)
                except Exception:
                    continue
            
            # Method 3: Look for div elements with specific patterns
            div_elements = soup.find_all('div')
            for div in div_elements:
                text = div.get_text(strip=True)
                # Clean UI elements from the extracted text
                text = self._clean_ui_elements_from_text(text)
                if re.search(r'??d+Ï°?, text) and len(text) > 50:
                    article_texts.append(text)
            
            # Method 4: Look for paragraph elements with article content
            p_elements = soup.find_all('p')
            for p in p_elements:
                text = p.get_text(strip=True)
                # Clean UI elements from the extracted text
                text = self._clean_ui_elements_from_text(text)
                if re.search(r'??d+Ï°?, text) and len(text) > 50:
                    article_texts.append(text)
            
            # If we found article texts, parse them
            if article_texts:
                for article_text in article_texts:
                    if self._is_valid_article_text(article_text):
                        parsed_article = self._parse_single_article(article_text)
                        if parsed_article:
                            articles.append(parsed_article)
            
            # If no articles found, try alternative HTML structure detection
            if not articles:
                # Look for div elements with article-related classes
                article_elements = soup.find_all(['div', 'p'], class_=re.compile(r'article|Ï°?law'))
                
                for element in article_elements:
                    article_text = element.get_text(strip=True)
                    # Clean UI elements from the extracted text
                    article_text = self._clean_ui_elements_from_text(article_text)
                    if self._is_valid_article_text(article_text):
                        parsed_article = self._parse_single_article(article_text)
                        if parsed_article:
                            articles.append(parsed_article)
            
            logger.debug(f"HTML parsing found {len(articles)} articles")
            return articles
            
        except Exception as e:
            logger.error(f"Error parsing HTML content: {e}")
            return []
    
    def _is_valid_article_text(self, text: str) -> bool:
        """
        Check if text contains valid article structure
        
        Args:
            text (str): Text to validate
            
        Returns:
            bool: True if text appears to be a valid article
        """
        if not text or len(text.strip()) < 10:
            return False
        
        # Check for article number pattern
        if re.search(r'??d+Ï°?, text):
            return True
        
        # Check for Korean legal structure patterns
        if re.search(r'[?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©]', text):
            return True
        
        return False
    
    def _parse_articles_from_text(self, law_content: str) -> List[Dict[str, Any]]:
        """
        Parse articles from text content
        
        Args:
            law_content (str): Raw law content text
            
        Returns:
            List[Dict[str, Any]]: List of parsed articles
        """
        try:
            # Remove amendment markers for cleaner parsing
            clean_content = self._remove_amendment_markers(law_content)
            
            # Split content into articles
            articles = self._split_into_articles(clean_content)
            
            # Parse each article
            parsed_articles = []
            for article_text in articles:
                parsed_article = self._parse_single_article(article_text)
                if parsed_article:
                    parsed_articles.append(parsed_article)
            
            return parsed_articles
            
        except Exception as e:
            logger.error(f"Error parsing articles: {e}")
            return []
    
    def _remove_amendment_markers(self, content: str) -> str:
        """
        Remove amendment markers from content
        
        Args:
            content (str): Raw content
            
        Returns:
            str: Content with amendment markers removed
        """
        return re.sub(self.amendment_pattern, '', content)
    
    def _split_into_articles(self, content: str) -> List[str]:
        """
        Split content into individual articles with improved UI element removal
        
        Args:
            content (str): Full law content
            
        Returns:
            List[str]: List of article texts
        """
        # First, remove UI elements from content
        clean_content = self._remove_ui_elements(content)
        
        # Find all article positions
        article_positions = []
        
        for match in self.article_pattern.finditer(clean_content):
            article_positions.append((match.start(), match.group(1)))
        
        # If no articles with titles found, try without titles
        if not article_positions:
            for match in self.article_pattern_no_title.finditer(clean_content):
                article_positions.append((match.start(), match.group(1)))
        
        # Sort by position
        article_positions.sort(key=lambda x: x[0])
        
        # Split content into articles
        articles = []
        for i, (start_pos, article_num) in enumerate(article_positions):
            if i + 1 < len(article_positions):
                end_pos = article_positions[i + 1][0]
                article_text = clean_content[start_pos:end_pos].strip()
            else:
                article_text = clean_content[start_pos:].strip()
            
            # Clean the article text
            article_text = self._clean_article_text(article_text)
            
            if article_text and len(article_text) > 10:  # Minimum meaningful length
                articles.append(article_text)
        
        return articles
    
    def _remove_ui_elements(self, content: str) -> str:
        """
        Remove UI elements from content
        
        Args:
            content (str): Input content
            
        Returns:
            str: Content with UI elements removed
        """
        ui_patterns = [
            r'Ï°∞Î¨∏Î≤ÑÌäº?†ÌÉùÏ≤¥ÌÅ¨',
            r'?ºÏπòÍ∏∞Ï†ëÍ∏?,
            r'?†ÌÉùÏ≤¥ÌÅ¨',
            r'?ºÏπòÍ∏?,
            r'?ëÍ∏∞',
            r'Î≤ÑÌäº',
            r'?†ÌÉù',
            r'Ï≤¥ÌÅ¨',
            r'Ï°∞Î¨∏ Î≤ÑÌäº ?åÍ∞ú',
            r'Ï°∞Î¨∏?∞ÌòÅ',
            r'Ï°∞Î¨∏?êÎ?',
            r'\[?ºÏπòÍ∏?]',
            r'\[?ëÍ∏∞\]',
            r'\[?†ÌÉù\]',
            r'\[Ï≤¥ÌÅ¨\]',
            r'\[Ï°∞Î¨∏\]',
            r'??,
            r'?Ä',
            r'??,
            r'??,
            r'??,
            r'??,
            r'??,
            r'??
        ]
        
        clean_content = content
        for pattern in ui_patterns:
            clean_content = re.sub(pattern, '', clean_content, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        clean_content = re.sub(r'\s+', ' ', clean_content).strip()
        
        return clean_content
    
    def _clean_article_text(self, article_text: str) -> str:
        """
        Clean individual article text
        
        Args:
            article_text (str): Raw article text
            
        Returns:
            str: Cleaned article text
        """
        # Remove extra whitespace
        article_text = re.sub(r'\s+', ' ', article_text)
        
        # Remove UI elements again (in case some were missed)
        article_text = self._remove_ui_elements(article_text)
        
        # Clean up any remaining artifacts
        article_text = re.sub(r'\s+', ' ', article_text)
        
        return article_text.strip()
    
    def _parse_single_article(self, article_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single article text with improved validation
        
        Args:
            article_text (str): Single article text
            
        Returns:
            Optional[Dict[str, Any]]: Parsed article data
        """
        try:
            # Extract article number and title with validation
            article_match = self.article_pattern.search(article_text)
            if not article_match:
                article_match = self.article_pattern_no_title.search(article_text)
                if not article_match:
                    return None
            
            article_number = f"??article_match.group(1)}Ï°?
            article_title = article_match.group(2) if len(article_match.groups()) > 1 else ""
            
            # Validate article number (should be reasonable range)
            article_num = int(article_match.group(1))
            if article_num > 1000:  # Unlikely to have more than 1000 articles
                logger.warning(f"Suspicious article number: {article_number}")
                return None
            
            # Extract main content - get the entire article content
            main_content = self._extract_complete_article_content_v8(article_text, article_match.end())
            
            # Special handling for definition articles (??Ï°??? - temporarily disabled
            # if self._is_definition_article(article_number, article_title, main_content):
            #     # For definition articles, use more aggressive content extraction
            #     # Use the content-only version of the method
            #     main_content = self._extract_definition_article_content(main_content)
            
            # Extract sub-articles with improved validation
            sub_articles = self._extract_sub_articles(main_content)
            
            # Extract references
            references = self._extract_references(article_text)
            
            # Create complete article content with all paragraphs (???ïÎ≥¥ ?¨Ìï®)
            # Use the original main_content instead of creating from sub_articles
            complete_content = f"{article_number}({article_title})"
            if main_content.strip():
                complete_content += f"\n{main_content.strip()}"
            
            # Validate content quality (??Í¥Ä?Ä??Í∏∞Ï? ?ÅÏö©)
            if len(complete_content.strip()) < self.min_content_length:
                # ÏßßÏ? Ï°∞Î¨∏???†Ìö®?????àÏúºÎØÄÎ°?Í≤ΩÍ≥†Îß??úÏãú?òÍ≥† Í≥ÑÏÜç ÏßÑÌñâ
                logger.debug(f"Article {article_number} has minimal content ({len(complete_content.strip())} chars)")
                # return None  # Ï£ºÏÑù Ï≤òÎ¶¨?òÏó¨ ÏßßÏ? Ï°∞Î¨∏??Ï≤òÎ¶¨?òÎèÑÎ°???
            
            return {
                'article_number': article_number,
                'article_title': article_title,
                'article_content': complete_content,
                'sub_articles': sub_articles,
                'references': references
            }
            
        except Exception as e:
            logger.error(f"Error parsing single article: {e}")
            return None
    
    def _extract_complete_article_content_v2(self, article_text: str, start_pos: int) -> str:
        """
        Enhanced article content extraction with better boundary detection
        
        Args:
            article_text (str): Full article text
            start_pos (int): Position after article header
            
        Returns:
            str: Complete article content
        """
        # Use a more sophisticated approach to find article boundaries
        # Look for complete article patterns: "?úÏà´?êÏ°∞(?úÎ™©)" at the beginning of lines
        article_pattern = re.compile(r'^??d+Ï°??:\([^)]*\))?', re.MULTILINE)
        
        # Find all article positions
        article_matches = list(article_pattern.finditer(article_text))
        
        if not article_matches:
            # No articles found, return everything from start_pos
            return article_text[start_pos:].strip()
        
        # Find the current article position
        current_article_pos = None
        for i, match in enumerate(article_matches):
            if match.start() >= start_pos:
                current_article_pos = i
                break
        
        if current_article_pos is None:
            # Current article not found, return everything from start_pos
            return article_text[start_pos:].strip()
        
        # Find the next article position
        if current_article_pos + 1 < len(article_matches):
            next_article_start = article_matches[current_article_pos + 1].start()
            complete_content = article_text[start_pos:next_article_start].strip()
        else:
            # No next article found, take until end of text
            complete_content = article_text[start_pos:].strip()
        
        # Validate content completeness
        if self._validate_article_content(complete_content):
            return self._clean_article_content(complete_content)
        else:
            # If validation fails, try alternative extraction method
            return self._extract_with_context_awareness(article_text, start_pos)
    
    def _extract_complete_article_content_v3(self, article_text: str, start_pos: int) -> str:
        """
        Ultimate article content extraction with comprehensive boundary detection
        
        Args:
            article_text (str): Full article text
            start_pos (int): Position after article header
            
        Returns:
            str: Complete article content
        """
        # Strategy 1: Look for next article with line-based detection
        remaining_text = article_text[start_pos:]
        
        # Find the next "?úÏà´?êÏ°∞" that appears at the beginning of a line
        next_article_pattern = re.compile(r'^??d+Ï°?, re.MULTILINE)
        next_match = next_article_pattern.search(remaining_text[1:])  # Skip first character
        
        if next_match:
            # Found next article, extract content up to that point
            end_pos = start_pos + 1 + next_match.start()
            content = article_text[start_pos:end_pos].strip()
        else:
            # No next article found, look for other boundaries
            content = self._find_content_with_alternative_boundaries(article_text, start_pos)
        
        # Clean and validate the content
        cleaned_content = self._clean_article_content(content)
        
        # Final validation - if still incomplete, try more aggressive extraction
        if not self._validate_article_content(cleaned_content):
            cleaned_content = self._extract_with_fallback_strategy(article_text, start_pos)
        
        return cleaned_content
    
    def _extract_complete_article_content_v5(self, article_text: str, start_pos: int) -> str:
        """
        Enhanced article content extraction with improved paragraph completeness detection
        
        Args:
            article_text (str): Full article text
            start_pos (int): Position after article header
            
        Returns:
            str: Complete article content
        """
        # Strategy 1: Look for next article with enhanced pattern matching
        remaining_text = article_text[start_pos:]
        
        # Enhanced pattern to find next article - look for "?úÏà´?êÏ°∞" at line start
        next_article_pattern = re.compile(r'^??d+Ï°?, re.MULTILINE)
        next_match = next_article_pattern.search(remaining_text[1:])  # Skip first character
        
        if next_match:
            # Found next article, extract content up to that point
            end_pos = start_pos + 1 + next_match.start()
            content = article_text[start_pos:end_pos].strip()
        else:
            # No next article found, look for other boundaries
            content = self._find_content_with_alternative_boundaries(article_text, start_pos)
        
        # Enhanced validation: check if content contains incomplete paragraphs
        if self._has_incomplete_paragraphs_v2(content):
            # Try to extend content to include complete paragraphs
            content = self._extend_to_complete_paragraphs_v2(article_text, start_pos, content)
        
        # Clean and validate the content
        cleaned_content = self._clean_article_content(content)
        
        # Final validation - if still incomplete, try more aggressive extraction
        if not self._validate_article_content(cleaned_content):
            cleaned_content = self._extract_with_fallback_strategy(article_text, start_pos)
        
        return cleaned_content
    
    def _extract_complete_article_content_v6(self, article_text: str, start_pos: int) -> str:
        """
        Aggressive article content extraction with comprehensive completeness detection
        
        Args:
            article_text (str): Full article text
            start_pos (int): Position after article header
            
        Returns:
            str: Complete article content
        """
        # Strategy 1: Look for next article with more aggressive pattern matching
        remaining_text = article_text[start_pos:]
        
        # More aggressive pattern to find next article
        next_article_pattern = re.compile(r'^??d+Ï°?, re.MULTILINE)
        next_match = next_article_pattern.search(remaining_text[1:])  # Skip first character
        
        if next_match:
            # Found next article, but be more conservative about where to cut
            end_pos = start_pos + 1 + next_match.start()
            content = article_text[start_pos:end_pos].strip()
            
            # Check if this content looks complete
            if self._is_content_likely_complete(content):
                return self._clean_article_content(content)
            else:
                # Content seems incomplete, try to extend it
                content = self._extend_content_aggressively(article_text, start_pos, content)
        else:
            # No next article found, look for other boundaries
            content = self._find_content_with_alternative_boundaries(article_text, start_pos)
        
        # Enhanced validation: check if content contains incomplete paragraphs
        if self._has_incomplete_paragraphs_v2(content):
            # Try to extend content to include complete paragraphs
            content = self._extend_to_complete_paragraphs_v2(article_text, start_pos, content)
        
        # Clean and validate the content
        cleaned_content = self._clean_article_content(content)
        
        # Final validation - if still incomplete, try more aggressive extraction
        if not self._validate_article_content(cleaned_content):
            cleaned_content = self._extract_with_fallback_strategy(article_text, start_pos)
        
    def _extract_complete_article_content_v8(self, article_text: str, start_pos: int) -> str:
        """
        Enhanced article content extraction with improved boundary detection
        
        Args:
            article_text (str): Full article text (clean text without HTML)
            start_pos (int): Position after article header
            
        Returns:
            str: Complete article content
        """
        logger.debug(f"_extract_complete_article_content_v8 called with start_pos: {start_pos}")
        logger.debug(f"Article text length: {len(article_text)}")
        logger.debug(f"Text from start_pos: {repr(article_text[start_pos:start_pos+100])}")
        
        # Extract content from start position to end of article
        content = article_text[start_pos:].strip()
        
        logger.debug(f"Extracted content length: {len(content)}")
        logger.debug(f"First 200 chars of content: {repr(content[:200])}")
        
        return content
    
    
    def _extract_definition_article_content(self, article_text: str, start_pos: int) -> str:
        """
        Special extraction for definition articles with comprehensive content
        
        Args:
            article_text (str): Full article text
            start_pos (int): Start position
            
        Returns:
            str: Complete definition article content
        """
        remaining_text = article_text[start_pos:]
        
        # For definition articles, be more aggressive in finding complete content
        # Look for patterns that indicate the end of a definition article
        end_patterns = [
            # Look for the next article with more specific patterns
            r'^??Ï°?s*\([^)]+\)',     # ??Ï°??§Î•∏ Î≤ïÎ•†Í≥ºÏùò Í¥ÄÍ≥?
            r'^??Ï°?s*[Í∞Ä-??',       # ??Ï°??§Î•∏
            r'^??Ï°?,                 # ??Ï°?
            
            # Look for other structural boundaries
            r'^??d+Ï°?s*\([^)]+\)',  # Any article with parentheses
            r'^??d+Ï°?s*[Í∞Ä-??',    # Any article with Korean text
            r'^??d+Ï°?,              # Any article
        ]
        
        best_end = len(remaining_text)
        best_score = 0
        
        for pattern in end_patterns:
            matches = list(re.finditer(pattern, remaining_text, re.MULTILINE))
            for match in matches:
                # Check if this would create a more complete content
                potential_end = match.start()
                potential_content = remaining_text[:potential_end].strip()
                
                # Score based on definition article completeness
                score = self._score_definition_completeness(potential_content)
                if score > best_score:
                    best_end = potential_end
                    best_score = score
        
        # Extract content up to the best boundary
        content = remaining_text[:best_end].strip()
        
        # For definition articles, be more lenient with validation
        # If we have a reasonable amount of content, accept it
        if len(content) > 500 and self._has_definition_structure(content):
            return content
        
        # If validation fails, try to extend content more aggressively
        return self._extend_definition_content_aggressive(article_text, start_pos, content)
    
    def _score_definition_completeness(self, content: str) -> int:
        """
        Score definition article completeness
        
        Args:
            content (str): Content to score
            
        Returns:
            int: Completeness score (higher is better)
        """
        score = 0
        
        # Bonus for having numbered items (1., 2., 3., etc.)
        numbered_items = len(re.findall(r'\d+\.', content))
        score += numbered_items * 5
        
        # Bonus for having sub-items (Í∞Ä., ??, ??, etc.)
        sub_items = len(re.findall(r'[Í∞Ä-??\.', content))
        score += sub_items * 3
        
        # Bonus for having paragraph numbers (?? ?? ?? etc.)
        paragraph_numbers = len(re.findall(r'[?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©]', content))
        score += paragraph_numbers * 4
        
        # Bonus for having multiple paragraphs (?? ?? etc.)
        multiple_paragraphs = len(re.findall(r'??????????????????, content))
        score += multiple_paragraphs * 6
        
        # Bonus for content length (definition articles are typically long)
        if len(content) > 2000:
            score += 20
        elif len(content) > 1000:
            score += 15
        elif len(content) > 500:
            score += 10
        
        # Bonus for having "?§ÏùåÍ≥?Í∞ôÎã§" pattern
        if re.search(r'?§ÏùåÍ≥?s*Í∞ôÎã§', content):
            score += 10
        
        # Bonus for having "??Î≤ïÏóê???¨Ïö©?òÎäî ?©Ïñ¥" pattern
        if re.search(r'??s*Î≤ïÏóê??s*?¨Ïö©?òÎäî\s*?©Ïñ¥', content):
            score += 15
        
        # Penalty for incomplete patterns
        if re.search(r'?§Ïùå\s*$', content.strip()):
            score -= 20
        if re.search(r'Í∞?s*$', content.strip()):
            score -= 20
        
        return score
    
    def _validate_definition_content(self, content: str) -> bool:
        """
        Validate definition article content
        
        Args:
            content (str): Content to validate
            
        Returns:
            bool: True if content appears complete
        """
        # Check if content has proper definition structure
        if not re.search(r'?©Ïñ¥??s*???©Ïñ¥??s*?ïÏùò', content):
            return False
        
        # Check if content has numbered items
        if not re.search(r'\d+\.', content):
            return False
        
        # Check if content is long enough (definition articles are typically long)
        if len(content) < 200:
            return False
        
        # Check if content ends properly
        if re.search(r'?§ÏùåÍ≥?s*Í∞ôÎã§\s*$', content.strip()):
            return False  # Incomplete
        
        return True
    
    def _extend_definition_content(self, article_text: str, start_pos: int, current_content: str) -> str:
        """
        Extend definition content to include more complete information
        
        Args:
            article_text (str): Full article text
            start_pos (int): Start position
            current_content (str): Current content
            
        Returns:
            str: Extended content
        """
        # Find the end position of current content
        current_end = start_pos + len(current_content)
        remaining_text = article_text[current_end:]
        
        # Look for patterns that typically end definition articles
        end_patterns = [
            r'??s*??s*Î≤ïÏóê??s*?¨Ïö©?òÎäî\s*?©Ïñ¥??s*?ªÏ?\s*??d+??óê??s*Í∑úÏ†ï??s*Í≤ÉÏùÑ\s*?úÏô∏?òÍ≥†??,
            r'??s*??s*Î≤ïÏóê??s*?¨Ïö©?òÎäî\s*?©Ïñ¥??s*?ªÏ?\s*??d+??óê??s*Í∑úÏ†ï??s*Í≤ÉÏùÑ\s*?úÏô∏?òÍ≥†??,
            r'??d+??óê??s*Í∑úÏ†ï??s*Í≤ÉÏùÑ\s*?úÏô∏?òÍ≥†??,
            r'Ï∂ïÏÇ∞Î≤?*?ÑÏÉùÍ¥ÄÎ¶¨Î≤ï.*?∞Î•∏??,
            r'?∞Î•∏??s*$',
        ]
        
        best_end = len(remaining_text)
        best_score = 0
        
        for pattern in end_patterns:
            matches = list(re.finditer(pattern, remaining_text, re.MULTILINE))
            for match in matches:
                potential_end = match.end()
                potential_content = remaining_text[:potential_end].strip()
                
                score = self._score_definition_completeness(potential_content)
                if score > best_score:
                    best_end = potential_end
                    best_score = score
        
        return remaining_text[:best_end].strip()
    
    def _has_definition_structure(self, content: str) -> bool:
        """
        Check if content has proper definition article structure
        
        Args:
            content (str): Content to check
            
        Returns:
            bool: True if content has definition structure
        """
        # Check for definition article patterns
        definition_patterns = [
            r'?©Ïñ¥??s*?ªÏ?\s*?§ÏùåÍ≥?s*Í∞ôÎã§',
            r'?©Ïñ¥??s*?ïÏùò??s*?§ÏùåÍ≥?s*Í∞ôÎã§',
            r'??s*Î≤ïÏóê??s*?¨Ïö©?òÎäî\s*?©Ïñ¥',
        ]
        
        for pattern in definition_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    def _extend_definition_content_aggressive(self, article_text: str, start_pos: int, current_content: str) -> str:
        """
        More aggressive extension for definition content
        
        Args:
            article_text (str): Full article text
            start_pos (int): Start position
            current_content (str): Current content
            
        Returns:
            str: Extended content
        """
        # Find the end position of current content
        current_end = start_pos + len(current_content)
        remaining_text = article_text[current_end:]
        
        # Look for patterns that typically end definition articles more aggressively
        end_patterns = [
            # Look for the next article (??Ï°?
            r'??Ï°?s*\([^)]+\)',
            r'??Ï°?s*[Í∞Ä-??',
            r'??Ï°?,
            
            # Look for other articles
            r'??d+Ï°?s*\([^)]+\)',
            r'??d+Ï°?s*[Í∞Ä-??',
            r'??d+Ï°?,
            
            # Look for specific ending patterns
            r'??s*??s*Î≤ïÏóê??s*?¨Ïö©?òÎäî\s*?©Ïñ¥??s*?ªÏ?\s*??d+??óê??s*Í∑úÏ†ï??s*Í≤ÉÏùÑ\s*?úÏô∏?òÍ≥†??,
            r'??d+??óê??s*Í∑úÏ†ï??s*Í≤ÉÏùÑ\s*?úÏô∏?òÍ≥†??,
            r'Ï∂ïÏÇ∞Î≤?*?ÑÏÉùÍ¥ÄÎ¶¨Î≤ï.*?∞Î•∏??,
            r'?∞Î•∏??s*$',
        ]
        
        best_end = len(remaining_text)
        best_score = 0
        
        for pattern in end_patterns:
            matches = list(re.finditer(pattern, remaining_text, re.MULTILINE))
            for match in matches:
                potential_end = match.end()
                potential_content = remaining_text[:potential_end].strip()
                
                score = self._score_definition_completeness(potential_content)
                if score > best_score:
                    best_end = potential_end
                    best_score = score
        
        return remaining_text[:best_end].strip()
    
    def _has_incomplete_paragraphs_v3(self, content: str) -> bool:
        """
        Enhanced check for incomplete paragraphs with original text structure awareness
        
        Args:
            content (str): Content to check
            
        Returns:
            bool: True if content appears to have incomplete paragraphs
        """
        # Look for patterns that suggest incomplete content in original text
        incomplete_patterns = [
            r'??*?$',  # Ends with incomplete first paragraph
            r'??*?$',  # Ends with incomplete second paragraph
            r'??*?$',  # Ends with incomplete third paragraph
            r'??*?$',  # Ends with incomplete fourth paragraph
            r'??*?$',  # Ends with incomplete fifth paragraph
            r'??*?$',  # Ends with incomplete sixth paragraph
            r'??*?$',  # Ends with incomplete seventh paragraph
            r'??*?$',  # Ends with incomplete eighth paragraph
            r'??*?$',  # Ends with incomplete ninth paragraph
            r'??*?$',  # Ends with incomplete tenth paragraph
            r'1\.\s*$',  # Ends with incomplete first item
            r'2\.\s*$',  # Ends with incomplete second item
            r'3\.\s*$',  # Ends with incomplete third item
            r'4\.\s*$',  # Ends with incomplete fourth item
            r'5\.\s*$',  # Ends with incomplete fifth item
            r'6\.\s*$',  # Ends with incomplete sixth item
            r'7\.\s*$',  # Ends with incomplete seventh item
            r'8\.\s*$',  # Ends with incomplete eighth item
            r'9\.\s*$',  # Ends with incomplete ninth item
            r'10\.\s*$', # Ends with incomplete tenth item
            r'?§Ïùå\s*$',  # Ends with "?§Ïùå" (incomplete)
            r'Í∞?s*$',   # Ends with "Í∞? (incomplete)
            r'?∏Ïùò\s*$', # Ends with "?∏Ïùò" (incomplete)
            r'?¨Ìï≠\s*$', # Ends with "?¨Ìï≠" (incomplete)
            r'Í≤ΩÏö∞\s*$', # Ends with "Í≤ΩÏö∞" (incomplete)
            r'Î™©Ïùò\s*$', # Ends with "Î™©Ïùò" (incomplete)
            r'Í∞Ä\.\s*$', # Ends with "Í∞Ä." (incomplete)
            r'??.\s*$', # Ends with "??" (incomplete)
            r'??.\s*$', # Ends with "??" (incomplete)
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, content.strip()):
                return True
        
        # Check for paragraph number sequences that suggest missing content
        paragraph_numbers = re.findall(r'[?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©]', content)
        if len(paragraph_numbers) > 0:
            # If we have paragraph numbers but content seems incomplete
            if not re.search(r'?úÎã§\.\s*$|Í∑úÏ†ï?úÎã§\.\s*$|?úÌñâ?úÎã§\.\s*$|Î≥∏Îã§\.\s*$', content.strip()):
                return True
        
        # Check for numbered list sequences that suggest missing content
        numbered_items = re.findall(r'\d+\.', content)
        if len(numbered_items) > 0:
            # If we have numbered items but content seems incomplete
            if not re.search(r'?úÎã§\.\s*$|Í∑úÏ†ï?úÎã§\.\s*$|?úÌñâ?úÎã§\.\s*$|Î≥∏Îã§\.\s*$', content.strip()):
                return True
        
        return False
    
    def _extend_to_complete_paragraphs_v3(self, article_text: str, start_pos: int, current_content: str) -> str:
        """
        Enhanced content extension to include complete paragraphs with original text structure
        
        Args:
            article_text (str): Full article text
            start_pos (int): Start position
            current_content (str): Current content
            
        Returns:
            str: Extended content
        """
        # Find the end position of current content
        current_end = start_pos + len(current_content)
        remaining_text = article_text[current_end:]
        
        # Look for patterns that typically end paragraphs in original text
        paragraph_end_patterns = [
            r'?úÎã§\.\s*$',         # Ends with "?úÎã§."
            r'Í∑úÏ†ï?úÎã§\.\s*$',     # Ends with "Í∑úÏ†ï?úÎã§."
            r'?úÌñâ?úÎã§\.\s*$',     # Ends with "?úÌñâ?úÎã§."
            r'Î≥∏Îã§\.\s*$',         # Ends with "Î≥∏Îã§."
            r'\.\s*$',             # Ends with period
        ]
        
        # Try to find a better end point
        best_end = current_end
        best_score = 0
        
        for pattern in paragraph_end_patterns:
            matches = list(re.finditer(pattern, remaining_text, re.MULTILINE))
            for match in matches:
                # Check if this would create a more complete content
                potential_end = current_end + match.end()
                potential_content = article_text[start_pos:potential_end].strip()
                
                # Score based on content completeness
                score = self._score_content_completeness_v4(potential_content)
                if score > best_score:
                    best_end = potential_end
                    best_score = score
        
        return article_text[start_pos:best_end].strip()
    
    def _score_content_completeness_v4(self, content: str) -> int:
        """
        Enhanced content completeness scoring with original text structure awareness
        
        Args:
            content (str): Content to score
            
        Returns:
            int: Completeness score (higher is better)
        """
        score = 0
        
        # Bonus for proper endings
        if re.search(r'?úÎã§\.\s*$', content.strip()):
            score += 20
        elif re.search(r'Í∑úÏ†ï?úÎã§\.\s*$', content.strip()):
            score += 20
        elif re.search(r'?úÌñâ?úÎã§\.\s*$', content.strip()):
            score += 20
        elif re.search(r'Î≥∏Îã§\.\s*$', content.strip()):
            score += 20
        elif re.search(r'\.\s*$', content.strip()):
            score += 15
        
        # Bonus for complete paragraph structure
        paragraph_count = len(re.findall(r'[?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©]', content))
        score += paragraph_count * 6
        
        # Bonus for complete numbered lists
        numbered_list_count = len(re.findall(r'\d+\.', content))
        score += numbered_list_count * 4
        
        # Bonus for complete alphabet lists (Í∞Ä, ?? ?? etc.)
        alphabet_list_count = len(re.findall(r'[Í∞Ä-??\.', content))
        score += alphabet_list_count * 3
        
        # Bonus for "?§Ïùå Í∞??? pattern completion
        if re.search(r'?§Ïùå\s+Í∞?s+??, content):
            score += 10
        
        # Bonus for content length (longer content is generally more complete)
        if len(content) > 2000:
            score += 15
        elif len(content) > 1000:
            score += 12
        elif len(content) > 500:
            score += 8
        elif len(content) > 200:
            score += 5
        
        # Penalty for incomplete patterns
        if re.search(r'?§Ïùå\s*$', content.strip()):
            score -= 20
        if re.search(r'Í∞?s*$', content.strip()):
            score -= 20
        if re.search(r'?∏Ïùò\s*$', content.strip()):
            score -= 20
        if re.search(r'?¨Ìï≠\s*$', content.strip()):
            score -= 15
        if re.search(r'Í≤ΩÏö∞\s*$', content.strip()):
            score -= 15
        if re.search(r'Î™©Ïùò\s*$', content.strip()):
            score -= 15
        
        return score
    
    def _clean_article_content_v2(self, content: str) -> str:
        """
        Enhanced content cleaning with original text structure awareness
        
        Args:
            content (str): Content to clean
            
        Returns:
            str: Cleaned content
        """
        # Remove excessive whitespace but preserve structure
        content = re.sub(r'\s+', ' ', content)
        
        # Remove HTML-like artifacts that might remain
        content = re.sub(r'<[^>]+>', '', content)
        content = re.sub(r'javascript:[^"\']*', '', content)
        content = re.sub(r'href="[^"]*"', '', content)
        content = re.sub(r'class="[^"]*"', '', content)
        content = re.sub(r'id="[^"]*"', '', content)
        
        # Clean up any remaining artifacts
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        return content
    
    def _validate_article_content_v2(self, content: str) -> bool:
        """
        Enhanced content validation with original text structure awareness
        
        Args:
            content (str): Content to validate
            
        Returns:
            bool: True if content is valid
        """
        # Check if content is too short
        if len(content.strip()) < 10:
            return False
        
        # Check if content contains meaningful Korean text
        korean_chars = len(re.findall(r'[Í∞Ä-??', content))
        if korean_chars < 5:
            return False
        
        # Check if content has proper structure
        if re.search(r'??d+Ï°?, content):
            return True
        
        # Check if content has paragraph structure
        if re.search(r'[?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©]', content):
            return True
        
        # Check if content has numbered list structure
        if re.search(r'\d+\.', content):
            return True
        
        # Check if content has alphabet list structure
        if re.search(r'[Í∞Ä-??\.', content):
            return True
        
        return True
    
    def _extract_with_fallback_strategy_v2(self, article_text: str, start_pos: int) -> str:
        """
        Enhanced fallback extraction strategy with original text structure awareness
        
        Args:
            article_text (str): Full article text
            start_pos (int): Start position
            
        Returns:
            str: Extracted content
        """
        # Try to find content by looking for complete sentences
        remaining_text = article_text[start_pos:]
        
        # Look for patterns that typically end articles in original text
        end_patterns = [
            r'?úÎã§\.\s*$',         # Ends with "?úÎã§."
            r'Í∑úÏ†ï?úÎã§\.\s*$',     # Ends with "Í∑úÏ†ï?úÎã§."
            r'?úÌñâ?úÎã§\.\s*$',     # Ends with "?úÌñâ?úÎã§."
            r'Î≥∏Îã§\.\s*$',         # Ends with "Î≥∏Îã§."
            r'\.\s*$',             # Ends with period
        ]
        
        # Find the longest content that ends properly
        best_content = remaining_text
        best_score = 0
        
        for pattern in end_patterns:
            matches = list(re.finditer(pattern, remaining_text, re.MULTILINE))
            for match in matches:
                content = remaining_text[:match.end()].strip()
                score = self._score_content_completeness_v4(content)
                if score > best_score:
                    best_content = content
                    best_score = score
        
        return best_content
        """
        HTML-aware article content extraction using HTML structure for better boundary detection
        
        Args:
            article_text (str): Full article text (may contain HTML)
            start_pos (int): Position after article header
            
        Returns:
            str: Complete article content
        """
        # Strategy 1: Use HTML structure if available
        if self._has_html_structure(article_text):
            return self._extract_with_html_structure(article_text, start_pos)
        
        # Strategy 2: Enhanced text-based extraction with better boundary detection
        remaining_text = article_text[start_pos:]
        
        # Look for next article with more sophisticated pattern matching
        next_article_pattern = re.compile(r'^??d+Ï°?, re.MULTILINE)
        next_match = next_article_pattern.search(remaining_text[1:])  # Skip first character
        
        if next_match:
            # Found next article, but validate completeness
            end_pos = start_pos + 1 + next_match.start()
            content = article_text[start_pos:end_pos].strip()
            
            # Check if content is complete based on structure analysis
            if self._is_content_structurally_complete(content):
                return self._clean_article_content(content)
            else:
                # Content seems incomplete, try to extend it
                content = self._extend_content_with_structure_analysis(article_text, start_pos, content)
        else:
            # No next article found, look for other boundaries
            content = self._find_content_with_alternative_boundaries(article_text, start_pos)
        
        # Enhanced validation: check if content contains incomplete paragraphs
        if self._has_incomplete_paragraphs_v2(content):
            # Try to extend content to include complete paragraphs
            content = self._extend_to_complete_paragraphs_v2(article_text, start_pos, content)
        
        # Clean and validate the content
        cleaned_content = self._clean_article_content(content)
        
        # Final validation - if still incomplete, try more aggressive extraction
        if not self._validate_article_content(cleaned_content):
            cleaned_content = self._extract_with_fallback_strategy(article_text, start_pos)
        
        return cleaned_content
    
    def _has_html_structure(self, text: str) -> bool:
        """
        Check if text contains HTML structure that can be used for parsing
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if HTML structure is available
        """
        # Look for common HTML tags that indicate structure
        html_indicators = [
            r'<div[^>]*class[^>]*>',  # div with class
            r'<p[^>]*>',             # paragraph tags
            r'<span[^>]*>',          # span tags
            r'<br[^>]*>',            # line breaks
            r'<strong[^>]*>',        # strong tags
            r'<em[^>]*>',           # emphasis tags
        ]
        
        for pattern in html_indicators:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _extract_with_html_structure(self, article_text: str, start_pos: int) -> str:
        """
        Extract article content using HTML structure for better boundary detection
        
        Args:
            article_text (str): Full article text with HTML
            start_pos (int): Position after article header
            
        Returns:
            str: Complete article content
        """
        try:
            from bs4 import BeautifulSoup
            
            # Parse HTML
            soup = BeautifulSoup(article_text, 'html.parser')
            
            # Find the current article element
            current_article = self._find_current_article_element(soup, start_pos)
            
            if current_article:
                # Extract content from the article element
                content = self._extract_content_from_element(current_article)
                return self._clean_article_content(content)
            else:
                # Fallback to text-based extraction
                return self._extract_complete_article_content_v6(article_text, start_pos)
                
        except Exception as e:
            logger.debug(f"HTML structure extraction failed: {e}")
            # Fallback to text-based extraction
            return self._extract_complete_article_content_v6(article_text, start_pos)
    
    def _find_current_article_element(self, soup, start_pos: int):
        """
        Find the HTML element containing the current article
        
        Args:
            soup: BeautifulSoup object
            start_pos (int): Position after article header
            
        Returns:
            Element or None: The article element
        """
        # Look for elements that might contain article content
        # This is a simplified approach - in practice, you'd need to analyze
        # the specific HTML structure of the law documents
        
        # Look for div elements that might contain article content
        divs = soup.find_all('div')
        for div in divs:
            if div.get_text().strip():
                # Check if this div contains article-like content
                text = div.get_text()
                if re.search(r'??d+Ï°?, text):
                    return div
        
        # Look for paragraph elements
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            if p.get_text().strip():
                text = p.get_text()
                if re.search(r'??d+Ï°?, text):
                    return p
        
        return None
    
    def _extract_content_from_element(self, element):
        """
        Extract content from an HTML element
        
        Args:
            element: BeautifulSoup element
            
        Returns:
            str: Extracted content
        """
        # Extract text content while preserving structure
        content = element.get_text(separator='\n', strip=True)
        
        # Clean up the content
        content = re.sub(r'\n+', '\n', content)  # Remove multiple newlines
        content = content.strip()
        
        return content
    
    def _is_content_structurally_complete(self, content: str) -> bool:
        """
        Check if content is structurally complete based on various indicators
        
        Args:
            content (str): Content to check
            
        Returns:
            bool: True if content is structurally complete
        """
        # Check for proper endings
        if re.search(r'?úÎã§\.\s*$|Í∑úÏ†ï?úÎã§\.\s*$|?úÌñâ?úÎã§\.\s*$|Î≥∏Îã§\.\s*$', content.strip()):
            return True
        
        # Check for complete numbered lists (1. 2. 3. etc.)
        numbered_items = re.findall(r'\d+\.', content)
        if len(numbered_items) > 0:
            # Check if the list seems complete
            numbers = [int(item.replace('.', '')) for item in numbered_items]
            if len(numbers) > 1:
                # If we have multiple items, check if they're sequential
                if numbers == list(range(min(numbers), max(numbers) + 1)):
                    return True
        
        # Check for complete paragraph structure (?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©)
        paragraph_numbers = re.findall(r'[?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©]', content)
        if len(paragraph_numbers) > 0:
            # Check if paragraphs seem complete
            return True
        
        # Check for "?§Ïùå Í∞??? pattern completion
        if re.search(r'?§Ïùå\s+Í∞?s+??, content):
            # Look for numbered items after this pattern
            if re.search(r'\d+\.', content):
                return True
        
        # Check content length and structure
        if len(content) > 300:  # Longer content is more likely to be complete
            return True
        
        return False
    
    def _extend_content_with_structure_analysis(self, article_text: str, start_pos: int, current_content: str) -> str:
        """
        Extend content using structure analysis for better completeness
        
        Args:
            article_text (str): Full article text
            start_pos (int): Start position
            current_content (str): Current content
            
        Returns:
            str: Extended content
        """
        # Find the end position of current content
        current_end = start_pos + len(current_content)
        remaining_text = article_text[current_end:]
        
        # Look for structural patterns that indicate article completion
        structural_patterns = [
            r'?úÎã§\.\s*$',         # Ends with "?úÎã§."
            r'Í∑úÏ†ï?úÎã§\.\s*$',     # Ends with "Í∑úÏ†ï?úÎã§."
            r'?úÌñâ?úÎã§\.\s*$',     # Ends with "?úÌñâ?úÎã§."
            r'Î≥∏Îã§\.\s*$',         # Ends with "Î≥∏Îã§."
            r'\.\s*$',             # Ends with period
        ]
        
        # Try to find a better end point based on structure
        best_end = current_end
        best_score = 0
        
        for pattern in structural_patterns:
            matches = list(re.finditer(pattern, remaining_text, re.MULTILINE))
            for match in matches:
                # Check if this would create a more complete content
                potential_end = current_end + match.end()
                potential_content = article_text[start_pos:potential_end].strip()
                
                # Score based on structural completeness
                score = self._score_structural_completeness(potential_content)
                if score > best_score:
                    best_end = potential_end
                    best_score = score
        
        return article_text[start_pos:best_end].strip()
    
    def _score_structural_completeness(self, content: str) -> int:
        """
        Score content based on structural completeness indicators
        
        Args:
            content (str): Content to score
            
        Returns:
            int: Completeness score (higher is better)
        """
        score = 0
        
        # Bonus for proper endings
        if re.search(r'?úÎã§\.\s*$', content.strip()):
            score += 25
        elif re.search(r'Í∑úÏ†ï?úÎã§\.\s*$', content.strip()):
            score += 25
        elif re.search(r'?úÌñâ?úÎã§\.\s*$', content.strip()):
            score += 25
        elif re.search(r'Î≥∏Îã§\.\s*$', content.strip()):
            score += 25
        elif re.search(r'\.\s*$', content.strip()):
            score += 20
        
        # Bonus for complete paragraph structure
        paragraph_count = len(re.findall(r'[?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©]', content))
        score += paragraph_count * 6
        
        # Bonus for complete numbered lists
        numbered_list_count = len(re.findall(r'\d+\.', content))
        score += numbered_list_count * 4
        
        # Bonus for "?§Ïùå Í∞??? pattern completion
        if re.search(r'?§Ïùå\s+Í∞?s+??, content):
            score += 10
        
        # Bonus for content length (longer content is generally more complete)
        if len(content) > 1500:
            score += 15
        elif len(content) > 1000:
            score += 12
        elif len(content) > 500:
            score += 8
        elif len(content) > 200:
            score += 5
        
        # Penalty for incomplete patterns
        if re.search(r'?§Ïùå\s*$', content.strip()):
            score -= 20
        if re.search(r'Í∞?s*$', content.strip()):
            score -= 20
        if re.search(r'?∏Ïùò\s*$', content.strip()):
            score -= 20
        if re.search(r'?¨Ìï≠\s*$', content.strip()):
            score -= 15
        if re.search(r'Í≤ΩÏö∞\s*$', content.strip()):
            score -= 15
        
        return score
    
    def _is_content_likely_complete(self, content: str) -> bool:
        """
        Check if content is likely complete based on various indicators
        
        Args:
            content (str): Content to check
            
        Returns:
            bool: True if content is likely complete
        """
        # Check for proper endings
        if re.search(r'?úÎã§\.\s*$|Í∑úÏ†ï?úÎã§\.\s*$|?úÌñâ?úÎã§\.\s*$|Î≥∏Îã§\.\s*$', content.strip()):
            return True
        
        # Check for complete numbered lists
        numbered_items = re.findall(r'\d+\.', content)
        if len(numbered_items) > 0:
            # If we have numbered items, check if they seem complete
            last_number = int(numbered_items[-1].replace('.', ''))
            if last_number > 1:  # If we have more than one item, likely complete
                return True
        
        # Check for paragraph completeness
        paragraph_numbers = re.findall(r'[?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©]', content)
        if len(paragraph_numbers) > 0:
            # If we have paragraph numbers, check if they seem complete
            return True
        
        # Check content length
        if len(content) > 200:  # Longer content is more likely to be complete
            return True
        
        return False
    
    def _extend_content_aggressively(self, article_text: str, start_pos: int, current_content: str) -> str:
        """
        Aggressively extend content to ensure completeness
        
        Args:
            article_text (str): Full article text
            start_pos (int): Start position
            current_content (str): Current content
            
        Returns:
            str: Extended content
        """
        # Find the end position of current content
        current_end = start_pos + len(current_content)
        remaining_text = article_text[current_end:]
        
        # Look for patterns that typically end articles more aggressively
        end_patterns = [
            r'?úÎã§\.\s*$',         # Ends with "?úÎã§."
            r'Í∑úÏ†ï?úÎã§\.\s*$',     # Ends with "Í∑úÏ†ï?úÎã§."
            r'?úÌñâ?úÎã§\.\s*$',     # Ends with "?úÌñâ?úÎã§."
            r'Î≥∏Îã§\.\s*$',         # Ends with "Î≥∏Îã§."
            r'\.\s*$',             # Ends with period
        ]
        
        # Try to find a better end point
        best_end = current_end
        best_score = 0
        
        for pattern in end_patterns:
            matches = list(re.finditer(pattern, remaining_text, re.MULTILINE))
            for match in matches:
                # Check if this would create a more complete content
                potential_end = current_end + match.end()
                potential_content = article_text[start_pos:potential_end].strip()
                
                # Score based on content completeness
                score = self._score_content_completeness_v3(potential_content)
                if score > best_score:
                    best_end = potential_end
                    best_score = score
        
        return article_text[start_pos:best_end].strip()
    
    def _score_content_completeness_v3(self, content: str) -> int:
        """
        Enhanced content completeness scoring with more aggressive scoring
        
        Args:
            content (str): Content to score
            
        Returns:
            int: Completeness score (higher is better)
        """
        score = 0
        
        # Bonus for proper endings
        if re.search(r'?úÎã§\.\s*$', content.strip()):
            score += 20
        elif re.search(r'Í∑úÏ†ï?úÎã§\.\s*$', content.strip()):
            score += 20
        elif re.search(r'?úÌñâ?úÎã§\.\s*$', content.strip()):
            score += 20
        elif re.search(r'Î≥∏Îã§\.\s*$', content.strip()):
            score += 20
        elif re.search(r'\.\s*$', content.strip()):
            score += 15
        
        # Bonus for complete paragraph structure
        paragraph_count = len(re.findall(r'[?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©]', content))
        score += paragraph_count * 5
        
        # Bonus for complete numbered lists
        numbered_list_count = len(re.findall(r'\d+\.', content))
        score += numbered_list_count * 3
        
        # Bonus for content length (longer content is generally more complete)
        if len(content) > 1000:
            score += 10
        elif len(content) > 500:
            score += 7
        elif len(content) > 200:
            score += 5
        
        # Penalty for incomplete patterns
        if re.search(r'?§Ïùå\s*$', content.strip()):
            score -= 15
        if re.search(r'Í∞?s*$', content.strip()):
            score -= 15
        if re.search(r'?∏Ïùò\s*$', content.strip()):
            score -= 15
        if re.search(r'?¨Ìï≠\s*$', content.strip()):
            score -= 10
        if re.search(r'Í≤ΩÏö∞\s*$', content.strip()):
            score -= 10
        
        return score
    
    def _has_incomplete_paragraphs_v2(self, content: str) -> bool:
        """
        Enhanced check for incomplete paragraphs with better pattern detection
        
        Args:
            content (str): Content to check
            
        Returns:
            bool: True if content appears to have incomplete paragraphs
        """
        # Look for patterns that suggest incomplete content
        incomplete_patterns = [
            r'??*?$',  # Ends with incomplete first paragraph
            r'??*?$',  # Ends with incomplete second paragraph
            r'??*?$',  # Ends with incomplete third paragraph
            r'??*?$',  # Ends with incomplete fourth paragraph
            r'??*?$',  # Ends with incomplete fifth paragraph
            r'??*?$',  # Ends with incomplete sixth paragraph
            r'??*?$',  # Ends with incomplete seventh paragraph
            r'??*?$',  # Ends with incomplete eighth paragraph
            r'??*?$',  # Ends with incomplete ninth paragraph
            r'??*?$',  # Ends with incomplete tenth paragraph
            r'?§Ïùå\s*$',  # Ends with "?§Ïùå" (incomplete)
            r'Í∞?s*$',   # Ends with "Í∞? (incomplete)
            r'?∏Ïùò\s*$', # Ends with "?∏Ïùò" (incomplete)
            r'?¨Ìï≠\s*$', # Ends with "?¨Ìï≠" (incomplete)
            r'Í≤ΩÏö∞\s*$', # Ends with "Í≤ΩÏö∞" (incomplete)
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, content.strip()):
                return True
        
        # Check for paragraph number sequences that suggest missing content
        paragraph_numbers = re.findall(r'[?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©]', content)
        if len(paragraph_numbers) > 0:
            # If we have paragraph numbers but content seems incomplete
            if not re.search(r'?úÎã§\.\s*$|Í∑úÏ†ï?úÎã§\.\s*$|?úÌñâ?úÎã§\.\s*$|Î≥∏Îã§\.\s*$', content.strip()):
                return True
        
        return False
    
    def _extend_to_complete_paragraphs_v2(self, article_text: str, start_pos: int, current_content: str) -> str:
        """
        Enhanced content extension to include complete paragraphs
        
        Args:
            article_text (str): Full article text
            start_pos (int): Start position
            current_content (str): Current content
            
        Returns:
            str: Extended content
        """
        # Find the end position of current content
        current_end = start_pos + len(current_content)
        remaining_text = article_text[current_end:]
        
        # Look for patterns that typically end paragraphs
        paragraph_end_patterns = [
            r'?úÎã§\.\s*$',         # Ends with "?úÎã§."
            r'Í∑úÏ†ï?úÎã§\.\s*$',     # Ends with "Í∑úÏ†ï?úÎã§."
            r'?úÌñâ?úÎã§\.\s*$',     # Ends with "?úÌñâ?úÎã§."
            r'Î≥∏Îã§\.\s*$',         # Ends with "Î≥∏Îã§."
            r'\.\s*$',             # Ends with period
        ]
        
        # Try to find a better end point
        best_end = current_end
        best_score = 0
        
        for pattern in paragraph_end_patterns:
            matches = list(re.finditer(pattern, remaining_text, re.MULTILINE))
            for match in matches:
                # Check if this would create a more complete content
                potential_end = current_end + match.end()
                potential_content = article_text[start_pos:potential_end].strip()
                
                # Score based on content completeness
                score = self._score_content_completeness_v2(potential_content)
                if score > best_score:
                    best_end = potential_end
                    best_score = score
        
        return article_text[start_pos:best_end].strip()
    
    def _score_content_completeness_v2(self, content: str) -> int:
        """
        Enhanced content completeness scoring
        
        Args:
            content (str): Content to score
            
        Returns:
            int: Completeness score (higher is better)
        """
        score = 0
        
        # Bonus for proper endings
        if re.search(r'?úÎã§\.\s*$', content.strip()):
            score += 15
        elif re.search(r'Í∑úÏ†ï?úÎã§\.\s*$', content.strip()):
            score += 15
        elif re.search(r'?úÌñâ?úÎã§\.\s*$', content.strip()):
            score += 15
        elif re.search(r'Î≥∏Îã§\.\s*$', content.strip()):
            score += 15
        elif re.search(r'\.\s*$', content.strip()):
            score += 10
        
        # Bonus for complete paragraph structure
        paragraph_count = len(re.findall(r'[?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©]', content))
        score += paragraph_count * 3
        
        # Bonus for complete numbered lists
        numbered_list_count = len(re.findall(r'\d+\.', content))
        score += numbered_list_count * 2
        
        # Penalty for incomplete patterns
        if re.search(r'?§Ïùå\s*$', content.strip()):
            score -= 10
        if re.search(r'Í∞?s*$', content.strip()):
            score -= 10
        if re.search(r'?∏Ïùò\s*$', content.strip()):
            score -= 10
        if re.search(r'?¨Ìï≠\s*$', content.strip()):
            score -= 5
        if re.search(r'Í≤ΩÏö∞\s*$', content.strip()):
            score -= 5
        
        # Bonus for content length (longer content is generally more complete)
        if len(content) > 500:
            score += 5
        elif len(content) > 200:
            score += 3
        
        return score
    
    def _extract_complete_article_content_v4(self, article_text: str, start_pos: int) -> str:
        """
        Enhanced article content extraction with improved paragraph detection
        
        Args:
            article_text (str): Full article text
            start_pos (int): Position after article header
            
        Returns:
            str: Complete article content
        """
        # Strategy 1: Look for next article with enhanced pattern matching
        remaining_text = article_text[start_pos:]
        
        # Enhanced pattern to find next article - look for "?úÏà´?êÏ°∞" at line start
        next_article_pattern = re.compile(r'^??d+Ï°?, re.MULTILINE)
        next_match = next_article_pattern.search(remaining_text[1:])  # Skip first character
        
        if next_match:
            # Found next article, extract content up to that point
            end_pos = start_pos + 1 + next_match.start()
            content = article_text[start_pos:end_pos].strip()
        else:
            # No next article found, look for other boundaries
            content = self._find_content_with_alternative_boundaries(article_text, start_pos)
        
        # Additional validation: check if content contains incomplete paragraphs
        if self._has_incomplete_paragraphs(content):
            # Try to extend content to include complete paragraphs
            content = self._extend_to_complete_paragraphs(article_text, start_pos, content)
        
        # Clean and validate the content
        cleaned_content = self._clean_article_content(content)
        
        # Final validation - if still incomplete, try more aggressive extraction
        if not self._validate_article_content(cleaned_content):
            cleaned_content = self._extract_with_fallback_strategy(article_text, start_pos)
        
        return cleaned_content
    
    def _has_incomplete_paragraphs(self, content: str) -> bool:
        """
        Check if content has incomplete paragraphs (missing sub-items)
        
        Args:
            content (str): Content to check
            
        Returns:
            bool: True if content appears to have incomplete paragraphs
        """
        # Look for patterns that suggest incomplete content
        incomplete_patterns = [
            r'??*?$',  # Ends with incomplete first paragraph
            r'??*?$',  # Ends with incomplete second paragraph
            r'??*?$',  # Ends with incomplete third paragraph
            r'??*?$',  # Ends with incomplete fourth paragraph
            r'??*?$',  # Ends with incomplete fifth paragraph
            r'?§Ïùå\s*$',  # Ends with "?§Ïùå" (incomplete)
            r'Í∞?s*$',   # Ends with "Í∞? (incomplete)
            r'?∏Ïùò\s*$', # Ends with "?∏Ïùò" (incomplete)
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, content.strip()):
                return True
        
        return False
    
    def _extend_to_complete_paragraphs(self, article_text: str, start_pos: int, current_content: str) -> str:
        """
        Extend content to include complete paragraphs
        
        Args:
            article_text (str): Full article text
            start_pos (int): Start position
            current_content (str): Current content
            
        Returns:
            str: Extended content
        """
        # Find the end position of current content
        current_end = start_pos + len(current_content)
        remaining_text = article_text[current_end:]
        
        # Look for patterns that typically end paragraphs
        paragraph_end_patterns = [
            r'?úÎã§\.\s*$',         # Ends with "?úÎã§."
            r'Í∑úÏ†ï?úÎã§\.\s*$',     # Ends with "Í∑úÏ†ï?úÎã§."
            r'?úÌñâ?úÎã§\.\s*$',     # Ends with "?úÌñâ?úÎã§."
            r'Î≥∏Îã§\.\s*$',         # Ends with "Î≥∏Îã§."
            r'\.\s*$',             # Ends with period
        ]
        
        # Try to find a better end point
        best_end = current_end
        best_score = 0
        
        for pattern in paragraph_end_patterns:
            matches = list(re.finditer(pattern, remaining_text, re.MULTILINE))
            for match in matches:
                # Check if this would create a more complete content
                potential_end = current_end + match.end()
                potential_content = article_text[start_pos:potential_end].strip()
                
                # Score based on content completeness
                score = self._score_content_completeness(potential_content)
                if score > best_score:
                    best_end = potential_end
                    best_score = score
        
        return article_text[start_pos:best_end].strip()
    
    def _score_content_completeness(self, content: str) -> int:
        """
        Score content completeness
        
        Args:
            content (str): Content to score
            
        Returns:
            int: Completeness score (higher is better)
        """
        score = 0
        
        # Bonus for proper endings
        if re.search(r'?úÎã§\.\s*$', content.strip()):
            score += 10
        elif re.search(r'Í∑úÏ†ï?úÎã§\.\s*$', content.strip()):
            score += 10
        elif re.search(r'?úÌñâ?úÎã§\.\s*$', content.strip()):
            score += 10
        elif re.search(r'Î≥∏Îã§\.\s*$', content.strip()):
            score += 10
        elif re.search(r'\.\s*$', content.strip()):
            score += 5
        
        # Bonus for complete paragraph structure
        paragraph_count = len(re.findall(r'[?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©]', content))
        score += paragraph_count * 2
        
        # Penalty for incomplete patterns
        if re.search(r'?§Ïùå\s*$', content.strip()):
            score -= 5
        if re.search(r'Í∞?s*$', content.strip()):
            score -= 5
        if re.search(r'?∏Ïùò\s*$', content.strip()):
            score -= 5
        
        return score
    
    def _find_content_with_alternative_boundaries(self, article_text: str, start_pos: int) -> str:
        """
        Find content using alternative boundary detection methods
        
        Args:
            article_text (str): Full article text
            start_pos (int): Start position
            
        Returns:
            str: Extracted content
        """
        remaining_text = article_text[start_pos:]
        
        # Look for common document boundaries
        boundary_patterns = [
            r'^Î∂ÄÏπ?,              # Supplementary provisions
            r'^Î≥ÑÌëú',              # Attached tables
            r'^Î≥ÑÏ?',              # Attached forms
            r'^??d+Ï°?,           # Next article (fallback)
        ]
        
        for pattern in boundary_patterns:
            match = re.search(pattern, remaining_text, re.MULTILINE)
            if match:
                end_pos = start_pos + match.start()
                return article_text[start_pos:end_pos].strip()
        
        # If no boundary found, return everything from start_pos
        return article_text[start_pos:].strip()
    
    def _extract_with_fallback_strategy(self, article_text: str, start_pos: int) -> str:
        """
        Fallback extraction strategy when normal methods fail
        
        Args:
            article_text (str): Full article text
            start_pos (int): Start position
            
        Returns:
            str: Extracted content
        """
        # Try to find content by looking for complete sentences
        remaining_text = article_text[start_pos:]
        
        # Look for patterns that typically end articles
        end_patterns = [
            r'?úÎã§\.\s*$',         # Ends with "?úÎã§."
            r'Í∑úÏ†ï?úÎã§\.\s*$',     # Ends with "Í∑úÏ†ï?úÎã§."
            r'?úÌñâ?úÎã§\.\s*$',     # Ends with "?úÌñâ?úÎã§."
            r'\.\s*$',             # Ends with period
        ]
        
        # Find the longest content that ends properly
        best_content = remaining_text
        best_score = 0
        
        for pattern in end_patterns:
            matches = list(re.finditer(pattern, remaining_text, re.MULTILINE))
            for match in matches:
                content = remaining_text[:match.end()].strip()
                if len(content) > best_score:
                    best_content = content
                    best_score = len(content)
        
        return best_content
    
    def _validate_article_content(self, content: str) -> bool:
        """
        Validate if article content appears complete
        
        Args:
            content (str): Content to validate
            
        Returns:
            bool: True if content appears complete
        """
        if not content or len(content.strip()) < 10:
            return False
        
        # Check for common incomplete patterns
        incomplete_patterns = [
            r'??^??*$',           # Unclosed quotation marks
            r'\([^)]*$',            # Unclosed parentheses
            r'\d+\.\s*$',           # Ends with incomplete numbered list
            r'?§Ïùå\s*$',            # Ends with "?§Ïùå" (incomplete)
            r'Í∞?s*$',              # Ends with "Í∞? (incomplete)
            r'?∏Ïùò\s*$',            # Ends with "?∏Ïùò" (incomplete)
            r'?¨Ìï≠\s*$',            # Ends with "?¨Ìï≠" (incomplete)
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, content.strip()):
                return False
        
        # Check for proper ending patterns
        proper_endings = [
            r'?úÎã§\.\s*$',          # Ends with "?úÎã§."
            r'Í∑úÏ†ï?úÎã§\.\s*$',      # Ends with "Í∑úÏ†ï?úÎã§."
            r'?úÌñâ?úÎã§\.\s*$',      # Ends with "?úÌñâ?úÎã§."
            r'\.\s*$',              # Ends with period
        ]
        
        for pattern in proper_endings:
            if re.search(pattern, content.strip()):
                return True
        
        # If no proper ending found but content is substantial, consider it valid
        return len(content.strip()) > 50
    
    def _extract_with_context_awareness(self, article_text: str, start_pos: int) -> str:
        """
        Extract article content with context awareness
        
        Args:
            article_text (str): Full article text
            start_pos (int): Start position
            
        Returns:
            str: Complete article content
        """
        # Look for the next article with more context
        remaining_text = article_text[start_pos:]
        
        # Find the next "?úÏà´?êÏ°∞" pattern that appears at the beginning of a line
        next_article_pattern = re.compile(r'^??d+Ï°?, re.MULTILINE)
        next_match = next_article_pattern.search(remaining_text[1:])  # Skip first character
        
        if next_match:
            # Found next article, extract content up to that point
            end_pos = start_pos + 1 + next_match.start()
            content = article_text[start_pos:end_pos].strip()
        else:
            # No next article found, take everything from start_pos to end
            content = article_text[start_pos:].strip()
        
        return self._clean_article_content(content)
    
    def _is_content_truncated(self, content: str) -> bool:
        """
        Check if content appears to be truncated
        
        Args:
            content (str): Content to check
            
        Returns:
            bool: True if content appears truncated
        """
        if not content:
            return True
        
        # Check for common truncation patterns
        truncation_patterns = [
            r'??^??*$',  # Unclosed quotation marks
            r'\([^)]*$',   # Unclosed parentheses
            r'\d+\.\s*$',  # Ends with incomplete numbered list
            r'?§Ïùå\s*$',   # Ends with "?§Ïùå" (incomplete)
            r'Í∞?s*$',     # Ends with "Í∞? (incomplete)
        ]
        
        for pattern in truncation_patterns:
            if re.search(pattern, content.strip()):
                return True
        
        return False
    
    def _find_better_content_boundary(self, article_text: str, start_pos: int, current_content: str) -> int:
        """
        Find a better content boundary when content appears truncated
        
        Args:
            article_text (str): Full article text
            start_pos (int): Start position
            current_content (str): Current content that might be truncated
            
        Returns:
            int: Better end position
        """
        # Look for common article endings
        ending_patterns = [
            r'\.\s*$',           # Ends with period
            r'?úÎã§\.\s*$',       # Ends with "?úÎã§."
            r'?úÎã§\s*$',         # Ends with "?úÎã§"
            r'Í∑úÏ†ï?úÎã§\.\s*$',   # Ends with "Í∑úÏ†ï?úÎã§."
            r'?úÌñâ?úÎã§\.\s*$',   # Ends with "?úÌñâ?úÎã§."
        ]
        
        # Search backwards from the current end to find a better boundary
        current_end = start_pos + len(current_content)
        
        # Look for the next article pattern but with more context
        remaining_text = article_text[current_end:]
        
        # Find patterns that typically indicate end of article
        end_patterns = [
            r'??d+Ï°?,           # Next article
            r'Î∂ÄÏπ?,              # Supplementary provisions
            r'Î≥ÑÌëú',              # Attached tables
            r'Î≥ÑÏ?',              # Attached forms
        ]
        
        for pattern in end_patterns:
            match = re.search(pattern, remaining_text)
            if match:
                return current_end + match.start()
        
        # If no better boundary found, return current end
        return current_end
    
    def _clean_article_content(self, content: str) -> str:
        """
        Clean article content while preserving paragraph structure
        
        Args:
            content (str): Raw article content
            
        Returns:
            str: Cleaned article content
        """
        # Remove common prefixes that might interfere
        prefixes_to_remove = [
            r'^??s+',
            r'^?§Ïùå\s+',
            r'^?§ÏùåÍ≥?s+Í∞ôÎã§\s*',
        ]
        
        for prefix in prefixes_to_remove:
            content = re.sub(prefix, '', content)
        
        # Remove HTML tags and unwanted elements
        content = re.sub(r'<[^>]+>', '', content)  # Remove HTML tags
        content = re.sub(r'&nbsp;', ' ', content)  # Replace &nbsp; with space
        content = re.sub(r'&amp;', '&', content)   # Replace &amp; with &
        content = re.sub(r'&lt;', '<', content)    # Replace &lt; with <
        content = re.sub(r'&gt;', '>', content)    # Replace &gt; with >
        content = re.sub(r'&quot;', '"', content)  # Replace &quot; with "
        content = re.sub(r'&apos;', "'", content)  # Replace &apos; with '
        
        # Remove control characters (both actual and escaped)
        # Actual control characters
        content = content.replace('\n', ' ')  # Replace actual newline with space
        content = content.replace('\t', ' ')  # Replace actual tab with space
        content = content.replace('\r', ' ')  # Replace actual carriage return with space
        content = content.replace('\f', ' ')  # Replace form feed with space
        content = content.replace('\v', ' ')  # Replace vertical tab with space
        
        # Escaped control characters
        content = content.replace('\\n', ' ')  # Replace escaped newline with space
        content = content.replace('\\t', ' ')  # Replace escaped tab with space
        content = content.replace('\\r', ' ')  # Replace escaped carriage return with space
        content = content.replace('\\"', '"')  # Replace escaped quotes
        content = content.replace("\\'", "'")  # Replace escaped single quotes
        content = content.replace('\\\\', '\\')  # Replace escaped backslashes
        
        # Remove other control characters (ASCII 0-31 except space)
        import string
        control_chars = ''.join(chr(i) for i in range(32) if chr(i) not in string.whitespace)
        for char in control_chars:
            content = content.replace(char, ' ')
        
        # Clean up whitespace but preserve paragraph structure
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content
    
    def _create_complete_article_content(self, article_number: str, article_title: str, main_content: str, sub_articles: List[Dict[str, Any]]) -> str:
        """
        Create complete article content with all paragraphs, sub-paragraphs, and items properly formatted
        
        Args:
            article_number (str): Article number (e.g., "??Ï°?)
            article_title (str): Article title
            main_content (str): Main content
            sub_articles (List[Dict[str, Any]]): List of sub-articles
            
        Returns:
            str: Complete formatted article content
        """
        # Start with article header
        complete_content = f"{article_number}({article_title})"
        
        # Group sub_articles by type and number for hierarchical structure
        hang_items = [item for item in sub_articles if item.get('type') == '??]
        ho_items = [item for item in sub_articles if item.get('type') == '??]
        mok_items = [item for item in sub_articles if item.get('type') == 'Î™?]
        
        # Sort by position to maintain order
        hang_items.sort(key=lambda x: x.get('position', 0))
        ho_items.sort(key=lambda x: x.get('position', 0))
        mok_items.sort(key=lambda x: x.get('position', 0))
        
        # Add main content if no structured content exists
        if not hang_items and not ho_items and not mok_items and main_content.strip():
            complete_content += f"\n{main_content.strip()}"
        
        # Add all paragraphs (??
        for hang_item in hang_items:
            paragraph_number = hang_item.get('number', 1)
            paragraph_content = hang_item.get('content', '')
            
            # Convert number to Korean symbol
            korean_symbols = ['', '??, '??, '??, '??, '??, '??, '??, '??, '??, '??, 
                            '??, '??, '??, '??, '??, '??, '??, '??, '??, '??]
            symbol = korean_symbols[paragraph_number] if paragraph_number <= 20 else f"{paragraph_number}."
            
            # Add amendment info if present
            amendment_info = hang_item.get('amendment_info', {})
            if amendment_info.get('has_amendment'):
                if amendment_info.get('amendment_type') == '??†ú':
                    complete_content += f"\n{symbol} ??†ú<{amendment_info.get('amendment_date', '')}>"
                else:
                    complete_content += f"\n{symbol} {paragraph_content} <Í∞úÏ†ï {amendment_info.get('amendment_date', '')}>"
            else:
                complete_content += f"\n{symbol} {paragraph_content}"
            
            # Add ??items that belong to this ??
            hang_position = hang_item.get('position', 0)
            related_ho_items = [ho for ho in ho_items if ho.get('position', 0) > hang_position]
            
            # Find the next ??to determine the boundary
            next_hang_position = float('inf')
            for next_hang in hang_items:
                if next_hang.get('position', 0) > hang_position:
                    next_hang_position = min(next_hang_position, next_hang.get('position', 0))
            
            # Add ??items that are between current ??and next ??
            for ho_item in related_ho_items:
                ho_position = ho_item.get('position', 0)
                if ho_position < next_hang_position:
                    ho_number = ho_item.get('number', 1)
                    ho_content = ho_item.get('content', '')
                    
                    # Add amendment info if present
                    ho_amendment_info = ho_item.get('amendment_info', {})
                    if ho_amendment_info.get('has_amendment'):
                        if ho_amendment_info.get('amendment_type') == '??†ú':
                            complete_content += f"\n  {ho_number}. ??†ú<{ho_amendment_info.get('amendment_date', '')}>"
                        else:
                            complete_content += f"\n  {ho_number}. {ho_content} <Í∞úÏ†ï {ho_amendment_info.get('amendment_date', '')}>"
                    else:
                        complete_content += f"\n  {ho_number}. {ho_content}"
                    
                    # Add Î™?items that belong to this ??
                    related_mok_items = [mok for mok in mok_items if mok.get('position', 0) > ho_position]
                    
                    # Find the next ??to determine the boundary
                    next_ho_position = float('inf')
                    for next_ho in ho_items:
                        if next_ho.get('position', 0) > ho_position:
                            next_ho_position = min(next_ho_position, next_ho.get('position', 0))
                    
                    # Add Î™?items that are between current ??and next ??
                    for mok_item in related_mok_items:
                        mok_position = mok_item.get('position', 0)
                        if mok_position < next_ho_position:
                            mok_letter = mok_item.get('letter', mok_item.get('number', ''))
                            mok_content = mok_item.get('content', '')
                            
                            # Add amendment info if present
                            mok_amendment_info = mok_item.get('amendment_info', {})
                            if mok_amendment_info.get('has_amendment'):
                                if mok_amendment_info.get('amendment_type') == '??†ú':
                                    complete_content += f"\n    {mok_letter}. ??†ú<{mok_amendment_info.get('amendment_date', '')}>"
                                else:
                                    complete_content += f"\n    {mok_letter}. {mok_content} <Í∞úÏ†ï {mok_amendment_info.get('amendment_date', '')}>"
                            else:
                                complete_content += f"\n    {mok_letter}. {mok_content}"
        
        # Add standalone ??items (not under any ??
        standalone_ho_items = []
        for ho_item in ho_items:
            ho_position = ho_item.get('position', 0)
            # Check if this ??is not under any ??
            is_standalone = True
            for hang_item in hang_items:
                hang_position = hang_item.get('position', 0)
                # Find next ??position
                next_hang_position = float('inf')
                for next_hang in hang_items:
                    if next_hang.get('position', 0) > hang_position:
                        next_hang_position = min(next_hang_position, next_hang.get('position', 0))
                
                if hang_position < ho_position < next_hang_position:
                    is_standalone = False
                    break
            
            if is_standalone:
                standalone_ho_items.append(ho_item)
        
        for ho_item in standalone_ho_items:
            ho_number = ho_item.get('number', 1)
            ho_content = ho_item.get('content', '')
            
            # Add amendment info if present
            ho_amendment_info = ho_item.get('amendment_info', {})
            if ho_amendment_info.get('has_amendment'):
                if ho_amendment_info.get('amendment_type') == '??†ú':
                    complete_content += f"\n{ho_number}. ??†ú<{ho_amendment_info.get('amendment_date', '')}>"
                else:
                    complete_content += f"\n{ho_number}. {ho_content} <Í∞úÏ†ï {ho_amendment_info.get('amendment_date', '')}>"
            else:
                complete_content += f"\n{ho_number}. {ho_content}"
        
        # Add standalone Î™?items (not under any ??
        standalone_mok_items = []
        for mok_item in mok_items:
            mok_position = mok_item.get('position', 0)
            # Check if this Î™?is not under any ??
            is_standalone = True
            for ho_item in ho_items:
                ho_position = ho_item.get('position', 0)
                # Find next ??position
                next_ho_position = float('inf')
                for next_ho in ho_items:
                    if next_ho.get('position', 0) > ho_position:
                        next_ho_position = min(next_ho_position, next_ho.get('position', 0))
                
                if ho_position < mok_position < next_ho_position:
                    is_standalone = False
                    break
            
            if is_standalone:
                standalone_mok_items.append(mok_item)
        
        for mok_item in standalone_mok_items:
            mok_letter = mok_item.get('letter', mok_item.get('number', ''))
            mok_content = mok_item.get('content', '')
            
            # Add amendment info if present
            mok_amendment_info = mok_item.get('amendment_info', {})
            if mok_amendment_info.get('has_amendment'):
                if mok_amendment_info.get('amendment_type') == '??†ú':
                    complete_content += f"\n{mok_letter}. ??†ú<{mok_amendment_info.get('amendment_date', '')}>"
                else:
                    complete_content += f"\n{mok_letter}. {mok_content} <Í∞úÏ†ï {mok_amendment_info.get('amendment_date', '')}>"
            else:
                complete_content += f"\n{mok_letter}. {mok_content}"
        
        return complete_content.strip()
    
    def _extract_main_content(self, article_text: str, start_pos: int) -> str:
        """
        Extract main content from article text
        
        Args:
            article_text (str): Full article text
            start_pos (int): Position after article header
            
        Returns:
            str: Main content text
        """
        content = article_text[start_pos:].strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            r'^??s+',
            r'^?§Ïùå\s+',
            r'^?§ÏùåÍ≥?s+Í∞ôÎã§\s*',
            r'^?§Ïùå\s+Í∞?s+?∏Ï?\s+Í∞ôÎã§\s*',
        ]
        
        for prefix in prefixes_to_remove:
            content = re.sub(prefix, '', content)
        
        return content.strip()
    
    def _validate_sub_article_data(self, number: int, content: str, sub_type: str) -> bool:
        """
        Validate sub-article data to avoid false matches
        
        Args:
            number (int): Sub-article number
            content (str): Sub-article content
            sub_type (str): Type of sub-article
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Check for unreasonable numbers
        if number > 100:  # Unlikely to have more than 100 sub-articles
            return False
        
        # Check content quality
        if len(content.strip()) < self.min_content_length:
            return False
        
        # Check for invalid content patterns
        content_lower = content.lower()
        invalid_content_patterns = [
            r'^\s*[,\.\s]+\s*$',  # Only punctuation and spaces
            r'^\s*\d{4,}\s*$',    # Only years
            r'^\s*\d+\.\d+\.\d+\s*$',  # Only dates
        ]
        
        for pattern in invalid_content_patterns:
            if re.match(pattern, content):
                return False
        
        return True
    
    def _extract_sub_articles(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract sub-articles (?? ?? Î™? from content with improved validation
        
        Args:
            content (str): Article content
            
        Returns:
            List[Dict[str, Any]]: List of validated sub-articles
        """
        logger.debug(f"_extract_sub_articles called with content length: {len(content)}")
        logger.debug(f"First 200 chars: {repr(content[:200])}")
        sub_articles = []
        
        # Extract paragraphs (?? - Korean legal format only
        paragraphs = self._extract_paragraphs_korean(content)
        sub_articles.extend(paragraphs)
        
        # Extract sub-paragraphs (?? - 1., 2., 3. etc.
        sub_paragraphs = self._extract_sub_paragraphs_korean(content)
        logger.debug(f"Found {len(sub_paragraphs)} sub-paragraphs (?? in content")
        sub_articles.extend(sub_paragraphs)
        
        # Extract items (Î™? - Í∞Ä., ??, ?? etc.
        items = self._extract_items_korean(content)
        sub_articles.extend(items)
        
        # Remove duplicates and sort by position
        sub_articles = self._remove_duplicate_sub_articles(sub_articles)
        sub_articles.sort(key=lambda x: x['position'])
        
        return sub_articles
    
    def _remove_duplicate_sub_articles(self, sub_articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate sub-articles based on position and content
        
        Args:
            sub_articles (List[Dict[str, Any]]): List of sub-articles
            
        Returns:
            List[Dict[str, Any]]: Deduplicated list
        """
        seen_positions = set()
        unique_articles = []
        
        for article in sub_articles:
            # Use position as primary key for deduplication
            pos = article.get('position', 0)
            if pos not in seen_positions:
                seen_positions.add(pos)
                unique_articles.append(article)
        
        return unique_articles
    
    def _extract_paragraphs_korean(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract paragraphs (?? using improved Korean legal format parsing
        
        Args:
            content (str): Article content
            
        Returns:
            List[Dict[str, Any]]: List of paragraphs
        """
        paragraphs = []
        
        # Use only the Korean paragraph symbols pattern to avoid duplicates
        paragraph_pattern = re.compile(r'[?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©?™‚ë´?¨‚ë≠??ëØ?∞‚ë±?≤‚ë≥]')
        
        for match in paragraph_pattern.finditer(content):
            paragraph_number = self._extract_paragraph_number_enhanced(match)
            paragraph_content = self._extract_sub_content_enhanced(content, match.start())
            
            if self._validate_paragraph_content(paragraph_content):
                paragraph_data = {
                    'type': '??,
                    'number': paragraph_number,
                    'content': paragraph_content,
                    'position': match.start()
                }
                
                # Only include amendment info if there's an actual amendment
                amendment_info = self._extract_amendment_info(paragraph_content)
                if amendment_info and amendment_info.get('has_amendment'):
                    paragraph_data['amendment'] = {
                        'date': amendment_info.get('amendment_date'),
                        'type': amendment_info.get('amendment_type')
                    }
                
                paragraphs.append(paragraph_data)
        
        # Sort by position to maintain order
        paragraphs.sort(key=lambda x: x['position'])
        
        logger.debug(f"Found {len(paragraphs)} paragraphs in content")
        
        return paragraphs
    
    def _extract_sub_paragraphs_korean(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract sub-paragraphs (?? using Korean legal format parsing
        Pattern: 1., 2., 3. etc.
        
        Args:
            content (str): Article content
            
        Returns:
            List[Dict[str, Any]]: List of sub-paragraphs
        """
        logger.debug(f"_extract_sub_paragraphs_korean called with content length: {len(content)}")
        logger.debug(f"First 200 chars: {repr(content[:200])}")
        sub_paragraphs = []
        
        # Enhanced patterns for ??(?? - 1., 2., 3. etc.
        ho_patterns = [
            # Pattern 1: 1. "content" (with quotes) - most common in definition articles
            re.compile(r'(\d+)\.\s*"([^"]+)"', re.MULTILINE),
            
            # Pattern 2: 1. content (without quotes, until next number or end) - improved
            re.compile(r'(\d+)\.\s*([^0-9?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©Í∞Ä-??+?)(?=\d+\.|$)', re.MULTILINE | re.DOTALL),
            
            # Pattern 3: 1. content (more flexible, until next pattern) - improved
            re.compile(r'(\d+)\.\s*([^?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©Í∞Ä-??+?)(?=[?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©Í∞Ä-??|$)', re.MULTILINE | re.DOTALL),
            
            # Pattern 4: Enhanced pattern for Korean legal documents - more permissive
            re.compile(r'(\d+)\.\s*([^?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©Í∞Ä-??d]+?)(?=\d+\.|$|?§Ïùå|Í∞??∏Ïùò|?¨Ìï≠|Í≤ΩÏö∞|Î™©Ïùò)', re.MULTILINE | re.DOTALL),
            
            # Pattern 5: Pattern for content with Korean characters mixed
            re.compile(r'(\d+)\.\s*([^?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©\d]+?)(?=\d+\.|$)', re.MULTILINE | re.DOTALL),
            
            # Pattern 6: Very permissive pattern - captures until next number or end of content
            re.compile(r'(\d+)\.\s*([^0-9]+?)(?=\d+\.|$)', re.MULTILINE | re.DOTALL),
            
            # Pattern 7: Pattern for content with line breaks
            re.compile(r'(\d+)\.\s*([^0-9]+?)(?=\d+\.|$)', re.MULTILINE | re.DOTALL | re.VERBOSE)
        ]
        
        for pattern_idx, pattern in enumerate(ho_patterns):
            logger.debug(f"Testing pattern {pattern_idx + 1} for ??(??")
            matches = list(pattern.finditer(content))
            logger.debug(f"Pattern {pattern_idx + 1} found {len(matches)} matches")
            
            for match in matches:
                ho_number = int(match.group(1))
                ho_content = match.group(2).strip()
                
                # Clean up content
                ho_content = re.sub(r'^\s*["\']|["\']\s*$', '', ho_content)
                ho_content = ho_content.strip()
                
                logger.debug(f"Processing ??{ho_number}: '{ho_content[:50]}...' (length: {len(ho_content)})")
                
                # Enhanced validation for ??(?? items
                if self._validate_ho_number_and_content(ho_number, ho_content):
                    # Check if this is not already captured
                    if not any(sp['number'] == ho_number and sp['position'] == match.start() for sp in sub_paragraphs):
                        logger.debug(f"Adding ??{ho_number} to results")
                        
                        ho_data = {
                            'type': '??,
                            'number': ho_number,
                            'content': ho_content,
                            'position': match.start()
                        }
                        
                        # Only include amendment info if there's an actual amendment
                        amendment_info = self._extract_amendment_info(ho_content)
                        if amendment_info and amendment_info.get('has_amendment'):
                            ho_data['amendment'] = {
                                'date': amendment_info.get('amendment_date'),
                                'type': amendment_info.get('amendment_type')
                            }
                        
                        sub_paragraphs.append(ho_data)
                    else:
                        logger.debug(f"??{ho_number} already captured, skipping")
                else:
                    logger.debug(f"??{ho_number} failed validation (number: {ho_number}, content: '{ho_content[:30]}...')")
        
        return sub_paragraphs
    
    def _extract_items_korean(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract items (Î™? using Korean legal format parsing
        Pattern: Í∞Ä., ??, ?? etc.
        
        Args:
            content (str): Article content
            
        Returns:
            List[Dict[str, Any]]: List of items
        """
        items = []
        
        # Pattern for Î™?(?? - Í∞Ä., ??, ?? etc. - More strict pattern
        mok_pattern = re.compile(r'^([Í∞Ä-??)\.\s+(.+?)(?=\n[Í∞Ä-??\.|\n\d+\.|\n[?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©]|$)', re.MULTILINE | re.DOTALL)
        
        # First pass: find all potential Î™?items
        potential_items = []
        for match in mok_pattern.finditer(content):
            mok_letter = match.group(1)
            mok_content = match.group(2).strip()
            
            # Convert Korean letter to number for sorting
            mok_number = ord(mok_letter) - ord('Í∞Ä') + 1
            
            potential_items.append({
                'letter': mok_letter,
                'number': mok_number,
                'content': mok_content,
                'position': match.start()
            })
        
        # Strict validation for Î™?sequence
        if potential_items:
            # Additional check: reject if content contains "Í∞??? (??items)
            if "Í∞??? in content or "?∏Ï?" in content or "Í∞ÅÎ™©" in content:
                logger.debug("Content contains 'Í∞???, '?∏Ï?', or 'Í∞ÅÎ™©', rejecting as Î™?items")
                return items
            
            # Additional check: reject if content contains ??patterns (1., 2., etc.)
            ho_pattern = re.compile(r'\d+\.\s+')
            if ho_pattern.search(content):
                logger.debug("Content contains ??patterns (1., 2., etc.), rejecting as Î™?items")
                return items
            
            # Additional check: reject if content contains ??patterns (?? ?? etc.)
            hang_pattern = re.compile(r'[?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©?™‚ë´?¨‚ë≠??ëØ?∞‚ë±?≤‚ë≥]')
            if hang_pattern.search(content):
                logger.debug("Content contains ??patterns (?? ?? etc.), rejecting as Î™?items")
                return items
            
            # Must have Í∞Ä. to be considered valid Î™?items
            has_ga = any(item['letter'] == 'Í∞Ä' for item in potential_items)
            
            if not has_ga:
                # No Í∞Ä. found, reject all Î™?items
                logger.debug("No Í∞Ä. found in potential Î™?items, rejecting all Î™?items")
                return items
            
            # Additional validation: Check for proper Korean legal sequence
            # Korean legal documents must follow Í∞Ä. -> ?? -> ?? sequence
            letters = [item['letter'] for item in potential_items]
            expected_sequence = ['Í∞Ä', '??, '??, '??, 'Îß?, 'Î∞?, '??, '??, '??, 'Ï∞?, 'Ïπ?, '?Ä', '??, '??]
            
            # Check if we have a proper sequence starting from Í∞Ä
            is_proper_sequence = True
            for i, letter in enumerate(letters):
                if i < len(expected_sequence) and letter != expected_sequence[i]:
                    is_proper_sequence = False
                    break
            
            if not is_proper_sequence:
                logger.debug(f"Invalid Î™?sequence: {letters}, rejecting all Î™?items")
                return items
            
            # Additional check: reject if only "??" is found without "Í∞Ä." and "??"
            if len(letters) == 1 and letters[0] == '??:
                logger.debug("Only '??' found without 'Í∞Ä.' and '??', rejecting as invalid Î™?sequence")
                return items
            
            # Additional check: reject if "??" appears before "Í∞Ä." or "??"
            if '?? in letters and ('Í∞Ä' not in letters or '?? not in letters):
                logger.debug(f"'??' found without proper preceding 'Í∞Ä.' and '??', rejecting as invalid Î™?sequence. Letters: {letters}")
                return items
            
            # Sort by position to maintain order
            potential_items.sort(key=lambda x: x['position'])
            
            # Additional validation: check for meaningful content
            valid_items = []
            for item in potential_items:
                content_clean = item['content'].strip()
                
                # Reject very short content or just punctuation
                if len(content_clean) < 5:
                    logger.debug(f"Rejecting Î™?item '{item['letter']}.' - content too short: '{content_clean}'")
                    continue
                
                # Reject content that's just punctuation or single characters
                if content_clean in ['??', '??, '', 'Í∞Ä.', '??', '??', 'Îß?', 'Î∞?', '??', '??', '??', 'Ï∞?', 'Ïπ?', '?Ä.', '??', '??']:
                    logger.debug(f"Rejecting Î™?item '{item['letter']}.' - content is just punctuation: '{content_clean}'")
                    continue
                
                # Reject UI elements
                if 'Ï°∞Î¨∏Î≤ÑÌäº?†ÌÉùÏ≤¥ÌÅ¨' in content_clean or '?ºÏπòÍ∏∞Ï†ëÍ∏? in content_clean:
                    logger.debug(f"Rejecting Î™?item '{item['letter']}.' - contains UI elements")
                    continue
                
                # Check for meaningful content
                meaningful_chars = re.sub(r'[^\wÍ∞Ä-??', '', content_clean)
                if len(meaningful_chars) < 3:
                    logger.debug(f"Rejecting Î™?item '{item['letter']}.' - not enough meaningful characters: '{content_clean}'")
                    continue
                
                # Additional check: reject if content ends with just punctuation and is short
                if len(content_clean) <= 5 and content_clean.endswith('.'):
                    logger.debug(f"Rejecting Î™?item '{item['letter']}.' - ends with punctuation and too short: '{content_clean}'")
                    continue
                
                valid_items.append(item)
            
            # Only add items if we have at least 2 valid items
            if len(valid_items) >= 2:
                for item in valid_items:
                    mok_data = {
                        'type': 'Î™?,
                        'number': item['number'],
                        'letter': item['letter'],
                        'content': item['content'],
                        'position': item['position']
                    }
                    
                    # Only include amendment info if there's an actual amendment
                    amendment_info = self._extract_amendment_info(item['content'])
                    if amendment_info and amendment_info.get('has_amendment'):
                        mok_data['amendment'] = {
                            'date': amendment_info.get('amendment_date'),
                            'type': amendment_info.get('amendment_type')
                        }
                    
                    items.append(mok_data)
            else:
                logger.debug(f"Not enough valid Î™?items ({len(valid_items)}), rejecting all Î™?items")
        
        return items
    
    def _is_definition_article(self, article_number: str, article_title: str, content: str) -> bool:
        """
        Check if this is a definition article (?ïÏùò Ï°∞Î¨∏)
        
        Args:
            article_number (str): Article number (e.g., "??Ï°?)
            article_title (str): Article title
            content (str): Article content
            
        Returns:
            bool: True if this is a definition article
        """
        # Check article number (??Ï°?is commonly used for definitions)
        if article_number == "??Ï°?:
            return True
        
        # Check title patterns
        definition_titles = ["?ïÏùò", "?©Ïñ¥???ïÏùò", "?ïÏùò Î∞?Î™ÖÏπ≠", "?©Ïñ¥????]
        if any(title in article_title for title in definition_titles):
            return True
        
        # Check content patterns
        definition_patterns = [
            r'?©Ïñ¥??s*?ªÏ?\s*?§ÏùåÍ≥?s*Í∞ôÎã§',
            r'?©Ïñ¥??s*?ïÏùò??s*?§ÏùåÍ≥?s*Í∞ôÎã§',
            r'??s*Î≤ïÏóê??s*?¨Ïö©?òÎäî\s*?©Ïñ¥',
            r'?ïÏùò',
            r'?©Ïñ¥??s*??,
        ]
        
        for pattern in definition_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
        """
        Extract sub-paragraphs (?? using Korean legal format
        
        Args:
            content (str): Article content
            
        Returns:
            List[Dict[str, Any]]: List of sub-paragraphs
        """
        subparagraphs = []
        
        # Extract numbered sub-paragraphs (1., 2., 3., etc.)
        for match in self.subparagraph_patterns['numbered'].finditer(content):
            subparagraph_number = int(match.group(1))
            subparagraph_content = self._extract_sub_content(content, match.start())
            
            if len(subparagraph_content.strip()) >= self.min_content_length:
                subparagraphs.append({
                    'type': '??,
                    'number': subparagraph_number,
                    'content': subparagraph_content,
                    'position': match.start()
                })
        
        return subparagraphs
    
    def _extract_items_korean(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract items (Î™? using Korean legal format
        
        Args:
            content (str): Article content
            
        Returns:
            List[Dict[str, Any]]: List of items
        """
        items = []
        
        # Extract lettered items (Í∞Ä., ??, ??, etc.)
        for match in self.item_patterns['lettered'].finditer(content):
            item_letter = match.group(1)
            item_content = self._extract_sub_content(content, match.start())
            
            if len(item_content.strip()) >= self.min_content_length:
                items.append({
                    'type': 'Î™?,
                    'number': item_letter,
                    'content': item_content,
                    'position': match.start()
                })
        
        return items
    
    def _get_paragraph_number(self, paragraph_symbol: str) -> int:
        """
        Convert Korean paragraph symbol to number
        
        Args:
            paragraph_symbol (str): Korean paragraph symbol (?? ?? etc.)
            
        Returns:
            int: Paragraph number
        """
        symbol_map = {
            '??: 1, '??: 2, '??: 3, '??: 4, '??: 5,
            '??: 6, '??: 7, '??: 8, '??: 9, '??: 10,
            '??: 11, '??: 12, '??: 13, '??: 14, '??: 15,
            '??: 16, '??: 17, '??: 18, '??: 19, '??: 20
        }
        return symbol_map.get(paragraph_symbol, 1)
    
    def _extract_paragraph_number_enhanced(self, match) -> int:
        """
        Enhanced paragraph number extraction handling various formats
        
        Args:
            match: Regex match object
            
        Returns:
            int: Paragraph number
        """
        matched_text = match.group()
        
        # Korean symbol to number mapping
        symbol_map = {
            '??: 1, '??: 2, '??: 3, '??: 4, '??: 5,
            '??: 6, '??: 7, '??: 8, '??: 9, '??: 10,
            '??: 11, '??: 12, '??: 13, '??: 14, '??: 15,
            '??: 16, '??: 17, '??: 18, '??: 19, '??: 20
        }
        
        if matched_text in symbol_map:
            return symbol_map[matched_text]
        
        # Extract number from patterns like "1??, "????
        number_match = re.search(r'(\d+)', matched_text)
        if number_match:
            return int(number_match.group(1))
        
        return 1
    
    def _extract_sub_content_enhanced(self, content: str, start_pos: int) -> str:
        """
        Enhanced content extraction with better boundary detection
        
        Args:
            content (str): Full content
            start_pos (int): Start position
            
        Returns:
            str: Extracted content
        """
        # Get the text starting from the current position
        remaining_text = content[start_pos:]
        
        # Find the next Korean paragraph symbol (?? or next article
        next_paragraph_pattern = re.compile(r'[?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©?™‚ë´?¨‚ë≠??ëØ?∞‚ë±?≤‚ë≥]')
        next_article_pattern = re.compile(r'??d+Ï°?)
        
        # Look for next paragraph symbol (skip the current one by starting from position 1)
        next_paragraph_match = next_paragraph_pattern.search(remaining_text[1:])
        next_article_match = next_article_pattern.search(remaining_text[1:])
        
        # Determine the end position
        end_pos = len(content)
        
        if next_paragraph_match and next_article_match:
            # Both found, take the earlier one
            para_end = start_pos + 1 + next_paragraph_match.start()
            art_end = start_pos + 1 + next_article_match.start()
            end_pos = min(para_end, art_end)
        elif next_paragraph_match:
            # Only next paragraph found
            end_pos = start_pos + 1 + next_paragraph_match.start()
        elif next_article_match:
            # Only next article found
            end_pos = start_pos + 1 + next_article_match.start()
        
        # Extract the content
        if end_pos < len(content):
            sub_content = content[start_pos:end_pos].strip()
        else:
            sub_content = content[start_pos:].strip()
        
        # Clean the content
        sub_content = self._clean_legal_content(sub_content)
        
        # Remove ???? items from ???? content to prevent duplication
        sub_content = self._remove_ho_items_from_hang_content(sub_content)
        
        return sub_content
    
    def _clean_legal_content(self, content: str) -> str:
        """
        Clean legal content removing unwanted elements
        
        Args:
            content (str): Raw content
            
        Returns:
            str: Cleaned content
        """
        # Keep amendment markers for now - they contain important information
        # content = re.sub(r'<Í∞úÏ†ï\s+[^>]+>', '', content)
        
        # Remove execution markers
        content = re.sub(r'\[?úÌñâ\s+[^\]]+\]', '', content)
        
        # Remove paragraph markers at the beginning - but be more careful
        # Only remove if it's at the very beginning and not part of content
        content = re.sub(r'^[?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©?™‚ë´?¨‚ë≠??ëØ?∞‚ë±?≤‚ë≥]\s*', '', content)
        # Don't remove "???? etc. as they are part of the content
        # content = re.sub(r'^??\d+[??ò∏Î™?\s*', '', content)
        
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content
    
    def _extract_amendment_info(self, content: str) -> Dict[str, Any]:
        """
        Extract amendment information from content
        
        Args:
            content (str): Content to analyze
            
        Returns:
            Dict[str, Any]: Amendment information
        """
        amendment_info = {
            'has_amendment': False,
            'amendment_date': None,
            'amendment_type': None
        }
        
        # Extract amendment markers (including deletion)
        amendment_match = re.search(r'<Í∞úÏ†ï\s+([^>]+)>', content)
        deletion_match = re.search(r'??†ú<([^>]+)>', content)
        
        if amendment_match:
            amendment_info['has_amendment'] = True
            amendment_info['amendment_date'] = amendment_match.group(1)
            amendment_info['amendment_type'] = 'Í∞úÏ†ï'
        elif deletion_match:
            amendment_info['has_amendment'] = True
            amendment_info['amendment_date'] = deletion_match.group(1)
            amendment_info['amendment_type'] = '??†ú'
        
        return amendment_info
    
    def _validate_paragraph_content(self, content: str) -> bool:
        """
        Validate paragraph content quality with optimized checks
        
        Args:
            content (str): Content to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if len(content.strip()) < self.min_content_length:
            return False
        
        # Skip expensive validation if disabled
        if not self.enable_expensive_validation:
            return True
        
        # Basic validation only
        meaningful_chars = re.sub(r'[^\wÍ∞Ä-??', '', content)
        if len(meaningful_chars) < 3:
            return False
        
        return True
    
    def _validate_ho_content(self, content: str) -> bool:
        """
        Validate ??(?? content with enhanced rules to prevent date misclassification
        
        Args:
            content (str): Content to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if len(content.strip()) < 3:  # Increased minimum length
            return False
        
        # Reject content that looks like dates or amendments
        content_lower = content.lower()
        date_patterns = [
            r'\d{4}\.\d{1,2}\.\d{1,2}',  # 1991.2.18
            r'\d{1,2}\.\d{1,2}\.\d{1,2}',  # 2.18.2007
            r'<Í∞úÏ†ï',  # Amendment markers
            r'>$',  # Ending with >
            r'Ï°∞Î¨∏Î≤ÑÌäº?†ÌÉùÏ≤¥ÌÅ¨',  # UI elements
            r'?ºÏπòÍ∏∞Ï†ëÍ∏?,  # UI elements
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, content_lower):
                return False
        
        # Reject content that's mostly punctuation or numbers
        if re.match(r'^[\d\.,\s>]+$', content.strip()):
            return False
        
        # Check for meaningful content
        meaningful_chars = re.sub(r'[^\wÍ∞Ä-??', '', content)
        if len(meaningful_chars) < 2:  # Require at least 2 meaningful characters
            return False
        
        # Check for common invalid patterns
        invalid_patterns = [
            r'^\s*$',  # Empty or whitespace only
            r'^[^\wÍ∞Ä-??*$',  # No meaningful characters
            r'^[0-9]+$',  # Only numbers
            r'^[?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©]+$',  # Only paragraph markers
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, content.strip()):
                return False
        
        return True
    
    def _validate_ho_number_and_content(self, number: int, content: str) -> bool:
        """
        Enhanced validation for ??(?? items to prevent date misclassification
        
        Args:
            number (int): ??Î≤àÌò∏
            content (str): ???¥Ïö©
            
        Returns:
            bool: True if valid ??item, False otherwise
        """
        # Reject dates as ??numbers (1900-2030 range)
        if 1900 <= number <= 2030:
            logger.debug(f"Rejecting date as ??number: {number}")
            return False
        
        # Reject unreasonable ??numbers
        if number < 1 or number > 50:
            logger.debug(f"Rejecting unreasonable ??number: {number}")
            return False
        
        # Use existing content validation
        return self._validate_ho_content(content)
    
    def _validate_mok_content(self, content: str) -> bool:
        """
        Enhanced validation for Î™?(?? items to prevent empty content
        
        Args:
            content (str): Î™??¥Ïö©
            
        Returns:
            bool: True if valid Î™?item, False otherwise
        """
        # Reject very short content
        if len(content.strip()) < 5:  # Increased minimum length
            return False
        
        # Reject content that's just punctuation
        if content.strip() in ['??', '??, '']:
            return False
        
        # Reject UI elements
        if 'Ï°∞Î¨∏Î≤ÑÌäº?†ÌÉùÏ≤¥ÌÅ¨' in content or '?ºÏπòÍ∏∞Ï†ëÍ∏? in content:
            return False
        
        # Reject content that ends with just punctuation
        if content.strip().endswith('.') and len(content.strip()) <= 5:
            return False
        
        # Check for meaningful content
        meaningful_chars = re.sub(r'[^\wÍ∞Ä-??', '', content)
        if len(meaningful_chars) < 3:  # Require at least 3 meaningful characters
            return False
        
        return True
    
    def _remove_ho_items_from_hang_content(self, content: str) -> str:
        """
        Remove ???? items from ???? content to prevent duplication
        
        Args:
            content (str): ???? content
            
        Returns:
            str: Cleaned content without ???? items
        """
        # Don't remove anything - let the sub_articles parsing handle ??items
        # Removing ??items here was causing issues with legal references like "??Ï°∞Ï†ú1??
        # which were being mistakenly identified as ??items and removed
        
        # Only clean up UI elements
        cleaned_content = re.sub(r'Ï°∞Î¨∏Î≤ÑÌäº?†ÌÉùÏ≤¥ÌÅ¨', '', content)
        cleaned_content = re.sub(r'?ºÏπòÍ∏∞Ï†ëÍ∏?, '', cleaned_content)
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()
        
        return cleaned_content
    
    def _remove_ui_elements_from_html(self, soup):
        """
        Remove UI elements and unwanted HTML elements from BeautifulSoup object
        
        Args:
            soup: BeautifulSoup object to clean
        """
        # Remove elements with UI-related classes or IDs (Korean legal document specific)
        ui_selectors = [
            # Korean legal document UI elements
            '[class*="article_icon"]', '[class*="article-icon"]',
            '[class*="button"]', '[class*="btn"]', '[class*="ui"]', '[class*="control"]',
            '[class*="toggle"]', '[class*="expand"]', '[class*="collapse"]',
            '[class*="hidden"]', '[class*="icon"]',
            '[id*="button"]', '[id*="btn"]', '[id*="ui"]', '[id*="control"]',
            '[id*="toggle"]', '[id*="expand"]', '[id*="collapse"]',
            '[id*="cacheArea"]', '[id*="cache-area"]',
            # Standard HTML elements
            'button', 'input[type="button"]', 'input[type="checkbox"]',
            'input[type="radio"]', 'input[type="submit"]'
        ]
        
        for selector in ui_selectors:
            try:
                elements = soup.select(selector)
                for element in elements:
                    element.decompose()  # Remove the element completely
            except Exception:
                continue
        
        # Remove specific Korean legal document UI elements by text content
        ui_text_patterns = [
            'Ï°∞Î¨∏ Î≤ÑÌäº ?åÍ∞ú', 'Ï°∞Î¨∏Î≤ÑÌäº?†ÌÉùÏ≤¥ÌÅ¨', '?†ÌÉùÏ≤¥ÌÅ¨', '?†ÌÉù',
            'Ï°∞Î¨∏?∞ÌòÅ', 'Ï°∞Î¨∏?êÎ?', '?ºÏπòÍ∏∞Ï†ëÍ∏?, '?ºÏπòÍ∏?, '?ëÍ∏∞',
            'Ï°∞Î¨∏?∞ÌòÅ', 'Ï°∞Î¨∏?êÎ?'
        ]
        
        # Find and remove elements containing UI text
        for text_pattern in ui_text_patterns:
            try:
                # Find elements containing the UI text
                elements = soup.find_all(text=lambda text: text and text_pattern in text)
                for text_element in elements:
                    parent = text_element.parent
                    if parent:
                        # Check if this is a UI-related element
                        if any(ui_class in str(parent.get('class', [])).lower() for ui_class in ['icon', 'button', 'btn', 'ui', 'control', 'hidden']):
                            parent.decompose()
                        elif any(ui_id in str(parent.get('id', '')).lower() for ui_id in ['button', 'btn', 'ui', 'control', 'icon']):
                            parent.decompose()
                        elif parent.name in ['dt', 'dd'] and 'article_icon' in str(parent.get('class', [])):
                            parent.decompose()
            except Exception:
                continue
        
        # Remove script, style, and noscript elements
        for element in soup(['script', 'style', 'noscript']):
            element.decompose()
        
        # Remove img elements with UI-related alt text
        img_elements = soup.find_all('img')
        for img in img_elements:
            alt_text = img.get('alt', '').lower()
            if any(ui_text in alt_text for ui_text in ['?†ÌÉùÏ≤¥ÌÅ¨', '?∞ÌòÅ', '?êÎ?', 'button', 'btn', 'icon']):
                img.decompose()
        
        # Remove dl elements with article_icon class (Korean legal document specific)
        dl_elements = soup.find_all('dl', class_='article_icon')
        for dl in dl_elements:
            dl.decompose()
    
    def _clean_ui_elements_from_text(self, text: str) -> str:
        """
        Clean UI elements from extracted text
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        # UI element patterns to remove (Korean legal document specific)
        ui_patterns = [
            r'Ï°∞Î¨∏Î≤ÑÌäº?†ÌÉùÏ≤¥ÌÅ¨',
            r'?ºÏπòÍ∏∞Ï†ëÍ∏?,
            r'?ºÏπòÍ∏?,
            r'?ëÍ∏∞',
            r'?†ÌÉùÏ≤¥ÌÅ¨',
            r'Î≤ÑÌäº?†ÌÉù',
            r'Ï°∞Î¨∏Î≤ÑÌäº',
            r'Ï°∞Î¨∏ Î≤ÑÌäº ?åÍ∞ú',
            r'Ï°∞Î¨∏?∞ÌòÅ',
            r'Ï°∞Î¨∏?êÎ?',
            r'\[?ºÏπòÍ∏?]',
            r'\[?ëÍ∏∞\]',
            r'\[?†ÌÉù\]',
            r'\[Ï≤¥ÌÅ¨\]',
            r'\[Ï°∞Î¨∏\]',
            r'??,
            r'?Ä',
            r'??,
            r'??,
            r'??,
            r'??,
            r'??,
            r'??,
            # Additional patterns found in raw data
            r'?†ÌÉù\s*$',  # "?†ÌÉù" at end of line
            r'?∞ÌòÅ\s*$',  # "?∞ÌòÅ" at end of line
            r'?êÎ?\s*$',  # "?êÎ?" at end of line
        ]
        
        cleaned_text = text
        for pattern in ui_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace and normalize
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Remove any remaining isolated UI text
        isolated_ui_patterns = [
            r'^\s*?†ÌÉù\s*$',
            r'^\s*?∞ÌòÅ\s*$',
            r'^\s*?êÎ?\s*$',
            r'^\s*Ï≤¥ÌÅ¨\s*$',
            r'^\s*Î≤ÑÌäº\s*$',
        ]
        
        for pattern in isolated_ui_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE)
        
        return cleaned_text
    
    def _extract_sub_content(self, content: str, start_pos: int) -> str:
        """
        Extract content for a sub-article
        
        Args:
            content (str): Full content
            start_pos (int): Starting position
            
        Returns:
            str: Sub-article content
        """
        # Find the end of this sub-article
        remaining_text = content[start_pos:]
        
        # Look for next sub-article pattern
        end_pos = len(content)
        
        # Use enhanced boundary detection
        next_patterns = [
            re.compile(r'[?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©?™‚ë´?¨‚ë≠??ëØ?∞‚ë±?≤‚ë≥]'),
            re.compile(r'??d+Ï°?),
            re.compile(r'<Í∞úÏ†ï\s+[^>]+>'),
            re.compile(r'\[[^\]]+\]')
        ]
        
        for pattern in next_patterns:
            next_match = pattern.search(remaining_text[1:])  # Skip first char
            if next_match:
                potential_end = start_pos + 1 + next_match.start()
                end_pos = min(end_pos, potential_end)
        
        sub_content = content[start_pos:end_pos].strip()
        
        # Use enhanced content cleaning
        sub_content = self._clean_legal_content(sub_content)
        
        return sub_content
    
    def _extract_references(self, text: str) -> List[str]:
        """
        Extract legal references from text - only substantial law references
        
        Args:
            text (str): Text to analyze
            
        Returns:
            List[str]: List of substantial law references found
        """
        references = []
        
        # Pattern for quoted law names (?§Ïßà?ÅÏù∏ Î≤ïÎ•†Î™ÖÎßå Ï∂îÏ∂ú)
        quoted_pattern = re.compile(r'??[^??+)??)
        quoted_matches = quoted_pattern.findall(text)
        
        # ?§Ïßà?ÅÏù∏ Î≤ïÎ•†Î™ÖÎßå ?ÑÌÑ∞Îß?
        substantial_laws = []
        for match in quoted_matches:
            law_name = match.strip()
            # ?ºÎ∞ò?ÅÏù∏ Ï∞∏Ï°∞Í∞Ä ?ÑÎãå Íµ¨Ï≤¥?ÅÏù∏ Î≤ïÎ•†Î™ÖÎßå ?¨Ìï®
            if (law_name and 
                law_name not in ['??Î≤?, 'Í∞ôÏ? Î≤?, '?ôÎ≤ï', '?ÅÎ≤ï', 'ÎØºÎ≤ï', '?ïÎ≤ï', '?âÏ†ïÎ≤?] and
                len(law_name) > 2 and  # ?àÎ¨¥ ÏßßÏ? Í≤ÉÏ? ?úÏô∏
                'Î≤? in law_name):  # Î≤ïÎ•†Î™ÖÏóê 'Î≤????¨Ìï®?òÏñ¥????
                substantial_laws.append(law_name)
        
        references.extend(substantial_laws)
        
        # Ï§ëÎ≥µ ?úÍ±∞ Î∞??ïÎ†¨
        references = list(set(ref.strip() for ref in references if ref.strip()))
        references.sort()
        
        return references
    
    def validate_parsing_results(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parsing results and provide quality metrics
        
        Args:
            parsed_data (Dict[str, Any]): Parsed data to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            'is_valid': True,
            'quality_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        # Check article continuity
        articles = parsed_data.get('articles', [])
        if len(articles) > 1:
            article_numbers = []
            for art in articles:
                try:
                    num = int(art['article_number'].replace('??, '').replace('Ï°?, ''))
                    article_numbers.append(num)
                except (ValueError, KeyError):
                    continue
            
            if len(article_numbers) > 1:
                expected_numbers = list(range(min(article_numbers), max(article_numbers) + 1))
                missing_articles = set(expected_numbers) - set(article_numbers)
                
                if missing_articles:
                    validation_results['issues'].append(f"Missing articles: {sorted(missing_articles)}")
                    validation_results['quality_score'] -= 0.1
        
        # Check paragraph continuity within articles
        for article in articles:
            paragraphs = article.get('sub_articles', [])
            paragraph_numbers = [p['number'] for p in paragraphs if p.get('type') == '??]
            
            if len(paragraph_numbers) > 1:
                max_para = max(paragraph_numbers)
                expected_paras = list(range(1, max_para + 1))
                missing_paras = set(expected_paras) - set(paragraph_numbers)
                
                if missing_paras:
                    validation_results['issues'].append(
                        f"Article {article.get('article_number', 'Unknown')}: Missing paragraphs {sorted(missing_paras)}"
                    )
        
        # Calculate quality score
        total_elements = len(articles) + sum(len(art.get('sub_articles', [])) for art in articles)
        
        if total_elements > 0:
            validation_results['quality_score'] = max(0.0, 1.0 - (len(validation_results['issues']) * 0.1))
        
        validation_results['is_valid'] = validation_results['quality_score'] > 0.7
        
        return validation_results
    
    def validate_article_structure(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate article structure and return quality metrics
        
        Args:
            articles (List[Dict[str, Any]]): List of parsed articles
            
        Returns:
            Dict[str, Any]: Validation results and metrics
        """
        validation_results = {
            'total_articles': len(articles),
            'valid_articles': 0,
            'missing_titles': 0,
            'empty_content': 0,
            'duplicate_numbers': [],
            'non_sequential_numbers': [],
            'quality_score': 0.0
        }
        
        article_numbers = []
        
        for article in articles:
            # Check if article has required fields
            if not article.get('article_number') or not article.get('article_content'):
                continue
            
            validation_results['valid_articles'] += 1
            
            # Check for missing titles
            if not article.get('article_title'):
                validation_results['missing_titles'] += 1
            
            # Check for empty content
            if not article.get('article_content').strip():
                validation_results['empty_content'] += 1
            
            # Extract article number for validation
            article_num_match = re.search(r'??\d+)Ï°?, article['article_number'])
            if article_num_match:
                article_num = int(article_num_match.group(1))
                article_numbers.append(article_num)
        
        # Check for duplicate numbers
        seen_numbers = set()
        for num in article_numbers:
            if num in seen_numbers:
                validation_results['duplicate_numbers'].append(num)
            seen_numbers.add(num)
        
        # Check for non-sequential numbers
        sorted_numbers = sorted(article_numbers)
        for i in range(1, len(sorted_numbers)):
            if sorted_numbers[i] - sorted_numbers[i-1] > 1:
                validation_results['non_sequential_numbers'].append(
                    (sorted_numbers[i-1], sorted_numbers[i])
                )
        
        # Calculate quality score
        if validation_results['total_articles'] > 0:
            quality_factors = [
                validation_results['valid_articles'] / validation_results['total_articles'],
                1.0 - (validation_results['missing_titles'] / validation_results['total_articles']),
                1.0 - (validation_results['empty_content'] / validation_results['total_articles']),
                1.0 - (len(validation_results['duplicate_numbers']) / validation_results['total_articles']),
                1.0 - (len(validation_results['non_sequential_numbers']) / validation_results['total_articles'])
            ]
            validation_results['quality_score'] = sum(quality_factors) / len(quality_factors)
        
        return validation_results
    
    def _extract_definition_article_content(self, content: str) -> str:
        """
        Special extraction for definition articles with comprehensive content
        
        Args:
            content (str): Article content
            
        Returns:
            str: Complete definition article content
        """
        # For definition articles, we need to be more aggressive in finding complete content
        # Look for patterns that indicate the end of a definition article
        end_patterns = [
            # Look for the next article with more specific patterns
            r'^??Ï°?s*\([^)]+\)',     # ??Ï°??§Î•∏ Î≤ïÎ•†Í≥ºÏùò Í¥ÄÍ≥?
            r'^??Ï°?s*[Í∞Ä-??',       # ??Ï°??§Î•∏
            r'^??Ï°?,                 # ??Ï°?
            
            # Look for other structural boundaries
            r'^??d+Ï°?s*\([^)]+\)',  # Any article with parentheses
            r'^??d+Ï°?s*[Í∞Ä-??',    # Any article with Korean text
            r'^??d+Ï°?,              # Any article
        ]
        
        best_end = len(content)
        best_score = 0
        
        for pattern in end_patterns:
            matches = list(re.finditer(pattern, content, re.MULTILINE))
            for match in matches:
                # Check if this would create a more complete content
                potential_end = match.start()
                potential_content = content[:potential_end].strip()
                
                # Score based on definition article completeness
                score = self._score_definition_completeness(potential_content)
                if score > best_score:
                    best_end = potential_end
                    best_score = score
        
        # Extract content up to the best boundary
        extracted_content = content[:best_end].strip()
        
        # Additional processing for definition articles
        # Ensure we capture all numbered items (??
        if self._has_definition_structure(extracted_content):
            extracted_content = self._extend_definition_content_aggressive(content, 0, extracted_content)
        
        return extracted_content
    
    def _score_definition_completeness(self, content: str) -> int:
        """
        Score definition article completeness
        
        Args:
            content (str): Content to score
            
        Returns:
            int: Completeness score (higher is better)
        """
        score = 0
        
        # Bonus for having numbered items (1., 2., 3., etc.)
        numbered_items = len(re.findall(r'\d+\.', content))
        score += numbered_items * 5
        
        # Bonus for having sub-items (Í∞Ä., ??, ??, etc.)
        sub_items = len(re.findall(r'[Í∞Ä-??\.', content))
        score += sub_items * 3
        
        # Bonus for having paragraph numbers (?? ?? ?? etc.)
        paragraph_numbers = len(re.findall(r'[?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©]', content))
        score += paragraph_numbers * 4
        
        # Bonus for having multiple paragraphs (?? ?? etc.)
        multiple_paragraphs = len(re.findall(r'??????????????????, content))
        score += multiple_paragraphs * 6
        
        # Bonus for content length (definition articles are typically long)
        if len(content) > 2000:
            score += 20
        elif len(content) > 1000:
            score += 15
        elif len(content) > 500:
            score += 10
        
        # Bonus for having "?§ÏùåÍ≥?Í∞ôÎã§" pattern
        if re.search(r'?§ÏùåÍ≥?s*Í∞ôÎã§', content):
            score += 10
        
        # Bonus for having "??Î≤ïÏóê???¨Ïö©?òÎäî ?©Ïñ¥" pattern
        if re.search(r'??s*Î≤ïÏóê??s*?¨Ïö©?òÎäî\s*?©Ïñ¥', content):
            score += 15
        
        # Penalty for incomplete patterns
        if re.search(r'?§Ïùå\s*$', content.strip()):
            score -= 20
        if re.search(r'Í∞?s*$', content.strip()):
            score -= 20
        
        return score
