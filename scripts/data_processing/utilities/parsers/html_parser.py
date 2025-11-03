"""
HTML Parser for Assembly Law Data

This module parses HTML content from Assembly law data to extract clean text
and structured article information.
"""

import re
import logging
from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup, Tag, NavigableString

logger = logging.getLogger(__name__)


class LawHTMLParser:
    """HTML parser for Assembly law data"""
    
    def __init__(self):
        """Initialize the HTML parser"""
        self.article_pattern = re.compile(r'??\d+)Ï°?)
        self.sub_article_patterns = {
            '??: re.compile(r'??\d+)??),
            '??: re.compile(r'??\d+)??),
            'Î™?: re.compile(r'^([Í∞Ä-??)\.\s+(.+?)(?=\n[Í∞Ä-??\.|\n\d+\.|\n[?†‚ë°?¢‚ë£?§‚ë•?¶‚ëß?®‚ë©]|$)', re.MULTILINE | re.DOTALL)
        }
    
    def parse_html(self, html_content: str) -> Dict[str, Any]:
        """
        Parse HTML content and extract structured information with enhanced cleaning
        
        Args:
            html_content (str): Raw HTML content from Assembly law data
            
        Returns:
            Dict[str, Any]: Parsed content with clean text, articles, and metadata
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Enhanced cleaning process
            self._remove_unwanted_elements_enhanced(soup)
            
            return {
                'clean_text': self._extract_clean_text_enhanced(soup),
                'articles': self._extract_articles_enhanced(soup),
                'metadata': self._extract_html_metadata(soup)
            }
        except Exception as e:
            logger.error(f"Error parsing HTML content: {e}")
            return {
                'clean_text': '',
                'articles': [],
                'metadata': {}
            }
    
    def parse_html_with_structure(self, html_content: str) -> Dict[str, Any]:
        """
        Parse HTML content while preserving structural information for better article parsing
        
        Args:
            html_content (str): Raw HTML content from Assembly law data
            
        Returns:
            Dict[str, Any]: Parsed content with structure-preserved text and articles
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Minimal cleaning - only remove obvious UI elements
            self._remove_minimal_unwanted_elements(soup)
            
            return {
                'structured_text': self._extract_structured_text(soup),
                'articles': self._extract_articles_with_structure(soup),
                'metadata': self._extract_html_metadata(soup)
            }
        except Exception as e:
            logger.error(f"Error parsing HTML content with structure: {e}")
            return {
                'structured_text': '',
                'articles': [],
                'metadata': {}
            }
    
    def _remove_minimal_unwanted_elements(self, soup: BeautifulSoup):
        """
        Minimal removal of unwanted HTML elements - preserve structure for parsing
        
        Args:
            soup (BeautifulSoup): Parsed HTML soup object
        """
        # Only remove script and style elements
        for element in soup(['script', 'style', 'noscript']):
            element.decompose()
        
        # Remove only obvious UI elements
        ui_tags = ['button', 'input', 'select', 'textarea']
        
        for tag in ui_tags:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remove elements with JavaScript attributes
        for element in soup.find_all(attrs={'onclick': True}):
            element.decompose()
    
    def _extract_structured_text(self, soup: BeautifulSoup) -> str:
        """
        Extract text while preserving HTML structure for better parsing
        
        Args:
            soup (BeautifulSoup): Parsed HTML soup object
            
        Returns:
            str: Structured text with HTML tags preserved
        """
        # Convert to string while preserving structure
        structured_text = str(soup)
        
        # Clean up excessive whitespace but preserve structure
        structured_text = re.sub(r'\s+', ' ', structured_text)
        structured_text = re.sub(r'>\s+<', '><', structured_text)
        
        return structured_text.strip()
    
    def _extract_articles_with_structure(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract articles using HTML structure for better accuracy
        
        Args:
            soup (BeautifulSoup): Parsed HTML soup object
            
        Returns:
            List[Dict[str, Any]]: List of extracted articles
        """
        articles = []
        
        # Look for article patterns in HTML structure
        # This is a simplified approach - in practice, you'd analyze the specific structure
        
        # Find all elements that might contain articles
        potential_articles = soup.find_all(['div', 'p', 'section'], string=re.compile(r'??d+Ï°?))
        
        for element in potential_articles:
            text = element.get_text(strip=True)
            if self.article_pattern.search(text):
                article_info = self._parse_article_from_element(element)
                if article_info:
                    articles.append(article_info)
        
        return articles
    
    def _parse_article_from_element(self, element) -> Optional[Dict[str, Any]]:
        """
        Parse article information from an HTML element
        
        Args:
            element: BeautifulSoup element containing article
            
        Returns:
            Optional[Dict[str, Any]]: Parsed article information
        """
        try:
            text = element.get_text(strip=True)
            
            # Extract article number and title
            article_match = self.article_pattern.search(text)
            if not article_match:
                return None
            
            article_number = article_match.group(0)
            
            # Extract title (text after article number)
            title_match = re.search(rf'{re.escape(article_number)}\s*\(([^)]+)\)', text)
            title = title_match.group(1) if title_match else ""
            
            # Extract content (everything after the title)
            content_start = title_match.end() if title_match else article_match.end()
            content = text[content_start:].strip()
            
            return {
                'number': article_number,
                'title': title,
                'content': content,
                'html_element': str(element)  # Preserve HTML structure
            }
            
        except Exception as e:
            logger.debug(f"Error parsing article from element: {e}")
            return None
    
    def _remove_unwanted_elements_enhanced(self, soup: BeautifulSoup):
        """
        Enhanced removal of unwanted HTML elements including UI components
        
        Args:
            soup (BeautifulSoup): Parsed HTML soup object
        """
        # Remove script and style elements
        for element in soup(['script', 'style', 'noscript']):
            element.decompose()
        
        # Remove UI elements and buttons with safer approach
        ui_tags = ['button', 'input', 'select', 'textarea', 'script', 'style', 'noscript']
        
        for tag in ui_tags:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remove elements with specific classes (safer approach) - only remove obvious UI elements
        ui_classes = ['btn', 'button', 'nav', 'menu', 'sidebar', 'header', 'footer']
        
        for class_name in ui_classes:
            try:
                for element in soup.find_all(class_=class_name):
                    element.decompose()
            except Exception as e:
                logger.debug(f"Error removing elements with class {class_name}: {e}")
                continue
        
        # Remove elements with specific text content
        unwanted_texts = [
            'Ï°∞Î¨∏Î≤ÑÌäº', '?†ÌÉùÏ≤¥ÌÅ¨', '?ºÏπòÍ∏?, '?ëÍ∏∞', '?†ÌÉù',
            'Ï°∞Î¨∏?∞ÌòÅ', 'Ï°∞Î¨∏?êÎ?', 'Ï°∞Î¨∏?¥ÏÑ§', 'Ï°∞Î¨∏?†Î?',
            '?úÌñâ?àÏ†ï', '?åÍ∞ú', 'Î≤ÑÌäº', 'javascript:', 'onclick',
            'href="#', 'href="javascript:', 'class="btn', 'class="button'
        ]
        
        for text in unwanted_texts:
            elements = soup.find_all(text=re.compile(text, re.IGNORECASE))
            for element in elements:
                if element.parent:
                    element.parent.decompose()
        
        # Remove elements with JavaScript attributes
        for element in soup.find_all(attrs={'onclick': True}):
            element.decompose()
        
        for element in soup.find_all(attrs={'onload': True}):
            element.decompose()
        
        # Remove elements with href attributes containing javascript or #
        for element in soup.find_all('a', href=lambda x: x and ('javascript:' in x or x.startswith('#'))):
            element.decompose()
        
        # Remove elements with specific IDs that indicate UI elements
        ui_ids = ['menu', 'nav', 'header', 'footer', 'sidebar', 'toolbar', 'breadcrumb', 'pagination']
        for element in soup.find_all(id=lambda x: x and any(ui_id in x.lower() for ui_id in ui_ids)):
            element.decompose()
    
    def _extract_clean_text_enhanced(self, soup: BeautifulSoup) -> str:
        """
        Extract clean text with enhanced cleaning
        
        Args:
            soup (BeautifulSoup): Parsed HTML soup object
            
        Returns:
            str: Clean text content
        """
        # Find main content area
        main_content = soup.find('div', class_='content') or soup.find('body')
        
        if not main_content:
            return soup.get_text(separator='\n', strip=True)
        
        # Extract text with better formatting
        text = main_content.get_text(separator='\n', strip=True)
        
        # Additional cleaning
        text = self._clean_extracted_text(text)
        
        return text
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean extracted text by removing unwanted patterns
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        # Remove unwanted text patterns
        unwanted_patterns = [
            r'Ï°∞Î¨∏Î≤ÑÌäº\s*',
            r'?†ÌÉùÏ≤¥ÌÅ¨\s*',
            r'?ºÏπòÍ∏?s*',
            r'?ëÍ∏∞\s*',
            r'?†ÌÉù\s*',
            r'?åÍ∞ú\s*',
            r'Î≤ÑÌäº\s*',
            r'javascript:\s*',
            r'onclick\s*',
            r'href="#\s*',
            r'href="javascript:\s*',
            r'class="btn\s*',
            r'class="button\s*',
            r'<.*?>',  # Remove any remaining HTML tags
            r'&nbsp;',
            r'&amp;',
            r'&lt;',
            r'&gt;',
            r'&quot;',
            r'&apos;',
            r'\n\s*\n\s*\n',  # Multiple empty lines
            r'^\s*\n',       # Leading empty lines
            r'\n\s*$',       # Trailing empty lines
        ]
        
        for pattern in unwanted_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()
    
    def _extract_articles_enhanced(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract articles with enhanced validation
        
        Args:
            soup (BeautifulSoup): Parsed HTML soup object
            
        Returns:
            List[Dict[str, Any]]: List of extracted articles
        """
        articles = []
        
        # Find article elements
        article_elements = soup.find_all(['div', 'section'], class_=re.compile(r'article|Ï°∞Î¨∏'))
        
        for element in article_elements:
            article_text = element.get_text(strip=True)
            
            # Extract article number and title
            article_match = self.article_pattern.search(article_text)
            if article_match:
                article_number = f"??article_match.group(1)}Ï°?
                article_title = article_match.group(2) if len(article_match.groups()) > 1 else ""
                
                # Extract content
                content_start = article_match.end()
                article_content = article_text[content_start:].strip()
                
                # Validate article content
                if len(article_content) > 10:  # Minimum content length
                    articles.append({
                        'article_number': article_number,
                        'article_title': article_title,
                        'article_content': article_content,
                        'sub_articles': self._extract_sub_articles_from_html(element)
                    })
        
        return articles
    
    def _extract_sub_articles_from_html(self, element: Tag) -> List[Dict[str, Any]]:
        """
        Extract sub-articles from HTML element
        
        Args:
            element (Tag): HTML element containing article
            
        Returns:
            List[Dict[str, Any]]: List of sub-articles
        """
        sub_articles = []
        
        # Find sub-article elements using article patterns
        article_pattern = re.compile(r'??\d+)Ï°?)
        matches = article_pattern.findall(element.get_text())
        for match in matches:
            sub_articles.append({
                'type': 'Ï°?,
                'number': match,
                'content': ''  # Content extraction can be enhanced
            })
        
        return sub_articles
    
    def _extract_clean_text(self, soup: BeautifulSoup) -> str:
        """
        Extract clean text from HTML, removing navigation and formatting elements
        
        Args:
            soup (BeautifulSoup): Parsed HTML soup object
            
        Returns:
            str: Clean text content
        """
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _extract_articles(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract individual articles (??Ï°? ??Ï°?etc) from HTML
        
        Args:
            soup (BeautifulSoup): Parsed HTML soup object
            
        Returns:
            List[Dict[str, Any]]: List of extracted articles
        """
        articles = []
        
        # Find all text nodes that contain article patterns
        text_content = soup.get_text()
        
        # Split by article patterns
        article_sections = self.article_pattern.split(text_content)
        
        for i in range(1, len(article_sections), 2):
            if i + 1 < len(article_sections):
                article_number = article_sections[i]
                article_content = article_sections[i + 1]
                
                # Extract article title (content in parentheses)
                title_match = re.search(r'\(([^)]+)\)', article_content)
                article_title = title_match.group(1) if title_match else ''
                
                # Extract main content (after title)
                main_content = re.sub(r'^\([^)]+\)\s*', '', article_content).strip()
                
                # Extract sub-articles
                sub_articles = self._extract_sub_articles(main_content)
                
                articles.append({
                    'article_number': f'??article_number}Ï°?,
                    'article_title': article_title,
                    'article_content': main_content,
                    'sub_articles': sub_articles
                })
        
        return articles
    
    def _extract_sub_articles(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract sub-articles (?? ?? Î™? from article content
        
        Args:
            content (str): Article content text
            
        Returns:
            List[Dict[str, Any]]: List of sub-articles
        """
        sub_articles = []
        
        # Extract ??(paragraphs)
        for match in self.sub_article_patterns['??].finditer(content):
            sub_articles.append({
                'type': '??,
                'number': int(match.group(1)),
                'content': self._extract_sub_content(content, match.start())
            })
        
        # Extract ??(sub-paragraphs)
        for match in self.sub_article_patterns['??].finditer(content):
            sub_articles.append({
                'type': '??,
                'number': int(match.group(1)),
                'content': self._extract_sub_content(content, match.start())
            })
        
        # Extract Î™?(items) with sequence validation
        mok_items = []
        for match in self.sub_article_patterns['Î™?].finditer(content):
            mok_letter = match.group(1)
            mok_content = match.group(2).strip()
            
            # Convert Korean letter to number for sorting
            mok_number = ord(mok_letter) - ord('Í∞Ä') + 1
            
            mok_items.append({
                'type': 'Î™?,
                'number': mok_number,
                'letter': mok_letter,
                'content': mok_content,
                'position': match.start()
            })
        
        # Sequence validation: check if starts with Í∞Ä., ??, ?? in proper order
        if mok_items:
            # Sort by position
            mok_items.sort(key=lambda x: x['position'])
            
            # Sequence validation
            letters = [item['letter'] for item in mok_items]
            expected_sequence = ['Í∞Ä', '??, '??, '??, 'Îß?, 'Î∞?, '??, '??, '??, 'Ï∞?, 'Ïπ?, '?Ä', '??, '??]
            
            # Must start with Í∞Ä.
            if letters[0] == 'Í∞Ä':
                # Check if sequence is proper
                is_proper_sequence = True
                for i, letter in enumerate(letters):
                    if i < len(expected_sequence) and letter != expected_sequence[i]:
                        is_proper_sequence = False
                        break
                
                # Must have at least 2 items
                if is_proper_sequence and len(mok_items) >= 2:
                    sub_articles.extend(mok_items)
                # Otherwise reject all Î™?items
        
        return sub_articles
    
    def _extract_sub_content(self, content: str, start_pos: int) -> str:
        """
        Extract content for a sub-article from a given position
        
        Args:
            content (str): Full content text
            start_pos (int): Starting position of sub-article
            
        Returns:
            str: Sub-article content
        """
        # Find the end of this sub-article (next sub-article or end of content)
        end_pos = len(content)
        
        # Look for next sub-article pattern
        remaining_text = content[start_pos:]
        for pattern in self.sub_article_patterns.values():
            next_match = pattern.search(remaining_text[1:])  # Skip first char to avoid matching itself
            if next_match:
                potential_end = start_pos + 1 + next_match.start()
                end_pos = min(end_pos, potential_end)
        
        return content[start_pos:end_pos].strip()
    
    def _extract_html_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract metadata from HTML structure
        
        Args:
            soup (BeautifulSoup): Parsed HTML soup object
            
        Returns:
            Dict[str, Any]: Extracted metadata
        """
        metadata = {}
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['html_title'] = title_tag.get_text().strip()
        
        # Extract meta tags
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[f'meta_{name}'] = content
        
        # Extract links
        links = soup.find_all('a', href=True)
        metadata['links'] = [{'text': link.get_text().strip(), 'href': link['href']} 
                           for link in links if link.get_text().strip()]
        
        return metadata
    
    def extract_references(self, text: str) -> List[str]:
        """
        Extract legal references from text
        
        Args:
            text (str): Text content to analyze
            
        Returns:
            List[str]: List of legal references found
        """
        references = []
        
        # Pattern for legal references
        reference_patterns = [
            r'??[^??+)??,  # Quoted law names
            r'Í∞ôÏ? Î≤?,      # Same law reference
            r'?ôÎ≤ï',         # Same law reference (alternative)
            r'??Î≤?,        # This law reference
        ]
        
        for pattern in reference_patterns:
            matches = re.findall(pattern, text)
            references.extend(matches)
        
        # Remove duplicates and empty strings
        references = list(set(ref.strip() for ref in references if ref.strip()))
        
        return references
