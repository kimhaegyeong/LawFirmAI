#!/usr/bin/env python3
"""
개선된 HTML 파서

이 스크립트는 법령 HTML에서 모든 조문을 정확히 추출하는 개선된 파서입니다.
"""

import re
import logging
from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup, Tag, NavigableString

logger = logging.getLogger(__name__)


class ImprovedLawHTMLParser:
    """개선된 법령 HTML 파서"""
    
    def __init__(self):
        """초기화"""
        # 조문 패턴들
        self.article_pattern = re.compile(r'제(\d+)조\s*(?:\(([^)]+)\))?')
        self.sub_article_patterns = {
            '항': re.compile(r'제(\d+)항'),
            '호': re.compile(r'제(\d+)호'),
            '목': re.compile(r'^([가-힣])\.\s+(.+?)(?=\n[가-힣]\.|\n\d+\.|\n[①②③④⑤⑥⑦⑧⑨⑩]|$)', re.MULTILINE | re.DOTALL)
        }
        
        # 불필요한 요소들 제거 패턴
        self.unwanted_patterns = [
            r'조문버튼\s*',
            r'선택체크\s*',
            r'펼치기\s*',
            r'접기\s*',
            r'선택\s*',
            r'소개\s*',
            r'버튼\s*',
            r'javascript:\s*',
            r'onclick\s*',
            r'href="#\s*',
            r'href="javascript:\s*',
            r'class="btn\s*',
            r'class="button\s*',
            r'<.*?>',  # HTML 태그
            r'&nbsp;',
            r'&amp;',
            r'&lt;',
            r'&gt;',
            r'&quot;',
            r'&apos;',
        ]
    
    def parse_html(self, html_content: str) -> Dict[str, Any]:
        """
        HTML 내용을 파싱하여 구조화된 정보 추출
        
        Args:
            html_content (str): 원본 HTML 내용
            
        Returns:
            Dict[str, Any]: 파싱된 내용
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 불필요한 요소 제거
            self._remove_unwanted_elements(soup)
            
            # 텍스트 추출
            clean_text = self._extract_clean_text(soup)
            
            # 조문 추출 (개선된 방법)
            articles = self._extract_articles_from_text(clean_text)
            
            return {
                'clean_text': clean_text,
                'articles': articles,
                'metadata': self._extract_metadata(soup)
            }
            
        except Exception as e:
            logger.error(f"HTML 파싱 중 오류: {e}")
            return {
                'clean_text': '',
                'articles': [],
                'metadata': {}
            }
    
    def _remove_unwanted_elements(self, soup: BeautifulSoup):
        """불필요한 HTML 요소 제거"""
        # 스크립트, 스타일 제거
        for element in soup(['script', 'style', 'noscript']):
            element.decompose()
        
        # UI 요소 제거
        ui_tags = ['button', 'input', 'select', 'textarea']
        for tag in ui_tags:
            for element in soup.find_all(tag):
                element.decompose()
        
        # 특정 클래스의 요소 제거
        ui_classes = ['button', 'btn', 'nav', 'menu', 'sidebar', 'header', 'footer']
        for class_name in ui_classes:
            try:
                for element in soup.find_all(class_=class_name):
                    element.decompose()
            except Exception:
                continue
        
        # JavaScript 속성 제거
        for element in soup.find_all(attrs={'onclick': True}):
            element.decompose()
        
        # 링크 정리
        for element in soup.find_all('a', href=lambda x: x and ('javascript:' in x or x.startswith('#'))):
            element.decompose()
    
    def _extract_clean_text(self, soup: BeautifulSoup) -> str:
        """깨끗한 텍스트 추출"""
        # 메인 콘텐츠 영역 찾기
        main_content = soup.find('div', class_='article') or soup.find('body')
        
        if not main_content:
            return soup.get_text(separator='\n', strip=True)
        
        # 텍스트 추출
        text = main_content.get_text(separator='\n', strip=True)
        
        # 추가 정리
        text = self._clean_text(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        # 불필요한 패턴 제거
        for pattern in self.unwanted_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        # 공백 정규화
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()
    
    def _extract_articles_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        텍스트에서 조문 추출 (개선된 방법)
        
        Args:
            text (str): 정리된 텍스트
            
        Returns:
            List[Dict[str, Any]]: 추출된 조문 목록
        """
        articles = []
        
        # 모든 조문 패턴 찾기
        article_matches = list(self.article_pattern.finditer(text))
        
        if not article_matches:
            logger.warning("조문을 찾을 수 없습니다.")
            return articles
        
        # 조문별로 내용 추출
        for i, match in enumerate(article_matches):
            try:
                article_number = f"제{match.group(1)}조"
                article_title = match.group(2) if match.group(2) else ""
                
                # 다음 조문까지의 내용 추출
                start_pos = match.start()
                if i + 1 < len(article_matches):
                    end_pos = article_matches[i + 1].start()
                else:
                    end_pos = len(text)
                
                article_content = text[start_pos:end_pos].strip()
                
                # 제목 제거 (이미 추출했으므로)
                if article_title:
                    title_pattern = rf'제{match.group(1)}조\s*\({re.escape(article_title)}\)'
                    article_content = re.sub(title_pattern, '', article_content).strip()
                else:
                    article_pattern = rf'제{match.group(1)}조'
                    article_content = re.sub(article_pattern, '', article_content).strip()
                
                # 내용이 충분한지 확인
                if len(article_content) > 10:
                    # 하위 조문 추출
                    sub_articles = self._extract_sub_articles(article_content)
                    
                    articles.append({
                        'article_number': article_number,
                        'article_title': article_title,
                        'article_content': article_content,
                        'sub_articles': sub_articles,
                        'word_count': len(article_content.split()),
                        'char_count': len(article_content)
                    })
                    
            except Exception as e:
                logger.error(f"조문 {match.group(1)} 추출 중 오류: {e}")
                continue
        
        # 중복 조문 제거 및 정렬
        articles = self._remove_duplicate_articles(articles)
        
        # 조문 번호순 정렬
        articles.sort(key=lambda x: int(re.search(r'제(\d+)조', x['article_number']).group(1)))
        
        logger.info(f"총 {len(articles)}개 조문 추출 완료")
        return articles
    
    def _remove_duplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """중복 조문 제거"""
        unique_articles = {}
        
        for article in articles:
            article_number = article['article_number']
            
            # 같은 번호의 조문이 있으면 더 긴 내용을 선택
            if article_number in unique_articles:
                existing = unique_articles[article_number]
                if len(article['article_content']) > len(existing['article_content']):
                    unique_articles[article_number] = article
            else:
                unique_articles[article_number] = article
        
        return list(unique_articles.values())
    
    def _extract_sub_articles(self, content: str) -> List[Dict[str, Any]]:
        """하위 조문 추출"""
        sub_articles = []
        
        # 항 추출
        for match in self.sub_article_patterns['항'].finditer(content):
            sub_articles.append({
                'type': '항',
                'number': int(match.group(1)),
                'content': self._extract_sub_content(content, match.start())
            })
        
        # 호 추출
        for match in self.sub_article_patterns['호'].finditer(content):
            sub_articles.append({
                'type': '호',
                'number': int(match.group(1)),
                'content': self._extract_sub_content(content, match.start())
            })
        
        # 목 추출 (시퀀스 검증 포함)
        mok_items = []
        for match in self.sub_article_patterns['목'].finditer(content):
            mok_letter = match.group(1)
            mok_content = match.group(2).strip()
            
            # Convert Korean letter to number for sorting
            mok_number = ord(mok_letter) - ord('가') + 1
            
            mok_items.append({
                'type': '목',
                'number': mok_number,
                'letter': mok_letter,
                'content': mok_content,
                'position': match.start()
            })
        
        # 시퀀스 검증: 가., 나., 다. 순서로 시작하는지 확인
        if mok_items:
            # 정렬
            mok_items.sort(key=lambda x: x['position'])
            
            # 시퀀스 검증
            letters = [item['letter'] for item in mok_items]
            expected_sequence = ['가', '나', '다', '라', '마', '바', '사', '아', '자', '차', '카', '타', '파', '하']
            
            # 가.로 시작하는지 확인
            if letters[0] == '가':
                # 순서가 올바른지 확인
                is_proper_sequence = True
                for i, letter in enumerate(letters):
                    if i < len(expected_sequence) and letter != expected_sequence[i]:
                        is_proper_sequence = False
                        break
                
                # 최소 2개 이상의 목이 있어야 함
                if is_proper_sequence and len(mok_items) >= 2:
                    sub_articles.extend(mok_items)
                else:
                    # 시퀀스가 올바르지 않거나 목이 1개뿐인 경우 제외
                    pass
        
        return sub_articles
    
    def _extract_sub_content(self, content: str, start_pos: int) -> str:
        """하위 조문 내용 추출"""
        end_pos = len(content)
        
        # 다음 하위 조문 찾기
        remaining_text = content[start_pos:]
        for pattern in self.sub_article_patterns.values():
            next_match = pattern.search(remaining_text[1:])
            if next_match:
                potential_end = start_pos + 1 + next_match.start()
                end_pos = min(end_pos, potential_end)
        
        return content[start_pos:end_pos].strip()
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """메타데이터 추출"""
        metadata = {}
        
        # 제목 추출
        title_tag = soup.find('title')
        if title_tag:
            metadata['html_title'] = title_tag.get_text().strip()
        
        # 메타 태그 추출
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[f'meta_{name}'] = content
        
        return metadata


def test_parser():
    """파서 테스트"""
    import json
    
    # 테스트 데이터 로드
    with open('data/raw/assembly/law/20251010/law_page_001_181503.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    law = data['laws'][0]
    html_content = law['content_html']
    
    # 파서 테스트
    parser = ImprovedLawHTMLParser()
    result = parser.parse_html(html_content)
    
    print(f"법령명: {law['law_name']}")
    print(f"추출된 조문 수: {len(result['articles'])}")
    print(f"깨끗한 텍스트 길이: {len(result['clean_text'])}")
    
    # 첫 5개 조문 출력
    for i, article in enumerate(result['articles'][:5]):
        print(f"\n{article['article_number']} {article['article_title']}")
        print(f"내용 길이: {len(article['article_content'])}")
        print(f"하위 조문 수: {len(article['sub_articles'])}")
        print(f"내용 미리보기: {article['article_content'][:100]}...")


if __name__ == "__main__":
    test_parser()
