#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
부칙 파싱 개선 스크립트
대한민국 법률 부칙 작성 규칙에 따른 부칙 인식 및 파싱 개선
"""

import json
import sys
import os
import re
from pathlib import Path
import logging

# Windows 콘솔에서 UTF-8 인코딩 설정
if os.name == 'nt':  # Windows
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# 파서 모듈 경로 추가
sys.path.append(str(Path(__file__).parent / 'parsers'))

from parsers.improved_article_parser import ImprovedArticleParser

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedArticleParser(ImprovedArticleParser):
    """부칙 파싱이 개선된 조문 파서"""
    
    def __init__(self):
        super().__init__()
        # 부칙 관련 정규식 패턴들
        self.supplementary_patterns = [
            # 기본 부칙 패턴
            r'부칙\s*<[^>]*>',
            r'부칙\s*$',
            r'부칙\s*펼치기접기',
            
            # 부칙 조문 패턴 (제1조, 제2조 등)
            r'제(\d+)조\s*\([^)]*\)',
            r'제(\d+)조\s*[가-힣]',
            
            # 시행일 관련 패턴
            r'이\s*[법령규칙]\s*은\s*공포한\s*날부터\s*시행한다',
            r'이\s*[법령규칙]\s*은\s*공포\s*후\s*\d+개월이\s*경과한\s*날부터\s*시행한다',
            r'이\s*[법령규칙]\s*은\s*\d{4}년\s*\d{1,2}월\s*\d{1,2}일부터\s*시행한다',
            
            # 경과조치 관련 패턴
            r'경과조치',
            r'적용례',
            r'종전의\s*규정',
            r'이\s*[법령규칙]\s*시행\s*당시',
        ]
        
        # 부칙 시작을 나타내는 키워드들
        self.supplementary_keywords = [
            '부칙', '시행일', '경과조치', '적용례', '준비행위', 
            '종전 처분', '다른 법률의 개정', '다른 법률의 폐지'
        ]
    
    def _identify_supplementary_section(self, content: str) -> tuple:
        """부칙 섹션 식별"""
        lines = content.split('\n')
        supplementary_start = -1
        supplementary_end = len(lines)
        
        # 부칙 시작점 찾기
        for i, line in enumerate(lines):
            line_clean = line.strip()
            
            # 부칙 시작 패턴 확인
            if re.search(r'부칙\s*<[^>]*>', line_clean) or \
               re.search(r'부칙\s*$', line_clean) or \
               re.search(r'부칙\s*펼치기접기', line_clean):
                supplementary_start = i
                break
        
        if supplementary_start == -1:
            return None, None
        
        # 부칙 끝점 찾기 (다음 섹션이나 문서 끝)
        for i in range(supplementary_start + 1, len(lines)):
            line_clean = lines[i].strip()
            
            # 부칙이 아닌 다른 섹션 시작 확인
            if line_clean and not self._is_supplementary_line(line_clean):
                # 부칙이 아닌 새로운 섹션인지 확인
                if re.match(r'제\d+조', line_clean) and not self._is_supplementary_article(line_clean):
                    supplementary_end = i
                    break
        
        return supplementary_start, supplementary_end
    
    def _is_supplementary_line(self, line: str) -> bool:
        """부칙 관련 라인인지 확인"""
        # 빈 라인은 부칙에 포함될 수 있음
        if not line.strip():
            return True
        
        # 부칙 키워드 확인
        for keyword in self.supplementary_keywords:
            if keyword in line:
                return True
        
        # 부칙 조문 패턴 확인
        if re.match(r'제\d+조\s*\([^)]*\)', line) or \
           re.match(r'제\d+조\s*[가-힣]', line):
            return True
        
        # 시행일 관련 패턴 확인
        if re.search(r'시행한다', line) or \
           re.search(r'공포한\s*날부터', line) or \
           re.search(r'경과한\s*날부터', line):
            return True
        
        return False
    
    def _is_supplementary_article(self, line: str) -> bool:
        """부칙 조문인지 확인"""
        # 부칙 조문의 특징적인 패턴들
        supplementary_patterns = [
            r'시행일',
            r'경과조치',
            r'적용례',
            r'준비행위',
            r'종전\s*처분',
            r'다른\s*법률의\s*개정',
            r'다른\s*법률의\s*폐지',
            r'벌칙\s*적용',
            r'과태료\s*적용'
        ]
        
        for pattern in supplementary_patterns:
            if re.search(pattern, line):
                return True
        
        return False
    
    def _parse_supplementary_articles(self, content: str) -> list:
        """부칙 조문 파싱"""
        supplementary_start, supplementary_end = self._identify_supplementary_section(content)
        
        if supplementary_start is None:
            return []
        
        lines = content.split('\n')
        supplementary_content = '\n'.join(lines[supplementary_start:supplementary_end])
        
        # 부칙 조문 추출
        articles = []
        
        # 부칙 조문 패턴으로 매칭 (더 정확한 패턴)
        article_pattern = r'제(\d+)조\s*\(([^)]*)\)\s*(.*?)(?=제\d+조\s*\(|$)'
        matches = re.finditer(article_pattern, supplementary_content, re.DOTALL)
        
        for match in matches:
            article_number = f"부칙제{match.group(1)}조"
            article_title = match.group(2).strip()
            article_content = match.group(3).strip()
            
            # 내용 정리
            article_content = self._clean_content(article_content)
            
            if article_content:
                articles.append({
                    'article_number': article_number,
                    'article_title': article_title,
                    'article_content': article_content,
                    'sub_articles': [],
                    'references': [],
                    'word_count': len(article_content.split()),
                    'char_count': len(article_content),
                    'is_supplementary': True
                })
        
        # 조문이 없는 단순 부칙 처리
        if not articles and supplementary_content.strip():
            # 부칙 헤더 제거
            clean_content = re.sub(r'부칙\s*<[^>]*>펼치기접기', '', supplementary_content)
            clean_content = re.sub(r'부칙\s*<[^>]*>', '', clean_content)
            clean_content = clean_content.strip()
            
            if clean_content:
                articles.append({
                    'article_number': '부칙',
                    'article_title': '',
                    'article_content': clean_content,
                    'sub_articles': [],
                    'references': [],
                    'word_count': len(clean_content.split()),
                    'char_count': len(clean_content),
                    'is_supplementary': True
                })
        
        return articles
    
    def parse_law_document(self, content: str) -> dict:
        """법률 문서 파싱 (부칙 포함)"""
        # 기본 조문 파싱
        result = super().parse_law_document(content)
        
        # 부칙 파싱
        supplementary_articles = self._parse_supplementary_articles(content)
        
        # 결과에 부칙 추가
        if supplementary_articles:
            result['all_articles'].extend(supplementary_articles)
            result['supplementary_articles'] = supplementary_articles
            result['total_supplementary_articles'] = len(supplementary_articles)
            result['total_articles'] += len(supplementary_articles)
        
        return result

def test_supplementary_parsing():
    """부칙 파싱 테스트"""
    
    # 테스트용 법률 문서 (부칙 포함)
    test_content = """제1조(목적) 이 법은 대한민국의 법치주의를 구현하기 위하여 필요한 사항을 규정함을 목적으로 한다.

제2조(정의) 이 법에서 사용하는 용어의 정의는 다음과 같다.
1. "법률"이란 국회에서 제정한 법을 말한다.
2. "명령"이란 행정부에서 제정한 규칙을 말한다.

제3조(적용 범위) 이 법은 대한민국 영토 내에서 적용한다.

부칙 <법률 제20000호, 2025. 1. 15.>

제1조(시행일) 이 법은 공포 후 6개월이 경과한 날부터 시행한다.

제2조(경과조치) 이 법 시행 당시 종전의 규정에 따라 행한 처분은 이 법에 따라 행한 것으로 본다.

제3조(다른 법률의 개정) 근로기준법 일부를 다음과 같이 개정한다.
제56조제1항 중 "8시간"을 "7시간"으로 한다."""
    
    print("=== 부칙 파싱 테스트 ===")
    print("원본 내용:")
    print(test_content)
    print("\n" + "="*80 + "\n")
    
    # 개선된 파서로 테스트
    parser = EnhancedArticleParser()
    result = parser.parse_law_document(test_content)
    
    print("파싱 결과:")
    print(f"총 조문 수: {result['total_articles']}")
    print(f"본칙 조문 수: {result['total_articles'] - result.get('total_supplementary_articles', 0)}")
    print(f"부칙 조문 수: {result.get('total_supplementary_articles', 0)}")
    print()
    
    for i, article in enumerate(result['all_articles']):
        print(f"조문 {i+1}:")
        print(f"  번호: {article['article_number']}")
        print(f"  제목: {article.get('article_title', 'N/A')}")
        print(f"  부칙 여부: {article.get('is_supplementary', False)}")
        print(f"  내용: {article['article_content'][:100]}...")
        print()

def test_real_law_with_supplementary():
    """실제 법률 문서로 부칙 파싱 테스트"""
    
    # 실제 법률 문서 로드
    test_file = "data/processed/assembly/law/ml_enhanced/20251013/_대한민국_법원의_날_제정에_관한_규칙_assembly_law_1951.json"
    
    if not Path(test_file).exists():
        print(f"테스트 파일이 없습니다: {test_file}")
        return
    
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 원본 내용 재구성 (부칙 포함)
    original_content = """제1조(목적) 이 규칙은 대한민국 법원이 사법주권을 회복한 날을 기념하기 위하여 『대한민국 법원의 날』을 제정하고, 사법독립과 법치주의의 중요성을 알리며 그 의의를 기념하기 위한 행사 등을 진행함에 있어 필요한 사항을 규정함을 목적으로 한다.
제2조(정의 및 명칭) ① 제1조에서 사법주권을 회복한 날이라 함은, 일제에 사법주권을 빼앗겼다가 대한민국이 1948년 9월 13일 미군정으로부터 사법권을 이양받음으로써 헌법기관인 대한민국 법원이 실질적으로 수립된 날을 의미한다.
② 『대한민국 법원의 날』은 매년 9월 13일로 한다.
제3조(기념식 및 행사) ① 법원은 『대한민국 법원의 날』에 기념식과 그에 부수되는 행사를 실시할 수 있다.
제4조(포상) ① 대법원장은 제2조제1항에 규정된 기념일의 의식에서 사법부의 발전 또는 법률문화의 향상에 공헌한 행적이 뚜렷한 사람에게 포상할 수 있다.
부칙 <제2605호, 2015.6.29.>펼치기접기
이 규칙은 공포한 날부터 시행한다."""
    
    print("=== 실제 법률 문서 부칙 파싱 테스트 ===")
    
    # 개선된 파서로 테스트
    parser = EnhancedArticleParser()
    result = parser.parse_law_document(original_content)
    
    print("파싱 결과:")
    print(f"총 조문 수: {result['total_articles']}")
    print(f"본칙 조문 수: {result['total_articles'] - result.get('total_supplementary_articles', 0)}")
    print(f"부칙 조문 수: {result.get('total_supplementary_articles', 0)}")
    print()
    
    for i, article in enumerate(result['all_articles']):
        print(f"조문 {i+1}:")
        print(f"  번호: {article['article_number']}")
        print(f"  제목: {article.get('article_title', 'N/A')}")
        print(f"  부칙 여부: {article.get('is_supplementary', False)}")
        print(f"  내용: {article['article_content'][:100]}...")
        print()

if __name__ == "__main__":
    print("부칙 파싱 개선 테스트")
    print("=" * 50)
    
    # 기본 테스트
    test_supplementary_parsing()
    
    print("\n" + "="*80 + "\n")
    
    # 실제 법률 문서 테스트
    test_real_law_with_supplementary()
    
    print("\n부칙 파싱 개선 완료!")
