#!/usr/bin/env python3
"""
assembly_articles 테이블에서 조문 데이터를 기반으로 키워드를 추출하는 스크립트
"""

import sys
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.data.database import DatabaseManager
from source.services.legal_text_preprocessor import LegalTextPreprocessor

class ArticleKeywordExtractor:
    """조문 기반 키워드 추출기"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.preprocessor = LegalTextPreprocessor()
        self.extracted_keywords = defaultdict(set)
        
    def extract_keywords_from_articles(self) -> Dict[str, List[str]]:
        """assembly_articles 테이블에서 키워드 추출"""
        print("📋 assembly_articles 테이블에서 조문 정보 조회 중...")
        
        # 조문 데이터 조회 (법률명과 함께)
        query = """
        SELECT 
            a.article_content,
            a.article_title,
            a.article_number,
            l.law_name,
            l.category,
            l.ministry,
            l.law_type
        FROM assembly_articles a
        JOIN assembly_laws l ON a.law_id = l.law_id
        WHERE a.article_content IS NOT NULL 
        AND a.article_content != ''
        ORDER BY l.law_name, a.article_number
        LIMIT 10000
        """
        
        articles = self.db_manager.execute_query(query)
        print(f"✅ 총 {len(articles)}개 조문 발견")
        
        # 도메인별 키워드 추출
        domain_keywords = defaultdict(set)
        
        for i, article in enumerate(articles):
            if i % 1000 == 0:
                print(f"  📄 처리 중: {i+1}/{len(articles)}")
            
            law_name = article['law_name']
            category = article['category'] or '기타'
            ministry = article['ministry'] or '기타'
            law_type = article['law_type'] or '법률'
            article_content = article['article_content']
            article_title = article['article_title'] or ''
            article_number = article['article_number']
            
            # 도메인 분류
            domain = self._classify_law_domain(law_name, category, ministry, law_type)
            
            # 키워드 추출
            extracted = self._extract_keywords_from_article(article_content, article_title, law_name, article_number)
            domain_keywords[domain].update(extracted)
        
        # Set을 List로 변환하고 정렬
        result = {}
        for domain, keywords in domain_keywords.items():
            result[domain] = sorted(list(keywords))
            
        return result
    
    def _classify_law_domain(self, law_name: str, category: str, ministry: str, law_type: str) -> str:
        """법령을 도메인별로 분류"""
        law_name_lower = law_name.lower()
        
        # 민사법 도메인
        if any(keyword in law_name_lower for keyword in ['민법', '계약', '채권', '채무', '손해배상', '불법행위', '물권']):
            return '민사법'
        
        # 형사법 도메인
        if any(keyword in law_name_lower for keyword in ['형법', '형사', '범죄', '처벌', '수사', '형벌']):
            return '형사법'
        
        # 가족법 도메인
        if any(keyword in law_name_lower for keyword in ['가족', '혼인', '이혼', '양육', '상속', '친족', '가족관계']):
            return '가족법'
        
        # 상사법 도메인
        if any(keyword in law_name_lower for keyword in ['상법', '회사', '주식', '어음', '수표', '보험', '해상', '상행위']):
            return '상사법'
        
        # 노동법 도메인
        if any(keyword in law_name_lower for keyword in ['근로', '노동', '임금', '고용', '해고', '산업안전', '근로기준']):
            return '노동법'
        
        # 부동산법 도메인
        if any(keyword in law_name_lower for keyword in ['부동산', '토지', '등기', '임대차', '전세', '매매', '부동산등기']):
            return '부동산법'
        
        # 지적재산권법 도메인
        if any(keyword in law_name_lower for keyword in ['특허', '상표', '저작권', '디자인', '영업비밀', '지적재산', '특허법', '상표법']):
            return '지적재산권법'
        
        # 세법 도메인
        if any(keyword in law_name_lower for keyword in ['세법', '소득세', '법인세', '부가가치세', '조세', '국세', '지방세']):
            return '세법'
        
        # 민사소송법 도메인
        if any(keyword in law_name_lower for keyword in ['민사소송', '소송', '집행', '강제집행', '민사집행', '민사소송법']):
            return '민사소송법'
        
        # 형사소송법 도메인
        if any(keyword in law_name_lower for keyword in ['형사소송', '수사', '기소', '공소', '변호', '재판', '형사소송법']):
            return '형사소송법'
        
        # 행정법 도메인
        if any(keyword in law_name_lower for keyword in ['행정', '허가', '인가', '면허', '신고', '처분', '행정법']):
            return '행정법'
        
        # 기타
        return '기타/일반'
    
    def _extract_keywords_from_article(self, content: str, title: str, law_name: str, article_number: int) -> Set[str]:
        """조문에서 키워드 추출"""
        if not content:
            return set()
        
        keywords = set()
        
        # 1. 조문 제목에서 키워드 추출
        if title:
            title_keywords = self._extract_keywords_from_text(title)
            keywords.update(title_keywords)
        
        # 2. 조문 내용에서 키워드 추출
        content_keywords = self._extract_keywords_from_text(content)
        keywords.update(content_keywords)
        
        # 3. 법률명에서 키워드 추출
        law_keywords = self._extract_keywords_from_law_name(law_name)
        keywords.update(law_keywords)
        
        # 4. 조문 번호 관련 키워드
        if article_number:
            keywords.add(f"제{article_number}조")
        
        return keywords
    
    def _extract_keywords_from_text(self, text: str) -> Set[str]:
        """텍스트에서 키워드 추출"""
        if not text:
            return set()
        
        # 텍스트 전처리
        cleaned_text = self.preprocessor.clean_text(text)
        
        keywords = set()
        
        # 1. 법률 조문 번호 패턴 (제1조, 제2조 등)
        article_pattern = r'제\d+조'
        articles = re.findall(article_pattern, cleaned_text)
        keywords.update(articles)
        
        # 2. 법률 용어 패턴 (조, 항, 호, 목 등)
        legal_terms = re.findall(r'\d+조|\d+항|\d+호|\d+목', cleaned_text)
        keywords.update(legal_terms)
        
        # 3. 한글 법률 용어 (2-6글자)
        korean_terms = re.findall(r'[가-힣]{2,6}(?=은|는|이|가|을|를|에|의|에서|으로|로|와|과|에|에게|에게서)', cleaned_text)
        keywords.update(korean_terms)
        
        # 4. 일반적인 법률 용어
        common_legal_terms = [
            '법률', '법령', '규정', '조항', '조문', '항목', '호목',
            '시행', '공포', '개정', '폐지', '제정', '부칙', '본칙',
            '권리', '의무', '책임', '처벌', '벌금', '징역', '형',
            '소송', '재판', '판결', '선고', '확정', '상고', '항소',
            '신청', '청구', '제출', '제기', '기각', '인용', '각하',
            '계약', '손해배상', '불법행위', '채권', '채무', '소유권',
            '혼인', '이혼', '양육', '상속', '친권', '양육권',
            '회사', '주식', '주주', '이사', '상행위', '법인',
            '근로', '고용', '임금', '해고', '부당해고', '근로계약',
            '부동산', '토지', '등기', '매매', '임대', '소유권이전',
            '특허', '상표', '저작권', '디자인', '영업비밀',
            '세금', '소득세', '법인세', '부가가치세', '신고', '납부',
            '수사', '기소', '공소', '변호', '재심', '자백'
        ]
        
        for term in common_legal_terms:
            if term in cleaned_text:
                keywords.add(term)
        
        # 5. 특수문자와 숫자가 포함된 용어 제거
        filtered_keywords = set()
        for keyword in keywords:
            if len(keyword) >= 2 and not re.search(r'[^\w가-힣]', keyword):
                filtered_keywords.add(keyword)
        
        return filtered_keywords
    
    def _extract_keywords_from_law_name(self, law_name: str) -> Set[str]:
        """법률명에서 키워드 추출"""
        keywords = set()
        
        # 법률명을 공백이나 특수문자로 분리
        words = re.split(r'[^\w가-힣]+', law_name)
        
        for word in words:
            if len(word) >= 2:
                keywords.add(word)
        
        return keywords
    
    def save_keywords_to_database(self, domain_keywords: Dict[str, List[str]]):
        """추출된 키워드를 데이터베이스에 저장"""
        print("💾 추출된 키워드를 데이터베이스에 저장 중...")
        
        # 기존 키워드 데이터베이스 로드
        db_path = "data/legal_terms_database.json"
        if os.path.exists(db_path):
            with open(db_path, 'r', encoding='utf-8') as f:
                existing_db = json.load(f)
        else:
            existing_db = {}
        
        # 새로운 키워드 추가
        for domain, keywords in domain_keywords.items():
            if domain not in existing_db:
                existing_db[domain] = {}
            
            for keyword in keywords:
                if keyword not in existing_db[domain]:
                    existing_db[domain][keyword] = {
                        "weight": 0.8,  # 조문 기반이므로 높은 가중치
                        "synonyms": [],
                        "related_terms": [],
                        "context_keywords": [],
                        "source": "assembly_articles",
                        "confidence": 0.9,
                        "verified": True,
                        "added_date": "2025-10-19"
                    }
        
        # 데이터베이스 저장
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(existing_db, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 키워드 데이터베이스 업데이트 완료: {db_path}")
    
    def print_statistics(self, domain_keywords: Dict[str, List[str]]):
        """추출 통계 출력"""
        print("\n" + "="*60)
        print("📊 조문 기반 키워드 추출 통계")
        print("="*60)
        
        total_keywords = 0
        for domain, keywords in domain_keywords.items():
            count = len(keywords)
            total_keywords += count
            print(f"  {domain}: {count:,}개 키워드")
        
        print(f"\n  총 키워드 수: {total_keywords:,}개")
        print("="*60)

def main():
    """메인 함수"""
    try:
        extractor = ArticleKeywordExtractor()
        
        # 키워드 추출
        domain_keywords = extractor.extract_keywords_from_articles()
        
        # 통계 출력
        extractor.print_statistics(domain_keywords)
        
        # 데이터베이스에 저장
        extractor.save_keywords_to_database(domain_keywords)
        
        print("\n✅ 조문 기반 키워드 추출 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
