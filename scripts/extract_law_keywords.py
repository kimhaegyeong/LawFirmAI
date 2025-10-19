#!/usr/bin/env python3
"""
assembly_laws 테이블에서 법령 정보를 기반으로 키워드를 추출하는 스크립트
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

class LawKeywordExtractor:
    """법령 기반 키워드 추출기"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.preprocessor = LegalTextPreprocessor()
        self.extracted_keywords = defaultdict(set)
        
    def extract_keywords_from_laws(self) -> Dict[str, List[str]]:
        """assembly_laws 테이블에서 키워드 추출"""
        print("📋 assembly_laws 테이블에서 법령 정보 조회 중...")
        
        # 법령 데이터 조회
        query = """
        SELECT 
            law_name,
            category,
            ministry,
            full_text,
            keywords,
            law_type
        FROM assembly_laws 
        WHERE full_text IS NOT NULL 
        AND full_text != ''
        ORDER BY law_name
        """
        
        laws = self.db_manager.execute_query(query)
        print(f"✅ 총 {len(laws)}개 법령 발견")
        
        # 도메인별 키워드 추출
        domain_keywords = defaultdict(set)
        
        for law in laws:
            law_name = law['law_name']
            category = law['category'] or '기타'
            ministry = law['ministry'] or '기타'
            full_text = law['full_text']
            keywords = law['keywords']
            law_type = law['law_type'] or '법률'
            
            print(f"  📄 처리 중: {law_name}")
            
            # 도메인 분류
            domain = self._classify_law_domain(law_name, category, ministry, law_type)
            
            # 키워드 추출
            extracted = self._extract_keywords_from_text(full_text, law_name)
            domain_keywords[domain].update(extracted)
            
            # 기존 키워드가 있다면 추가
            if keywords:
                try:
                    existing_keywords = json.loads(keywords)
                    if isinstance(existing_keywords, list):
                        domain_keywords[domain].update(existing_keywords)
                except:
                    pass
        
        # Set을 List로 변환하고 정렬
        result = {}
        for domain, keywords in domain_keywords.items():
            result[domain] = sorted(list(keywords))
            
        return result
    
    def _classify_law_domain(self, law_name: str, category: str, ministry: str, law_type: str) -> str:
        """법령을 도메인별로 분류"""
        law_name_lower = law_name.lower()
        
        # 민사법 도메인
        if any(keyword in law_name_lower for keyword in ['민법', '계약', '채권', '채무', '손해배상', '불법행위']):
            return '민사법'
        
        # 형사법 도메인
        if any(keyword in law_name_lower for keyword in ['형법', '형사', '범죄', '처벌', '수사']):
            return '형사법'
        
        # 가족법 도메인
        if any(keyword in law_name_lower for keyword in ['가족', '혼인', '이혼', '양육', '상속', '친족']):
            return '가족법'
        
        # 상사법 도메인
        if any(keyword in law_name_lower for keyword in ['상법', '회사', '주식', '어음', '수표', '보험', '해상']):
            return '상사법'
        
        # 노동법 도메인
        if any(keyword in law_name_lower for keyword in ['근로', '노동', '임금', '고용', '해고', '산업안전']):
            return '노동법'
        
        # 부동산법 도메인
        if any(keyword in law_name_lower for keyword in ['부동산', '토지', '등기', '임대차', '전세', '매매']):
            return '부동산법'
        
        # 지적재산권법 도메인
        if any(keyword in law_name_lower for keyword in ['특허', '상표', '저작권', '디자인', '영업비밀', '지적재산']):
            return '지적재산권법'
        
        # 세법 도메인
        if any(keyword in law_name_lower for keyword in ['세법', '소득세', '법인세', '부가가치세', '조세', '국세']):
            return '세법'
        
        # 민사소송법 도메인
        if any(keyword in law_name_lower for keyword in ['민사소송', '소송', '집행', '강제집행', '민사집행']):
            return '민사소송법'
        
        # 형사소송법 도메인
        if any(keyword in law_name_lower for keyword in ['형사소송', '수사', '기소', '공소', '변호', '재판']):
            return '형사소송법'
        
        # 행정법 도메인
        if any(keyword in law_name_lower for keyword in ['행정', '허가', '인가', '면허', '신고', '처분']):
            return '행정법'
        
        # 기타
        return '기타/일반'
    
    def _extract_keywords_from_text(self, text: str, law_name: str) -> Set[str]:
        """텍스트에서 키워드 추출"""
        if not text:
            return set()
        
        # 텍스트 전처리
        cleaned_text = self.preprocessor.clean_text(text)
        
        # 법률 용어 패턴 추출
        keywords = set()
        
        # 1. 법률 조문 번호 패턴 (제1조, 제2조 등)
        article_pattern = r'제\d+조'
        articles = re.findall(article_pattern, cleaned_text)
        keywords.update(articles)
        
        # 2. 법률 용어 패턴 (조, 항, 호, 목 등)
        legal_terms = re.findall(r'\d+조|\d+항|\d+호|\d+목', cleaned_text)
        keywords.update(legal_terms)
        
        # 3. 한글 법률 용어 (2-4글자)
        korean_terms = re.findall(r'[가-힣]{2,4}(?=은|는|이|가|을|를|에|의|에서|으로|로|와|과)', cleaned_text)
        keywords.update(korean_terms)
        
        # 4. 법률명에서 키워드 추출
        law_keywords = self._extract_keywords_from_law_name(law_name)
        keywords.update(law_keywords)
        
        # 5. 일반적인 법률 용어
        common_legal_terms = [
            '법률', '법령', '규정', '조항', '조문', '항목', '호목',
            '시행', '공포', '개정', '폐지', '제정', '부칙', '본칙',
            '권리', '의무', '책임', '처벌', '벌금', '징역', '형',
            '소송', '재판', '판결', '선고', '확정', '상고', '항소',
            '신청', '청구', '제출', '제기', '기각', '인용', '각하'
        ]
        
        for term in common_legal_terms:
            if term in cleaned_text:
                keywords.add(term)
        
        # 6. 특수문자와 숫자가 포함된 용어 제거
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
                        "weight": 0.7,  # 기본 가중치
                        "synonyms": [],
                        "related_terms": [],
                        "context_keywords": [],
                        "source": "assembly_laws",
                        "confidence": 0.8,
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
        print("📊 법령 기반 키워드 추출 통계")
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
        extractor = LawKeywordExtractor()
        
        # 키워드 추출
        domain_keywords = extractor.extract_keywords_from_laws()
        
        # 통계 출력
        extractor.print_statistics(domain_keywords)
        
        # 데이터베이스에 저장
        extractor.save_keywords_to_database(domain_keywords)
        
        print("\n✅ 법령 기반 키워드 추출 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
