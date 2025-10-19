#!/usr/bin/env python3
"""
precedent_cases 테이블에서 판례 데이터를 기반으로 키워드를 추출하는 스크립트
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

class PrecedentKeywordExtractor:
    """판례 기반 키워드 추출기"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.preprocessor = LegalTextPreprocessor()
        self.extracted_keywords = defaultdict(set)
        
    def extract_keywords_from_precedents(self) -> Dict[str, List[str]]:
        """precedent_cases 테이블에서 키워드 추출"""
        print("📋 precedent_cases 테이블에서 판례 정보 조회 중...")
        
        # 판례 데이터 조회 (섹션 데이터와 함께)
        query = """
        SELECT 
            p.case_name,
            p.case_number,
            p.category,
            p.field,
            p.court,
            p.full_text,
            p.searchable_text,
            s.section_type,
            s.section_content
        FROM precedent_cases p
        LEFT JOIN precedent_sections s ON p.case_id = s.case_id
        WHERE p.full_text IS NOT NULL 
        AND p.full_text != ''
        ORDER BY p.case_name, s.section_type
        LIMIT 20000
        """
        
        precedents = self.db_manager.execute_query(query)
        print(f"✅ 총 {len(precedents)}개 판례 데이터 발견")
        
        # 도메인별 키워드 추출
        domain_keywords = defaultdict(set)
        
        for i, precedent in enumerate(precedents):
            if i % 2000 == 0:
                print(f"  📄 처리 중: {i+1}/{len(precedents)}")
            
            case_name = precedent['case_name']
            case_number = precedent['case_number']
            category = precedent['category'] or 'civil'
            field = precedent['field'] or '민사'
            court = precedent['court'] or '대법원'
            full_text = precedent['full_text']
            searchable_text = precedent['searchable_text'] or ''
            section_type = precedent['section_type']
            section_content = precedent['section_content'] or ''
            
            # 도메인 분류
            domain = self._classify_precedent_domain(category, field, case_name)
            
            # 키워드 추출
            extracted = self._extract_keywords_from_precedent(
                full_text, searchable_text, section_content, 
                case_name, case_number, section_type
            )
            domain_keywords[domain].update(extracted)
        
        # Set을 List로 변환하고 정렬
        result = {}
        for domain, keywords in domain_keywords.items():
            result[domain] = sorted(list(keywords))
            
        return result
    
    def _classify_precedent_domain(self, category: str, field: str, case_name: str) -> str:
        """판례를 도메인별로 분류"""
        case_name_lower = case_name.lower()
        
        # 형사법 도메인
        if category == 'criminal' or field == '형사' or any(keyword in case_name_lower for keyword in [
            '형법', '형사', '범죄', '처벌', '수사', '형벌', '살인', '절도', '사기', '강도', '강간', '성폭력'
        ]):
            return '형사법'
        
        # 민사법 도메인
        if category == 'civil' or field == '민사' or any(keyword in case_name_lower for keyword in [
            '민법', '계약', '채권', '채무', '손해배상', '불법행위', '물권', '소유권', '점유권'
        ]):
            return '민사법'
        
        # 가족법 도메인
        if category == 'family' or field == '가사' or any(keyword in case_name_lower for keyword in [
            '가족', '혼인', '이혼', '양육', '상속', '친족', '가족관계', '양육권', '친권'
        ]):
            return '가족법'
        
        # 상사법 도메인
        if any(keyword in case_name_lower for keyword in [
            '상법', '회사', '주식', '어음', '수표', '보험', '해상', '상행위', '법인', '주주'
        ]):
            return '상사법'
        
        # 노동법 도메인
        if any(keyword in case_name_lower for keyword in [
            '근로', '노동', '임금', '고용', '해고', '산업안전', '근로기준', '부당해고'
        ]):
            return '노동법'
        
        # 부동산법 도메인
        if any(keyword in case_name_lower for keyword in [
            '부동산', '토지', '등기', '임대차', '전세', '매매', '부동산등기', '소유권이전'
        ]):
            return '부동산법'
        
        # 지적재산권법 도메인
        if category == 'patent' or field == '특허' or any(keyword in case_name_lower for keyword in [
            '특허', '상표', '저작권', '디자인', '영업비밀', '지적재산', '특허법', '상표법', '저작권법'
        ]):
            return '지적재산권법'
        
        # 세법 도메인
        if category == 'tax' or field == '조세' or any(keyword in case_name_lower for keyword in [
            '세법', '소득세', '법인세', '부가가치세', '조세', '국세', '지방세', '세무조사'
        ]):
            return '세법'
        
        # 민사소송법 도메인
        if any(keyword in case_name_lower for keyword in [
            '민사소송', '소송', '집행', '강제집행', '민사집행', '민사소송법', '소장', '항소'
        ]):
            return '민사소송법'
        
        # 형사소송법 도메인
        if any(keyword in case_name_lower for keyword in [
            '형사소송', '수사', '기소', '공소', '변호', '재판', '형사소송법', '공소제기'
        ]):
            return '형사소송법'
        
        # 행정법 도메인
        if category == 'administrative' or field == '행정' or any(keyword in case_name_lower for keyword in [
            '행정', '허가', '인가', '면허', '신고', '처분', '행정법', '행정처분'
        ]):
            return '행정법'
        
        # 기타
        return '기타/일반'
    
    def _extract_keywords_from_precedent(self, full_text: str, searchable_text: str, 
                                       section_content: str, case_name: str, 
                                       case_number: str, section_type: str) -> Set[str]:
        """판례에서 키워드 추출"""
        keywords = set()
        
        # 1. 판례명에서 키워드 추출
        case_keywords = self._extract_keywords_from_case_name(case_name)
        keywords.update(case_keywords)
        
        # 2. 사건번호에서 키워드 추출
        if case_number:
            keywords.add(case_number)
        
        # 3. 섹션 타입별 키워드 추출
        if section_type and section_content:
            section_keywords = self._extract_keywords_from_section(section_type, section_content)
            keywords.update(section_keywords)
        
        # 4. 전체 텍스트에서 키워드 추출
        if full_text:
            text_keywords = self._extract_keywords_from_text(full_text)
            keywords.update(text_keywords)
        
        # 5. 검색용 텍스트에서 키워드 추출
        if searchable_text:
            search_keywords = self._extract_keywords_from_text(searchable_text)
            keywords.update(search_keywords)
        
        return keywords
    
    def _extract_keywords_from_case_name(self, case_name: str) -> Set[str]:
        """판례명에서 키워드 추출"""
        keywords = set()
        
        # 판례명을 특수문자로 분리
        words = re.split(r'[^\w가-힣]+', case_name)
        
        for word in words:
            if len(word) >= 2:
                keywords.add(word)
        
        # 법률명 패턴 추출
        law_patterns = [
            r'([가-힣]+법)',
            r'([가-힣]+법률)',
            r'([가-힣]+규칙)',
            r'([가-힣]+령)',
            r'([가-힣]+시행령)',
            r'([가-힣]+시행규칙)'
        ]
        
        for pattern in law_patterns:
            matches = re.findall(pattern, case_name)
            keywords.update(matches)
        
        return keywords
    
    def _extract_keywords_from_section(self, section_type: str, content: str) -> Set[str]:
        """판례 섹션에서 키워드 추출"""
        keywords = set()
        
        if not content:
            return keywords
        
        # 섹션 타입별 특화 키워드 추출
        if section_type == 'points_at_issue':  # 판시사항
            keywords.update(self._extract_legal_terms_from_text(content))
        elif section_type == 'reasoning':  # 판결요지
            keywords.update(self._extract_legal_terms_from_text(content))
        elif section_type == 'decision_summary':  # 판결요약
            keywords.update(self._extract_legal_terms_from_text(content))
        
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
        
        # 3. 한글 법률 용어 (2-8글자)
        korean_terms = re.findall(r'[가-힣]{2,8}(?=은|는|이|가|을|를|에|의|에서|으로|로|와|과|에|에게|에게서)', cleaned_text)
        keywords.update(korean_terms)
        
        # 4. 법률 용어 추출
        legal_keywords = self._extract_legal_terms_from_text(cleaned_text)
        keywords.update(legal_keywords)
        
        # 5. 특수문자와 숫자가 포함된 용어 제거
        filtered_keywords = set()
        for keyword in keywords:
            if len(keyword) >= 2 and not re.search(r'[^\w가-힣]', keyword):
                filtered_keywords.add(keyword)
        
        return filtered_keywords
    
    def _extract_legal_terms_from_text(self, text: str) -> Set[str]:
        """텍스트에서 법률 용어 추출"""
        keywords = set()
        
        # 일반적인 법률 용어
        common_legal_terms = [
            # 기본 법률 용어
            '법률', '법령', '규정', '조항', '조문', '항목', '호목',
            '시행', '공포', '개정', '폐지', '제정', '부칙', '본칙',
            
            # 권리와 의무
            '권리', '의무', '책임', '권한', '범위', '효력', '효과',
            
            # 형사법 용어
            '처벌', '벌금', '징역', '형', '범죄', '구성요건', '고의', '과실',
            '미수', '기수', '공범', '정범', '교사범', '방조범',
            
            # 민사법 용어
            '계약', '손해배상', '불법행위', '채권', '채무', '소유권', '점유권',
            '물권', '채권자', '채무자', '이행', '변제', '대위', '취소',
            
            # 가족법 용어
            '혼인', '이혼', '양육', '상속', '친권', '양육권', '친생자', '양자',
            '친족', '가족관계', '혼인무효', '혼인취소',
            
            # 상사법 용어
            '회사', '주식', '주주', '이사', '상행위', '법인', '주식회사',
            '어음', '수표', '보험', '해상', '상법',
            
            # 노동법 용어
            '근로', '고용', '임금', '해고', '부당해고', '근로계약', '근로기준',
            '최저임금', '근로시간', '휴게시간', '연장근로',
            
            # 부동산법 용어
            '부동산', '토지', '등기', '매매', '임대', '소유권이전', '등기부',
            '임대차', '전세', '월세', '보증금',
            
            # 지적재산권법 용어
            '특허', '상표', '저작권', '디자인', '영업비밀', '지적재산',
            '특허권', '상표권', '저작권침해', '특허침해',
            
            # 세법 용어
            '세금', '소득세', '법인세', '부가가치세', '신고', '납부', '조세',
            '국세', '지방세', '세무조사', '과세', '비과세',
            
            # 소송법 용어
            '소송', '재판', '판결', '선고', '확정', '상고', '항소', '소장',
            '답변서', '증거', '입증', '절차', '집행', '강제집행',
            
            # 형사소송법 용어
            '수사', '기소', '공소', '변호', '재심', '자백', '증거능력',
            '증거력', '수사기관', '검찰', '경찰', '변호인',
            
            # 행정법 용어
            '행정', '허가', '인가', '면허', '신고', '처분', '행정처분',
            '행정행위', '재량행위', '기속행위',
            
            # 판례 특화 용어
            '판시사항', '판결요지', '판결요약', '참조판례', '참조조문',
            '대법원', '고등법원', '지방법원', '특허법원', '행정법원'
        ]
        
        for term in common_legal_terms:
            if term in text:
                keywords.add(term)
        
        # 법률명 패턴 추출
        law_name_patterns = [
            r'([가-힣]+법)',
            r'([가-힣]+법률)',
            r'([가-힣]+규칙)',
            r'([가-힣]+령)',
            r'([가-힣]+시행령)',
            r'([가-힣]+시행규칙)'
        ]
        
        for pattern in law_name_patterns:
            matches = re.findall(pattern, text)
            keywords.update(matches)
        
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
                        "weight": 0.9,  # 판례 기반이므로 높은 가중치
                        "synonyms": [],
                        "related_terms": [],
                        "context_keywords": [],
                        "source": "precedent_cases",
                        "confidence": 0.95,
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
        print("📊 판례 기반 키워드 추출 통계")
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
        extractor = PrecedentKeywordExtractor()
        
        # 키워드 추출
        domain_keywords = extractor.extract_keywords_from_precedents()
        
        # 통계 출력
        extractor.print_statistics(domain_keywords)
        
        # 데이터베이스에 저장
        extractor.save_keywords_to_database(domain_keywords)
        
        print("\n✅ 판례 기반 키워드 추출 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
