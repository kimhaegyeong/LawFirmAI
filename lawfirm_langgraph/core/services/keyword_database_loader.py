import json
import logging
from typing import Dict, List, Set, Optional
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

class KeywordDatabaseLoader:
    """데이터베이스에서 키워드를 로드하는 클래스"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
        
        # 데이터베이스 파일 경로들
        self.database_files = {
            'comprehensive': self.data_dir / 'comprehensive_legal_term_dictionary.json',
            'legal_terms': self.data_dir / 'legal_term_dictionary.json',
            'legal_terms_db': self.data_dir / 'legal_terms_database.json'
        }
    
    def load_all_keywords(self) -> Dict[str, List[str]]:
        """모든 데이터베이스에서 키워드 로드"""
        all_keywords = defaultdict(set)
        
        for db_name, file_path in self.database_files.items():
            if file_path.exists():
                try:
                    keywords = self._load_from_file(file_path, db_name)
                    for domain, terms in keywords.items():
                        all_keywords[domain].update(terms)
                    self.logger.info(f"로드된 키워드 수 ({db_name}): {sum(len(terms) for terms in keywords.values())}")
                except Exception as e:
                    self.logger.error(f"데이터베이스 로드 실패 ({db_name}): {e}")
            else:
                self.logger.warning(f"데이터베이스 파일 없음: {file_path}")
        
        # Set을 List로 변환
        return {domain: list(terms) for domain, terms in all_keywords.items()}
    
    def _load_from_file(self, file_path: Path, db_type: str) -> Dict[str, List[str]]:
        """파일에서 키워드 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        keywords = defaultdict(set)
        
        if db_type == 'comprehensive':
            keywords = self._load_comprehensive_format(data)
        elif db_type == 'legal_terms':
            keywords = self._load_legal_terms_format(data)
        elif db_type == 'legal_terms_db':
            keywords = self._load_legal_terms_db_format(data)
        
        return {domain: list(terms) for domain, terms in keywords.items()}
    
    def _load_comprehensive_format(self, data: Dict) -> Dict[str, Set[str]]:
        """comprehensive_legal_term_dictionary.json 형식 로드"""
        keywords = defaultdict(set)
        
        if 'dictionary' in data:
            for term, info in data['dictionary'].items():
                # 도메인 매핑
                domain = self._map_term_to_domain(term, info)
                
                # 기본 용어
                keywords[domain].add(term)
                
                # 동의어 추가
                if 'synonyms' in info:
                    keywords[domain].update(info['synonyms'])
                
                # 관련 용어 추가
                if 'related_terms' in info:
                    keywords[domain].update(info['related_terms'])
                
                # 판례 키워드 추가
                if 'precedent_keywords' in info:
                    keywords[domain].update(info['precedent_keywords'])
        
        return keywords
    
    def _load_legal_terms_format(self, data: Dict) -> Dict[str, Set[str]]:
        """legal_term_dictionary.json 형식 로드"""
        keywords = defaultdict(set)
        
        for term, info in data.items():
            # 도메인 매핑
            domain = self._map_term_to_domain(term, info)
            
            # 기본 용어
            keywords[domain].add(term)
            
            # 동의어 추가
            if 'synonyms' in info:
                keywords[domain].update(info['synonyms'])
            
            # 관련 용어 추가
            if 'related_terms' in info:
                keywords[domain].update(info['related_terms'])
            
            # 판례 키워드 추가
            if 'precedent_keywords' in info:
                keywords[domain].update(info['precedent_keywords'])
        
        return keywords
    
    def _load_legal_terms_db_format(self, data: Dict) -> Dict[str, Set[str]]:
        """legal_terms_database.json 형식 로드 (도메인별 구조)"""
        keywords = defaultdict(set)
        
        # 새로운 형식: 도메인별로 키워드가 구성됨
        if isinstance(data, dict):
            for domain, terms_dict in data.items():
                if isinstance(terms_dict, dict):
                    # 각 용어와 관련 정보 추출
                    for term, term_info in terms_dict.items():
                        if isinstance(term_info, dict):
                            # 기본 용어 추가
                            keywords[domain].add(term)
                            
                            # 동의어 추가
                            if 'synonyms' in term_info and isinstance(term_info['synonyms'], list):
                                keywords[domain].update(term_info['synonyms'])
                            
                            # 관련 용어 추가
                            if 'related_terms' in term_info and isinstance(term_info['related_terms'], list):
                                keywords[domain].update(term_info['related_terms'])
                            
                            # 문맥 키워드 추가
                            if 'context_keywords' in term_info and isinstance(term_info['context_keywords'], list):
                                keywords[domain].update(term_info['context_keywords'])
        
        return keywords
    
    def _map_term_to_domain(self, term: str, info: Dict) -> str:
        """용어를 도메인으로 매핑"""
        # 용어의 특성에 따라 도메인 분류
        term_lower = term.lower()
        
        # 민사법 관련
        if any(keyword in term_lower for keyword in ['계약', '손해', '소유권', '채권', '채무', '불법행위']):
            return '민사법'
        
        # 형사법 관련
        if any(keyword in term_lower for keyword in ['살인', '절도', '사기', '강도', '범죄', '형', '처벌']):
            return '형사법'
        
        # 가족법 관련
        if any(keyword in term_lower for keyword in ['혼인', '이혼', '양육', '친권', '상속', '유언']):
            return '가족법'
        
        # 상사법 관련
        if any(keyword in term_lower for keyword in ['회사', '주식', '주주', '이사', '상행위']):
            return '상사법'
        
        # 노동법 관련
        if any(keyword in term_lower for keyword in ['근로', '고용', '임금', '해고', '노동']):
            return '노동법'
        
        # 부동산법 관련
        if any(keyword in term_lower for keyword in ['부동산', '토지', '등기', '매매', '임대']):
            return '부동산법'
        
        # 지적재산권법 관련
        if any(keyword in term_lower for keyword in ['특허', '상표', '저작권', '디자인', '발명']):
            return '지적재산권법'
        
        # 세법 관련
        if any(keyword in term_lower for keyword in ['세금', '소득세', '법인세', '부가가치세', '세무']):
            return '세법'
        
        # 민사소송법 관련
        if any(keyword in term_lower for keyword in ['소송', '소', '증거', '입증', '절차']):
            return '민사소송법'
        
        # 형사소송법 관련
        if any(keyword in term_lower for keyword in ['수사', '기소', '공소', '변호', '재심']):
            return '형사소송법'
        
        # 기본값
        return '기타/일반'
    
    def get_domain_keyword_count(self, domain: str) -> int:
        """특정 도메인의 키워드 수 반환"""
        all_keywords = self.load_all_keywords()
        return len(all_keywords.get(domain, []))
    
    def get_minimum_keyword_threshold(self) -> int:
        """최소 키워드 임계값 반환"""
        return 20  # 도메인당 최소 20개 키워드 필요
    
    def should_expand_domain(self, domain: str) -> bool:
        """도메인 키워드 확장이 필요한지 판단"""
        current_count = self.get_domain_keyword_count(domain)
        threshold = self.get_minimum_keyword_threshold()
        return current_count < threshold
