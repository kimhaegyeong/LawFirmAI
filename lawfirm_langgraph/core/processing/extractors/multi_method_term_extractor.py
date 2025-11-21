import re
import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import konlpy
from konlpy.tag import Okt, Kkma

logger = get_logger(__name__)

class NERBasedExtractor:
    """개체명 인식 기반 용어 추출기"""
    
    def __init__(self):
        self.legal_entity_types = [
            "LAW_TERM",      # 법률 용어
            "LEGAL_ACT",     # 법률명
            "COURT_CASE",    # 판례
            "LEGAL_PROCEDURE" # 법적 절차
        ]
        
        # 한국어 형태소 분석기
        self.okt = Okt()
        self.kkma = Kkma()
        
        # 법률 용어 패턴
        self.legal_patterns = {
            'legal_terms': [
                r'[가-힣]{2,10}(?:권|법|절차|소송|계약|손해|배상)',
                r'[가-힣]{2,8}(?:죄|범|사건|처벌)',
                r'[가-힣]{2,6}(?:등기|신고|신청|제기)',
                r'[가-힣]{2,8}(?:특허|상표|저작권|디자인)',
                r'[가-힣]{2,6}(?:이혼|혼인|친자|양육)',
                r'[가-힣]{2,8}(?:회사|주식|상행위|상법)',
                r'[가-힣]{2,6}(?:근로|해고|임금|노동)',
                r'[가-힣]{2,8}(?:부동산|등기|매매|임대)',
                r'[가-힣]{2,6}(?:소득세|법인세|부가가치세)',
                r'[가-힣]{2,8}(?:수사|기소|변호인|재판)'
            ],
            'compound_terms': [
                r'[가-힣]+(?:계약|소송|절차|권리|의무)',
                r'[가-힣]+(?:침해|위반|해지|이행)',
                r'[가-힣]+(?:등기|신고|신청|제기)',
                r'[가-힣]+(?:손해|배상|구제|보상)'
            ]
        }
    
    def extract_legal_terms(self, text: str) -> List[str]:
        """법률 용어 추출"""
        legal_terms = []
        
        # 패턴 기반 추출
        for pattern_group, patterns in self.legal_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                legal_terms.extend(matches)
        
        # 형태소 분석을 통한 복합어 추출
        morphs = self.okt.morphs(text)
        pos_tags = self.okt.pos(text)
        
        # 명사 + 법률 관련 접미사 조합
        for i, (morph, pos) in enumerate(pos_tags):
            if pos == 'Noun' and len(morph) >= 2:
                # 다음 형태소가 법률 관련 접미사인지 확인
                if i + 1 < len(pos_tags):
                    next_morph, next_pos = pos_tags[i + 1]
                    if next_morph in ['권', '법', '절차', '소송', '계약']:
                        compound_term = morph + next_morph
                        legal_terms.append(compound_term)
        
        return list(set(legal_terms))
    
    def extract(self, text: str) -> List[str]:
        """용어 추출 메인 메서드"""
        return self.extract_legal_terms(text)

class PatternBasedExtractor:
    """패턴 기반 용어 추출기"""
    
    def __init__(self):
        self.legal_patterns = {
            'legal_terms': [
                r'[가-힣]{2,10}(?:권|법|절차|소송|계약|손해|배상)',
                r'[가-힣]{2,8}(?:죄|범|사건|처벌)',
                r'[가-힣]{2,6}(?:등기|신고|신청|제기)',
                r'[가-힣]{2,8}(?:특허|상표|저작권|디자인)',
                r'[가-힣]{2,6}(?:이혼|혼인|친자|양육)',
                r'[가-힣]{2,8}(?:회사|주식|상행위|상법)',
                r'[가-힣]{2,6}(?:근로|해고|임금|노동)',
                r'[가-힣]{2,8}(?:부동산|등기|매매|임대)',
                r'[가-힣]{2,6}(?:소득세|법인세|부가가치세)',
                r'[가-힣]{2,8}(?:수사|기소|변호인|재판)'
            ],
            'compound_terms': [
                r'[가-힣]+(?:계약|소송|절차|권리|의무)',
                r'[가-힣]+(?:침해|위반|해지|이행)',
                r'[가-힣]+(?:등기|신고|신청|제기)',
                r'[가-힣]+(?:손해|배상|구제|보상)'
            ],
            'legal_acts': [
                r'[가-힣]+법',
                r'[가-힣]+법령',
                r'[가-힣]+규칙',
                r'[가-힣]+조례'
            ],
            'court_cases': [
                r'대법원\s+\d+다\d+',
                r'헌법재판소\s+\d+헌바\d+',
                r'고등법원\s+\d+나\d+',
                r'지방법원\s+\d+가합\d+'
            ]
        }
    
    def extract_by_patterns(self, text: str) -> List[str]:
        """패턴 매칭으로 용어 추출"""
        terms = []
        
        for pattern_group, patterns in self.legal_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                terms.extend(matches)
        
        return list(set(terms))
    
    def extract(self, text: str) -> List[str]:
        """용어 추출 메인 메서드"""
        return self.extract_by_patterns(text)

class FrequencyBasedExtractor:
    """빈도 분석 기반 용어 추출기"""
    
    def __init__(self, min_frequency: int = 3, min_length: int = 2, max_length: int = 10):
        self.min_frequency = min_frequency
        self.min_length = min_length
        self.max_length = max_length
        self.okt = Okt()
    
    def extract_ngrams(self, texts: List[str], n_range: Tuple[int, int] = (2, 4)) -> List[str]:
        """n-gram 추출"""
        all_ngrams = []
        
        for text in texts:
            # 형태소 분석
            morphs = self.okt.morphs(text)
            
            # n-gram 생성
            for n in range(n_range[0], n_range[1] + 1):
                for i in range(len(morphs) - n + 1):
                    ngram = ' '.join(morphs[i:i+n])
                    if self.min_length <= len(ngram) <= self.max_length:
                        all_ngrams.append(ngram)
        
        return all_ngrams
    
    def extract_frequent_terms(self, texts: List[str]) -> List[str]:
        """빈도 기반 용어 추출"""
        # n-gram 추출
        ngrams = self.extract_ngrams(texts)
        
        # 빈도 계산
        term_freq = Counter(ngrams)
        
        # 필터링
        frequent_terms = [
            term for term, freq in term_freq.items()
            if freq >= self.min_frequency
        ]
        
        return frequent_terms
    
    def extract(self, text: str) -> List[str]:
        """용어 추출 메인 메서드"""
        return self.extract_frequent_terms([text])

class EmbeddingBasedExtractor:
    """임베딩 기반 용어 추출기"""
    
    def __init__(self):
        self.legal_seed_terms = [
            "계약", "손해배상", "소유권", "이혼", "살인", 
            "절도", "회사", "특허", "저작권", "소송",
            "수사", "기소", "변호인", "재판", "증거",
            "등기", "신고", "신청", "제기", "침해",
            "위반", "해지", "이행", "구제", "보상"
        ]
        self.similarity_threshold = 0.7
        self.okt = Okt()
    
    def extract_candidates(self, text: str) -> List[str]:
        """후보 용어 추출"""
        # 형태소 분석
        morphs = self.okt.morphs(text)
        pos_tags = self.okt.pos(text)
        
        candidates = []
        
        # 명사 추출
        for morph, pos in pos_tags:
            if pos == 'Noun' and len(morph) >= 2:
                candidates.append(morph)
        
        # 복합어 추출
        for i, (morph, pos) in enumerate(pos_tags):
            if pos == 'Noun' and i + 1 < len(pos_tags):
                next_morph, next_pos = pos_tags[i + 1]
                if next_pos == 'Noun':
                    compound = morph + next_morph
                    candidates.append(compound)
        
        return list(set(candidates))
    
    def calculate_similarity(self, term1: str, term2: str) -> float:
        """용어 간 유사도 계산 (간단한 문자 기반)"""
        # 자카드 유사도
        set1 = set(term1)
        set2 = set(term2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def extract_similar_terms(self, text: str) -> List[str]:
        """유사 용어 추출"""
        # 후보 용어 추출
        candidate_terms = self.extract_candidates(text)
        
        # 유사도 계산하여 법률 용어 필터링
        legal_terms = []
        for term in candidate_terms:
            max_similarity = max([
                self.calculate_similarity(term, seed_term)
                for seed_term in self.legal_seed_terms
            ])
            
            if max_similarity > self.similarity_threshold:
                legal_terms.append(term)
        
        return legal_terms
    
    def extract(self, text: str) -> List[str]:
        """용어 추출 메인 메서드"""
        return self.extract_similar_terms(text)

class MultiMethodTermExtractor:
    """다중 방법론 기반 용어 추출기"""
    
    def __init__(self):
        self.extractors = {
            "ner": NERBasedExtractor(),
            "pattern": PatternBasedExtractor(),
            "frequency": FrequencyBasedExtractor(),
            "embedding": EmbeddingBasedExtractor()
        }
        self.logger = get_logger(__name__)
    
    def extract_terms(self, text: str) -> Dict[str, List[str]]:
        """다중 방법으로 용어 추출"""
        all_terms = {}
        
        for method, extractor in self.extractors.items():
            try:
                terms = extractor.extract(text)
                all_terms[method] = terms
                self.logger.info(f"{method} 방법으로 {len(terms)}개 용어 추출")
            except Exception as e:
                self.logger.error(f"{method} 방법 추출 중 오류: {e}")
                all_terms[method] = []
        
        return all_terms
    
    def merge_and_deduplicate(self, all_terms: Dict[str, List[str]]) -> List[str]:
        """중복 제거 및 통합"""
        # 모든 용어 수집
        all_term_list = []
        for method, terms in all_terms.items():
            all_term_list.extend(terms)
        
        # 중복 제거
        unique_terms = list(set(all_term_list))
        
        # 빈도 기반 정렬
        term_freq = Counter(all_term_list)
        sorted_terms = sorted(unique_terms, key=lambda x: term_freq[x], reverse=True)
        
        return sorted_terms
    
    def extract_and_merge(self, text: str) -> List[str]:
        """용어 추출 및 통합"""
        all_terms = self.extract_terms(text)
        merged_terms = self.merge_and_deduplicate(all_terms)
        
        self.logger.info(f"총 {len(merged_terms)}개 용어 추출 완료")
        return merged_terms
    
    def batch_extract(self, texts: List[str]) -> List[Dict[str, Any]]:
        """배치 용어 추출"""
        results = []
        
        for i, text in enumerate(texts):
            self.logger.info(f"용어 추출 진행: {i+1}/{len(texts)}")
            
            # 개별 방법별 추출
            method_terms = self.extract_terms(text)
            
            # 통합 추출
            merged_terms = self.merge_and_deduplicate(method_terms)
            
            results.append({
                'text': text,
                'method_terms': method_terms,
                'merged_terms': merged_terms,
                'term_count': len(merged_terms)
            })
        
        return results
