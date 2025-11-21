import re
import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

logger = get_logger(__name__)

class LegalTextPreprocessor:
    """법률 텍스트 전처리기"""
    
    def __init__(self):
        self.legal_patterns = {
            'legal_articles': r'제\d+조',
            'legal_paragraphs': r'제\d+항',
            'legal_items': r'제\d+호',
            'court_cases': r'대법원\s+\d+다\d+',
            'constitutional_cases': r'헌법재판소\s+\d+헌바\d+',
            'legal_acts': r'[가-힣]+법',
            'legal_terms': r'[가-힣]{2,10}(?:권|법|절차|소송|계약|손해|배상|죄|범|사건|처벌|등기|신고|신청|제기)'
        }
        
        # 법률 문서 특화 정제 규칙
        self.cleaning_rules = {
            'remove_page_numbers': r'^\s*\d+\s*$',
            'remove_line_numbers': r'^\s*\d+\.\s*',
            'normalize_whitespace': r'\s+',
            'remove_special_chars': r'[^\w\s가-힣.,;:!?()\[\]{}]',
            'normalize_quotes': r'["""]',
            'normalize_dashes': r'[—–-]+'
        }
    
    def clean_text(self, text: str) -> str:
        """텍스트 정제"""
        if not text:
            return ""
        
        # 기본 정제
        cleaned = text.strip()
        
        # 법률 문서 특화 정제
        for rule_name, pattern in self.cleaning_rules.items():
            if rule_name == 'normalize_whitespace':
                cleaned = re.sub(pattern, ' ', cleaned)
            elif rule_name == 'normalize_quotes':
                cleaned = re.sub(pattern, '"', cleaned)
            elif rule_name == 'normalize_dashes':
                cleaned = re.sub(pattern, '-', cleaned)
            else:
                cleaned = re.sub(pattern, '', cleaned)
        
        return cleaned
    
    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """법률 개체명 추출"""
        entities = {}
        
        for entity_type, pattern in self.legal_patterns.items():
            matches = re.findall(pattern, text)
            entities[entity_type] = list(set(matches))
        
        return entities
    
    def tokenize_legal_text(self, text: str) -> List[str]:
        """법률 텍스트 토큰화"""
        # 문장 단위 분리
        sentences = re.split(r'[.!?]+', text)
        
        # 토큰화 (단어 단위)
        tokens = []
        for sentence in sentences:
            if sentence.strip():
                # 법률 용어는 하나의 토큰으로 유지
                legal_terms = re.findall(self.legal_patterns['legal_terms'], sentence)
                for term in legal_terms:
                    sentence = sentence.replace(term, f" {term} ")
                
                # 일반 토큰화
                sentence_tokens = re.findall(r'\S+', sentence)
                tokens.extend(sentence_tokens)
        
        return tokens
    
    def extract_legal_sentences(self, text: str) -> List[str]:
        """법률 관련 문장 추출"""
        sentences = re.split(r'[.!?]+', text)
        legal_sentences = []
        
        for sentence in sentences:
            if sentence.strip():
                # 법률 용어가 포함된 문장만 추출
                if re.search(self.legal_patterns['legal_terms'], sentence):
                    legal_sentences.append(sentence.strip())
        
        return legal_sentences
    
    def preprocess_document(self, text: str) -> Dict[str, Any]:
        """문서 전체 전처리"""
        try:
            # 텍스트 정제
            cleaned_text = self.clean_text(text)
            
            # 법률 개체명 추출
            entities = self.extract_legal_entities(cleaned_text)
            
            # 토큰화
            tokens = self.tokenize_legal_text(cleaned_text)
            
            # 법률 문장 추출
            legal_sentences = self.extract_legal_sentences(cleaned_text)
            
            return {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'entities': entities,
                'tokens': tokens,
                'legal_sentences': legal_sentences,
                'word_count': len(tokens),
                'sentence_count': len(legal_sentences)
            }
            
        except Exception as e:
            logger.error(f"문서 전처리 중 오류 발생: {e}")
            return {
                'original_text': text,
                'cleaned_text': '',
                'entities': {},
                'tokens': [],
                'legal_sentences': [],
                'word_count': 0,
                'sentence_count': 0,
                'error': str(e)
            }
    
    def batch_preprocess(self, texts: List[str]) -> List[Dict[str, Any]]:
        """배치 전처리"""
        results = []
        for i, text in enumerate(texts):
            logger.info(f"전처리 진행: {i+1}/{len(texts)}")
            result = self.preprocess_document(text)
            results.append(result)
        
        return results
    
    def save_preprocessed_data(self, data: List[Dict[str, Any]], output_path: str):
        """전처리된 데이터 저장"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"전처리된 데이터 저장 완료: {output_file}")
    
    def load_preprocessed_data(self, input_path: str) -> List[Dict[str, Any]]:
        """전처리된 데이터 로드"""
        input_file = Path(input_path)
        
        if not input_file.exists():
            logger.warning(f"파일이 존재하지 않습니다: {input_file}")
            return []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"전처리된 데이터 로드 완료: {input_file}")
        return data
