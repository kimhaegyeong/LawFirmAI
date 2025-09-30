"""
Data Processor
데이터 전처리 및 구조화 모듈

국가법령정보센터 OpenAPI에서 수집한 법률 데이터를 전처리하고 구조화합니다.
"""

import logging
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import hashlib

from .legal_term_normalizer import LegalTermNormalizer

logger = logging.getLogger(__name__)


class LegalDataProcessor:
    """법률 데이터 전처리 클래스"""
    
    def __init__(self, enable_term_normalization: bool = True):
        """데이터 프로세서 초기화"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("LegalDataProcessor initialized")
        
        # 법률 용어 정규화 패턴
        self.law_patterns = {
            'law_name': r'([가-힣]+법|([가-힣]+법률))',
            'article': r'제(\d+)조',
            'paragraph': r'제(\d+)항',
            'subparagraph': r'제(\d+)호',
            'case_number': r'([가-힣]+[0-9]+[가-힣]*[0-9]*[가-힣]*)',
            'court': r'([가-힣]+법원)',
            'date': r'(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)',
            'law_id': r'법률\s*제(\d+)호'
        }
        
        # 불용어 목록
        self.stopwords = {
            '의', '을', '를', '이', '가', '은', '는', '에', '에서', '로', '으로', 
            '와', '과', '도', '만', '부터', '까지', '에서', '에게', '한테', 
            '께', '에게서', '한테서', '께서', '이랑', '랑', '하고', '와', '과',
            '같이', '처럼', '만큼', '보다', '마다', '마다', '쯤', '정도',
            '뿐', '만', '밖에', '외에', '제외하고', '빼고', '말고'
        }
        
        # 법률 용어 정규화기 초기화
        self.enable_term_normalization = enable_term_normalization
        if enable_term_normalization:
            try:
                self.term_normalizer = LegalTermNormalizer()
                self.logger.info("법률 용어 정규화기 초기화 완료")
            except Exception as e:
                self.logger.warning(f"법률 용어 정규화기 초기화 실패: {e}")
                self.term_normalizer = None
                self.enable_term_normalization = False
        else:
            self.term_normalizer = None
    
    def clean_text(self, text: str, context: str = None) -> str:
        """텍스트 정리 및 정규화"""
        if not text:
            return ""
        
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # 특수 문자 정리 (법률 문서에 필요한 문자는 유지)
        text = re.sub(r'[^\w\s가-힣.,!?;:()「」『』\[\]()]', ' ', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        # 법률 용어 정규화 적용
        if self.enable_term_normalization and self.term_normalizer:
            try:
                normalization_result = self.term_normalizer.normalize_text(text, context)
                if normalization_result['success']:
                    text = normalization_result['normalized_text']
                    self.logger.debug(f"법률 용어 정규화 적용: {normalization_result['term_mappings']}")
                else:
                    self.logger.warning(f"법률 용어 정규화 실패: {normalization_result.get('error', 'Unknown error')}")
            except Exception as e:
                self.logger.error(f"법률 용어 정규화 중 오류: {e}")
        
        return text
    
    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """법률 엔티티 추출"""
        entities = {
            "laws": [],
            "articles": [],
            "cases": [],
            "courts": [],
            "dates": [],
            "keywords": []
        }
        
        # 법률명 추출
        law_matches = re.findall(self.law_patterns['law_name'], text)
        entities["laws"] = list(set([match[0] for match in law_matches]))
        
        # 조문 추출
        article_matches = re.findall(self.law_patterns['article'], text)
        entities["articles"] = list(set([f"제{match}조" for match in article_matches]))
        
        # 사건번호 추출
        case_matches = re.findall(self.law_patterns['case_number'], text)
        entities["cases"] = list(set(case_matches[:10]))  # 상위 10개만
        
        # 법원명 추출
        court_matches = re.findall(self.law_patterns['court'], text)
        entities["courts"] = list(set(court_matches))
        
        # 날짜 추출
        date_matches = re.findall(self.law_patterns['date'], text)
        entities["dates"] = list(set(date_matches))
        
        # 키워드 추출
        entities["keywords"] = self._extract_keywords(text)
        
        return entities
    
    def _extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """키워드 추출 (TF-IDF 기반)"""
        # 단어 분리 (한글 단어만)
        words = re.findall(r'[가-힣]+', text)
        
        # 불용어 제거 및 길이 필터링
        keywords = [word for word in words if len(word) > 1 and word not in self.stopwords]
        
        # 빈도 계산
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 빈도순 정렬
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in sorted_words[:max_keywords]]
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 1000, 
                              overlap: int = 100) -> List[Dict[str, Any]]:
        """텍스트를 청크로 분할"""
        if not text:
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 문장 경계에서 자르기
            if end < len(text):
                # 마지막 문장 끝 찾기
                last_sentence_end = text.rfind('.', start, end)
                if last_sentence_end > start:
                    end = last_sentence_end + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk = {
                    'id': f"chunk_{chunk_id}",
                    'text': chunk_text,
                    'start_pos': start,
                    'end_pos': end,
                    'length': len(chunk_text),
                    'entities': self.extract_legal_entities(chunk_text)
                }
                chunks.append(chunk)
                chunk_id += 1
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def process_law_data(self, law_data: Dict[str, Any]) -> Dict[str, Any]:
        """법령 데이터 처리"""
        try:
            # 기본 정보 추출
            basic_info = law_data.get('basic_info', {})
            current_text = law_data.get('current_text', {})
            
            processed_law = {
                'id': basic_info.get('id', ''),
                'law_name': basic_info.get('name', ''),
                'law_id': basic_info.get('id', ''),
                'mst': basic_info.get('mst', ''),
                'effective_date': basic_info.get('effective_date', ''),
                'promulgation_date': basic_info.get('promulgation_date', ''),
                'ministry': basic_info.get('ministry', ''),
                'category': basic_info.get('category', ''),
                'status': 'success',
                'processed_at': datetime.now().isoformat()
            }
            
            # 본문 내용 추출 및 처리
            if current_text and 'response' in current_text:
                response = current_text['response']
                if 'body' in response and 'items' in response['body']:
                    items = response['body']['items']
                    if 'item' in items:
                        item = items['item']
                        if isinstance(item, list):
                            item = item[0]  # 첫 번째 항목 사용
                        
                        # 조문별 내용 추출
                        articles = []
                        if '조문내용' in item:
                            articles.append({
                                'article_number': 1,
                                'content': item['조문내용'],
                                'title': item.get('조문제목', '')
                            })
                        
                        processed_law['articles'] = articles
                        processed_law['full_content'] = item.get('조문내용', '')
            
            # 텍스트 정리 및 청킹
            if processed_law.get('full_content'):
                context = f"law_{processed_law.get('law_name', '')}"
                cleaned_content = self.clean_text(processed_law['full_content'], context)
                processed_law['cleaned_content'] = cleaned_content
                processed_law['chunks'] = self.split_text_into_chunks(cleaned_content)
                processed_law['entities'] = self.extract_legal_entities(cleaned_content)
            
            # 연혁 정보 처리
            if law_data.get('history'):
                processed_law['history'] = self._process_law_history(law_data['history'])
            
            return processed_law
            
        except Exception as e:
            self.logger.error(f"Error processing law data: {e}")
            return {
                'error': str(e),
                'original_data': law_data,
                'status': 'failed'
            }
    
    def process_precedent_data(self, precedent_data: Dict[str, Any]) -> Dict[str, Any]:
        """판례 데이터 처리"""
        try:
            basic_info = precedent_data.get('basic_info', {})
            detail_info = precedent_data.get('detail_info', {})
            
            processed_precedent = {
                'id': basic_info.get('판례일련번호', ''),
                'case_name': basic_info.get('사건명', ''),
                'case_number': basic_info.get('사건번호', ''),
                'court': basic_info.get('법원명', ''),
                'court_code': basic_info.get('법원코드', ''),
                'decision_date': basic_info.get('선고일자', ''),
                'case_type': basic_info.get('사건유형명', ''),
                'case_type_code': basic_info.get('사건유형코드', ''),
                'decision_type': basic_info.get('판결유형', ''),
                'category': precedent_data.get('category', ''),
                'status': 'success',
                'processed_at': datetime.now().isoformat()
            }
            
            # 상세 정보 추출
            if detail_info and 'response' in detail_info:
                response = detail_info['response']
                if 'body' in response and 'items' in response['body']:
                    items = response['body']['items']
                    if 'item' in items:
                        item = items['item']
                        if isinstance(item, list):
                            item = item[0]
                        
                        processed_precedent.update({
                            'issue': item.get('판시사항', ''),
                            'reasoning': item.get('판결요지', ''),
                            'case_summary': item.get('사건개요', ''),
                            'dispute_point': item.get('쟁점', ''),
                            'conclusion': item.get('결론', ''),
                            'related_laws': item.get('참조조문', ''),
                            'related_precedents': item.get('참조판례', ''),
                            'keywords': item.get('키워드', ''),
                            'classification': item.get('분류', '')
                        })
            
            # 전체 텍스트 결합 및 처리
            full_text = f"{processed_precedent.get('issue', '')} {processed_precedent.get('reasoning', '')} {processed_precedent.get('case_summary', '')}"
            if full_text.strip():
                context = f"precedent_{processed_precedent.get('case_name', '')}"
                cleaned_content = self.clean_text(full_text, context)
                processed_precedent['cleaned_content'] = cleaned_content
                processed_precedent['chunks'] = self.split_text_into_chunks(cleaned_content)
                processed_precedent['entities'] = self.extract_legal_entities(cleaned_content)
            
            return processed_precedent
            
        except Exception as e:
            self.logger.error(f"Error processing precedent data: {e}")
            return {
                'error': str(e),
                'original_data': precedent_data,
                'status': 'failed'
            }
    
    def process_constitutional_decision_data(self, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """헌재결정례 데이터 처리"""
        try:
            basic_info = decision_data.get('basic_info', {})
            detail_info = decision_data.get('detail_info', {})
            
            processed_decision = {
                'id': basic_info.get('판례일련번호', ''),
                'case_name': basic_info.get('사건명', ''),
                'case_number': basic_info.get('사건번호', ''),
                'decision_date': basic_info.get('선고일자', ''),
                'decision_type': decision_data.get('decision_type', ''),
                'status': 'success',
                'processed_at': datetime.now().isoformat()
            }
            
            # 상세 정보 추출
            if detail_info and 'response' in detail_info:
                response = detail_info['response']
                if 'body' in response and 'items' in response['body']:
                    items = response['body']['items']
                    if 'item' in items:
                        item = items['item']
                        if isinstance(item, list):
                            item = item[0]
                        
                        processed_decision.update({
                            'issue': item.get('판시사항', ''),
                            'reasoning': item.get('판결요지', ''),
                            'case_summary': item.get('사건개요', ''),
                            'dispute_point': item.get('쟁점', ''),
                            'conclusion': item.get('결론', ''),
                            'related_laws': item.get('관련법령', ''),
                            'related_precedents': item.get('관련판례', ''),
                            'keywords': item.get('키워드', ''),
                            'classification': item.get('분류', '')
                        })
            
            # 전체 텍스트 결합 및 처리
            full_text = f"{processed_decision.get('issue', '')} {processed_decision.get('reasoning', '')} {processed_decision.get('case_summary', '')}"
            if full_text.strip():
                cleaned_content = self.clean_text(full_text)
                processed_decision['cleaned_content'] = cleaned_content
                processed_decision['chunks'] = self.split_text_into_chunks(cleaned_content)
                processed_decision['entities'] = self.extract_legal_entities(cleaned_content)
            
            return processed_decision
            
        except Exception as e:
            self.logger.error(f"Error processing constitutional decision data: {e}")
            return {
                'error': str(e),
                'original_data': decision_data,
                'status': 'failed'
            }
    
    def process_legal_interpretation_data(self, interpretation_data: Dict[str, Any]) -> Dict[str, Any]:
        """법령해석례 데이터 처리"""
        try:
            basic_info = interpretation_data.get('basic_info', {})
            detail_info = interpretation_data.get('detail_info', {})
            
            processed_interpretation = {
                'id': basic_info.get('판례일련번호', ''),
                'case_name': basic_info.get('사건명', ''),
                'case_number': basic_info.get('사건번호', ''),
                'decision_date': basic_info.get('선고일자', ''),
                'topic': interpretation_data.get('topic', ''),
                'ministry': interpretation_data.get('ministry', ''),
                'status': 'success',
                'processed_at': datetime.now().isoformat()
            }
            
            # 상세 정보 추출
            if detail_info and 'response' in detail_info:
                response = detail_info['response']
                if 'body' in response and 'items' in response['body']:
                    items = response['body']['items']
                    if 'item' in items:
                        item = items['item']
                        if isinstance(item, list):
                            item = item[0]
                        
                        processed_interpretation.update({
                            'issue': item.get('판시사항', ''),
                            'reasoning': item.get('판결요지', ''),
                            'case_summary': item.get('사건개요', ''),
                            'dispute_point': item.get('쟁점', ''),
                            'conclusion': item.get('결론', ''),
                            'related_laws': item.get('관련법령', ''),
                            'related_precedents': item.get('관련판례', ''),
                            'keywords': item.get('키워드', ''),
                            'classification': item.get('분류', '')
                        })
            
            # 전체 텍스트 결합 및 처리
            full_text = f"{processed_interpretation.get('issue', '')} {processed_interpretation.get('reasoning', '')} {processed_interpretation.get('case_summary', '')}"
            if full_text.strip():
                cleaned_content = self.clean_text(full_text)
                processed_interpretation['cleaned_content'] = cleaned_content
                processed_interpretation['chunks'] = self.split_text_into_chunks(cleaned_content)
                processed_interpretation['entities'] = self.extract_legal_entities(cleaned_content)
            
            return processed_interpretation
            
        except Exception as e:
            self.logger.error(f"Error processing legal interpretation data: {e}")
            return {
                'error': str(e),
                'original_data': interpretation_data,
                'status': 'failed'
            }
    
    def _process_law_history(self, history_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """법령 연혁 데이터 처리"""
        processed_history = []
        
        for history_item in history_data:
            try:
                processed_item = {
                    'history_id': history_item.get('연혁ID', ''),
                    'effective_date': history_item.get('시행일자', ''),
                    'promulgation_date': history_item.get('공포일자', ''),
                    'amendment_type': history_item.get('제개정구분', ''),
                    'content': history_item.get('내용', ''),
                    'reason': history_item.get('제개정이유', '')
                }
                processed_history.append(processed_item)
            except Exception as e:
                self.logger.error(f"Error processing history item: {e}")
                continue
        
        return processed_history
    
    def validate_document(self, document: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """문서 유효성 검사"""
        errors = []
        
        # 필수 필드 검사
        if not document.get('id'):
            errors.append("ID is required")
        
        if not document.get('cleaned_content'):
            errors.append("Content is required")
        
        # 내용 길이 검사
        content = document.get('cleaned_content', '')
        if len(content) < 10:
            errors.append("Content too short (minimum 10 characters)")
        
        if len(content) > 100000:
            errors.append("Content too long (maximum 100,000 characters)")
        
        # 청크 검사
        chunks = document.get('chunks', [])
        if not chunks:
            errors.append("No chunks generated")
        
        return len(errors) == 0, errors
    
    def generate_document_hash(self, document: Dict[str, Any]) -> str:
        """문서 해시 생성 (중복 검사용)"""
        content = document.get('cleaned_content', '')
        title = document.get('law_name', document.get('case_name', ''))
        
        hash_string = f"{title}_{content}"
        return hashlib.md5(hash_string.encode('utf-8')).hexdigest()
    
    def process_batch(self, data_list: List[Dict[str, Any]], data_type: str) -> List[Dict[str, Any]]:
        """배치 데이터 처리"""
        processed_list = []
        
        for i, data in enumerate(data_list):
            try:
                if data_type == 'law':
                    processed = self.process_law_data(data)
                elif data_type == 'precedent':
                    processed = self.process_precedent_data(data)
                elif data_type == 'constitutional_decision':
                    processed = self.process_constitutional_decision_data(data)
                elif data_type == 'legal_interpretation':
                    processed = self.process_legal_interpretation_data(data)
                else:
                    self.logger.warning(f"Unknown data type: {data_type}")
                    continue
                
                # 유효성 검사
                is_valid, errors = self.validate_document(processed)
                if is_valid:
                    processed['document_hash'] = self.generate_document_hash(processed)
                    processed_list.append(processed)
                else:
                    self.logger.warning(f"Document {i} validation failed: {errors}")
                
                # 진행률 로그
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(data_list)} documents")
                
            except Exception as e:
                self.logger.error(f"Error processing document {i}: {e}")
                continue
        
        self.logger.info(f"Batch processing completed: {len(processed_list)}/{len(data_list)} documents processed successfully")
        return processed_list
    
    def process_administrative_rule_data(self, rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """행정규칙 데이터 전처리"""
        try:
            processed_rule = {
                'id': rule_data.get('id', ''),
                'name': rule_data.get('name', ''),
                'category': 'administrative_rule',
                'status': 'success',
                'raw_data': rule_data,
                'processed_at': datetime.now().isoformat()
            }
            
            # 기본 정보 추출
            if 'administrativeRule' in rule_data:
                rule_info = rule_data['administrativeRule']
                processed_rule.update({
                    'rule_name': rule_info.get('행정규칙명', ''),
                    'rule_number': rule_info.get('행정규칙번호', ''),
                    'effective_date': rule_info.get('시행일자', ''),
                    'promulgation_date': rule_info.get('공포일자', ''),
                    'ministry': rule_info.get('소관부처', ''),
                    'content': rule_info.get('내용', ''),
                    'purpose': rule_info.get('제정목적', ''),
                    'scope': rule_info.get('적용범위', '')
                })
            
            # 텍스트 정리
            content = processed_rule.get('content', '')
            if content:
                processed_rule['cleaned_content'] = self.clean_text(content)
                processed_rule['chunks'] = self.create_chunks(processed_rule['cleaned_content'])
            else:
                processed_rule['cleaned_content'] = ''
                processed_rule['chunks'] = []
            
            # 메타데이터 생성
            processed_rule['metadata'] = {
                'word_count': len(processed_rule['cleaned_content'].split()) if processed_rule['cleaned_content'] else 0,
                'char_count': len(processed_rule['cleaned_content']) if processed_rule['cleaned_content'] else 0,
                'chunk_count': len(processed_rule['chunks']),
                'has_content': bool(processed_rule['cleaned_content'])
            }
            
            # 유효성 검사
            is_valid, errors = self.validate_document(processed_rule)
            processed_rule['is_valid'] = is_valid
            processed_rule['validation_errors'] = errors
            
            # 해시 생성
            processed_rule['document_hash'] = self.generate_document_hash(processed_rule)
            
            return processed_rule
            
        except Exception as e:
            self.logger.error(f"Error processing administrative rule data: {e}")
            return {
                'id': rule_data.get('id', ''),
                'name': rule_data.get('name', ''),
                'category': 'administrative_rule',
                'status': 'failed',
                'error': str(e),
                'processed_at': datetime.now().isoformat()
            }