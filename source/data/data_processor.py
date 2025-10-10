# -*- coding: utf-8 -*-
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
                'mst': basic_info.get('mst'),
                'effective_date': basic_info.get('effective_date'),
                'promulgation_date': basic_info.get('promulgation_date'),
                'ministry': basic_info.get('ministry'),
                'category': basic_info.get('category', ''),
                'status': 'success',
                'processed_at': datetime.now().isoformat()
            }
            
            # 본문 내용 추출 및 처리
            full_content = ""
            articles = []
            
            # current_text에서 법령 내용 추출
            if current_text and '법령' in current_text:
                law_content = current_text['법령']
                
                # 개정문 내용 추출
                if '개정문' in law_content and '개정문내용' in law_content['개정문']:
                    amendment_content = law_content['개정문']['개정문내용']
                    if isinstance(amendment_content, list) and len(amendment_content) > 0:
                        if isinstance(amendment_content[0], list):
                            # 2차원 배열인 경우 첫 번째 배열 사용
                            amendment_text = amendment_content[0]
                            full_content += " ".join(amendment_text)
                            # 개정문에서 조문 구조 파싱
                            articles = self._parse_amendment_content(amendment_text)
                        else:
                            amendment_text = amendment_content
                            full_content += " ".join(amendment_text)
                            # 개정문에서 조문 구조 파싱
                            articles = self._parse_amendment_content(amendment_text)
                
                # 조문 내용 추출 (기존 조문 구조가 있는 경우)
                if '조문' in law_content and not articles:
                    articles_data = law_content['조문']
                    if isinstance(articles_data, list):
                        for article in articles_data:
                            if isinstance(article, dict):
                                article_content = ""
                                article_title = article.get('조문제목', '')
                                article_number = article.get('조문번호', '')
                                
                                # 조문내용 추출
                                if '조문내용' in article:
                                    content = article['조문내용']
                                    if isinstance(content, list):
                                        article_content = " ".join(content)
                                    else:
                                        article_content = str(content)
                                
                                # 조문별 세부 구조 파싱 (항, 호 단위)
                                parsed_article = self._parse_article_structure(article_content, article_number, article_title)
                                
                                if article_content:
                                    articles.append(parsed_article)
                                    full_content += f"\n\n{article_title}\n{article_content}"
                
                # 부칙 내용 추출
                if '부칙' in law_content:
                    supplementary_content = law_content['부칙']
                    if isinstance(supplementary_content, list):
                        for item in supplementary_content:
                            if isinstance(item, dict) and '부칙내용' in item:
                                content = item['부칙내용']
                                if isinstance(content, list):
                                    full_content += "\n\n" + " ".join(content)
                                else:
                                    full_content += "\n\n" + str(content)
            
            processed_law['articles'] = articles
            processed_law['full_content'] = full_content
            
            # 텍스트 정리 및 청킹
            if full_content.strip():
                context = f"law_{processed_law.get('law_name', '')}"
                cleaned_content = self.clean_text(full_content, context)
                processed_law['cleaned_content'] = cleaned_content
                
                # 조문항단위 청킹 적용
                if articles:
                    processed_law['article_chunks'] = self.create_article_chunks(articles)
                    processed_law['chunks'] = processed_law['article_chunks']  # 조문항단위 청크를 기본 청크로 사용
                else:
                    processed_law['chunks'] = self.split_text_into_chunks(cleaned_content)
                
                processed_law['entities'] = self.extract_legal_entities(cleaned_content)
            else:
                processed_law['cleaned_content'] = ""
                processed_law['chunks'] = []
                processed_law['article_chunks'] = []
                processed_law['entities'] = {}
            
            # 연혁 정보 처리
            if law_data.get('history'):
                processed_law['history'] = self._process_law_history(law_data['history'])
            
            return processed_law
            
        except Exception as e:
            self.logger.error(f"Error processing law data: {e}")
            return {
                'id': law_data.get('basic_info', {}).get('id', ''),
                'law_name': law_data.get('basic_info', {}).get('name', ''),
                'error': str(e),
                'status': 'failed',
                'processed_at': datetime.now().isoformat()
            }
    
    def process_precedent_data(self, precedent_data: Dict[str, Any]) -> Dict[str, Any]:
        """판례 데이터 처리"""
        try:
            # 판례 데이터는 직접 필드에 접근
            processed_precedent = {
                'id': precedent_data.get('판례일련번호', ''),
                'case_name': precedent_data.get('사건명', ''),
                'case_number': precedent_data.get('사건번호', ''),
                'court': precedent_data.get('법원명', ''),
                'court_code': precedent_data.get('법원종류코드', ''),
                'decision_date': precedent_data.get('선고일자', ''),
                'case_type': precedent_data.get('사건종류명', ''),
                'case_type_code': precedent_data.get('사건종류코드', ''),
                'decision_type': precedent_data.get('판결유형', ''),
                'category': 'precedent',
                'status': 'success',
                'processed_at': datetime.now().isoformat()
            }
            
            # 상세 정보 추출
            detail_info = precedent_data.get('detail_info', {})
            if detail_info:
                processed_precedent.update({
                    'issue': detail_info.get('판시사항', ''),
                    'reasoning': detail_info.get('판결요지', ''),
                    'case_summary': detail_info.get('사건개요', ''),
                    'dispute_point': detail_info.get('쟁점', ''),
                    'conclusion': detail_info.get('결론', ''),
                    'related_laws': detail_info.get('참조조문', ''),
                    'related_precedents': detail_info.get('참조판례', ''),
                    'keywords': detail_info.get('키워드', ''),
                    'classification': detail_info.get('분류', ''),
                    'case_content': detail_info.get('판례내용', '')  # 실제 판례 내용
                })
            
            # 전체 텍스트 결합 및 처리 - 판례내용을 우선적으로 사용
            full_text_parts = [
                processed_precedent.get('case_content', ''),  # 판례내용이 가장 중요
                processed_precedent.get('issue', ''),
                processed_precedent.get('reasoning', ''),
                processed_precedent.get('case_summary', ''),
                processed_precedent.get('dispute_point', ''),
                processed_precedent.get('conclusion', '')
            ]
            full_text = " ".join([part for part in full_text_parts if part])
            
            if full_text.strip():
                context = f"precedent_{processed_precedent.get('case_name', '')}"
                cleaned_content = self.clean_text(full_text, context)
                processed_precedent['cleaned_content'] = cleaned_content
                processed_precedent['chunks'] = self.split_text_into_chunks(cleaned_content)
                processed_precedent['entities'] = self.extract_legal_entities(cleaned_content)
            else:
                processed_precedent['cleaned_content'] = ""
                processed_precedent['chunks'] = []
                processed_precedent['entities'] = {}
            
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
    
    def create_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[Dict[str, Any]]:
        """
        텍스트를 청크로 분할 (split_text_into_chunks의 별칭 메서드)

        Args:
            text: 분할할 텍스트
            chunk_size: 청크 크기 (기본값: 1000)
            overlap: 청크 간 오버랩 크기 (기본값: 100)

        Returns:
            청크 리스트
        """
        return self.split_text_into_chunks(text, chunk_size, overlap)
    
    def _parse_article_structure(self, content: str, article_number: str, article_title: str) -> Dict[str, Any]:
        """
        조문의 세부 구조를 파싱 (항, 호 단위)
        
        Args:
            content: 조문 내용
            article_number: 조문번호
            article_title: 조문제목
            
        Returns:
            파싱된 조문 구조
        """
        import re
        
        parsed_article = {
            'article_number': article_number,
            'title': article_title,
            'content': content,
            'paragraphs': [],
            'subparagraphs': [],
            'items': [],
            'searchable_text': content
        }
        
        # 항(paragraph) 추출 - "제X항" 패턴
        paragraph_pattern = r'제(\d+)항'
        paragraph_matches = list(re.finditer(paragraph_pattern, content))
        
        for i, match in enumerate(paragraph_matches):
            paragraph_num = match.group(1)
            start_pos = match.start()
            end_pos = paragraph_matches[i + 1].start() if i + 1 < len(paragraph_matches) else len(content)
            
            paragraph_text = content[start_pos:end_pos].strip()
            
            # 호(subparagraph) 추출 - "제X호" 패턴
            subparagraph_pattern = r'제(\d+)호'
            subparagraph_matches = list(re.finditer(subparagraph_pattern, paragraph_text))
            
            subparagraphs = []
            for j, sub_match in enumerate(subparagraph_matches):
                sub_num = sub_match.group(1)
                sub_start = sub_match.start()
                sub_end = subparagraph_matches[j + 1].start() if j + 1 < len(subparagraph_matches) else len(paragraph_text)
                
                sub_text = paragraph_text[sub_start:sub_end].strip()
                subparagraphs.append({
                    'number': sub_num,
                    'content': sub_text,
                    'searchable_text': sub_text
                })
            
            parsed_article['paragraphs'].append({
                'number': paragraph_num,
                'content': paragraph_text,
                'subparagraphs': subparagraphs,
                'searchable_text': paragraph_text
            })
        
        # 조문 전체에 대한 검색 최적화 텍스트 생성
        searchable_parts = [article_title, content]
        for p in parsed_article['paragraphs']:
            searchable_parts.append(f"제{p['number']}항 {p['content']}")
            for sp in p['subparagraphs']:
                searchable_parts.append(f"제{sp['number']}호 {sp['content']}")
        
        parsed_article['searchable_text'] = " ".join(searchable_parts)
        
        return parsed_article
    
    def create_article_chunks(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        조문항단위 검색을 위한 전용 청킹
        
        Args:
            articles: 파싱된 조문 리스트
            
        Returns:
            조문항단위 청크 리스트
        """
        chunks = []
        
        for article in articles:
            # 조문 전체 청크
            chunks.append({
                'id': f"article_{article['article_number']}_full",
                'type': 'article',
                'article_number': article['article_number'],
                'title': article['title'],
                'content': article['content'],
                'searchable_text': article['searchable_text'],
                'level': 'article',
                'parent_id': None
            })
            
            # 항별 청크 (조-항-호-목 계층 구조)
            for paragraph in article.get('paragraphs', []):
                if paragraph.get('level') == 'paragraph':
                    chunks.append({
                        'id': f"article_{article['article_number']}_para_{paragraph['paragraph_number']}",
                        'type': 'paragraph',
                        'article_number': article['article_number'],
                        'paragraph_number': paragraph['paragraph_number'],
                        'title': f"제{article['article_number']}조 제{paragraph['paragraph_number']}항",
                        'content': paragraph['content'],
                        'searchable_text': paragraph['searchable_text'],
                        'level': 'paragraph',
                        'parent_id': f"article_{article['article_number']}_full"
                    })
                    
                    # 호별 청크
                    for sub_paragraph in paragraph.get('sub_paragraphs', []):
                        if sub_paragraph.get('level') == 'sub_paragraph':
                            chunks.append({
                                'id': f"article_{article['article_number']}_para_{paragraph['paragraph_number']}_sub_{sub_paragraph['sub_paragraph_number']}",
                                'type': 'sub_paragraph',
                                'article_number': article['article_number'],
                                'paragraph_number': paragraph['paragraph_number'],
                                'sub_paragraph_number': sub_paragraph['sub_paragraph_number'],
                                'title': f"제{article['article_number']}조 제{paragraph['paragraph_number']}항 제{sub_paragraph['sub_paragraph_number']}호",
                                'content': sub_paragraph['content'],
                                'searchable_text': sub_paragraph['searchable_text'],
                                'level': 'sub_paragraph',
                                'parent_id': f"article_{article['article_number']}_para_{paragraph['paragraph_number']}"
                            })
                            
                            # 목별 청크
                            for item in sub_paragraph.get('items', []):
                                if item.get('level') == 'item':
                                    chunks.append({
                                        'id': f"article_{article['article_number']}_para_{paragraph['paragraph_number']}_sub_{sub_paragraph['sub_paragraph_number']}_item_{item['item_number']}",
                                        'type': 'item',
                                        'article_number': article['article_number'],
                                        'paragraph_number': paragraph['paragraph_number'],
                                        'sub_paragraph_number': sub_paragraph['sub_paragraph_number'],
                                        'item_number': item['item_number'],
                                        'title': f"제{article['article_number']}조 제{paragraph['paragraph_number']}항 제{sub_paragraph['sub_paragraph_number']}호 {item['item_number']}목",
                                        'content': item['content'],
                                        'searchable_text': item['searchable_text'],
                                        'level': 'item',
                                        'parent_id': f"article_{article['article_number']}_para_{paragraph['paragraph_number']}_sub_{sub_paragraph['sub_paragraph_number']}"
                                    })
                        
                        # 호에 직접 속하는 목
                        elif sub_paragraph.get('level') == 'item':
                            chunks.append({
                                'id': f"article_{article['article_number']}_para_{paragraph['paragraph_number']}_item_{sub_paragraph['item_number']}",
                                'type': 'item',
                                'article_number': article['article_number'],
                                'paragraph_number': paragraph['paragraph_number'],
                                'item_number': sub_paragraph['item_number'],
                                'title': f"제{article['article_number']}조 제{paragraph['paragraph_number']}항 {sub_paragraph['item_number']}목",
                                'content': sub_paragraph['content'],
                                'searchable_text': sub_paragraph['searchable_text'],
                                'level': 'item',
                                'parent_id': f"article_{article['article_number']}_para_{paragraph['paragraph_number']}"
                            })
                
                # 조문에 직접 속하는 호
                elif paragraph.get('level') == 'sub_paragraph':
                    chunks.append({
                        'id': f"article_{article['article_number']}_sub_{paragraph['sub_paragraph_number']}",
                        'type': 'sub_paragraph',
                        'article_number': article['article_number'],
                        'sub_paragraph_number': paragraph['sub_paragraph_number'],
                        'title': f"제{article['article_number']}조 제{paragraph['sub_paragraph_number']}호",
                        'content': paragraph['content'],
                        'searchable_text': paragraph['searchable_text'],
                        'level': 'sub_paragraph',
                        'parent_id': f"article_{article['article_number']}_full"
                    })
                
                # 조문에 직접 속하는 목
                elif paragraph.get('level') == 'item':
                    chunks.append({
                        'id': f"article_{article['article_number']}_item_{paragraph['item_number']}",
                        'type': 'item',
                        'article_number': article['article_number'],
                        'item_number': paragraph['item_number'],
                        'title': f"제{article['article_number']}조 {paragraph['item_number']}목",
                        'content': paragraph['content'],
                        'searchable_text': paragraph['searchable_text'],
                        'level': 'item',
                        'parent_id': f"article_{article['article_number']}_full"
                    })
        
        return chunks
    
    def _parse_amendment_content(self, amendment_content: List[str]) -> List[Dict[str, Any]]:
        """개정문 내용에서 조문 구조 파싱 (조-항-호-목 계층 구조)"""
        articles = []
        current_article = None
        current_paragraph = None
        current_sub_paragraph = None
        current_item = None
        
        for line in amendment_content:
            line = line.strip()
            if not line:
                continue
                
            # 조문 번호 패턴 (제X조, 제X조의X 등)
            if re.match(r'제\d+조(?:의\d+)?', line):
                # 이전 조문 저장
                if current_article:
                    articles.append(current_article)
                
                # 새 조문 시작
                article_match = re.match(r'제(\d+)조(?:의(\d+))?', line)
                if article_match:
                    article_num = article_match.group(1)
                    sub_article_num = article_match.group(2)
                    
                    current_article = {
                        'article_number': f"{article_num}" + (f"의{sub_article_num}" if sub_article_num else ""),
                        'title': line,
                        'content': '',
                        'searchable_text': '',
                        'paragraphs': [],
                        'level': 'article'
                    }
                    current_paragraph = None
                    current_sub_paragraph = None
                    current_item = None
                    
            # 항 번호 패턴 (①, ②, ③ 등) - 완성된 문장
            elif re.match(r'[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]', line):
                if current_article:
                    paragraph_num = self._extract_paragraph_number(line)
                    current_paragraph = {
                        'paragraph_number': paragraph_num,
                        'content': line,
                        'searchable_text': self._clean_text(line),
                        'sub_paragraphs': [],
                        'level': 'paragraph'
                    }
                    current_article['paragraphs'].append(current_paragraph)
                    current_sub_paragraph = None
                    current_item = None
                    
            # 호 번호 패턴 (1., 2., 3. 등) - 단어나 어절, "...할 것" 형식
            elif re.match(r'\d+\.', line):
                if current_paragraph:
                    sub_paragraph_num = re.match(r'(\d+)\.', line).group(1)
                    current_sub_paragraph = {
                        'sub_paragraph_number': sub_paragraph_num,
                        'content': line,
                        'searchable_text': self._clean_text(line),
                        'items': [],
                        'level': 'sub_paragraph'
                    }
                    current_paragraph['sub_paragraphs'].append(current_sub_paragraph)
                    current_item = None
                elif current_article:
                    # 조문에 직접 속하는 호
                    sub_paragraph_num = re.match(r'(\d+)\.', line).group(1)
                    current_sub_paragraph = {
                        'sub_paragraph_number': sub_paragraph_num,
                        'content': line,
                        'searchable_text': self._clean_text(line),
                        'items': [],
                        'level': 'sub_paragraph'
                    }
                    if not current_article.get('paragraphs'):
                        current_article['paragraphs'] = []
                    current_article['paragraphs'].append(current_sub_paragraph)
                    current_item = None
                    
            # 목 번호 패턴 (가., 나., 다. 등) - 단어나 어절, "...할 것" 형식
            elif re.match(r'[가-힣]\.', line):
                if current_sub_paragraph:
                    item_num = re.match(r'([가-힣])\.', line).group(1)
                    current_item = {
                        'item_number': item_num,
                        'content': line,
                        'searchable_text': self._clean_text(line),
                        'sub_items': [],
                        'level': 'item'
                    }
                    current_sub_paragraph['items'].append(current_item)
                elif current_paragraph:
                    # 항에 직접 속하는 목
                    item_num = re.match(r'([가-힣])\.', line).group(1)
                    current_item = {
                        'item_number': item_num,
                        'content': line,
                        'searchable_text': self._clean_text(line),
                        'sub_items': [],
                        'level': 'item'
                    }
                    if not current_paragraph.get('sub_paragraphs'):
                        current_paragraph['sub_paragraphs'] = []
                    current_paragraph['sub_paragraphs'].append(current_item)
                elif current_article:
                    # 조문에 직접 속하는 목
                    item_num = re.match(r'([가-힣])\.', line).group(1)
                    current_item = {
                        'item_number': item_num,
                        'content': line,
                        'searchable_text': self._clean_text(line),
                        'sub_items': [],
                        'level': 'item'
                    }
                    if not current_article.get('paragraphs'):
                        current_article['paragraphs'] = []
                    current_article['paragraphs'].append(current_item)
                    
            # 하위 목록 패턴 (1), 2), 3) 등)
            elif re.match(r'\d+\)', line):
                if current_item:
                    sub_item_num = re.match(r'(\d+)\)', line).group(1)
                    sub_item = {
                        'sub_item_number': sub_item_num,
                        'content': line,
                        'searchable_text': self._clean_text(line),
                        'level': 'sub_item'
                    }
                    current_item['sub_items'].append(sub_item)
                elif current_sub_paragraph:
                    # 호에 직접 속하는 하위 목록
                    sub_item_num = re.match(r'(\d+)\)', line).group(1)
                    sub_item = {
                        'sub_item_number': sub_item_num,
                        'content': line,
                        'searchable_text': self._clean_text(line),
                        'level': 'sub_item'
                    }
                    if not current_sub_paragraph.get('items'):
                        current_sub_paragraph['items'] = []
                    current_sub_paragraph['items'].append(sub_item)
                    
            # 하위 하위 목록 패턴 (가), 나), 다) 등)
            elif re.match(r'[가-힣]\)', line):
                if current_item and current_item.get('sub_items'):
                    # 1), 2), 3)의 하위 항목
                    sub_sub_item_num = re.match(r'([가-힣])\)', line).group(1)
                    sub_sub_item = {
                        'sub_sub_item_number': sub_sub_item_num,
                        'content': line,
                        'searchable_text': self._clean_text(line),
                        'level': 'sub_sub_item'
                    }
                    current_item['sub_items'][-1]['sub_sub_items'] = current_item['sub_items'][-1].get('sub_sub_items', [])
                    current_item['sub_items'][-1]['sub_sub_items'].append(sub_sub_item)
                    
            # 일반 내용
            else:
                if current_item and current_item.get('sub_items'):
                    # 하위 목록에 내용 추가
                    current_item['sub_items'][-1]['content'] += f" {line}"
                    current_item['sub_items'][-1]['searchable_text'] += f" {self._clean_text(line)}"
                elif current_item:
                    # 목에 내용 추가
                    current_item['content'] += f" {line}"
                    current_item['searchable_text'] += f" {self._clean_text(line)}"
                elif current_sub_paragraph:
                    # 호에 내용 추가
                    current_sub_paragraph['content'] += f" {line}"
                    current_sub_paragraph['searchable_text'] += f" {self._clean_text(line)}"
                elif current_paragraph:
                    # 항에 내용 추가
                    current_paragraph['content'] += f" {line}"
                    current_paragraph['searchable_text'] += f" {self._clean_text(line)}"
                elif current_article:
                    # 조문에 내용 추가
                    current_article['content'] += f" {line}"
                    current_article['searchable_text'] += f" {self._clean_text(line)}"
        
        # 마지막 조문 저장
        if current_article:
            articles.append(current_article)
            
        return articles
    
    def _extract_paragraph_number(self, line: str) -> str:
        """항 번호 추출"""
        # ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳ 매핑
        circle_numbers = {
            '①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5',
            '⑥': '6', '⑦': '7', '⑧': '8', '⑨': '9', '⑩': '10',
            '⑪': '11', '⑫': '12', '⑬': '13', '⑭': '14', '⑮': '15',
            '⑯': '16', '⑰': '17', '⑱': '18', '⑲': '19', '⑳': '20'
        }
        
        for circle, num in circle_numbers.items():
            if line.startswith(circle):
                return num
        return '1'
    
    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        if not text:
            return ""
        
        # 기본 정리
        cleaned = text.strip()
        
        # 특수문자 정리
        cleaned = re.sub(r'\s+', ' ', cleaned)  # 연속 공백을 하나로
        cleaned = re.sub(r'[^\w\s가-힣]', ' ', cleaned)  # 특수문자 제거 (한글, 영문, 숫자, 공백만 유지)
        
        return cleaned.strip()