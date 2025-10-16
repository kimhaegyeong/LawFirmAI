#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
법령해석례 수집기 클래스 (collect_precedents.py 구조 참고)
"""

import json
import time
import random
import signal
import atexit
import hashlib
import traceback
import gc  # 가비지 컬렉션
import re  # 정규표현식
from bs4 import BeautifulSoup  # HTML 파싱
try:
    import psutil  # 메모리 모니터링
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil이 설치되지 않았습니다. 메모리 모니터링이 비활성화됩니다.")
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from contextlib import contextmanager

import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIClient, LawOpenAPIConfig
from scripts.legal_interpretation.legal_interpretation_models import (
    CollectionStats, LegalInterpretationData, CollectionStatus, InterpretationCategory, MinistryType,
    INTERPRETATION_KEYWORDS, PRIORITY_KEYWORDS, KEYWORD_TARGET_COUNTS, DEFAULT_KEYWORD_COUNT,
    MINISTRY_KEYWORDS, INTERPRETATION_TOPIC_KEYWORDS, FALLBACK_KEYWORDS
)
from scripts.legal_interpretation.legal_interpretation_logger import setup_logging

logger = setup_logging()


class LegalInterpretationCollector:
    """법령해석례 수집 클래스 (개선된 버전)"""
    
    def __init__(self, config: LawOpenAPIConfig, output_dir: Optional[Path] = None):
        """
        법령해석례 수집기 초기화
        
        Args:
            config: API 설정 객체
            output_dir: 출력 디렉토리 (기본값: data/raw/legal_interpretations)
        """
        self.client = LawOpenAPIClient(config)
        self.output_dir = output_dir or Path("data/raw/legal_interpretations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 배치 저장 디렉토리
        self.batch_dir = self.output_dir / "batches"
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 관리
        self.collected_interpretations: Set[str] = set()  # 중복 방지
        self.processed_keywords: Set[str] = set()  # 처리된 키워드 추적
        self.pending_interpretations: List[LegalInterpretationData] = []  # 임시 저장소
        
        # 설정
        self.batch_size = 5  # 배치 저장 크기 (상세 정보 수집으로 인한 메모리 최적화)
        self.max_retries = 3  # 최대 재시도 횟수
        self.retry_delay = 5  # 재시도 간격 (초)
        self.api_delay_range = (1.0, 3.0)  # API 요청 간 지연 범위
        
        # 메모리 관리 설정
        self.memory_check_interval = 50  # 메모리 체크 간격 (API 요청 수)
        self.max_memory_usage = 80  # 최대 메모리 사용률 (%)
        self.api_request_count = 0  # API 요청 카운터
        
        # 배치 관리
        self.batch_counter = 0  # 배치 카운터
        
        # 통계 및 상태
        self.stats = CollectionStats()
        self.stats.total_keywords = len(INTERPRETATION_KEYWORDS)
        self.checkpoint_file: Optional[Path] = None
        self.resume_mode = False
        
        # 에러 처리
        self.error_count = 0
        self.max_errors = 50  # 최대 허용 에러 수
        
        # Graceful shutdown 관련
        self.shutdown_requested = False
        self.shutdown_reason = None
        self._setup_signal_handlers()
        
        # 기존 수집된 데이터 로드
        self._load_existing_data()
        
        # 체크포인트 파일 확인 및 복구
        self._check_and_resume_from_checkpoint()
        
        logger.info(f"법령해석례 수집기 초기화 완료 - 목표: {self.stats.target_count}건")
    
    def _setup_signal_handlers(self):
        """시그널 핸들러 설정 (Graceful shutdown)"""
        def signal_handler(signum, frame):
            """시그널 핸들러"""
            signal_name = signal.Signals(signum).name
            logger.warning(f"시그널 {signal_name} ({signum}) 수신됨. Graceful shutdown 시작...")
            self.shutdown_requested = True
            self.shutdown_reason = f"Signal {signal_name} ({signum})"
        
        # SIGINT (Ctrl+C), SIGTERM 처리
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Windows에서 SIGBREAK 처리
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
        
        # 프로그램 종료 시 정리 함수 등록
        atexit.register(self._cleanup_on_exit)
    
    def _cleanup_on_exit(self):
        """프로그램 종료 시 정리 작업"""
        if self.pending_interpretations:
            logger.info("프로그램 종료 시 임시 데이터 저장 중...")
            self._save_batch_interpretations()
        
        if self.checkpoint_file:
            logger.info("최종 체크포인트 저장 중...")
            self._save_checkpoint(self.checkpoint_file)
    
    def _check_shutdown_requested(self) -> bool:
        """종료 요청 확인"""
        return self.shutdown_requested
    
    def _get_next_batch_number(self) -> int:
        """다음 배치 번호 생성"""
        self.batch_counter += 1
        return self.batch_counter
    
    def _extract_content_from_json(self, json_data: Dict[str, Any]) -> Tuple[str, str]:
        """JSON에서 질의내용과 회신내용 추출"""
        try:
            question_content = ""
            answer_content = ""
            
            # ExpcService 필드 내부 확인
            expc_service = json_data.get('ExpcService', {})
            if isinstance(expc_service, dict):
                # ExpcService 내부에서 질의요지와 회답 찾기
                question_content = expc_service.get('질의요지', '').strip()
                answer_content = expc_service.get('회답', '').strip()
                
                # 충분한 길이가 있으면 사용
                if len(question_content) > 10 and len(answer_content) > 10:
                    return question_content, answer_content
            
            # 최상위 레벨에서도 확인
            # 다양한 필드명으로 질의내용 찾기
            question_fields = [
                '질의요지', '질의내용', '질의 내용', '질의사항', '질의 사항',
                'question', '질문', '문의내용', '문의 내용',
                '질의', '문의', '요청내용', '요청 내용'
            ]
            
            for field in question_fields:
                if field in json_data and json_data[field]:
                    question_content = str(json_data[field]).strip()
                    if len(question_content) > 10:  # 충분한 길이의 내용만
                        break
            
            # 다양한 필드명으로 회신내용 찾기
            answer_fields = [
                '회답', '회신내용', '회신 내용', '답변내용', '답변 내용',
                'answer', 'reply', '해석내용', '해석 내용',
                '회신', '답변', '해석', '결론'
            ]
            
            for field in answer_fields:
                if field in json_data and json_data[field]:
                    answer_content = str(json_data[field]).strip()
                    if len(answer_content) > 10:  # 충분한 길이의 내용만
                        break
            
            # 중첩된 객체에서 찾기
            if not question_content or not answer_content:
                for key, value in json_data.items():
                    if isinstance(value, dict):
                        sub_question, sub_answer = self._extract_content_from_json(value)
                        if sub_question and not question_content:
                            question_content = sub_question
                        if sub_answer and not answer_content:
                            answer_content = sub_answer
            
            # 최종 정리
            question_content = self._clean_text(question_content)
            answer_content = self._clean_text(answer_content)
            
            return question_content, answer_content
            
        except Exception as e:
            logger.error(f"JSON 본문 추출 실패: {e}")
            return "", ""
    
    def _extract_content_from_html(self, html_content: str) -> Tuple[str, str]:
        """HTML에서 질의내용과 회신내용 추출"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            question_content = ""
            answer_content = ""
            
            # 질의내용 추출 (다양한 패턴 시도)
            question_patterns = [
                '질의내용', '질의 내용', '질의사항', '질의 사항',
                'question', '질문', '문의내용', '문의 내용'
            ]
            
            for pattern in question_patterns:
                # 텍스트로 패턴 검색
                question_elements = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
                for element in question_elements:
                    parent = element.parent
                    if parent:
                        # 다음 형제 요소들에서 내용 추출
                        content = self._extract_text_from_element(parent)
                        if content and len(content) > 50:  # 충분한 길이의 내용만
                            question_content = content
                            break
                if question_content:
                    break
            
            # 회신내용 추출 (다양한 패턴 시도)
            answer_patterns = [
                '회신내용', '회신 내용', '답변내용', '답변 내용',
                'answer', 'reply', '해석내용', '해석 내용'
            ]
            
            for pattern in answer_patterns:
                # 텍스트로 패턴 검색
                answer_elements = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
                for element in answer_elements:
                    parent = element.parent
                    if parent:
                        # 다음 형제 요소들에서 내용 추출
                        content = self._extract_text_from_element(parent)
                        if content and len(content) > 50:  # 충분한 길이의 내용만
                            answer_content = content
                            break
                if answer_content:
                    break
            
            # 패턴으로 찾지 못한 경우 테이블 구조에서 추출 시도
            if not question_content or not answer_content:
                tables = soup.find_all('table')
                for table in tables:
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all(['td', 'th'])
                        if len(cells) >= 2:
                            label = cells[0].get_text(strip=True)
                            content = cells[1].get_text(strip=True)
                            
                            if any(pattern in label for pattern in question_patterns):
                                question_content = content
                            elif any(pattern in label for pattern in answer_patterns):
                                answer_content = content
            
            # 최종 정리
            question_content = self._clean_text(question_content)
            answer_content = self._clean_text(answer_content)
            
            return question_content, answer_content
            
        except Exception as e:
            logger.error(f"HTML 본문 추출 실패: {e}")
            return "", ""
    
    def _extract_text_from_element(self, element) -> str:
        """요소에서 텍스트 추출"""
        try:
            # 요소의 모든 텍스트 추출
            text = element.get_text(separator=' ', strip=True)
            
            # 다음 형제 요소들도 확인
            next_sibling = element.find_next_sibling()
            if next_sibling:
                next_text = next_sibling.get_text(separator=' ', strip=True)
                if next_text and len(next_text) > len(text):
                    text = next_text
            
            return text
        except:
            return ""
    
    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        if not text:
            return ""
        
        # 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 특수 문자 정리
        text = re.sub(r'[^\w\s가-힣.,!?()\[\]{}:";]', '', text)
        
        return text
    
    def _collect_interpretation_detail(self, interpretation: Dict[str, Any]) -> Dict[str, Any]:
        """법령해석례 상세 정보 수집 (본문 포함)"""
        try:
            # 기본 정보 복사
            detailed_data = interpretation.copy()
            
            # 해석례 ID 추출 (상세 조회용으로는 법령해석례일련번호 사용)
            interpretation_id = (interpretation.get('법령해석례일련번호') or
                               interpretation.get('일련번호') or
                               interpretation.get('id'))
            
            if not interpretation_id:
                logger.warning("해석례 ID가 없어 상세 정보를 수집할 수 없습니다")
                return detailed_data
            
            logger.debug(f"해석례 상세 정보 수집 중: ID={interpretation_id}")
            
            # JSON 형식으로 상세 정보 조회 (본문 추출용)
            detail_json = self.client.get_interpretation_detail(str(interpretation_id))
            if detail_json and isinstance(detail_json, dict):
                # JSON 상세 정보를 기본 데이터에 병합
                detailed_data.update(detail_json)
                
                # JSON 구조 로깅 (디버깅용)
                logger.debug(f"JSON 응답 필드: {list(detail_json.keys())}")
                
                # JSON에서 본문 내용 추출
                question_content, answer_content = self._extract_content_from_json(detail_json)
                detailed_data['question_content'] = question_content
                detailed_data['answer_content'] = answer_content
                
                logger.debug(f"JSON 상세 정보 수집 완료: ID={interpretation_id}")
                logger.debug(f"질의내용 길이: {len(question_content)}, 회신내용 길이: {len(answer_content)}")
                
                # 질의내용과 회신내용이 없으면 JSON 구조 출력
                if not question_content and not answer_content:
                    logger.debug(f"본문 추출 실패 - JSON 구조: {str(detail_json)[:500]}...")
            else:
                logger.debug(f"JSON 상세 정보 수집 실패: ID={interpretation_id}")
            
            # HTML 형식은 에러 페이지가 반환되므로 건너뛰기
            # detail_html = self.client.get_interpretation_detail_html(str(interpretation_id))
            # if detail_html and not detail_html.strip().startswith('<!DOCTYPE html'):
            #     detailed_data['html_content'] = detail_html
            #     logger.debug(f"HTML 상세 정보 수집 완료: ID={interpretation_id}")
            # else:
            #     logger.debug(f"HTML 상세 정보 수집 실패: ID={interpretation_id}")
            
            # 기본 정보에 상세 링크 추가
            detail_url = interpretation.get('법령해석례상세링크', '')
            if detail_url:
                detailed_data['detail_url'] = f"http://www.law.go.kr{detail_url}"
            
            # 질의기관, 회신기관 정보 추가
            detailed_data['inquiry_agency'] = interpretation.get('질의기관명', '')
            detailed_data['reply_agency'] = interpretation.get('회신기관명', '')
            
            # API 요청 간 지연
            self._random_delay()
            
            return detailed_data
            
        except Exception as e:
            logger.error(f"해석례 상세 정보 수집 실패 (ID: {interpretation_id}): {e}")
            return interpretation  # 실패 시 기본 정보만 반환
    
    def _check_memory_usage(self) -> bool:
        """메모리 사용량 확인 및 관리"""
        if not PSUTIL_AVAILABLE:
            # psutil이 없는 경우 가비지 컬렉션만 실행
            if len(self.pending_interpretations) > self.batch_size:
                logger.info("psutil 없이 배치 저장 실행")
                self._save_batch_interpretations()
                return True
            return False
            
        try:
            # 현재 프로세스의 메모리 사용량 확인
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # 메모리 사용률이 임계값을 초과한 경우
            if memory_percent > self.max_memory_usage:
                logger.warning(f"메모리 사용률이 높습니다: {memory_percent:.1f}% (임계값: {self.max_memory_usage}%)")
                logger.warning(f"RSS 메모리: {memory_info.rss / 1024 / 1024:.1f}MB")
                
                # 가비지 컬렉션 강제 실행
                logger.info("가비지 컬렉션 실행 중...")
                collected = gc.collect()
                logger.info(f"가비지 컬렉션 완료: {collected}개 객체 정리")
                
                # 메모리 사용량 재확인
                memory_percent_after = process.memory_percent()
                logger.info(f"가비지 컬렉션 후 메모리 사용률: {memory_percent_after:.1f}%")
                
                # 여전히 높은 경우 배치 저장 강제 실행
                if memory_percent_after > self.max_memory_usage and len(self.pending_interpretations) > 0:
                    logger.warning("메모리 사용률이 여전히 높아 배치 저장을 강제 실행합니다")
                    self._save_batch_interpretations()
                    return True
                
            return False
            
        except Exception as e:
            logger.error(f"메모리 사용량 확인 중 오류: {e}")
            return False
    
    def _request_shutdown(self, reason: str):
        """종료 요청"""
        self.shutdown_requested = True
        self.shutdown_reason = reason
        logger.warning(f"종료 요청됨: {reason}")
    
    def _load_existing_data(self):
        """기존 수집된 데이터 로드하여 중복 방지"""
        logger.info("기존 수집된 데이터 확인 중...")
        
        loaded_count = 0
        error_count = 0
        
        # 다양한 파일 패턴에서 데이터 로드
        file_patterns = [
            "legal_interpretation_*.json",
            "batch_*.json", 
            "checkpoints/**/*.json",
            "*.json"
        ]
        
        for pattern in file_patterns:
            files = list(self.output_dir.glob(pattern))
            for file_path in files:
                try:
                    loaded_count += self._load_interpretations_from_file(file_path)
                except Exception as e:
                    error_count += 1
                    logger.debug(f"파일 로드 실패 {file_path}: {e}")
        
        logger.info(f"기존 데이터 로드 완료: {loaded_count:,}건, 오류: {error_count:,}건")
        self.stats.collected_count = len(self.collected_interpretations)
        logger.info(f"중복 방지를 위한 해석례 ID {len(self.collected_interpretations):,}개 로드됨")
    
    def _load_interpretations_from_file(self, file_path: Path) -> int:
        """파일에서 법령해석례 데이터 로드"""
        loaded_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 다양한 데이터 구조 처리
        interpretations = []
        
        if isinstance(data, dict):
            if 'interpretations' in data:
                interpretations = data['interpretations']
            elif 'basic_info' in data:
                interpretations = [data]
            elif 'by_category' in data:
                for category_data in data['by_category'].values():
                    interpretations.extend(category_data)
        elif isinstance(data, list):
            interpretations = data
        
        # 해석례 ID 추출
        for interpretation in interpretations:
            if isinstance(interpretation, dict):
                interpretation_id = interpretation.get('판례일련번호') or interpretation.get('interpretation_id')
                if interpretation_id:
                    self.collected_interpretations.add(str(interpretation_id))
                    loaded_count += 1
        
        return loaded_count
    
    def _is_duplicate_interpretation(self, interpretation: Dict[str, Any]) -> bool:
        """법령해석례 중복 여부 확인"""
        interpretation_id = (interpretation.get('id') or
                           interpretation.get('법령해석례일련번호') or
                           interpretation.get('판례일련번호') or 
                           interpretation.get('interpretation_id') or
                           interpretation.get('일련번호'))
        
        if interpretation_id and str(interpretation_id).strip() != '':
            if str(interpretation_id) in self.collected_interpretations:
                logger.debug(f"해석례일련번호로 중복 확인: {interpretation_id}")
                return True
        
        return False
    
    def _mark_interpretation_collected(self, interpretation: Dict[str, Any]):
        """법령해석례를 수집됨으로 표시"""
        interpretation_id = (interpretation.get('id') or
                           interpretation.get('법령해석례일련번호') or
                           interpretation.get('판례일련번호') or 
                           interpretation.get('interpretation_id') or
                           interpretation.get('일련번호'))
        
        if interpretation_id and str(interpretation_id).strip() != '':
            self.collected_interpretations.add(str(interpretation_id))
            logger.debug(f"해석례일련번호로 저장: {interpretation_id}")
    
    def _validate_interpretation_data(self, interpretation: Dict[str, Any]) -> bool:
        """법령해석례 데이터 검증"""
        # 법령해석례 API 응답 필드명 확인 (실제 API 응답 기준)
        interpretation_id = (interpretation.get('id') or
                           interpretation.get('법령해석례일련번호') or
                           interpretation.get('판례일련번호') or 
                           interpretation.get('interpretation_id') or
                           interpretation.get('일련번호'))
        case_name = (interpretation.get(' 안건명') or  # 실제 API 응답 필드명 (앞에 공백)
                    interpretation.get('안건명') or
                    interpretation.get('사건명') or 
                    interpretation.get('case_name') or
                    interpretation.get('제목') or
                    interpretation.get('title'))
        
        if not interpretation_id or str(interpretation_id).strip() == '':
            if not case_name:
                logger.warning(f"법령해석례 식별 정보 부족 - 해석례ID: {interpretation_id}, 사건명: {case_name}")
                logger.warning(f"사용 가능한 필드: {list(interpretation.keys())}")
                logger.warning(f"전체 데이터: {str(interpretation)[:200]}...")
                return False
            logger.debug(f"해석례일련번호 없음, 사건명으로 대체: {case_name}")
        elif not case_name:
            logger.warning(f"사건명이 없습니다: {str(interpretation)[:200]}...")
            return False
        
        return True
    
    def _create_interpretation_data(self, raw_data: Dict[str, Any]) -> Optional[LegalInterpretationData]:
        """원시 데이터에서 LegalInterpretationData 객체 생성"""
        try:
            # 데이터 검증
            if not self._validate_interpretation_data(raw_data):
                return None
            
            # 해석례 ID 추출 (실제 API 응답 필드명 우선)
            interpretation_id = (raw_data.get('id') or
                               raw_data.get('법령해석례일련번호') or
                               raw_data.get('판례일련번호') or 
                               raw_data.get('interpretation_id') or
                               raw_data.get('일련번호'))
            
            # 해석례 ID가 없는 경우 대체 ID 생성
            if not interpretation_id or str(interpretation_id).strip() == '':
                case_name = (raw_data.get(' 안건명', '') or  # 실제 API 응답 필드명 (앞에 공백)
                            raw_data.get('안건명', '') or
                            raw_data.get('사건명', '') or 
                            raw_data.get('case_name', '') or
                            raw_data.get('제목', '') or
                            raw_data.get('title', ''))
                if case_name:
                    interpretation_id = f"interpretation_{case_name}"
                else:
                    interpretation_id = f"interpretation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                logger.debug(f"대체 ID 생성: {interpretation_id}")
            
            # 카테고리 분류
            category = self.categorize_interpretation(raw_data)
            
            # 주제 분류
            topic = self.classify_interpretation_topic(raw_data)
            
            # 부처 분류
            ministry = self.classify_interpretation_ministry(raw_data)
            
            # LegalInterpretationData 객체 생성 (실제 API 응답 필드명 사용)
            interpretation_data = LegalInterpretationData(
                interpretation_id=str(interpretation_id),
                case_name=(raw_data.get(' 안건명', '') or  # 실제 API 응답 필드명 (앞에 공백)
                          raw_data.get('안건명', '') or
                          raw_data.get('사건명', '') or 
                          raw_data.get('case_name', '') or
                          raw_data.get('제목', '') or
                          raw_data.get('title', '')),
                case_number=(raw_data.get('안건번호', '') or
                           raw_data.get('사건번호', '') or 
                           raw_data.get('case_number', '') or
                           raw_data.get('번호', '')),
                ministry=ministry,
                decision_date=(raw_data.get('회신일자', '') or
                             raw_data.get('판결일자', '') or 
                             raw_data.get('선고일자', '') or
                             raw_data.get('decision_date', '') or
                             raw_data.get('일자', '')),
                category=category,
                topic=topic,
                raw_data=raw_data,
                # 상세 정보 필드들
                question_content=raw_data.get('질의내용', '') or raw_data.get('question', ''),
                answer_content=raw_data.get('회신내용', '') or raw_data.get('answer', ''),
                related_laws=raw_data.get('관련법령', '') or raw_data.get('related_laws', ''),
                html_content=raw_data.get('html_content', ''),
                detail_url=raw_data.get('법령해석례상세링크', '') or raw_data.get('detail_url', ''),
                inquiry_agency=raw_data.get('질의기관명', '') or raw_data.get('inquiry_agency', ''),
                reply_agency=raw_data.get('회신기관명', '') or raw_data.get('reply_agency', '')
            )
            
            return interpretation_data
            
        except Exception as e:
            logger.error(f"LegalInterpretationData 생성 실패: {e}")
            logger.error(f"원시 데이터: {raw_data}")
            return None
    
    def categorize_interpretation(self, interpretation: Dict[str, Any]) -> InterpretationCategory:
        """법령해석례 카테고리 분류"""
        case_name = interpretation.get('사건명', '').lower()
        case_content = interpretation.get('판시사항', '') + ' ' + interpretation.get('판결요지', '')
        case_content = case_content.lower()
        
        # 키워드 기반 분류
        category_keywords = {
            InterpretationCategory.ADMINISTRATIVE: [
                '행정처분', '허가', '인가', '신고', '행정심판', '행정소송', '국세', '지방세'
            ],
            InterpretationCategory.CIVIL: [
                '계약', '손해배상', '소유권', '물권', '채권', '상속', '혼인', '이혼'
            ],
            InterpretationCategory.CRIMINAL: [
                '절도', '강도', '사기', '살인', '상해', '교통사고', '음주운전'
            ],
            InterpretationCategory.COMMERCIAL: [
                '회사', '주식', '어음', '수표', '보험', '상행위'
            ],
            InterpretationCategory.LABOR: [
                '근로계약', '임금', '해고', '노동조합', '산업재해'
            ],
            InterpretationCategory.INTELLECTUAL_PROPERTY: [
                '특허', '저작권', '상표', '영업비밀', '부정경쟁'
            ],
            InterpretationCategory.CONSUMER: [
                '소비자', '계약', '약관', '표시광고', '할부거래'
            ],
            InterpretationCategory.ENVIRONMENT: [
                '환경', '대기', '수질', '폐기물', '환경영향평가'
            ],
            InterpretationCategory.FINANCE: [
                '금융', '은행', '보험', '증권', '자본시장'
            ],
            InterpretationCategory.INFORMATION_TECHNOLOGY: [
                '정보통신', '전자거래', '개인정보', '사이버보안'
            ]
        }
        
        # 키워드 매칭으로 카테고리 결정
        for category, keywords in category_keywords.items():
            if any(keyword in case_name or keyword in case_content for keyword in keywords):
                return category
        
        return InterpretationCategory.OTHER
    
    def classify_interpretation_topic(self, interpretation: Dict[str, Any]) -> str:
        """법령해석례 주제 분류"""
        case_name = interpretation.get('사건명', '').lower()
        case_content = interpretation.get('판시사항', '') + ' ' + interpretation.get('판결요지', '')
        case_content = case_content.lower()
        
        # 주제별 키워드 매칭
        for topic, keywords in INTERPRETATION_TOPIC_KEYWORDS.items():
            for keyword in keywords:
                if keyword in case_name or keyword in case_content:
                    return topic
        
        return "기타"
    
    def classify_interpretation_ministry(self, interpretation: Dict[str, Any]) -> str:
        """법령해석례 부처 분류"""
        case_name = interpretation.get('사건명', '').lower()
        case_content = interpretation.get('판시사항', '') + ' ' + interpretation.get('판결요지', '')
        case_content = case_content.lower()
        
        # 부처별 키워드 매칭
        for ministry, keywords in MINISTRY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in case_name or keyword in case_content:
                    return ministry
        
        return "기타"
    
    def _random_delay(self, min_seconds: Optional[float] = None, max_seconds: Optional[float] = None):
        """API 요청 간 랜덤 지연"""
        min_delay = min_seconds or self.api_delay_range[0]
        max_delay = max_seconds or self.api_delay_range[1]
        delay = random.uniform(min_delay, max_delay)
        logger.debug(f"API 요청 간 {delay:.2f}초 대기...")
        time.sleep(delay)
    
    @contextmanager
    def _api_request_with_retry(self, operation_name: str):
        """API 요청 재시도 컨텍스트 매니저"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                self.stats.api_requests_made += 1
                yield
                return
            except Exception as e:
                last_exception = e
                self.stats.api_errors += 1
                self.error_count += 1
                
                if attempt < self.max_retries - 1:
                    logger.warning(f"{operation_name} 실패 (시도 {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))  # 지수 백오프
                else:
                    logger.error(f"{operation_name} 최종 실패: {e}")
                    break
        
        # 모든 재시도가 실패한 경우 예외 발생
        if last_exception:
            raise last_exception
    
    def collect_all_interpretations(self, target_count: int = 2000):
        """모든 법령해석례 수집 (최신 판례 우선 수집)"""
        self.stats.target_count = target_count
        self.stats.status = CollectionStatus.IN_PROGRESS
        
        # 체크포인트 파일 설정
        checkpoint_file = self._setup_checkpoint_file()
        
        try:
            logger.info(f"법령해석례 수집 시작 - 목표: {target_count}건")
            logger.info("최신 판례부터 수집하는 방식으로 변경됨")
            logger.info("Graceful shutdown 지원: Ctrl+C 또는 SIGTERM으로 안전하게 중단 가능")
            logger.info("중단 시 현재까지 수집된 데이터가 자동으로 저장됩니다")
            
            # 최신 판례부터 수집 (키워드 기반이 아닌 날짜 기반)
            self._collect_by_date_range(target_count, checkpoint_file)
            
            # 목표 달성하지 못한 경우 추가 수집 시도
            if self.stats.collected_count < target_count:
                remaining_count = target_count - self.stats.collected_count
                logger.info(f"목표 달성 실패. 추가 수집으로 {remaining_count}건 더 수집 시도")
                self._collect_additional_interpretations(remaining_count, checkpoint_file)
            
            # 종료 요청 확인
            if self._check_shutdown_requested():
                logger.warning(f"수집이 중단되었습니다: {self.shutdown_reason}")
                self.stats.status = CollectionStatus.CANCELLED
                self.stats.end_time = datetime.now()
                self._save_final_checkpoint(checkpoint_file)
                return
            
            # 최종 통계 출력
            self._print_final_stats()
            
            # 체크포인트 파일 정리
            self._cleanup_checkpoint_file(checkpoint_file)
            
            self.stats.status = CollectionStatus.COMPLETED
            self.stats.end_time = datetime.now()
            
        except KeyboardInterrupt:
            logger.warning("사용자에 의해 수집이 중단되었습니다.")
            self.stats.status = CollectionStatus.CANCELLED
            self.stats.end_time = datetime.now()
            self._save_final_checkpoint(checkpoint_file)
            return
        except Exception as e:
            logger.error(f"법령해석례 수집 중 오류 발생: {e}")
            self.stats.status = CollectionStatus.FAILED
            self.stats.end_time = datetime.now()
            self._save_final_checkpoint(checkpoint_file)
            raise
    
    def _setup_checkpoint_file(self) -> Path:
        """체크포인트 파일 설정"""
        if self.resume_mode and self.checkpoint_file:
            return self.checkpoint_file
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return self.output_dir / f"collection_checkpoint_{timestamp}.json"
    
    def _collect_by_date_range(self, target_count: int, checkpoint_file: Path):
        """날짜 범위 기반으로 최신 판례부터 수집"""
        logger.info("최신 판례부터 날짜 기반 수집 시작")
        
        # 최근 3년간의 데이터부터 수집
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365)  # 3년 전
        
        logger.info(f"수집 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        # 날짜별로 수집 (최신부터 역순으로)
        current_date = end_date
        collected_count = 0
        
        while current_date >= start_date and collected_count < target_count:
            if self._check_shutdown_requested():
                logger.warning(f"종료 요청으로 날짜 기반 수집 중단: {self.shutdown_reason}")
                break
            
            # 해당 날짜의 판례 수집
            date_str = current_date.strftime('%Y%m%d')
            remaining_count = target_count - collected_count
            
            logger.info(f"날짜 {date_str} 판례 수집 중... (목표: {remaining_count}건)")
            
            try:
                # 해당 날짜의 판례 검색
                start_time = time.time()
                interpretations = self.collect_interpretations_by_date(date_str, min(100, remaining_count))
                elapsed_time = time.time() - start_time
                
                if interpretations:
                    # 수집된 해석례를 임시 저장소에 추가
                    for interpretation in interpretations:
                        self.pending_interpretations.append(interpretation)
                        self.stats.collected_count += 1
                        collected_count += 1
                        
                        if collected_count >= target_count:
                            break
                    
                    # 배치 저장
                    if len(self.pending_interpretations) >= self.batch_size:
                        logger.info(f"배치 크기 도달로 저장 실행 (대기 중: {len(self.pending_interpretations)}건)")
                        self._save_batch_interpretations()
                    
                    progress_percent = (collected_count / target_count) * 100
                    logger.info(f"날짜 {date_str}: {len(interpretations)}건 수집 (누적: {collected_count}/{target_count}건, {progress_percent:.1f}%, 소요시간: {elapsed_time:.1f}초)")
                else:
                    logger.debug(f"날짜 {date_str}: 수집된 판례 없음 (소요시간: {elapsed_time:.1f}초)")
                
                # 체크포인트 저장
                self._save_checkpoint(checkpoint_file)
                
                # API 요청 제한 확인
                if self._check_api_limits():
                    logger.warning("API 요청 제한에 도달하여 날짜 기반 수집 중단")
                    break
                
            except Exception as e:
                logger.error(f"날짜 {date_str} 수집 중 오류: {e}")
                continue
            
            # 다음 날로 이동
            current_date -= timedelta(days=1)
            
            # API 요청 간 지연
            self._random_delay()
        
        logger.info(f"날짜 기반 수집 완료: {collected_count}건 수집")
    
    def collect_interpretations_by_date(self, date_str: str, max_count: int = 100) -> List[LegalInterpretationData]:
        """특정 날짜의 법령해석례 검색 및 수집"""
        logger.debug(f"날짜 {date_str}로 법령해석례 검색 시작 (목표: {max_count}건)")
        
        interpretations = []
        page = 1
        consecutive_empty_pages = 0
        max_empty_pages = 3
        
        while len(interpretations) < max_count and consecutive_empty_pages < max_empty_pages:
            # 종료 요청 확인
            if self._check_shutdown_requested():
                logger.warning(f"종료 요청으로 날짜 {date_str} 검색 중단: {self.shutdown_reason}")
                break
            
            try:
                # API 요청 간 랜덤 지연
                if page > 1:
                    self._random_delay()
                
                # 메모리 사용량 체크
                self.api_request_count += 1
                if self.api_request_count % self.memory_check_interval == 0:
                    if self._check_memory_usage():
                        logger.info("메모리 최적화로 인한 배치 저장 완료")
                
                # API 요청 실행
                try:
                    with self._api_request_with_retry(f"날짜 {date_str} 검색"):
                        results = self.client.get_interpretation_list(
                            display=100,
                            page=page,
                            from_date=date_str,
                            to_date=date_str
                        )
                except Exception as api_error:
                    logger.error(f"API 요청 실패: {api_error}")
                    consecutive_empty_pages += 1
                    page += 1
                    continue
                
                if not results or len(results) == 0:
                    consecutive_empty_pages += 1
                    logger.debug(f"날짜 {date_str} 페이지 {page}에서 결과 없음 (연속 빈 페이지: {consecutive_empty_pages})")
                    page += 1
                    continue
                else:
                    consecutive_empty_pages = 0
                    logger.info(f"날짜 {date_str} 페이지 {page}: {len(results)}건 발견")
                    
                    # 첫 번째 결과의 구조 확인
                    if results and len(results) > 0:
                        first_result = results[0]
                        logger.debug(f"첫 번째 결과 구조: {list(first_result.keys()) if isinstance(first_result, dict) else type(first_result)}")
                        if isinstance(first_result, dict):
                            logger.debug(f"첫 번째 결과 샘플: {str(first_result)[:200]}...")
                
                # 결과 처리
                new_count, duplicate_count = self._process_search_results(results, interpretations, max_count)
                
                # 페이지별 결과 로깅
                logger.debug(f"페이지 {page}: {new_count}건 신규, {duplicate_count}건 중복 (누적: {len(interpretations)}/{max_count}건)")
                
                page += 1
                
                # API 요청 제한 확인
                if self._check_api_limits():
                    break
                    
            except KeyboardInterrupt:
                logger.warning(f"날짜 {date_str} 검색이 중단되었습니다.")
                break
            except Exception as e:
                logger.error(f"날짜 {date_str} 검색 중 오류: {e}")
                self.stats.failed_count += 1
                break
        
        # 수집된 해석례를 임시 저장소에 추가
        for interpretation in interpretations:
            self.pending_interpretations.append(interpretation)
            self.stats.collected_count += 1
        
        logger.info(f"날짜 {date_str} 수집 완료: {len(interpretations)}건")
        return interpretations
    
    def _collect_additional_interpretations(self, remaining_count: int, checkpoint_file: Path):
        """추가 수집 (키워드 기반)"""
        logger.info(f"추가 수집 시작 - 목표: {remaining_count}건")
        
        # 우선순위 키워드로 추가 수집
        priority_keywords = [kw for kw in PRIORITY_KEYWORDS if kw in INTERPRETATION_KEYWORDS]
        
        for keyword in priority_keywords:
            if self.stats.collected_count >= self.stats.target_count:
                logger.info(f"목표 수량 {self.stats.target_count:,}건 달성으로 추가 수집 중단")
                break
            
            if self._check_shutdown_requested():
                logger.warning(f"종료 요청으로 추가 수집 중단: {self.shutdown_reason}")
                break
            
            try:
                # 키워드별 목표 건수 (추가 수집은 적게)
                keyword_target = min(10, remaining_count)
                remaining_count -= keyword_target
                
                logger.info(f"추가 키워드 '{keyword}' 처리 시작 (목표: {keyword_target}건)")
                
                # 해석례 수집
                interpretations = self.collect_interpretations_by_keyword(keyword, keyword_target)
                
                # 배치 저장
                if len(self.pending_interpretations) >= self.batch_size:
                    self._save_batch_interpretations()
                
                # 체크포인트 저장
                self._save_checkpoint(checkpoint_file)
                
                # 진행 상황 로깅
                progress_percent = (self.stats.collected_count / self.stats.target_count) * 100
                logger.info(f"추가 키워드 '{keyword}' 완료: {len(interpretations)}건 수집 (총 {self.stats.collected_count:,}건, {progress_percent:.1f}%)")
                
                # API 요청 제한 확인
                if self._check_api_limits():
                    logger.warning("API 요청 제한에 도달하여 추가 수집 중단")
                    break
                    
            except Exception as e:
                logger.error(f"추가 키워드 '{keyword}' 수집 중 오류: {e}")
                self.stats.failed_count += 1
                continue
        
        logger.info(f"추가 수집 완료 - 총 {self.stats.collected_count:,}건 수집")
    
    def _collect_by_keywords(self, target_count: int, checkpoint_file: Path):
        """우선순위 기반 키워드별 법령해석례 수집"""
        # 우선순위 키워드와 일반 키워드 분리
        priority_keywords = [kw for kw in PRIORITY_KEYWORDS if kw in INTERPRETATION_KEYWORDS]
        remaining_keywords = [kw for kw in INTERPRETATION_KEYWORDS if kw not in PRIORITY_KEYWORDS]
        
        # 전체 키워드 목록 (우선순위 먼저, 나머지 나중에)
        ordered_keywords = priority_keywords + remaining_keywords
        
        # 이미 처리된 키워드 제외
        unprocessed_keywords = [kw for kw in ordered_keywords if kw not in self.processed_keywords]
        
        logger.info(f"우선순위 기반 키워드 수집 시작")
        logger.info(f"1순위 키워드: {len(priority_keywords)}개 (우선 수집)")
        logger.info(f"2순위 키워드: {len(remaining_keywords)}개 (추가 수집)")
        logger.info(f"총 키워드: {len(ordered_keywords)}개")
        logger.info(f"이미 처리된 키워드: {len(self.processed_keywords)}개")
        logger.info(f"처리 대기 키워드: {len(unprocessed_keywords)}개")
        
        if not unprocessed_keywords:
            logger.info("모든 키워드가 이미 처리되었습니다.")
            return
        
        # 진행 상황 추적
        total_keywords = len(unprocessed_keywords)
        logger.info(f"총 {total_keywords}개 미처리 키워드 처리 시작")
        
        for i, keyword in enumerate(unprocessed_keywords):
            # 종료 요청 확인
            if self._check_shutdown_requested():
                logger.warning(f"종료 요청으로 키워드 검색 중단: {self.shutdown_reason}")
                break
            
            if self.stats.collected_count >= target_count:
                logger.info(f"목표 수량 {target_count:,}건 달성으로 키워드 검색 중단")
                break
            
            try:
                # 키워드별 목표 건수 결정
                if keyword in KEYWORD_TARGET_COUNTS:
                    keyword_target = KEYWORD_TARGET_COUNTS[keyword]
                    priority_level = "우선순위"
                else:
                    keyword_target = DEFAULT_KEYWORD_COUNT
                    priority_level = "일반"
                
                # 진행 상황 로깅
                progress_percent = ((i + 1) / total_keywords) * 100
                logger.info(f"[{i+1}/{total_keywords}] ({progress_percent:.1f}%) 키워드 '{keyword}' 처리 시작 ({priority_level}, 목표: {keyword_target}건)")
                
                if i > 0:
                    self._random_delay()
                
                # 키워드별 수집
                interpretations = self.collect_interpretations_by_keyword(keyword, keyword_target)
                
                # 통계 업데이트
                self.stats.keywords_processed = len(self.processed_keywords)
                
                # 체크포인트 저장 (매 키워드마다)
                self._save_checkpoint(checkpoint_file)
                
                # 키워드 완료 로깅
                logger.info(f"키워드 '{keyword}' 완료 ({priority_level}, 목표: {keyword_target}건). 누적: {self.stats.collected_count:,}/{target_count:,}건")
                
                # API 요청 제한 확인
                if self._check_api_limits():
                    break
                    
            except KeyboardInterrupt:
                logger.warning(f"키워드 '{keyword}' 검색이 중단되었습니다.")
                break
            except Exception as e:
                logger.error(f"키워드 '{keyword}' 검색 실패: {e}")
                continue
    
    def collect_interpretations_by_keyword(self, keyword: str, max_count: int = 50) -> List[LegalInterpretationData]:
        """키워드로 법령해석례 검색 및 수집"""
        # 이미 처리된 키워드인지 확인
        if keyword in self.processed_keywords:
            logger.info(f"키워드 '{keyword}'는 이미 처리되었습니다. 건너뜁니다.")
            return []
        
        logger.info(f"키워드 '{keyword}'로 법령해석례 검색 시작 (목표: {max_count}건)")
        
        interpretations = []
        page = 1
        consecutive_empty_pages = 0
        max_empty_pages = 3
        
        while len(interpretations) < max_count and consecutive_empty_pages < max_empty_pages:
            # 종료 요청 확인
            if self._check_shutdown_requested():
                logger.warning(f"종료 요청으로 '{keyword}' 검색 중단: {self.shutdown_reason}")
                break
            
            try:
                # API 요청 간 랜덤 지연
                if page > 1:
                    self._random_delay()
                
                # 진행 상황 로깅
                logger.debug(f"키워드 '{keyword}' 페이지 {page} 검색 중...")
                
                # API 요청 실행
                try:
                    with self._api_request_with_retry(f"키워드 '{keyword}' 검색"):
                        results = self.client.get_interpretation_list(
                            query=keyword,
                            display=100,
                            page=page
                        )
                except Exception as api_error:
                    logger.error(f"API 요청 실패: {api_error}")
                    consecutive_empty_pages += 1
                    page += 1
                    continue
                
                if not results:
                    consecutive_empty_pages += 1
                    logger.debug(f"키워드 '{keyword}' 페이지 {page}에서 결과 없음 (연속 빈 페이지: {consecutive_empty_pages})")
                    page += 1
                    continue
                else:
                    consecutive_empty_pages = 0
                
                # 결과 처리
                new_count, duplicate_count = self._process_search_results(results, interpretations, max_count)
                
                # 페이지별 결과 로깅
                logger.debug(f"페이지 {page}: {new_count}건 신규, {duplicate_count}건 중복 (누적: {len(interpretations)}/{max_count}건)")
                
                page += 1
                
                # API 요청 제한 확인
                if self._check_api_limits():
                    break
                    
            except KeyboardInterrupt:
                logger.warning(f"키워드 '{keyword}' 검색이 중단되었습니다.")
                break
            except Exception as e:
                logger.error(f"키워드 '{keyword}' 검색 중 오류: {e}")
                self.stats.failed_count += 1
                break
        
        # 수집된 해석례를 임시 저장소에 추가
        for interpretation in interpretations:
            self.pending_interpretations.append(interpretation)
            self.stats.collected_count += 1
        
        # 키워드 처리 완료 표시
        self.processed_keywords.add(keyword)
        
        logger.info(f"키워드 '{keyword}' 수집 완료: {len(interpretations)}건")
        return interpretations
    
    def _process_search_results(self, results: List[Dict[str, Any]], interpretations: List[LegalInterpretationData], 
                              max_count: int) -> Tuple[int, int]:
        """검색 결과 처리"""
        new_count = 0
        duplicate_count = 0
        
        for result in results:
            # 종료 요청 확인
            if self._check_shutdown_requested():
                logger.warning(f"종료 요청으로 결과 처리 중단: {self.shutdown_reason}")
                break
            
            # result가 메타데이터인 경우 expc 배열에서 실제 데이터 추출
            if 'expc' in result and isinstance(result['expc'], list):
                expc_items = result['expc']
                logger.debug(f"expc 배열에서 {len(expc_items)}개 항목 처리")
                
                for expc_item in expc_items:
                    if self._check_shutdown_requested():
                        break
                    
                    # 중복 확인
                    if self._is_duplicate_interpretation(expc_item):
                        duplicate_count += 1
                        self.stats.duplicate_count += 1
                        continue
                    
                    # 상세 정보 수집
                    logger.debug(f"해석례 {new_count + 1} 상세 정보 수집 중...")
                    detailed_data = self._collect_interpretation_detail(expc_item)
                    
                    # LegalInterpretationData 객체 생성
                    interpretation_data = self._create_interpretation_data(detailed_data)
                    if not interpretation_data:
                        logger.warning(f"해석례 데이터 생성 실패")
                        self.stats.failed_count += 1
                        continue
                    
                    # 신규 해석례 추가
                    interpretations.append(interpretation_data)
                    self._mark_interpretation_collected(expc_item)
                    new_count += 1
                    
                    # 진행 상황 로깅 (5건마다)
                    if new_count % 5 == 0:
                        logger.info(f"진행 상황: {new_count}건 처리 완료 (누적: {len(interpretations)}/{max_count}건)")
                    
                    if len(interpretations) >= max_count:
                        logger.info(f"목표 수량 {max_count}건 달성으로 처리 중단")
                        break
            else:
                # result가 직접 법령해석례 데이터인 경우 (기존 로직)
                # 중복 확인
                if self._is_duplicate_interpretation(result):
                    duplicate_count += 1
                    self.stats.duplicate_count += 1
                    continue
                
                # LegalInterpretationData 객체 생성
                interpretation_data = self._create_interpretation_data(result)
                if not interpretation_data:
                    self.stats.failed_count += 1
                    continue
                
                # 신규 해석례 추가
                interpretations.append(interpretation_data)
                self._mark_interpretation_collected(result)
                new_count += 1
                
                if len(interpretations) >= max_count:
                    break
        
        return new_count, duplicate_count
    
    def _save_batch_interpretations(self):
        """배치 단위로 법령해석례 저장"""
        if not self.pending_interpretations:
            return
        
        try:
            # 카테고리별로 그룹화
            by_category = {}
            for interpretation in self.pending_interpretations:
                category = interpretation.category.value
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(interpretation)
            
            # 배치 파일 저장 (카테고리별로 분리)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            batch_number = self._get_next_batch_number()
            saved_files = []
            
            for category, interpretations in by_category.items():
                # 안전한 파일명 생성
                safe_category = category.replace('_', '-')
                filename = f"batch_{batch_number:03d}_{safe_category}_{len(interpretations)}건_{timestamp}.json"
                filepath = self.batch_dir / filename
                
                batch_data = {
                    'metadata': {
                        'category': category,
                        'count': len(interpretations),
                        'saved_at': datetime.now().isoformat(),
                        'batch_id': timestamp,
                        'batch_number': batch_number,
                        'total_collected': self.stats.collected_count,
                        'api_requests': self.api_request_count,
                        'collection_progress': {
                            'target_count': self.stats.target_count,
                            'completion_percentage': (self.stats.collected_count / self.stats.target_count * 100) if self.stats.target_count > 0 else 0
                        }
                    },
                    'interpretations': [i.raw_data for i in interpretations]
                }
                
                with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
                    json.dump(batch_data, f, ensure_ascii=False, indent=2)
                
                saved_files.append(filepath)
                logger.info(f"📁 배치 저장 완료: {category} 카테고리 {len(interpretations):,}건 -> {filename}")
            
            # 통계 업데이트
            self.stats.saved_count += len(self.pending_interpretations)
            
            # 임시 저장소 초기화
            self.pending_interpretations = []
            
            # 배치 저장 요약 로깅
            total_saved = sum(len(interpretations) for interpretations in by_category.values())
            logger.info(f"✅ 배치 저장 완료: 총 {len(saved_files):,}개 파일, {total_saved:,}건 저장")
            logger.info(f"📊 누적 수집: {self.stats.collected_count:,}건 / {self.stats.target_count:,}건 ({self.stats.collected_count/self.stats.target_count*100:.1f}%)")
            logger.info(f"📂 저장 위치: {self.batch_dir}")
            
        except Exception as e:
            logger.error(f"배치 저장 실패: {e}")
            logger.error(traceback.format_exc())
    
    def _check_api_limits(self) -> bool:
        """API 요청 제한 확인"""
        try:
            stats = self.client.get_request_stats()
            remaining = stats.get('remaining_requests', 0)
            if remaining < 100:
                logger.warning(f"API 요청 한도가 부족합니다. 남은 요청: {remaining}회")
                return True
        except Exception as e:
            logger.warning(f"API 요청 제한 확인 실패: {e}")
        return False
    
    def _print_final_stats(self):
        """최종 통계 출력"""
        logger.info("=" * 60)
        logger.info("수집 완료 통계")
        logger.info("=" * 60)
        logger.info(f"목표 수집 건수: {self.stats.target_count:,}건")
        logger.info(f"실제 수집 건수: {self.stats.collected_count:,}건")
        logger.info(f"중복 제외 건수: {self.stats.duplicate_count:,}건")
        logger.info(f"실패 건수: {self.stats.failed_count:,}건")
        logger.info(f"처리된 키워드: {len(self.processed_keywords):,}개")
        logger.info(f"저장된 배치 수: {self.stats.saved_count:,}건")
        logger.info(f"API 요청 수: {self.stats.api_requests_made:,}회")
        logger.info(f"API 오류 수: {self.stats.api_errors:,}회")
        logger.info(f"성공률: {self.stats.success_rate:.1f}%")
        if self.stats.duration:
            logger.info(f"소요 시간: {self.stats.duration}")
        logger.info("=" * 60)
    
    def _cleanup_checkpoint_file(self, checkpoint_file: Path):
        """체크포인트 파일 정리"""
        if checkpoint_file and checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                logger.info("체크포인트 파일 정리 완료")
            except Exception as e:
                logger.warning(f"체크포인트 파일 정리 실패: {e}")
    
    def _save_final_checkpoint(self, checkpoint_file: Path):
        """최종 체크포인트 저장"""
        try:
            self._save_checkpoint(checkpoint_file)
            logger.info(f"현재까지 수집된 데이터는 {checkpoint_file}에 저장되었습니다.")
            logger.info("나중에 다시 실행하면 이어서 계속할 수 있습니다.")
        except Exception as e:
            logger.error(f"최종 체크포인트 저장 실패: {e}")
    
    def _save_checkpoint(self, checkpoint_file: Path):
        """진행 상황 체크포인트 저장"""
        try:
            checkpoint_data = {
                'stats': {
                    'start_time': self.stats.start_time.isoformat(),
                    'end_time': self.stats.end_time.isoformat() if self.stats.end_time else None,
                    'target_count': self.stats.target_count,
                    'collected_count': self.stats.collected_count,
                    'saved_count': self.stats.saved_count,
                    'duplicate_count': self.stats.duplicate_count,
                    'failed_count': self.stats.failed_count,
                    'keywords_processed': self.stats.keywords_processed,
                    'total_keywords': self.stats.total_keywords,
                    'api_requests_made': self.stats.api_requests_made,
                    'api_errors': self.stats.api_errors,
                    'status': self.stats.status.value,
                    'processed_keywords': list(self.processed_keywords),
                    'collected_interpretations_count': len(self.collected_interpretations)
                },
                'interpretations': [i.raw_data for i in self.pending_interpretations],
                'saved_at': datetime.now().isoformat(),
                'resume_info': {
                    'can_resume': True,
                    'last_keyword_processed': list(self.processed_keywords)[-1] if self.processed_keywords else None,
                    'progress_percentage': (self.stats.collected_count / self.stats.target_count) * 100 if self.stats.target_count > 0 else 0
                },
                'shutdown_info': {
                    'shutdown_requested': self.shutdown_requested,
                    'shutdown_reason': self.shutdown_reason,
                    'graceful_shutdown_supported': True
                }
            }
            
            with open(checkpoint_file, 'w', encoding='utf-8', newline='\n') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"체크포인트 저장 완료: {checkpoint_file}")
            
        except Exception as e:
            logger.error(f"체크포인트 저장 실패: {e}")
            logger.error(traceback.format_exc())
    
    def _check_and_resume_from_checkpoint(self):
        """체크포인트 파일 확인 및 복구"""
        logger.info("체크포인트 파일 확인 중...")
        
        # 체크포인트 파일 찾기
        checkpoint_files = list(self.output_dir.glob("collection_checkpoint_*.json"))
        
        if not checkpoint_files:
            logger.info("체크포인트 파일이 없습니다. 새로 시작합니다.")
            return
        
        # 가장 최근 체크포인트 파일 선택
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        self.checkpoint_file = latest_checkpoint
        
        logger.info(f"체크포인트 파일 발견: {latest_checkpoint.name}")
        
        try:
            with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # 체크포인트 데이터 복구
            self._restore_from_checkpoint(checkpoint_data)
            
            self.resume_mode = True
            self.stats.status = CollectionStatus.IN_PROGRESS
            
            logger.info("=" * 60)
            logger.info("이전 작업 복구 완료")
            logger.info("=" * 60)
            logger.info(f"복구된 수집 건수: {self.stats.collected_count:,}건")
            logger.info(f"처리된 키워드: {len(self.processed_keywords):,}개")
            logger.info(f"저장된 배치 수: {self.stats.saved_count:,}건")
            logger.info(f"중복 제외 건수: {self.stats.duplicate_count:,}건")
            logger.info(f"API 요청 수: {self.stats.api_requests_made:,}회")
            logger.info("=" * 60)
            
            # 자동으로 계속 진행
            logger.info("이전 작업을 이어서 진행합니다.")
            
        except Exception as e:
            logger.error(f"체크포인트 파일 복구 실패: {e}")
            logger.info("새로 시작합니다.")
            self.resume_mode = False
            self.checkpoint_file = None
    
    def _restore_from_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """체크포인트 데이터에서 상태 복구"""
        stats_data = checkpoint_data.get('stats', {})
        interpretations = checkpoint_data.get('interpretations', [])
        
        # 통계 복구
        self.stats.collected_count = stats_data.get('collected_count', 0)
        self.stats.saved_count = stats_data.get('saved_count', 0)
        self.stats.duplicate_count = stats_data.get('duplicate_count', 0)
        self.stats.failed_count = stats_data.get('failed_count', 0)
        self.stats.keywords_processed = stats_data.get('keywords_processed', 0)
        self.stats.api_requests_made = stats_data.get('api_requests_made', 0)
        self.stats.api_errors = stats_data.get('api_errors', 0)
        
        # 처리된 키워드 복구
        processed_keywords = stats_data.get('processed_keywords', [])
        self.processed_keywords = set(processed_keywords)
        
        # 수집된 해석례 ID 복구
        for interpretation in interpretations:
            if isinstance(interpretation, dict):
                interpretation_id = interpretation.get('판례일련번호') or interpretation.get('interpretation_id')
                if interpretation_id:
                    self.collected_interpretations.add(str(interpretation_id))
    
    def _collect_with_fallback_keywords(self, remaining_count: int, checkpoint_file: Path):
        """백업 키워드로 추가 수집"""
        logger.info(f"백업 키워드 수집 시작 - 목표: {remaining_count}건")
        
        # 백업 키워드 중 아직 처리되지 않은 것들만 선택
        unprocessed_fallback = [kw for kw in FALLBACK_KEYWORDS if kw not in self.processed_keywords]
        
        if not unprocessed_fallback:
            logger.info("모든 백업 키워드가 이미 처리되었습니다.")
            return
        
        logger.info(f"백업 키워드 {len(unprocessed_fallback)}개로 추가 수집 시도")
        
        for i, keyword in enumerate(unprocessed_fallback):
            if self.stats.collected_count >= self.stats.target_count:
                logger.info(f"목표 수량 {self.stats.target_count:,}건 달성으로 백업 키워드 수집 중단")
                break
            
            if self._check_shutdown_requested():
                logger.warning(f"종료 요청으로 백업 키워드 수집 중단: {self.shutdown_reason}")
                break
            
            try:
                # 백업 키워드는 각각 10건씩 수집
                keyword_target = min(10, remaining_count)
                remaining_count -= keyword_target
                
                logger.info(f"백업 키워드 '{keyword}' 처리 시작 (목표: {keyword_target}건)")
                
                # 해석례 수집
                interpretations = self.collect_interpretations_by_keyword(keyword, keyword_target)
                
                # 배치 저장
                if len(self.pending_interpretations) >= self.batch_size:
                    self._save_batch_interpretations()
                
                # 체크포인트 저장
                self._save_checkpoint(checkpoint_file)
                
                # 진행 상황 로깅
                progress_percent = (self.stats.collected_count / self.stats.target_count) * 100
                logger.info(f"백업 키워드 '{keyword}' 완료: {len(interpretations)}건 수집 (총 {self.stats.collected_count:,}건, {progress_percent:.1f}%)")
                
                # API 요청 제한 확인
                if self._check_api_limits():
                    logger.warning("API 요청 제한에 도달하여 백업 키워드 수집 중단")
                    break
                    
            except Exception as e:
                logger.error(f"백업 키워드 '{keyword}' 수집 중 오류: {e}")
                self.stats.failed_count += 1
                continue
        
        logger.info(f"백업 키워드 수집 완료 - 총 {self.stats.collected_count:,}건 수집")
            
        # 배치 저장 (임시 저장소가 가득 찬 경우)
        if len(self.pending_interpretations) >= self.batch_size:
            self._save_batch_interpretations()
