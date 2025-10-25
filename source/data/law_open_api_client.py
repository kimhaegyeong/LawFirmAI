#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
국가법령정보센터 OPEN API 클라이언트

국가법령정보센터의 OPEN API를 통해 법령용어 데이터를 수집하는 클라이언트입니다.
- 법령용어 목록 조회
- 법령용어 상세 조회
- 요청 제한 관리
- 에러 처리 및 재시도 로직
"""

import os
import time
import requests
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import sys

logger = logging.getLogger(__name__)


class LawOpenAPIClient:
    """국가법령정보센터 OPEN API 기본 클라이언트"""
    
    def __init__(self, oc_parameter: str = None):
        """
        API 클라이언트 초기화
        
        Args:
            oc_parameter: OC 파라미터 (이메일 ID). None이면 환경변수에서 로드
        """
        self.oc_parameter = oc_parameter or os.getenv("LAW_OPEN_API_OC")
        if not self.oc_parameter:
            raise ValueError("LAW_OPEN_API_OC 환경변수가 설정되지 않았습니다.")
        
        self.base_url = "http://www.law.go.kr/DRF/lawSearch.do"
        self.detail_url = "http://www.law.go.kr/DRF/lawService.do"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LawFirmAI/1.0'
        })
        
        # 서버 과부하 방지를 위한 요청 간격 관리
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 최소 요청 간격 (초) - 서버 과부하 방지
        
        # 체크포인트 관리자 (지연 로딩)
        self._checkpoint_manager = None
        
        logger.info(f"LawOpenAPIClient 초기화 완료 - OC: {self.oc_parameter}")
    
    @property
    def checkpoint_manager(self):
        """체크포인트 관리자 지연 로딩"""
        if self._checkpoint_manager is None:
            # 프로젝트 루트를 Python 경로에 추가
            project_root = Path(__file__).parent.parent.parent
            sys.path.append(str(project_root))
            
            from scripts.data_collection.law_open_api.utils.checkpoint_manager import CheckpointManager
            self._checkpoint_manager = CheckpointManager()
        
        return self._checkpoint_manager
    
    def _wait_for_request_interval(self):
        """서버 과부하 방지를 위한 요청 간격 대기"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        API 요청 실행
        
        Args:
            params: 요청 파라미터
            
        Returns:
            API 응답 데이터
            
        Raises:
            requests.HTTPError: HTTP 에러 발생 시
            requests.RequestException: 요청 에러 발생 시
        """
        self._wait_for_request_interval()
        
        # OC 파라미터 추가
        params['OC'] = self.oc_parameter
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # JSON 응답 파싱
            data = response.json()
            
            logger.debug(f"API 요청 성공: {params.get('target', 'unknown')}")
            return data
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP 에러 발생: {e}, 응답: {response.text if 'response' in locals() else 'N/A'}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"요청 에러 발생: {e}")
            raise
        except ValueError as e:
            logger.error(f"JSON 파싱 에러: {e}")
            raise
    
    def get_legal_term_list(self, query: str = "", page: int = 1, 
                           per_page: int = 100, sort: str = "rasc") -> Dict[str, Any]:
        """
        법령용어 목록 조회
        
        Args:
            query: 검색 쿼리
            page: 페이지 번호 (1부터 시작)
            per_page: 페이지당 항목 수
            sort: 정렬옵션 (rasc: 등록일자 오름차순, rdes: 등록일자 내림차순, 
                           lasc: 법령용어명 오름차순, ldes: 법령용어명 내림차순)
            
        Returns:
            법령용어 목록 데이터
        """
        params = {
            'OC': self.oc_parameter,
            'target': 'lstrm',
            'type': 'JSON',
            'query': query,
            'page': page,
            'display': per_page,
            'sort': sort
        }
        
        logger.info(f"법령용어 목록 조회 - 쿼리: '{query}', 페이지: {page}, 크기: {per_page}, 정렬: {sort}")
        
        return self._make_request(params)
    
    def get_legal_terms_page(self, page: int, sort: str = "rasc", per_page: int = 100) -> List[Dict[str, Any]]:
        """
        특정 페이지의 법령용어 목록 조회
        
        Args:
            page: 페이지 번호 (1부터 시작)
            sort: 정렬옵션 (기본값: rasc - 등록일자 오름차순)
            per_page: 페이지당 항목 수 (기본값: 100)
            
        Returns:
            해당 페이지의 법령용어 목록
        """
        try:
            result = self.get_legal_term_list(query="", page=page, per_page=per_page, sort=sort)
            
            if result.get('LsTrmSearch') and result['LsTrmSearch'].get('lstrm'):
                lstrm = result['LsTrmSearch']['lstrm']
                # lstrm이 단일 객체인 경우 배열로 변환
                if isinstance(lstrm, dict):
                    terms = [lstrm]
                else:
                    terms = lstrm
                logger.info(f"페이지 {page} 조회 완료: {len(terms)}개 항목")
                return terms
            else:
                logger.warning(f"페이지 {page}에서 데이터가 없습니다")
                return []
                
        except Exception as e:
            logger.error(f"페이지 {page} 조회 실패: {e}")
            raise
    
    def get_legal_terms_count(self) -> int:
        """
        전체 법령용어 수 조회
        
        Returns:
            전체 법령용어 수
        """
        try:
            result = self.get_legal_term_list(query="", page=1, per_page=1, sort="rasc")
            
            if result.get('LsTrmSearch') and 'totalCnt' in result['LsTrmSearch']:
                total_count = int(result['LsTrmSearch']['totalCnt'])
                logger.info(f"전체 법령용어 수: {total_count:,}개")
                return total_count
            else:
                logger.warning("전체 법령용어 수를 조회할 수 없습니다")
                return 0
                
        except Exception as e:
            logger.error(f"전체 법령용어 수 조회 실패: {e}")
            raise
    
    def get_legal_term_detail(self, term_name: str) -> Dict[str, Any]:
        """
        법령용어 상세 조회
        
        Args:
            term_name: 법령용어명
            
        Returns:
            법령용어 상세 데이터
        """
        # 상세 조회는 별도 URL 사용
        detail_url = "http://www.law.go.kr/DRF/lawService.do"
        
        params = {
            'OC': self.oc_parameter,
            'target': 'lstrm',
            'type': 'JSON',
            'query': term_name
        }
        
        logger.debug(f"법령용어 상세 조회 - 용어명: {term_name}")
        
        # 상세 조회는 별도 URL이므로 직접 요청
        self._wait_for_request_interval()
        
        try:
            response = self.session.get(detail_url, params=params, timeout=30)
            response.raise_for_status()
            
            # 응답 내용 확인
            response_text = response.text.strip()
            if not response_text:
                logger.warning(f"법령용어 상세 조회 응답이 비어있음: {term_name}")
                return {"error": "empty_response", "term_name": term_name}
            
            # JSON 파싱 시도
            try:
                data = response.json()
                logger.debug(f"법령용어 상세 조회 성공: {term_name}")
                return data
            except ValueError as e:
                logger.warning(f"JSON 파싱 실패, 응답 내용: {response_text[:200]}...")
                return {
                    "error": "json_parse_error", 
                    "term_name": term_name,
                    "response_text": response_text[:500]  # 처음 500자만 저장
                }
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP 에러 발생: {e}, 응답: {response.text if 'response' in locals() else 'N/A'}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"요청 에러 발생: {e}")
            raise
    
    def get_legal_terms_with_details(self, query: str = "", max_pages: int = None, 
                                   sort: str = "rasc", batch_size: int = 1000, 
                                   save_batches: bool = True, resume_from_checkpoint: bool = False,
                                   resume_from_page: int = 1) -> List[Dict[str, Any]]:
        """
        법령용어 목록과 상세 정보를 함께 조회 (배치 저장 지원)
        
        Args:
            query: 검색 쿼리
            max_pages: 최대 페이지 수 (None이면 모든 페이지)
            sort: 정렬옵션 (기본값: rasc - 등록일자 오름차순)
            batch_size: 배치 크기 (기본값: 1000개)
            save_batches: 배치별 저장 여부 (기본값: True)
            resume_from_checkpoint: 체크포인트에서 재시작 여부
            resume_from_page: 재시작할 페이지 번호
            
        Returns:
            상세 정보가 포함된 법령용어 목록
        """
        # 먼저 목록 조회 (배치 저장)
        terms_list = self.get_all_legal_terms(query, max_pages, sort, True, batch_size, save_batches)
        
        logger.info(f"법령용어 상세 정보 조회 시작 - 총 {len(terms_list)}개 용어")
        
        # 체크포인트에서 재시작하는 경우
        start_index = 0
        if resume_from_checkpoint:
            # 상세 배치 체크포인트 확인
            from scripts.data_collection.law_open_api.utils.checkpoint_manager import CheckpointManager
            checkpoint_manager = CheckpointManager()
            detailed_cp = checkpoint_manager.load_latest_detailed_batch_checkpoint("legal_terms")
            
            if detailed_cp:
                start_index = detailed_cp.get("end_index", 0)
                print(f"🔄 상세 배치 체크포인트에서 재시작: 인덱스 {start_index}부터")
                logger.info(f"상세 배치 체크포인트에서 재시작: 인덱스 {start_index}부터")
        
        detailed_terms = []
        batch_count = 0
        current_batch = []
        
        # 배치 저장 디렉토리 설정
        if save_batches:
            batch_dir = Path("data/raw/law_open_api/legal_terms/detailed_batches")
            batch_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        error_count = 0
        
        for i, term in enumerate(terms_list, 1):
            # 체크포인트에서 재시작하는 경우 건너뛰기
            if i <= start_index:
                continue
                
            try:
                term_name = term.get('법령용어명', '')
                if term_name:
                    # 상세 정보 조회
                    detail = self.get_legal_term_detail(term_name)
                    
                    # 목록 정보와 상세 정보 결합
                    combined_term = {
                        **term,  # 목록 정보
                        'detailed_info': detail  # 상세 정보
                    }
                    detailed_terms.append(combined_term)
                    current_batch.append(combined_term)
                    
                    logger.info(f"상세 정보 조회 완료 ({i}/{len(terms_list)}): {term_name}")
                else:
                    logger.warning(f"용어명이 없어 상세 조회 건너뜀: {term}")
                    detailed_terms.append(term)
                    current_batch.append(term)
                
                # 배치 크기에 도달하면 파일로 저장
                if save_batches and len(current_batch) >= batch_size:
                    batch_count += 1
                    batch_file = batch_dir / f"detailed_batch_{timestamp}_{batch_count:03d}.json"
                    
                    batch_data = {
                        "batch_number": batch_count,
                        "batch_size": len(current_batch),
                        "start_index": i - len(current_batch) + 1,
                        "end_index": i,
                        "timestamp": datetime.now().isoformat(),
                        "terms": current_batch
                    }
                    
                    with open(batch_file, 'w', encoding='utf-8') as f:
                        json.dump(batch_data, f, ensure_ascii=False, indent=2)
                    
                    print(f"  💾 상세 배치 {batch_count} 저장: {len(current_batch):,}개 항목 -> {batch_file.name}")
                    logger.info(f"상세 배치 {batch_count} 저장 완료: {len(current_batch)}개 항목")
                    
                    # 상세 배치 체크포인트 저장
                    if resume_from_checkpoint:
                        from scripts.data_collection.law_open_api.utils.checkpoint_manager import CheckpointManager
                        checkpoint_manager = CheckpointManager()
                        checkpoint_manager.save_detailed_batch_checkpoint(
                            "legal_terms", batch_count, batch_size, 
                            i - len(current_batch) + 1, i, current_batch, error_count
                        )
                    
                    current_batch = []  # 배치 초기화
                
                # 진행률 표시
                if i % 100 == 0 or i == len(terms_list):
                    progress = (i / len(terms_list)) * 100
                    print(f"  상세 정보 수집 진행: {i}/{len(terms_list)} ({progress:.1f}%)")
                    
                # 서버 과부하 방지
                time.sleep(1.0)
                    
            except Exception as e:
                error_count += 1
                logger.error(f"법령용어 상세 조회 실패: {term.get('법령용어명', 'Unknown')} - {e}")
                # 상세 조회 실패해도 목록 정보는 포함
                detailed_terms.append(term)
                current_batch.append(term)
        
        # 마지막 배치 저장 (남은 데이터가 있는 경우)
        if save_batches and current_batch:
            batch_count += 1
            batch_file = batch_dir / f"detailed_batch_{timestamp}_{batch_count:03d}.json"
            
            batch_data = {
                "batch_number": batch_count,
                "batch_size": len(current_batch),
                "start_index": len(terms_list) - len(current_batch) + 1,
                "end_index": len(terms_list),
                "timestamp": datetime.now().isoformat(),
                "terms": current_batch
            }
            
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, ensure_ascii=False, indent=2)
            
            print(f"  💾 마지막 상세 배치 {batch_count} 저장: {len(current_batch):,}개 항목 -> {batch_file.name}")
            logger.info(f"마지막 상세 배치 {batch_count} 저장 완료: {len(current_batch)}개 항목")
            
            # 마지막 상세 배치 체크포인트 저장
            if resume_from_checkpoint:
                from scripts.data_collection.law_open_api.utils.checkpoint_manager import CheckpointManager
                checkpoint_manager = CheckpointManager()
                checkpoint_manager.save_detailed_batch_checkpoint(
                    "legal_terms", batch_count, batch_size, 
                    len(terms_list) - len(current_batch) + 1, len(terms_list), current_batch, error_count
                )
        
        # 상세 배치 요약 정보 저장
        if save_batches and batch_count > 0:
            summary_file = batch_dir / f"detailed_batch_summary_{timestamp}.json"
            summary_data = {
                "total_batches": batch_count,
                "total_terms": len(detailed_terms),
                "batch_size": batch_size,
                "timestamp": timestamp,
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "query": query,
                "sort": sort,
                "max_pages": max_pages
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
            print(f"  📊 상세 배치 요약 저장: {batch_count}개 배치, {len(detailed_terms):,}개 항목 -> {summary_file.name}")
        
        logger.info(f"법령용어 상세 정보 조회 완료 - 총 {len(detailed_terms)}개 용어")
        return detailed_terms
    
    def get_all_legal_terms(self, query: str = "", max_pages: int = None, 
                           sort: str = "rasc", resume_from_checkpoint: bool = True,
                           batch_size: int = 1000, save_batches: bool = True) -> List[Dict[str, Any]]:
        """
        모든 법령용어 조회 (페이지네이션 처리, 체크포인트 지원, 배치 저장)
        
        Args:
            query: 검색 쿼리
            max_pages: 최대 페이지 수 (None이면 모든 페이지)
            sort: 정렬옵션 (기본값: rasc - 등록일자 오름차순)
            resume_from_checkpoint: 체크포인트에서 재시작 여부
            batch_size: 배치 크기 (기본값: 1000개)
            save_batches: 배치별 저장 여부 (기본값: True)
            
        Returns:
            모든 법령용어 목록
        """
        all_terms = []
        page = 1
        total_pages = 0
        batch_count = 0
        current_batch = []
        
        # 배치 저장 디렉토리 설정
        if save_batches:
            batch_dir = Path("data/raw/law_open_api/legal_terms/batches")
            batch_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 체크포인트에서 재시작 여부 확인
        if resume_from_checkpoint:
            checkpoint = self.checkpoint_manager.load_page_checkpoint("legal_terms")
            if checkpoint:
                page = checkpoint.get("current_page", 1)
                total_pages = checkpoint.get("total_pages", 0)
                collected_count = checkpoint.get("collected_count", 0)
                batch_count = checkpoint.get("batch_count", 0)
                print(f"🔄 체크포인트에서 재시작: 페이지 {page}부터 (이미 수집: {collected_count:,}개, 배치: {batch_count}개)")
                logger.info(f"체크포인트에서 재시작: 페이지 {page}, 수집된 항목: {collected_count}, 배치: {batch_count}")
        
        logger.info(f"전체 법령용어 조회 시작 - 쿼리: '{query}', 최대페이지: {max_pages or '무제한'}, 정렬: {sort}, 배치크기: {batch_size}")
        
        # 첫 페이지로 전체 개수 확인 (체크포인트가 없을 때만)
        if page == 1:
            first_response = self.get_legal_term_list(query, 1, 100, sort)
            if first_response and 'LsTrmSearch' in first_response:
                total_count = int(first_response['LsTrmSearch'].get('totalCnt', 0))
                total_pages = (total_count + 99) // 100  # 페이지당 100개
                print(f"  전체 법령용어 수: {total_count:,}개 (총 {total_pages}페이지)")
        
        while True:
            if max_pages and page > max_pages:
                break
            
            try:
                if page == 1 and not resume_from_checkpoint:
                    response = first_response
                else:
                    response = self.get_legal_term_list(query, page, 100, sort)
                
                # 응답 데이터 확인
                if not response or 'LsTrmSearch' not in response:
                    logger.warning(f"페이지 {page}에서 데이터 없음")
                    break
                
                search_result = response['LsTrmSearch']
                if 'lstrm' not in search_result:
                    logger.info(f"페이지 {page}에서 빈 결과 - 수집 완료")
                    break
                
                # lstrm이 단일 객체인 경우 리스트로 변환
                terms = search_result['lstrm']
                if isinstance(terms, dict):
                    terms = [terms]
                
                all_terms.extend(terms)
                current_batch.extend(terms)
                
                # 배치 크기에 도달하면 파일로 저장
                if save_batches and len(current_batch) >= batch_size:
                    batch_count += 1
                    batch_file = batch_dir / f"batch_{timestamp}_{batch_count:03d}.json"
                    
                    batch_data = {
                        "batch_number": batch_count,
                        "batch_size": len(current_batch),
                        "start_page": page - len(current_batch) // 100 + 1,
                        "end_page": page,
                        "timestamp": datetime.now().isoformat(),
                        "terms": current_batch
                    }
                    
                    with open(batch_file, 'w', encoding='utf-8') as f:
                        json.dump(batch_data, f, ensure_ascii=False, indent=2)
                    
                    print(f"  💾 배치 {batch_count} 저장: {len(current_batch):,}개 항목 -> {batch_file.name}")
                    logger.info(f"배치 {batch_count} 저장 완료: {len(current_batch)}개 항목")
                    
                    current_batch = []  # 배치 초기화
                
                # 체크포인트 저장 (매 10페이지마다)
                if page % 10 == 0:
                    last_term_id = terms[-1].get('법령용어ID', '') if terms else ''
                    checkpoint_data = {
                        "data_type": "legal_terms",
                        "current_page": page,
                        "total_pages": total_pages,
                        "collected_count": len(all_terms),
                        "batch_count": batch_count,
                        "last_term_id": last_term_id,
                        "timestamp": datetime.now().isoformat(),
                        "status": "in_progress"
                    }
                    self.checkpoint_manager.save_page_checkpoint(
                        "legal_terms", page, total_pages, len(all_terms), last_term_id
                    )
                    print(f"  💾 체크포인트 저장: 페이지 {page}")
                
                # 진행률 표시
                if page % 10 == 0 or (total_pages > 0 and page >= total_pages):
                    progress = (page / total_pages * 100) if total_pages > 0 else 0
                    print(f"  페이지 {page} 수집 완료 - 누적: {len(all_terms):,}개 ({progress:.1f}%)")
                
                logger.info(f"페이지 {page} 수집 완료 - {len(terms)}개 용어, 누적: {len(all_terms)}개")
                
                page += 1
                
                # 서버 과부하 방지를 위한 대기 (1초)
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"페이지 {page} 수집 실패: {e}")
                break
        
        logger.info(f"전체 법령용어 조회 완료 - 총 {len(all_terms)}개 용어")
        
        # 마지막 배치 저장 (남은 데이터가 있는 경우)
        if save_batches and current_batch:
            batch_count += 1
            batch_file = batch_dir / f"batch_{timestamp}_{batch_count:03d}.json"
            
            batch_data = {
                "batch_number": batch_count,
                "batch_size": len(current_batch),
                "start_page": page - len(current_batch) // 100,
                "end_page": page - 1,
                "timestamp": datetime.now().isoformat(),
                "terms": current_batch
            }
            
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, ensure_ascii=False, indent=2)
            
            print(f"  💾 마지막 배치 {batch_count} 저장: {len(current_batch):,}개 항목 -> {batch_file.name}")
            logger.info(f"마지막 배치 {batch_count} 저장 완료: {len(current_batch)}개 항목")
        
        # 배치 요약 정보 저장
        if save_batches and batch_count > 0:
            summary_file = batch_dir / f"batch_summary_{timestamp}.json"
            summary_data = {
                "total_batches": batch_count,
                "total_terms": len(all_terms),
                "batch_size": batch_size,
                "timestamp": timestamp,
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "query": query,
                "sort": sort,
                "max_pages": max_pages
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
            print(f"  📊 배치 요약 저장: {batch_count}개 배치, {len(all_terms):,}개 항목 -> {summary_file.name}")
        
        # 수집 완료 시 체크포인트 삭제
        if len(all_terms) > 0:
            self.checkpoint_manager.clear_page_checkpoint("legal_terms")
            print(f"✅ 수집 완료 - 체크포인트 삭제됨")
        
        return all_terms
    
    def search_constitutional_decisions(self, 
                                      query: str = "",
                                      search: int = 1,
                                      display: int = 20,
                                      page: int = 1,
                                      sort: str = "dasc",
                                      date: Optional[str] = None,
                                      edYd: Optional[str] = None,
                                      nb: Optional[int] = None) -> Dict[str, Any]:
        """
        헌재결정례 목록 조회
        
        Args:
            query: 검색 쿼리
            search: 검색범위 (1: 헌재결정례명, 2: 본문검색)
            display: 검색된 결과 개수 (기본: 20, 최대: 100)
            page: 검색 결과 페이지 (기본: 1)
            sort: 정렬옵션 (dasc: 선고일자 오름차순, ddes: 선고일자 내림차순, 
                           lasc: 사건명 오름차순, ldes: 사건명 내림차순,
                           nasc: 사건번호 오름차순, ndes: 사건번호 내림차순,
                           efasc: 종국일자 오름차순, efdes: 종국일자 내림차순)
            date: 종국일자 (YYYYMMDD 형식)
            edYd: 종국일자 기간 검색
            nb: 사건번호
            
        Returns:
            헌재결정례 목록 데이터
        """
        params = {
            'OC': self.oc_parameter,
            'target': 'detc',
            'type': 'JSON',
            'query': query,
            'search': search,
            'display': display,
            'page': page,
            'sort': sort
        }
        
        if date:
            params['date'] = date
        if edYd:
            params['edYd'] = edYd
        if nb:
            params['nb'] = nb
            
        logger.info(f"헌재결정례 목록 조회 - 쿼리: '{query}', 페이지: {page}, 크기: {display}, 정렬: {sort}")
        
        return self._make_request(params)
    
    def get_constitutional_decision_detail(self, 
                                        decision_id: str,
                                        decision_name: Optional[str] = None) -> Dict[str, Any]:
        """
        헌재결정례 상세 조회
        
        Args:
            decision_id: 헌재결정례 일련번호
            decision_name: 헌재결정례명 (선택사항)
            
        Returns:
            헌재결정례 상세 데이터
        """
        # 상세 조회는 별도 URL 사용
        detail_url = "http://www.law.go.kr/DRF/lawService.do"
        
        params = {
            'OC': self.oc_parameter,
            'target': 'detc',
            'type': 'JSON',
            'ID': decision_id
        }
        
        if decision_name:
            params['LM'] = decision_name
            
        logger.debug(f"헌재결정례 상세 조회 - ID: {decision_id}, 이름: {decision_name}")
        
        # 상세 조회는 별도 URL이므로 직접 요청
        self._wait_for_request_interval()
        
        try:
            response = self.session.get(detail_url, params=params, timeout=30)
            response.raise_for_status()
            
            # 응답 내용 확인
            response_text = response.text.strip()
            if not response_text:
                logger.warning(f"헌재결정례 상세 조회 응답이 비어있음: {decision_id}")
                return {"error": "empty_response", "decision_id": decision_id}
            
            # JSON 파싱 시도
            try:
                data = response.json()
                logger.debug(f"헌재결정례 상세 조회 성공: {decision_id}")
                return data
            except ValueError as e:
                logger.warning(f"JSON 파싱 실패, 응답 내용: {response_text[:200]}...")
                return {
                    "error": "json_parse_error", 
                    "decision_id": decision_id,
                    "response_text": response_text[:500]  # 처음 500자만 저장
                }
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP 에러 발생: {e}, 응답: {response.text if 'response' in locals() else 'N/A'}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"요청 에러 발생: {e}")
            raise
    
    def get_all_constitutional_decisions(self, 
                                       query: str = "", 
                                       max_pages: int = None,
                                       sort: str = "dasc",
                                       include_details: bool = True,
                                       batch_size: int = 100,
                                       save_batches: bool = True) -> List[Dict[str, Any]]:
        """
        모든 헌재결정례 조회 (선고일자 오름차순, 배치 저장 지원)
        
        Args:
            query: 검색 쿼리
            max_pages: 최대 페이지 수 (None이면 모든 페이지)
            sort: 정렬옵션 (기본값: dasc - 선고일자 오름차순)
            include_details: 상세 정보 포함 여부
            batch_size: 배치 크기 (기본값: 100개)
            save_batches: 배치별 저장 여부 (기본값: True)
            
        Returns:
            헌재결정례 목록 (상세 정보 포함)
        """
        all_decisions = []
        page = 1
        total_pages = 0
        batch_count = 0
        current_batch = []
        
        # 배치 저장 디렉토리 설정
        if save_batches:
            batch_dir = Path("data/raw/constitutional_decisions/batches")
            batch_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"전체 헌재결정례 조회 시작 - 쿼리: '{query}', 최대페이지: {max_pages or '무제한'}, 정렬: {sort}, 배치크기: {batch_size}")
        
        # 첫 페이지로 전체 개수 확인
        if page == 1:
            first_response = self.search_constitutional_decisions(query, 1, 100, 1, sort)
            if first_response and 'DetcSearch' in first_response:
                total_count = int(first_response['DetcSearch'].get('totalCnt', 0))
                total_pages = (total_count + 99) // 100  # 페이지당 100개
                print(f"  전체 헌재결정례 수: {total_count:,}개 (총 {total_pages}페이지)")
        
        while True:
            if max_pages and page > max_pages:
                break
            
            try:
                if page == 1:
                    response = first_response
                else:
                    response = self.search_constitutional_decisions(query, 1, 100, page, sort)
                
                # 응답 데이터 확인
                if not response or 'DetcSearch' not in response:
                    logger.warning(f"페이지 {page}에서 데이터 없음")
                    break
                
                search_result = response['DetcSearch']
                if 'detc' not in search_result:
                    logger.info(f"페이지 {page}에서 빈 결과 - 수집 완료")
                    break
                
                # detc가 단일 객체인 경우 리스트로 변환
                decisions = search_result['detc']
                if isinstance(decisions, dict):
                    decisions = [decisions]
                
                # 상세 정보 포함 여부에 따라 처리
                if include_details:
                    detailed_decisions = []
                    for decision in decisions:
                        decision_id = decision.get('헌재결정례일련번호')
                        if decision_id:
                            try:
                                # 상세 정보 조회
                                detail = self.get_constitutional_decision_detail(decision_id)
                                
                                # 목록 정보와 상세 정보 결합
                                combined_decision = {
                                    **decision,  # 목록 정보
                                    'detailed_info': detail  # 상세 정보
                                }
                                detailed_decisions.append(combined_decision)
                                
                                # 서버 과부하 방지
                                time.sleep(1.0)
                                
                            except Exception as e:
                                logger.error(f"헌재결정례 상세 조회 실패: {decision_id} - {e}")
                                detailed_decisions.append(decision)
                        else:
                            detailed_decisions.append(decision)
                    
                    all_decisions.extend(detailed_decisions)
                    current_batch.extend(detailed_decisions)
                else:
                    all_decisions.extend(decisions)
                    current_batch.extend(decisions)
                
                # 배치 크기에 도달하면 파일로 저장
                if save_batches and len(current_batch) >= batch_size:
                    batch_count += 1
                    batch_file = batch_dir / f"constitutional_batch_{timestamp}_{batch_count:03d}.json"
                    
                    batch_data = {
                        "batch_number": batch_count,
                        "batch_size": len(current_batch),
                        "start_page": page - len(current_batch) // 100 + 1,
                        "end_page": page,
                        "timestamp": datetime.now().isoformat(),
                        "decisions": current_batch
                    }
                    
                    with open(batch_file, 'w', encoding='utf-8') as f:
                        json.dump(batch_data, f, ensure_ascii=False, indent=2)
                    
                    print(f"  💾 헌재결정례 배치 {batch_count} 저장: {len(current_batch):,}개 항목 -> {batch_file.name}")
                    logger.info(f"헌재결정례 배치 {batch_count} 저장 완료: {len(current_batch)}개 항목")
                    
                    current_batch = []  # 배치 초기화
                
                # 진행률 표시
                if page % 10 == 0 or (total_pages > 0 and page >= total_pages):
                    progress = (page / total_pages * 100) if total_pages > 0 else 0
                    print(f"  페이지 {page} 수집 완료 - 누적: {len(all_decisions):,}개 ({progress:.1f}%)")
                
                logger.info(f"페이지 {page} 수집 완료 - {len(decisions)}개 결정례, 누적: {len(all_decisions)}개")
                
                page += 1
                
                # 서버 과부하 방지를 위한 대기 (1초)
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"페이지 {page} 수집 실패: {e}")
                break
        
        logger.info(f"전체 헌재결정례 조회 완료 - 총 {len(all_decisions)}개 결정례")
        
        # 마지막 배치 저장 (남은 데이터가 있는 경우)
        if save_batches and current_batch:
            batch_count += 1
            batch_file = batch_dir / f"constitutional_batch_{timestamp}_{batch_count:03d}.json"
            
            batch_data = {
                "batch_number": batch_count,
                "batch_size": len(current_batch),
                "start_page": page - len(current_batch) // 100,
                "end_page": page - 1,
                "timestamp": datetime.now().isoformat(),
                "decisions": current_batch
            }
            
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, ensure_ascii=False, indent=2)
            
            print(f"  💾 마지막 헌재결정례 배치 {batch_count} 저장: {len(current_batch):,}개 항목 -> {batch_file.name}")
            logger.info(f"마지막 헌재결정례 배치 {batch_count} 저장 완료: {len(current_batch)}개 항목")
        
        # 배치 요약 정보 저장
        if save_batches and batch_count > 0:
            summary_file = batch_dir / f"constitutional_batch_summary_{timestamp}.json"
            summary_data = {
                "total_batches": batch_count,
                "total_decisions": len(all_decisions),
                "batch_size": batch_size,
                "timestamp": timestamp,
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "query": query,
                "sort": sort,
                "max_pages": max_pages,
                "include_details": include_details
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
            print(f"  📊 헌재결정례 배치 요약 저장: {batch_count}개 배치, {len(all_decisions):,}개 항목 -> {summary_file.name}")
        
        return all_decisions
    
    def search_current_laws(self, 
                           query: str = "",
                           search: int = 1,
                           display: int = 20,
                           page: int = 1,
                           sort: str = "ldes",
                           nw: int = 3,
                           knd: str = "A0002",
                           efYd: str = None,
                           date: str = None,
                           ancYd: str = None,
                           ancNo: str = None,
                           rrClsCd: str = None,
                           nb: int = None,
                           org: str = None,
                           gana: str = None) -> Dict[str, Any]:
        """
        현행법령 목록 조회
        
        Args:
            query: 검색 질의
            search: 검색 범위 (1: 법령명, 2: 본문검색)
            display: 검색 결과 개수 (기본 20, 최대 100)
            page: 검색 결과 페이지 (기본 1)
            sort: 정렬 옵션 (기본 ldes: 법령내림차순)
            nw: 검색 범위 (3: 현행)
            knd: 법령종류 (A0002: 법률)
            efYd: 시행일자 범위 검색
            date: 공포일자 검색
            ancYd: 공포일자 범위 검색
            ancNo: 공포번호 범위 검색
            rrClsCd: 법령 제개정 종류
            nb: 법령의 공포번호 검색
            org: 소관부처별 검색
            gana: 사전식 검색
            
        Returns:
            현행법령 목록 데이터
        """
        params = {
            'target': 'eflaw',
            'type': 'JSON',
            'query': query,
            'search': search,
            'display': min(display, 100),  # 최대 100개로 제한
            'page': page,
            'sort': sort,
            'nw': nw,
            'knd': knd
        }
        
        # 선택적 파라미터 추가
        if efYd:
            params['efYd'] = efYd
        if date:
            params['date'] = date
        if ancYd:
            params['ancYd'] = ancYd
        if ancNo:
            params['ancNo'] = ancNo
        if rrClsCd:
            params['rrClsCd'] = rrClsCd
        if nb:
            params['nb'] = nb
        if org:
            params['org'] = org
        if gana:
            params['gana'] = gana
        
        logger.info(f"현행법령 목록 조회 요청 - 페이지: {page}, 검색어: '{query}', 정렬: {sort}")
        
        try:
            response = self._make_request(params)
            logger.info(f"현행법령 목록 조회 성공 - 페이지: {page}")
            return response
        except Exception as e:
            logger.error(f"현행법령 목록 조회 실패 - 페이지: {page}, 에러: {e}")
            raise
    
    def get_current_law_detail(self, 
                              law_id: str = None,
                              mst: str = None,
                              efYd: int = None,
                              jo: str = None,
                              chrClsCd: str = None) -> Dict[str, Any]:
        """
        현행법령 본문 조회
        
        Args:
            law_id: 법령 ID (ID 또는 MST 중 하나는 반드시 입력)
            mst: 법령 마스터 번호 - 법령테이블의 lsi_seq 값을 의미함
            efYd: 법령의 시행일자 (ID 입력시에는 무시하는 값으로 입력하지 않음)
            jo: 조번호 (생략시 모든 조 표시, 6자리숫자: 조번호(4자리)+조가지번호(2자리))
            chrClsCd: 원문/한글 여부 (생략시 기본값: 한글, 010202: 한글, 010201: 원문)
            
        Returns:
            현행법령 본문 데이터
        """
        if not law_id and not mst:
            raise ValueError("law_id 또는 mst 중 하나는 반드시 입력해야 합니다.")
        
        params = {
            'target': 'eflaw',
            'type': 'JSON'
        }
        
        if law_id:
            params['ID'] = law_id
        else:
            params['MST'] = mst
            if efYd:
                params['efYd'] = efYd
        
        if jo:
            params['JO'] = jo
        if chrClsCd:
            params['chrClsCd'] = chrClsCd
        
        logger.info(f"현행법령 본문 조회 요청 - ID: {law_id}, MST: {mst}")
        
        try:
            # 본문 조회는 별도 엔드포인트 사용
            self._wait_for_request_interval()
            params['OC'] = self.oc_parameter
            
            response = self.session.get(self.detail_url, params=params, timeout=30)
            response.raise_for_status()
            
            # JSON 응답 파싱
            data = response.json()
            logger.info(f"현행법령 본문 조회 성공 - ID: {law_id}, MST: {mst}")
            return data
        except Exception as e:
            logger.error(f"현행법령 본문 조회 실패 - ID: {law_id}, MST: {mst}, 에러: {e}")
            raise
    
    def get_all_current_laws(self, 
                           query: str = "",
                           max_pages: int = None,
                           start_page: int = 1,
                           sort: str = "ldes",
                           batch_size: int = 10,
                           save_batches: bool = True,
                           include_details: bool = True,
                           resume_from_checkpoint: bool = False) -> List[Dict[str, Any]]:
        """
        모든 현행법령 조회 (배치 처리)
        
        Args:
            query: 검색 질의
            max_pages: 최대 페이지 수 (None이면 전체)
            sort: 정렬 옵션
            batch_size: 배치 크기
            save_batches: 배치 저장 여부
            include_details: 상세 정보 포함 여부
            resume_from_checkpoint: 체크포인트부터 재시작 여부
            
        Returns:
            모든 현행법령 목록
        """
        logger.info(f"전체 현행법령 조회 시작 - 검색어: '{query}', 배치크기: {batch_size}, 상세정보: {include_details}")
        
        all_laws = []
        page = start_page
        batch_count = 0
        current_batch = []
        
        # 배치 저장 디렉토리 설정
        if save_batches:
            batch_dir = Path("data/raw/law_open_api/current_laws/batches")
            batch_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 기존 배치 파일들에서 마지막 배치 번호 찾기
            existing_batches = list(batch_dir.glob("current_law_batch_*_*.json"))
            if existing_batches:
                # 파일명에서 배치 번호 추출 (예: current_law_batch_20251023_220223_001.json)
                batch_numbers = []
                for batch_file in existing_batches:
                    try:
                        # 파일명에서 마지막 숫자 부분 추출
                        parts = batch_file.stem.split('_')
                        if len(parts) >= 4 and parts[-1].isdigit():
                            batch_num = int(parts[-1])
                            # 비정상적으로 큰 배치 번호는 무시 (220811 같은)
                            if batch_num < 10000:  # 합리적인 범위 내에서만
                                batch_numbers.append(batch_num)
                    except:
                        continue
                
                if batch_numbers:
                    batch_count = max(batch_numbers)
                    logger.info(f"기존 배치 파일 발견 - 마지막 배치 번호: {batch_count}")
        
        # 체크포인트부터 재시작
        if resume_from_checkpoint:
            checkpoint_info = self.checkpoint_manager.get_resume_info("current_laws")
            if checkpoint_info["has_page_checkpoint"]:
                page = checkpoint_info["resume_from_page"]
                logger.info(f"체크포인트부터 재시작 - 페이지: {page}")
        
        while True:
            if max_pages and page > start_page + max_pages - 1:
                logger.info(f"최대 페이지 수({max_pages}) 도달 - 수집 중단 (시작: {start_page}, 현재: {page})")
                break
            
            try:
                # API 요청
                response = self.search_current_laws(
                    query=query,
                    display=100,  # 한 번에 최대 100개씩 조회
                    page=page,
                    sort=sort,
                    nw=3,  # 현행법령만
                    knd="A0002"  # 법률만
                )
                
                if not response or 'LawSearch' not in response:
                    logger.warning(f"페이지 {page}에서 응답 데이터 없음")
                    break
                
                search_result = response['LawSearch']
                if 'law' not in search_result:
                    logger.info(f"페이지 {page}에서 빈 결과 - 수집 완료")
                    break
                
                # law가 단일 객체인 경우 리스트로 변환
                page_laws = search_result['law']
                if isinstance(page_laws, dict):
                    page_laws = [page_laws]
                
                for law in page_laws:
                    if include_details:
                        try:
                            # 상세 정보 조회
                            law_id = law.get('법령ID')
                            if law_id:
                                detail = self.get_current_law_detail(law_id=law_id)
                                
                                # 목록 정보와 상세 정보 결합
                                combined_law = {
                                    **law,  # 목록 정보
                                    'detailed_info': detail,  # 상세 정보 (API 문서의 모든 필드 포함)
                                    'document_type': 'current_law',
                                    'collected_at': datetime.now().isoformat()
                                }
                                all_laws.append(combined_law)
                                current_batch.append(combined_law)
                                
                                # 서버 과부하 방지
                                time.sleep(1.0)
                                
                        except Exception as e:
                            logger.error(f"현행법령 상세 조회 실패: {law_id} - {e}")
                            law['document_type'] = 'current_law'
                            law['collected_at'] = datetime.now().isoformat()
                            all_laws.append(law)
                            current_batch.append(law)
                    else:
                        law['document_type'] = 'current_law'
                        law['collected_at'] = datetime.now().isoformat()
                        all_laws.append(law)
                        current_batch.append(law)
                    
                    # 배치 크기에 도달하면 파일로 저장
                    if save_batches and len(current_batch) >= batch_size:
                        batch_count += 1
                        batch_file = batch_dir / f"current_law_batch_{timestamp}_{batch_count:03d}.json"
                        
                        # 실제 페이지 범위 계산
                        laws_per_page = 100  # display=100으로 설정했으므로
                        start_page_for_batch = page - (len(current_batch) - 1) // laws_per_page
                        
                        batch_data = {
                            "batch_number": batch_count,
                            "batch_size": len(current_batch),
                            "start_page": start_page_for_batch,
                            "end_page": page,
                            "timestamp": datetime.now().isoformat(),
                            "laws": current_batch
                        }
                        
                        with open(batch_file, 'w', encoding='utf-8') as f:
                            json.dump(batch_data, f, ensure_ascii=False, indent=2)
                        
                        print(f"  💾 현행법령 배치 {batch_count} 저장: {len(current_batch):,}개 항목 -> {batch_file.name}")
                        logger.info(f"현행법령 배치 {batch_count} 저장 완료: {len(current_batch)}개 항목")
                        
                        current_batch = []  # 배치 초기화
                        
                        # 체크포인트 저장
                        if resume_from_checkpoint:
                            self.checkpoint_manager.save_checkpoint("current_laws", page + 1, batch_count)
                
                logger.info(f"페이지 {page} 완료: {len(page_laws)}개 법령 수집")
                logger.info(f"누적 수집: {len(all_laws)}개 법령")
                
                page += 1
                
                # 서버 과부하 방지를 위한 대기
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"페이지 {page} 수집 실패: {e}")
                break
        
        logger.info(f"전체 현행법령 조회 완료 - 총 {len(all_laws)}개 법령")
        
        # 마지막 배치 저장 (남은 데이터가 있는 경우)
        if save_batches and current_batch:
            batch_count += 1
            batch_file = batch_dir / f"current_law_batch_{timestamp}_{batch_count:03d}.json"
            
            batch_data = {
                "batch_number": batch_count,
                "batch_size": len(current_batch),
                "start_page": page - len(current_batch) // 100,
                "end_page": page - 1,
                "timestamp": datetime.now().isoformat(),
                "laws": current_batch
            }
            
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, ensure_ascii=False, indent=2)
            
            print(f"  💾 마지막 현행법령 배치 {batch_count} 저장: {len(current_batch):,}개 항목 -> {batch_file.name}")
            logger.info(f"마지막 현행법령 배치 {batch_count} 저장 완료: {len(current_batch)}개 항목")
        
        # 배치 요약 정보 저장
        if save_batches and batch_count > 0:
            summary_file = batch_dir / f"current_law_batch_summary_{timestamp}.json"
            summary_data = {
                "total_batches": batch_count,
                "total_laws": len(all_laws),
                "batch_size": batch_size,
                "timestamp": timestamp,
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "query": query,
                "sort": sort,
                "max_pages": max_pages,
                "include_details": include_details
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
            print(f"  📊 현행법령 배치 요약 저장: {batch_count}개 배치, {len(all_laws):,}개 항목 -> {summary_file.name}")
        
        return all_laws

    def test_connection(self) -> bool:
        """
        API 연결 테스트
        
        Returns:
            연결 성공 여부
        """
        try:
            logger.info("API 연결 테스트 시작")
            response = self.get_legal_term_list("", 1, 1)
            
            if response and 'LsTrmSearch' in response:
                logger.info("API 연결 테스트 성공")
                return True
            else:
                logger.error("API 연결 테스트 실패 - 응답 데이터 없음")
                return False
                
        except Exception as e:
            logger.error(f"API 연결 테스트 실패: {e}")
            return False


class LawOpenAPIConfig:
    """Law Open API 설정 클래스"""
    
    def __init__(self):
        self.base_url = "http://www.law.go.kr/DRF/lawSearch.do"
        self.timeout = 30
        self.max_retries = 3
        self.retry_delay = 5
        self.min_request_interval = 0.1
        self.page_size = 100
        self.max_pages = None  # None이면 모든 페이지 수집


# 편의 함수들
def create_client(oc_parameter: str = None) -> LawOpenAPIClient:
    """
    Law Open API 클라이언트 생성
    
    Args:
        oc_parameter: OC 파라미터
        
    Returns:
        LawOpenAPIClient 인스턴스
    """
    return LawOpenAPIClient(oc_parameter)


def test_api_connection(oc_parameter: str = None) -> bool:
    """
    API 연결 테스트
    
    Args:
        oc_parameter: OC 파라미터
        
    Returns:
        연결 성공 여부
    """
    try:
        client = create_client(oc_parameter)
        return client.test_connection()
    except Exception as e:
        logger.error(f"API 연결 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    # 테스트 실행
    logging.basicConfig(level=logging.INFO)
    
    print("Law Open API 클라이언트 테스트")
    print("=" * 40)
    
    # 환경변수 확인
    oc_param = os.getenv("LAW_OPEN_API_OC")
    if not oc_param:
        print("❌ LAW_OPEN_API_OC 환경변수가 설정되지 않았습니다.")
        print("다음과 같이 설정해주세요:")
        print("export LAW_OPEN_API_OC='your_email@example.com'")
        exit(1)
    
    print(f"✅ OC 파라미터: {oc_param}")
    
    # 클라이언트 생성 및 테스트
    try:
        client = create_client()
        
        # 연결 테스트
        if client.test_connection():
            print("✅ API 연결 테스트 성공")
            
            # 샘플 데이터 조회
            print("\n샘플 법령용어 조회:")
            terms = client.get_legal_term_list("", 1, 5)
            
            if terms and 'data' in terms:
                for i, term in enumerate(terms['data'][:3], 1):
                    print(f"  {i}. {term.get('termName', 'N/A')}")
                print("✅ 샘플 데이터 조회 성공")
            else:
                print("❌ 샘플 데이터 조회 실패")
        else:
            print("❌ API 연결 테스트 실패")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")




