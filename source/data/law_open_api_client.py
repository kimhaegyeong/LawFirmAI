#!/usr/bin/env python3
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
                                   save_batches: bool = True) -> List[Dict[str, Any]]:
        """
        법령용어 목록과 상세 정보를 함께 조회 (배치 저장 지원)
        
        Args:
            query: 검색 쿼리
            max_pages: 최대 페이지 수 (None이면 모든 페이지)
            sort: 정렬옵션 (기본값: rasc - 등록일자 오름차순)
            batch_size: 배치 크기 (기본값: 1000개)
            save_batches: 배치별 저장 여부 (기본값: True)
            
        Returns:
            상세 정보가 포함된 법령용어 목록
        """
        # 먼저 목록 조회 (배치 저장)
        terms_list = self.get_all_legal_terms(query, max_pages, sort, True, batch_size, save_batches)
        
        logger.info(f"법령용어 상세 정보 조회 시작 - 총 {len(terms_list)}개 용어")
        
        detailed_terms = []
        batch_count = 0
        current_batch = []
        
        # 배치 저장 디렉토리 설정
        if save_batches:
            batch_dir = Path("data/raw/law_open_api/legal_terms/detailed_batches")
            batch_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, term in enumerate(terms_list, 1):
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
                    
                    current_batch = []  # 배치 초기화
                
                # 진행률 표시
                if i % 100 == 0 or i == len(terms_list):
                    progress = (i / len(terms_list)) * 100
                    print(f"  상세 정보 수집 진행: {i}/{len(terms_list)} ({progress:.1f}%)")
                    
                # 서버 과부하 방지
                time.sleep(1.0)
                    
            except Exception as e:
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




