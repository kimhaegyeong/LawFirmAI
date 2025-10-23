#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
헌재결정례 수집기

국가법령정보센터 OPEN API를 통해 헌재결정례 데이터를 수집하는 클래스입니다.
- 선고일자 오름차순으로 정렬된 수집
- 100개 단위 배치 처리
- 상세 정보 포함 수집
- 체크포인트 지원
- 배치 저장 시스템
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIClient

logger = logging.getLogger(__name__)


@dataclass
class CollectionConfig:
    """수집 설정"""
    base_output_dir: Path = field(default_factory=lambda: Path("data/raw/constitutional_decisions"))
    batch_size: int = 100
    max_pages: Optional[int] = None
    include_details: bool = True
    sort_order: str = "dasc"  # 선고일자 오름차순
    save_batches: bool = True
    resume_from_checkpoint: bool = True


@dataclass
class CollectionStats:
    """수집 통계"""
    total_collected: int = 0
    total_pages: int = 0
    batch_count: int = 0
    api_requests_made: int = 0
    errors: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    collected_decisions: Set[str] = field(default_factory=set)


class ConstitutionalDecisionCollector:
    """헌재결정례 수집 클래스"""
    
    def __init__(self, config: CollectionConfig = None):
        """
        헌재결정례 수집기 초기화
        
        Args:
            config: 수집 설정
        """
        self.config = config or CollectionConfig()
        self.client = LawOpenAPIClient()
        self.stats = CollectionStats()
        self.collected_decisions: Set[str] = set()
        
        # 출력 디렉토리 생성
        self.config.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ConstitutionalDecisionCollector 초기화 완료 - 출력 디렉토리: {self.config.base_output_dir}")
    
    def collect_decisions_by_keyword(self, 
                                   keyword: str = "", 
                                   max_count: int = 1000,
                                   include_details: bool = True) -> List[Dict[str, Any]]:
        """
        키워드 기반 헌재결정례 수집
        
        Args:
            keyword: 검색 키워드
            max_count: 최대 수집 개수
            include_details: 상세 정보 포함 여부
            
        Returns:
            수집된 헌재결정례 목록
        """
        logger.info(f"키워드 기반 헌재결정례 수집 시작 - 키워드: '{keyword}', 최대개수: {max_count}")
        
        decisions = []
        page = 1
        batch_count = 0
        current_batch = []
        
        # 배치 저장 디렉토리 설정
        if self.config.save_batches:
            batch_dir = self.config.base_output_dir / "batches"
            batch_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        while len(decisions) < max_count:
            try:
                # API 요청
                response = self.client.search_constitutional_decisions(
                    query=keyword,
                    search=1,  # 헌재결정례명 검색
                    display=min(100, max_count - len(decisions)),
                    page=page,
                    sort=self.config.sort_order
                )
                
                self.stats.api_requests_made += 1
                
                if not response or 'DetcSearch' not in response:
                    logger.warning(f"페이지 {page}에서 응답 데이터 없음")
                    break
                
                search_result = response['DetcSearch']
                if 'detc' not in search_result:
                    logger.info(f"페이지 {page}에서 빈 결과 - 수집 완료")
                    break
                
                # detc가 단일 객체인 경우 리스트로 변환
                page_decisions = search_result['detc']
                if isinstance(page_decisions, dict):
                    page_decisions = [page_decisions]
                
                new_decisions = 0
                for decision in page_decisions:
                    if len(decisions) >= max_count:
                        break
                    
                    decision_id = decision.get('헌재결정례일련번호')
                    if decision_id and decision_id not in self.collected_decisions:
                        if include_details:
                            try:
                                # 상세 정보 조회
                                detail = self.client.get_constitutional_decision_detail(decision_id)
                                
                                # 목록 정보와 상세 정보 결합
                                combined_decision = {
                                    **decision,  # 목록 정보
                                    'detailed_info': detail,  # 상세 정보
                                    'document_type': 'constitutional_decision',
                                    'collected_at': datetime.now().isoformat()
                                }
                                decisions.append(combined_decision)
                                current_batch.append(combined_decision)
                                
                                # 서버 과부하 방지
                                time.sleep(1.0)
                                
                            except Exception as e:
                                logger.error(f"헌재결정례 상세 조회 실패: {decision_id} - {e}")
                                decision['document_type'] = 'constitutional_decision'
                                decision['collected_at'] = datetime.now().isoformat()
                                decisions.append(decision)
                                current_batch.append(decision)
                        else:
                            decision['document_type'] = 'constitutional_decision'
                            decision['collected_at'] = datetime.now().isoformat()
                            decisions.append(decision)
                            current_batch.append(decision)
                        
                        self.collected_decisions.add(decision_id)
                        self.stats.collected_decisions.add(decision_id)
                        new_decisions += 1
                        
                        logger.info(f"새로운 헌재결정례 수집: {decision.get('사건명', 'Unknown')} (ID: {decision_id})")
                        
                        # 배치 크기에 도달하면 파일로 저장
                        if self.config.save_batches and len(current_batch) >= self.config.batch_size:
                            batch_count += 1
                            batch_file = batch_dir / f"constitutional_batch_{timestamp}_{batch_count:03d}.json"
                            
                            batch_data = {
                                "batch_number": batch_count,
                                "batch_size": len(current_batch),
                                "keyword": keyword,
                                "timestamp": datetime.now().isoformat(),
                                "decisions": current_batch
                            }
                            
                            with open(batch_file, 'w', encoding='utf-8') as f:
                                json.dump(batch_data, f, ensure_ascii=False, indent=2)
                            
                            print(f"  💾 헌재결정례 배치 {batch_count} 저장: {len(current_batch):,}개 항목 -> {batch_file.name}")
                            logger.info(f"헌재결정례 배치 {batch_count} 저장 완료: {len(current_batch)}개 항목")
                            
                            current_batch = []  # 배치 초기화
                
                logger.info(f"페이지 {page} 완료: {new_decisions}건의 새로운 결정례 수집")
                logger.info(f"누적 수집: {len(decisions)}/{max_count}건 ({len(decisions)/max_count*100:.1f}%)")
                
                page += 1
                
                # 서버 과부하 방지를 위한 대기
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"페이지 {page} 수집 실패: {e}")
                self.stats.errors.append(f"Page {page} collection error: {e}")
                break
        
        # 마지막 배치 저장 (남은 데이터가 있는 경우)
        if self.config.save_batches and current_batch:
            batch_count += 1
            batch_file = batch_dir / f"constitutional_batch_{timestamp}_{batch_count:03d}.json"
            
            batch_data = {
                "batch_number": batch_count,
                "batch_size": len(current_batch),
                "keyword": keyword,
                "timestamp": datetime.now().isoformat(),
                "decisions": current_batch
            }
            
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, ensure_ascii=False, indent=2)
            
            print(f"  💾 마지막 헌재결정례 배치 {batch_count} 저장: {len(current_batch):,}개 항목 -> {batch_file.name}")
        
        # 배치 요약 정보 저장
        if self.config.save_batches and batch_count > 0:
            summary_file = batch_dir / f"constitutional_batch_summary_{timestamp}.json"
            summary_data = {
                "total_batches": batch_count,
                "total_decisions": len(decisions),
                "batch_size": self.config.batch_size,
                "keyword": keyword,
                "timestamp": timestamp,
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "sort_order": self.config.sort_order,
                "include_details": include_details
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
            print(f"  📊 헌재결정례 배치 요약 저장: {batch_count}개 배치, {len(decisions):,}개 항목 -> {summary_file.name}")
        
        self.stats.total_collected = len(decisions)
        self.stats.batch_count = batch_count
        
        logger.info(f"키워드 기반 헌재결정례 수집 완료 - 총 {len(decisions)}개 결정례")
        return decisions
    
    def collect_all_decisions(self, 
                            query: str = "",
                            include_details: bool = True) -> List[Dict[str, Any]]:
        """
        모든 헌재결정례 수집 (선고일자 오름차순)
        
        Args:
            query: 검색 쿼리 (빈 문자열이면 모든 결정례)
            include_details: 상세 정보 포함 여부
            
        Returns:
            수집된 헌재결정례 목록
        """
        logger.info(f"전체 헌재결정례 수집 시작 - 쿼리: '{query}', 상세정보: {include_details}")
        
        self.stats.start_time = datetime.now()
        
        try:
            # API 클라이언트의 전체 수집 메서드 사용
            decisions = self.client.get_all_constitutional_decisions(
                query=query,
                max_pages=self.config.max_pages,
                sort=self.config.sort_order,
                include_details=include_details,
                batch_size=self.config.batch_size,
                save_batches=self.config.save_batches
            )
            
            self.stats.total_collected = len(decisions)
            self.stats.end_time = datetime.now()
            
            logger.info(f"전체 헌재결정례 수집 완료 - 총 {len(decisions)}개 결정례")
            return decisions
            
        except Exception as e:
            logger.error(f"전체 헌재결정례 수집 실패: {e}")
            self.stats.errors.append(f"Full collection error: {e}")
            self.stats.end_time = datetime.now()
            return []
    
    def collect_decisions_by_date_range(self, 
                                      start_date: str, 
                                      end_date: str,
                                      include_details: bool = True) -> List[Dict[str, Any]]:
        """
        날짜 범위 기반 헌재결정례 수집
        
        Args:
            start_date: 시작 날짜 (YYYYMMDD 형식)
            end_date: 종료 날짜 (YYYYMMDD 형식)
            include_details: 상세 정보 포함 여부
            
        Returns:
            수집된 헌재결정례 목록
        """
        logger.info(f"날짜 범위 기반 헌재결정례 수집 시작 - {start_date} ~ {end_date}")
        
        decisions = []
        page = 1
        batch_count = 0
        current_batch = []
        
        # 배치 저장 디렉토리 설정
        if self.config.save_batches:
            batch_dir = self.config.base_output_dir / "batches"
            batch_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        while True:
            try:
                # 날짜 범위 검색
                response = self.client.search_constitutional_decisions(
                    query="",
                    search=1,
                    display=100,
                    page=page,
                    sort=self.config.sort_order,
                    edYd=f"{start_date}-{end_date}"
                )
                
                self.stats.api_requests_made += 1
                
                if not response or 'DetcSearch' not in response:
                    logger.warning(f"페이지 {page}에서 응답 데이터 없음")
                    break
                
                search_result = response['DetcSearch']
                if 'detc' not in search_result:
                    logger.info(f"페이지 {page}에서 빈 결과 - 수집 완료")
                    break
                
                # detc가 단일 객체인 경우 리스트로 변환
                page_decisions = search_result['detc']
                if isinstance(page_decisions, dict):
                    page_decisions = [page_decisions]
                
                for decision in page_decisions:
                    decision_id = decision.get('헌재결정례일련번호')
                    if decision_id and decision_id not in self.collected_decisions:
                        if include_details:
                            try:
                                # 상세 정보 조회
                                detail = self.client.get_constitutional_decision_detail(decision_id)
                                
                                # 목록 정보와 상세 정보 결합
                                combined_decision = {
                                    **decision,  # 목록 정보
                                    'detailed_info': detail,  # 상세 정보
                                    'document_type': 'constitutional_decision',
                                    'collected_at': datetime.now().isoformat()
                                }
                                decisions.append(combined_decision)
                                current_batch.append(combined_decision)
                                
                                # 서버 과부하 방지
                                time.sleep(1.0)
                                
                            except Exception as e:
                                logger.error(f"헌재결정례 상세 조회 실패: {decision_id} - {e}")
                                decision['document_type'] = 'constitutional_decision'
                                decision['collected_at'] = datetime.now().isoformat()
                                decisions.append(decision)
                                current_batch.append(decision)
                        else:
                            decision['document_type'] = 'constitutional_decision'
                            decision['collected_at'] = datetime.now().isoformat()
                            decisions.append(decision)
                            current_batch.append(decision)
                        
                        self.collected_decisions.add(decision_id)
                        self.stats.collected_decisions.add(decision_id)
                        
                        # 배치 크기에 도달하면 파일로 저장
                        if self.config.save_batches and len(current_batch) >= self.config.batch_size:
                            batch_count += 1
                            batch_file = batch_dir / f"constitutional_batch_{timestamp}_{batch_count:03d}.json"
                            
                            batch_data = {
                                "batch_number": batch_count,
                                "batch_size": len(current_batch),
                                "date_range": f"{start_date}-{end_date}",
                                "timestamp": datetime.now().isoformat(),
                                "decisions": current_batch
                            }
                            
                            with open(batch_file, 'w', encoding='utf-8') as f:
                                json.dump(batch_data, f, ensure_ascii=False, indent=2)
                            
                            print(f"  💾 헌재결정례 배치 {batch_count} 저장: {len(current_batch):,}개 항목 -> {batch_file.name}")
                            logger.info(f"헌재결정례 배치 {batch_count} 저장 완료: {len(current_batch)}개 항목")
                            
                            current_batch = []  # 배치 초기화
                
                logger.info(f"페이지 {page} 완료: {len(page_decisions)}건의 결정례 수집")
                
                page += 1
                
                # 서버 과부하 방지를 위한 대기
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"페이지 {page} 수집 실패: {e}")
                self.stats.errors.append(f"Page {page} collection error: {e}")
                break
        
        # 마지막 배치 저장 (남은 데이터가 있는 경우)
        if self.config.save_batches and current_batch:
            batch_count += 1
            batch_file = batch_dir / f"constitutional_batch_{timestamp}_{batch_count:03d}.json"
            
            batch_data = {
                "batch_number": batch_count,
                "batch_size": len(current_batch),
                "date_range": f"{start_date}-{end_date}",
                "timestamp": datetime.now().isoformat(),
                "decisions": current_batch
            }
            
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, ensure_ascii=False, indent=2)
            
            print(f"  💾 마지막 헌재결정례 배치 {batch_count} 저장: {len(current_batch):,}개 항목 -> {batch_file.name}")
        
        self.stats.total_collected = len(decisions)
        self.stats.batch_count = batch_count
        
        logger.info(f"날짜 범위 기반 헌재결정례 수집 완료 - 총 {len(decisions)}개 결정례")
        return decisions
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """수집 통계 반환"""
        return {
            "total_collected": self.stats.total_collected,
            "total_pages": self.stats.total_pages,
            "batch_count": self.stats.batch_count,
            "api_requests_made": self.stats.api_requests_made,
            "errors": self.stats.errors,
            "start_time": self.stats.start_time.isoformat() if self.stats.start_time else None,
            "end_time": self.stats.end_time.isoformat() if self.stats.end_time else None,
            "collected_decisions_count": len(self.stats.collected_decisions)
        }
    
    def clear_stats(self):
        """통계 초기화"""
        self.stats = CollectionStats()
        self.collected_decisions.clear()


def create_collector(config: CollectionConfig = None) -> ConstitutionalDecisionCollector:
    """
    헌재결정례 수집기 생성
    
    Args:
        config: 수집 설정
        
    Returns:
        ConstitutionalDecisionCollector 인스턴스
    """
    return ConstitutionalDecisionCollector(config)


if __name__ == "__main__":
    # 테스트 실행
    logging.basicConfig(level=logging.INFO)
    
    print("헌재결정례 수집기 테스트")
    print("=" * 40)
    
    # 환경변수 확인
    oc_param = os.getenv("LAW_OPEN_API_OC")
    if not oc_param:
        print("❌ LAW_OPEN_API_OC 환경변수가 설정되지 않았습니다.")
        print("다음과 같이 설정해주세요:")
        print("export LAW_OPEN_API_OC='your_email@example.com'")
        exit(1)
    
    print(f"✅ OC 파라미터: {oc_param}")
    
    # 수집기 생성 및 테스트
    try:
        config = CollectionConfig(
            batch_size=100,
            include_details=True,
            sort_order="dasc"  # 선고일자 오름차순
        )
        
        collector = create_collector(config)
        
        # API 연결 테스트
        if collector.client.test_connection():
            print("✅ API 연결 테스트 성공")
            
            # 샘플 수집 테스트
            print("\n샘플 헌재결정례 수집 테스트:")
            decisions = collector.collect_decisions_by_keyword(
                keyword="헌법",
                max_count=5,
                include_details=True
            )
            
            if decisions:
                print(f"✅ 샘플 수집 성공: {len(decisions)}개 결정례")
                for i, decision in enumerate(decisions[:3], 1):
                    print(f"  {i}. {decision.get('사건명', 'N/A')}")
            else:
                print("❌ 샘플 수집 실패")
        else:
            print("❌ API 연결 테스트 실패")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
