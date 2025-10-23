#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
현행법령 수집기 클래스

현행법령 목록 조회 후 각 법령의 본문을 수집하는 수집기입니다.
데이터베이스나 벡터 저장소 업데이트는 별도로 처리합니다.
"""

import os
import sys
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIClient

logger = logging.getLogger(__name__)


@dataclass
class CollectionConfig:
    """수집 설정 클래스"""
    batch_size: int = 10
    include_details: bool = True
    sort_order: str = "ldes"
    save_batches: bool = True
    max_pages: Optional[int] = None
    query: str = ""
    resume_from_checkpoint: bool = False


class CurrentLawCollector:
    """현행법령 수집기"""
    
    def __init__(self, config: CollectionConfig):
        """
        수집기 초기화
        
        Args:
            config: 수집 설정
        """
        self.config = config
        self.client = LawOpenAPIClient()
        
        # 통계 정보
        self.stats = {
            'total_collected': 0,
            'api_requests_made': 0,
            'batch_count': 0,
            'errors': [],
            'start_time': None,
            'end_time': None
        }
        
        # 배치 저장 디렉토리 설정
        if self.config.save_batches:
            self.batch_dir = Path("data/raw/law_open_api/current_laws/batches")
            self.batch_dir.mkdir(parents=True, exist_ok=True)
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"CurrentLawCollector 초기화 완료 - 배치크기: {config.batch_size}, 상세정보: {config.include_details}")
    
    def collect_laws_by_query(self, query: str = "", max_pages: int = None) -> List[Dict[str, Any]]:
        """
        검색어로 현행법령 수집
        
        Args:
            query: 검색 질의
            max_pages: 최대 페이지 수
            
        Returns:
            수집된 현행법령 목록
        """
        logger.info(f"검색어 기반 현행법령 수집 시작: '{query}'")
        
        # 설정 업데이트
        self.config.query = query
        if max_pages:
            self.config.max_pages = max_pages
        
        return self._collect_laws()
    
    def collect_all_laws(self, max_pages: int = None, start_page: int = 1) -> List[Dict[str, Any]]:
        """
        모든 현행법령 수집
        
        Args:
            max_pages: 최대 페이지 수
            start_page: 시작 페이지
            
        Returns:
            수집된 현행법령 목록
        """
        logger.info(f"전체 현행법령 수집 시작 - 시작 페이지: {start_page}")
        
        # 설정 업데이트
        self.config.query = ""
        if max_pages:
            self.config.max_pages = max_pages
        
        return self._collect_laws(start_page=start_page)
    
    def _collect_laws(self, start_page: int = 1) -> List[Dict[str, Any]]:
        """실제 수집 로직"""
        self.stats['start_time'] = datetime.now()
        logger.info("=" * 60)
        logger.info("현행법령 수집 시작")
        logger.info(f"검색어: '{self.config.query}'")
        logger.info(f"최대 페이지: {self.config.max_pages or '무제한'}")
        logger.info(f"배치 크기: {self.config.batch_size}개")
        logger.info(f"상세 정보: {'포함' if self.config.include_details else '제외'}")
        logger.info(f"정렬 순서: {self.config.sort_order}")
        logger.info("=" * 60)
        
        try:
            # API 클라이언트를 통한 데이터 수집
            laws = self.client.get_all_current_laws(
                query=self.config.query,
                max_pages=self.config.max_pages,
                start_page=start_page,
                sort=self.config.sort_order,
                batch_size=self.config.batch_size,
                save_batches=self.config.save_batches,
                include_details=self.config.include_details,
                resume_from_checkpoint=self.config.resume_from_checkpoint
            )
            
            self.stats['total_collected'] = len(laws)
            self.stats['end_time'] = datetime.now()
            
            logger.info(f"현행법령 수집 완료: {len(laws):,}개")
            
            if laws:
                print(f"\n✅ 수집 완료: {len(laws):,}개 현행법령")
                
                # 수집된 데이터 요약 출력
                self._print_collection_summary(laws)
            else:
                logger.warning("수집된 현행법령이 없습니다.")
                print("❌ 수집된 현행법령이 없습니다.")
            
            return laws
            
        except Exception as e:
            error_msg = f"현행법령 수집 실패: {e}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            self.stats['errors'].append(error_msg)
            raise
    
    def _print_collection_summary(self, laws: List[Dict[str, Any]]):
        """수집 결과 요약 출력"""
        if not laws:
            return
        
        print(f"\n📊 수집 결과 요약:")
        print(f"  총 수집: {len(laws):,}개")
        
        # 소관부처별 통계
        ministry_stats = {}
        for law in laws:
            ministry = law.get('소관부처명', '미분류')
            ministry_stats[ministry] = ministry_stats.get(ministry, 0) + 1
        
        print(f"  소관부처별 분포: {len(ministry_stats)}개 부처")
        for ministry, count in sorted(ministry_stats.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    - {ministry}: {count:,}개")
        
        # 상세 정보 포함 여부
        detailed_count = sum(1 for law in laws if law.get('detailed_info'))
        print(f"  상세 정보 포함: {detailed_count:,}개 ({detailed_count/len(laws)*100:.1f}%)")
        
        # 수집 시간
        if self.stats['start_time'] and self.stats['end_time']:
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            print(f"  수집 시간: {duration:.2f}초")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """수집 통계 반환"""
        stats_copy = self.stats.copy()
        
        # datetime 객체를 문자열로 변환
        if 'start_time' in stats_copy and isinstance(stats_copy['start_time'], datetime):
            stats_copy['start_time'] = stats_copy['start_time'].isoformat()
        if 'end_time' in stats_copy and isinstance(stats_copy['end_time'], datetime):
            stats_copy['end_time'] = stats_copy['end_time'].isoformat()
            
        return stats_copy
    
    def save_collection_report(self, laws: List[Dict[str, Any]], output_file: str = None) -> str:
        """
        수집 결과 보고서 저장
        
        Args:
            laws: 수집된 법령 목록
            output_file: 출력 파일명 (None이면 자동 생성)
            
        Returns:
            저장된 파일 경로
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"results/current_laws_collection_report_{timestamp}.json"
        
        # 결과 디렉토리 생성
        Path("results").mkdir(exist_ok=True)
        
        # 보고서 데이터 구성
        report = {
            "collection_info": {
                "timestamp": datetime.now().isoformat(),
                "query": self.config.query,
                "max_pages": self.config.max_pages,
                "batch_size": self.config.batch_size,
                "include_details": self.config.include_details,
                "sort_order": self.config.sort_order
            },
            "statistics": self.get_collection_stats(),  # datetime 객체가 변환된 통계 사용
            "summary": {
                "total_laws": len(laws),
                "ministries": len(set(law.get('소관부처명', '') for law in laws)),
                "detailed_count": sum(1 for law in laws if law.get('detailed_info')),
                "sample_laws": laws[:5] if laws else []  # 샘플 5개
            }
        }
        
        # 파일 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"수집 보고서 저장: {output_file}")
        print(f"📄 수집 보고서 저장: {output_file}")
        
        return output_file


def create_collector(config: CollectionConfig = None) -> CurrentLawCollector:
    """
    수집기 생성 편의 함수
    
    Args:
        config: 수집 설정 (None이면 기본값 사용)
        
    Returns:
        CurrentLawCollector 인스턴스
    """
    if config is None:
        config = CollectionConfig()
    
    return CurrentLawCollector(config)


if __name__ == "__main__":
    # 테스트 실행
    logging.basicConfig(level=logging.INFO)
    
    print("현행법령 수집기 테스트")
    print("=" * 40)
    
    # 환경변수 확인
    oc_param = os.getenv("LAW_OPEN_API_OC")
    if not oc_param:
        print("❌ LAW_OPEN_API_OC 환경변수가 설정되지 않았습니다.")
        exit(1)
    
    print(f"✅ OC 파라미터: {oc_param}")
    
    # 수집기 생성 및 테스트
    try:
        config = CollectionConfig(
            batch_size=5,
            include_details=True,
            save_batches=True,
            max_pages=1  # 테스트용으로 1페이지만
        )
        
        collector = create_collector(config)
        
        # 샘플 수집
        print("\n샘플 현행법령 수집 중...")
        laws = collector.collect_laws_by_query("자동차", max_pages=1)
        
        if laws:
            print(f"✅ 샘플 수집 성공: {len(laws)}개")
            
            # 보고서 저장
            report_file = collector.save_collection_report(laws)
            print(f"📄 보고서 저장: {report_file}")
        else:
            print("❌ 샘플 수집 실패")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
