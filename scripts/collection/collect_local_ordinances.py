#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
자치법규 데이터 수집 스크립트 (국가법령정보센터 OpenAPI 기반)

이 스크립트는 국가법령정보센터의 OpenAPI를 통해 자치법규 데이터를 수집합니다.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIClient, LawOpenAPIConfig
from source.data.data_processor import DataProcessor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/local_ordinance_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LocalOrdinanceCollector:
    """자치법규 데이터 수집 클래스"""
    
    def __init__(self):
        self.config = APIConfig()
        self.client = LawOpenAPIClient(self.config)
        self.data_processor = DataProcessor()
        
        # 수집 목표 설정
        self.target_ordinances = 500  # 자치법규 500건
        
        # 데이터 저장 디렉토리 생성
        self.raw_data_dir = Path("data/raw/local_ordinances")
        self.processed_data_dir = Path("data/processed/local_ordinances")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_local_ordinances(self, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """자치법규 데이터 수집"""
        logger.info("자치법규 데이터 수집 시작")
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y%m%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        
        all_ordinances = []
        page = 1
        per_page = 50
        
        while len(all_ordinances) < self.target_ordinances:
            logger.info(f"자치법규 수집 중... (페이지 {page}, 현재 {len(all_ordinances)}건)")
            
            response = self.client.get_local_ordinance_list(page=page, per_page=per_page)
            
            if not response:
                logger.error(f"페이지 {page} 수집 실패")
                break
            
            # 응답에서 자치법규 목록 추출
            ordinances = response.get('localOrdinanceList', {}).get('localOrdinance', [])
            if not ordinances:
                logger.info("더 이상 수집할 자치법규가 없습니다.")
                break
            
            # 단일 자치법규인 경우 리스트로 변환
            if isinstance(ordinances, dict):
                ordinances = [ordinances]
            
            # 각 자치법규의 상세 정보 수집
            for ordinance in ordinances:
                if len(all_ordinances) >= self.target_ordinances:
                    break
                
                ordinance_id = ordinance.get('id')
                if ordinance_id:
                    detail = self.client.get_local_ordinance_detail(ordinance_id)
                    if detail:
                        detail['category'] = 'local_ordinance'
                        all_ordinances.append(detail)
                        
                        # 원본 데이터 저장
                        self._save_raw_data(detail, f"local_ordinance_{ordinance_id}")
            
            page += 1
            
            # API 요청 제한 확인
            stats = self.client.get_request_stats()
            if stats['remaining_requests'] <= 10:
                logger.warning("API 요청 한도에 근접했습니다. 수집을 중단합니다.")
                break
        
        logger.info(f"자치법규 {len(all_ordinances)}건 수집 완료")
        return all_ordinances
    
    def _save_raw_data(self, data: Dict[str, Any], filename: str):
        """원본 데이터 저장"""
        file_path = self.raw_data_dir / f"{filename}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"원본 데이터 저장: {file_path}")
    
    def process_collected_data(self, ordinances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """수집된 데이터 전처리"""
        logger.info("수집된 자치법규 데이터 전처리 시작")
        
        processed_ordinances = []
        
        for ordinance in ordinances:
            try:
                # 데이터 정제 및 구조화
                processed_ordinance = self.data_processor.process_local_ordinance_data(ordinance)
                processed_ordinances.append(processed_ordinance)
                
            except Exception as e:
                logger.error(f"자치법규 데이터 전처리 실패: {e}")
                continue
        
        # 전처리된 데이터 저장
        self._save_processed_data(processed_ordinances)
        
        logger.info(f"자치법규 데이터 {len(processed_ordinances)}건 전처리 완료")
        return processed_ordinances
    
    def _save_processed_data(self, data: List[Dict[str, Any]]):
        """전처리된 데이터 저장"""
        file_path = self.processed_data_dir / "processed_local_ordinances.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"전처리된 데이터 저장: {file_path}")
    
    def generate_collection_report(self, ordinances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """수집 결과 보고서 생성"""
        report = {
            "collection_date": datetime.now().isoformat(),
            "total_ordinances": len(ordinances),
            "api_requests_used": self.client.get_request_stats()['request_count'],
            "collection_summary": {
                "successful_collections": len([o for o in ordinances if o.get('status') == 'success']),
                "failed_collections": len([o for o in ordinances if o.get('status') == 'failed']),
            },
            "target_achievement": f"{len(ordinances)}/{self.target_ordinances}",
            "completion_rate": f"{(len(ordinances) / self.target_ordinances) * 100:.1f}%"
        }
        
        # 보고서 저장
        report_path = Path("docs/local_ordinance_collection_report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 자치법규 데이터 수집 보고서\n\n")
            f.write(f"**수집 일시**: {report['collection_date']}\n")
            f.write(f"**수집된 자치법규 수**: {report['total_ordinances']}건\n")
            f.write(f"**API 요청 수**: {report['api_requests_used']}회\n")
            f.write(f"**목표 달성률**: {report['completion_rate']}\n\n")
            f.write(f"## 수집 결과 요약\n")
            f.write(f"- 성공: {report['collection_summary']['successful_collections']}건\n")
            f.write(f"- 실패: {report['collection_summary']['failed_collections']}건\n")
            f.write(f"- 목표: {report['target_achievement']}\n")
        
        logger.info(f"수집 보고서 생성: {report_path}")
        return report


def main():
    """메인 실행 함수"""
    logger.info("자치법규 데이터 수집 스크립트 시작")
    
    try:
        # 수집기 초기화
        collector = LocalOrdinanceCollector()
        
        # 자치법규 수집
        ordinances = collector.collect_local_ordinances()
        
        # 데이터 전처리
        processed_ordinances = collector.process_collected_data(ordinances)
        
        # 수집 보고서 생성
        report = collector.generate_collection_report(processed_ordinances)
        
        logger.info("자치법규 데이터 수집 완료")
        logger.info(f"수집된 자치법규 수: {len(processed_ordinances)}건")
        logger.info(f"API 요청 수: {report['api_requests_used']}회")
        logger.info(f"목표 달성률: {report['completion_rate']}")
        
    except Exception as e:
        logger.error(f"자치법규 데이터 수집 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
