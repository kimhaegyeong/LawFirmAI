#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
증분 법령용어 수집기

국가법령정보센터 OPEN API를 통해 법령용어 데이터를 증분 수집하는 모듈입니다.
- 증분 업데이트 수집
- 전체 데이터 수집
- 변경사항 분석 및 저장
- 수집 상태 관리
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIClient
from scripts.data_collection.law_open_api.utils import (
        TimestampManager, 
        ChangeDetector, 
        CollectionLogger,
        CheckpointManager
    )

logger = logging.getLogger(__name__)


class IncrementalLegalTermCollector:
    """증분 법령용어 수집기"""
    
    def __init__(self, client: LawOpenAPIClient = None, 
                 data_dir: str = "data/raw/law_open_api/legal_terms",
                 metadata_dir: str = "data/raw/law_open_api/metadata"):
        """
        증분 수집기 초기화
        
        Args:
            client: Law Open API 클라이언트
            data_dir: 데이터 저장 디렉토리
            metadata_dir: 메타데이터 저장 디렉토리
        """
        self.client = client or LawOpenAPIClient()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 유틸리티 초기화
        self.timestamp_manager = TimestampManager(metadata_dir)
        self.change_detector = ChangeDetector(str(self.data_dir))
        self.logger = CollectionLogger("IncrementalLegalTermCollector")
        self.checkpoint_manager = CheckpointManager()
        
        # 수집 설정
        self.data_type = "legal_terms"
        self.max_retries = 3
        self.retry_delay = 5
        
        logger.info(f"IncrementalLegalTermCollector 초기화 완료 - 데이터 디렉토리: {self.data_dir}")
    
    def collect_incremental_updates(self, include_details: bool = True, 
                                  resume_from_checkpoint: bool = True,
                                  batch_size: int = 1000) -> Dict[str, Any]:
        """
        증분 업데이트 수집
        
        Args:
            include_details: 상세 정보 포함 여부 (기본값: True)
            resume_from_checkpoint: 체크포인트에서 재시작 여부 (기본값: True)
            batch_size: 배치 크기 (기본값: 1000개)
        
        Returns:
            수집 결과 정보
        """
        self.logger.log_collection_start(self.data_type, "incremental")
        
        try:
            # 체크포인트에서 재시작 여부 확인
            if resume_from_checkpoint:
                collection_checkpoint = self.checkpoint_manager.load_collection_checkpoint(self.data_type)
                if collection_checkpoint:
                    print(f"🔄 수집 체크포인트에서 재시작")
                    logger.info("수집 체크포인트에서 재시작")
            
            # 마지막 수집 시간 확인
            last_collection = self.timestamp_manager.get_last_collection_time(self.data_type)
            
            if last_collection:
                print(f"마지막 수집 시간: {last_collection}")
                self.logger.info(f"마지막 수집 시간: {last_collection}")
            else:
                print("첫 수집 - 전체 데이터 수집")
                self.logger.info("첫 수집 - 전체 데이터 수집")
            
            # 전체 법령용어 목록 조회 (등록일자 오름차순, 배치 저장)
            print(f"\n📋 법령용어 목록 조회 시작 (상세정보: {include_details}, 배치크기: {batch_size})")
            self.logger.info(f"법령용어 목록 조회 시작 (상세정보: {include_details}, 배치크기: {batch_size})")
            all_terms = self._fetch_all_terms(include_details, batch_size)
            
            if not all_terms:
                print("❌ 수집된 법령용어가 없습니다")
                self.logger.warning("수집된 법령용어가 없습니다")
                return self._create_error_result("수집된 데이터 없음")
            
            print(f"✅ 법령용어 목록 조회 완료: {len(all_terms):,}개")
            self.logger.info(f"법령용어 목록 조회 완료: {len(all_terms)}개")
            
            # 변경사항 분석
            print(f"\n🔍 변경사항 분석 중...")
            self.logger.info("변경사항 분석 시작")
            changes = self.change_detector.analyze_changes(
                self.data_type, all_terms, last_collection
            )
            
            print(f"  - 새로운 레코드: {len(changes.get('new_records', []))}개")
            print(f"  - 업데이트된 레코드: {len(changes.get('updated_records', []))}개")
            print(f"  - 삭제된 레코드: {len(changes.get('deleted_records', []))}개")
            
            # 새로운 용어와 업데이트된 용어의 상세 정보 수집
            if include_details:
                print(f"\n📚 상세 정보 수집 중...")
            detailed_terms = self._collect_detailed_terms(changes)
            
            # 데이터 저장
            print(f"\n💾 데이터 저장 중...")
            self._save_collection_data(changes, detailed_terms, include_details)
            
            # 타임스탬프 업데이트
            self.timestamp_manager.update_collection_time(self.data_type, success=True)
            
            # 결과 생성
            result = self._create_success_result(changes)
            
            print(f"\n✅ 수집 완료!")
            print(f"  - 새로운 레코드: {result['new_records']}개")
            print(f"  - 업데이트된 레코드: {result['updated_records']}개")
            print(f"  - 삭제된 레코드: {result['deleted_records']}개")
            
            # 수집 완료 시 체크포인트 삭제
            self.checkpoint_manager.clear_collection_checkpoint(self.data_type)
            print(f"  - 체크포인트 삭제됨")
            
            self.logger.log_collection_end(self.data_type, result)
            
            return result
            
        except Exception as e:
            self.logger.log_collection_error(self.data_type, e)
            self.timestamp_manager.update_collection_time(self.data_type, success=False)
            return self._create_error_result(str(e))
    
    def collect_full_data(self) -> Dict[str, Any]:
        """
        전체 데이터 수집
        
        Returns:
            수집 결과 정보
        """
        self.logger.log_collection_start(self.data_type, "full")
        
        try:
            # 전체 법령용어 목록 조회
            self.logger.info("전체 법령용어 목록 조회 시작")
            all_terms = self._fetch_all_terms()
            
            if not all_terms:
                self.logger.warning("수집된 법령용어가 없습니다")
                return self._create_error_result("수집된 데이터 없음")
            
            self.logger.info(f"전체 법령용어 목록 조회 완료: {len(all_terms)}개")
            
            # 상세 정보 수집
            detailed_terms = self._collect_all_detailed_terms(all_terms)
            
            # 전체 데이터 저장
            self.change_detector.save_full_data(self.data_type, detailed_terms)
            
            # 타임스탬프 업데이트
            self.timestamp_manager.update_collection_time(self.data_type, success=True)
            
            # 결과 생성
            result = {
                "status": "success",
                "data_type": self.data_type,
                "total_records": len(detailed_terms),
                "collection_time": datetime.now().isoformat(),
                "mode": "full"
            }
            
            self.logger.log_collection_end(self.data_type, result)
            return result
            
        except Exception as e:
            self.logger.log_collection_error(self.data_type, e)
            self.timestamp_manager.update_collection_time(self.data_type, success=False)
            return self._create_error_result(str(e))
    
    def _fetch_all_terms(self, include_details: bool = False, batch_size: int = 1000) -> List[Dict[str, Any]]:
        """전체 법령용어 목록 조회 (등록일자 오름차순, 배치 저장)"""
        try:
            if include_details:
                # 상세 정보 포함하여 조회 (배치 저장)
                all_terms = self.client.get_legal_terms_with_details(
                    query="", 
                    max_pages=None, 
                    sort="rasc",
                    batch_size=batch_size,
                    save_batches=True
                )
            else:
                # 목록만 조회 (등록일자 오름차순, 배치 저장)
                all_terms = self.client.get_all_legal_terms(
                    query="", 
                    max_pages=None, 
                    sort="rasc",
                    batch_size=batch_size,
                    save_batches=True
                )
            return all_terms
        except Exception as e:
            self.logger.error(f"법령용어 목록 조회 실패: {e}")
            raise
    
    def _collect_detailed_terms(self, changes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """변경된 용어들의 상세 정보 수집"""
        detailed_terms = []
        
        # 새로운 레코드와 업데이트된 레코드의 상세 정보 수집
        terms_to_detail = []
        
        # 새로운 레코드
        for record in changes["new_records"]:
            terms_to_detail.append(record)
        
        # 업데이트된 레코드
        for change_record in changes["updated_records"]:
            terms_to_detail.append(change_record["new"])
        
        print(f"  대상: {len(terms_to_detail)}개")
        self.logger.info(f"상세 정보 수집 대상: {len(terms_to_detail)}개")
        
        for i, term in enumerate(terms_to_detail, 1):
            try:
                term_id = term.get("termId")
                if not term_id:
                    self.logger.warning(f"용어 ID가 없습니다: {term}")
                    continue
                
                # 상세 정보 조회
                detail = self.client.get_legal_term_detail(term_id)
                if detail:
                    detailed_terms.append(detail)
                
                # 진행률 표시
                if i % 10 == 0 or i == len(terms_to_detail):
                    progress = (i / len(terms_to_detail)) * 100
                    print(f"  진행: {i}/{len(terms_to_detail)} ({progress:.1f}%)")
                    self.logger.log_progress(i, len(terms_to_detail), "용어")
                
                # 서버 과부하 방지
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"용어 상세 정보 수집 실패 (ID: {term.get('termId', 'unknown')}): {e}")
                continue
        
        print(f"✅ 상세 정보 수집 완료: {len(detailed_terms)}개")
        self.logger.info(f"상세 정보 수집 완료: {len(detailed_terms)}개")
        return detailed_terms
    
    def _collect_all_detailed_terms(self, all_terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """모든 용어의 상세 정보 수집"""
        detailed_terms = []
        
        self.logger.info(f"전체 용어 상세 정보 수집 시작: {len(all_terms)}개")
        
        for i, term in enumerate(all_terms, 1):
            try:
                term_id = term.get("termId")
                if not term_id:
                    self.logger.warning(f"용어 ID가 없습니다: {term}")
                    continue
                
                # 상세 정보 조회
                detail = self.client.get_legal_term_detail(term_id)
                if detail:
                    detailed_terms.append(detail)
                
                # 진행률 로깅
                if i % 50 == 0 or i == len(all_terms):
                    self.logger.log_progress(i, len(all_terms), "용어")
                
                # 서버 과부하 방지
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"용어 상세 정보 수집 실패 (ID: {term.get('termId', 'unknown')}): {e}")
                continue
        
        self.logger.info(f"전체 용어 상세 정보 수집 완료: {len(detailed_terms)}개")
        return detailed_terms
    
    def _save_collection_data(self, changes: Dict[str, Any], detailed_terms: List[Dict[str, Any]], 
                            include_details: bool = True):
        """수집 데이터 저장"""
        try:
            # 증분 데이터 저장
            self.change_detector.save_incremental_data(self.data_type, changes)
            
            # 상세 정보가 있으면 별도 저장
            if include_details and detailed_terms:
                current_date = datetime.now().strftime("%Y-%m-%d")
                detailed_dir = self.data_dir / "incremental" / "daily" / current_date
                detailed_dir.mkdir(parents=True, exist_ok=True)
                
                # 상세 정보를 용어별로 분리하여 저장
                detailed_data = {
                    "collection_date": current_date,
                    "total_terms": len(detailed_terms),
                    "terms_with_details": []
                }
                
                for term in detailed_terms:
                    if 'detailed_info' in term:
                        detailed_data["terms_with_details"].append({
                            "term_id": term.get('법령용어ID'),
                            "term_name": term.get('법령용어명'),
                            "detailed_info": term['detailed_info']
                        })
                
                import json
                detailed_file = detailed_dir / "detailed_terms.json"
                with open(detailed_file, 'w', encoding='utf-8') as f:
                    json.dump(detailed_data, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"상세 정보 저장 완료: {detailed_file}")
            
            self.logger.info("수집 데이터 저장 완료")
            
        except Exception as e:
            self.logger.error(f"수집 데이터 저장 실패: {e}")
            raise
    
    def _create_success_result(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """성공 결과 생성"""
        return {
            "status": "success",
            "data_type": self.data_type,
            "new_records": changes["summary"]["new_count"],
            "updated_records": changes["summary"]["updated_count"],
            "deleted_records": changes["summary"]["deleted_count"],
            "unchanged_records": changes["summary"]["unchanged_count"],
            "collection_time": datetime.now().isoformat(),
            "mode": "incremental",
            "summary": changes["summary"]
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """에러 결과 생성"""
        return {
            "status": "error",
            "data_type": self.data_type,
            "error": error_message,
            "collection_time": datetime.now().isoformat(),
            "new_records": 0,
            "updated_records": 0,
            "deleted_records": 0,
            "unchanged_records": 0
        }
    
    def get_collection_status(self) -> Dict[str, Any]:
        """수집 상태 조회"""
        stats = self.timestamp_manager.get_collection_stats(self.data_type)
        last_collection = self.timestamp_manager.get_last_collection_time(self.data_type)
        
        return {
            "data_type": self.data_type,
            "last_collection": last_collection.isoformat() if last_collection else None,
            "stats": stats,
            "data_directory": str(self.data_dir),
            "metadata_directory": str(self.timestamp_manager.metadata_dir)
        }
    
    def test_connection(self) -> bool:
        """API 연결 테스트"""
        try:
            return self.client.test_connection()
        except Exception as e:
            self.logger.error(f"API 연결 테스트 실패: {e}")
            return False


# 편의 함수들
def create_collector(oc_parameter: str = None, 
                    data_dir: str = "data/raw/law_open_api/legal_terms") -> IncrementalLegalTermCollector:
    """
    증분 수집기 생성 (편의 함수)
    
    Args:
        oc_parameter: OC 파라미터
        data_dir: 데이터 디렉토리
        
    Returns:
        IncrementalLegalTermCollector 인스턴스
    """
    client = LawOpenAPIClient(oc_parameter)
    return IncrementalLegalTermCollector(client, data_dir)


def collect_incremental_updates(oc_parameter: str = None) -> Dict[str, Any]:
    """
    증분 업데이트 수집 (편의 함수)
    
    Args:
        oc_parameter: OC 파라미터
        
    Returns:
        수집 결과
    """
    collector = create_collector(oc_parameter)
    return collector.collect_incremental_updates()


def collect_full_data(oc_parameter: str = None) -> Dict[str, Any]:
    """
    전체 데이터 수집 (편의 함수)
    
    Args:
        oc_parameter: OC 파라미터
        
    Returns:
        수집 결과
    """
    collector = create_collector(oc_parameter)
    return collector.collect_full_data()


if __name__ == "__main__":
    # 테스트 실행
    logging.basicConfig(level=logging.INFO)
    
    print("IncrementalLegalTermCollector 테스트")
    print("=" * 50)
    
    # 환경변수 확인
    import os
    if not os.getenv("LAW_OPEN_API_OC"):
        print("❌ LAW_OPEN_API_OC 환경변수가 설정되지 않았습니다.")
        print("다음과 같이 설정해주세요:")
        print("export LAW_OPEN_API_OC='your_email@example.com'")
        exit(1)
    
    try:
        # 수집기 생성
        collector = create_collector()
        
        # 연결 테스트
        if collector.test_connection():
            print("✅ API 연결 테스트 성공")
            
            # 증분 수집 테스트 (샘플)
            print("\n증분 수집 테스트 시작...")
            result = collector.collect_incremental_updates()
            
            print(f"\n수집 결과:")
            print(f"  - 상태: {result['status']}")
            print(f"  - 새로운 레코드: {result['new_records']}개")
            print(f"  - 업데이트된 레코드: {result['updated_records']}개")
            print(f"  - 삭제된 레코드: {result['deleted_records']}개")
            print(f"  - 수집 시간: {result['collection_time']}")
            
            # 상태 조회
            status = collector.get_collection_status()
            print(f"\n수집 상태:")
            print(f"  - 마지막 수집: {status['last_collection']}")
            print(f"  - 수집 횟수: {status['stats']['collection_count']}")
            print(f"  - 성공률: {status['stats']['success_rate']}%")
            
        else:
            print("❌ API 연결 테스트 실패")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
    
    print("\n✅ IncrementalLegalTermCollector 테스트 완료")




