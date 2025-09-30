#!/usr/bin/env python3
"""
행정규칙 데이터 수집 스크립트 (국가법령정보센터 OpenAPI 기반)

이 스크립트는 국가법령정보센터의 OpenAPI를 통해 행정규칙 데이터를 수집합니다.
"""

import os
import sys
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIClient, LawOpenAPIConfig
from source.data.data_processor import LegalDataProcessor

# 로깅 설정
def setup_logging():
    """로깅 설정 함수"""
    # logs 디렉토리 생성
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # 로그 포맷 설정
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    
    # 파일 핸들러 설정
    file_handler = logging.FileHandler(
        logs_dir / 'administrative_rule_collection.log',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

logger = setup_logging()


class AdministrativeRuleCollector:
    """행정규칙 데이터 수집 클래스"""
    
    def __init__(self):
        self.config = LawOpenAPIConfig()
        self.client = LawOpenAPIClient(self.config)
        self.data_processor = LegalDataProcessor()
        
        # 수집 목표 설정
        self.target_rules = 1000  # 행정규칙 1,000건
        
        # 데이터 저장 디렉토리 생성
        self.raw_data_dir = Path("data/raw/administrative_rules")
        self.processed_data_dir = Path("data/processed/administrative_rules")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_administrative_rules(self, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """행정규칙 데이터 수집"""
        logger.info("=" * 60)
        logger.info("행정규칙 데이터 수집 시작")
        logger.info(f"목표 수집 건수: {self.target_rules:,}건")
        logger.info(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y%m%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        
        logger.info(f"수집 기간: {start_date} ~ {end_date}")
        
        all_rules = []
        page = 1
        display = 50
        failed_requests = 0
        successful_requests = 0
        start_time = datetime.now()
        
        while len(all_rules) < self.target_rules:
            logger.info(f"📄 페이지 {page} 수집 시작 (현재 수집: {len(all_rules):,}/{self.target_rules:,}건)")
            
            try:
                response = self.client.get_administrative_rule_list(page=page, display=display)
                successful_requests += 1
                
                if not response:
                    logger.error(f"❌ 페이지 {page} API 응답이 비어있습니다.")
                    failed_requests += 1
                    if failed_requests >= 3:
                        logger.error("연속 3회 실패로 수집을 중단합니다.")
                        break
                    page += 1
                    continue
                
                # 응답에서 행정규칙 목록 추출
                if isinstance(response, list) and len(response) > 0:
                    # API 응답이 리스트 형태인 경우
                    api_response = response[0]
                    if 'AdmRulSearch' in api_response:
                        search_result = api_response['AdmRulSearch']
                        rules = search_result.get('admrul', [])
                        total_count = search_result.get('totalCnt', '0')
                        try:
                            total_count_int = int(total_count)
                            logger.info(f"📊 총 행정규칙 수: {total_count_int:,}건")
                        except (ValueError, TypeError):
                            logger.info(f"📊 총 행정규칙 수: {total_count}건")
                    else:
                        rules = []
                else:
                    rules = []
                
                if not rules:
                    logger.info("📭 더 이상 수집할 행정규칙이 없습니다.")
                    break
                
                # 단일 규칙인 경우 리스트로 변환
                if isinstance(rules, dict):
                    rules = [rules]
                
                logger.info(f"📋 페이지 {page}에서 {len(rules)}개 규칙 발견")
                
                # 각 규칙의 상세 정보 수집
                page_success_count = 0
                for i, rule in enumerate(rules, 1):
                    if len(all_rules) >= self.target_rules:
                        logger.info(f"🎯 목표 수집 건수 {self.target_rules:,}건 달성!")
                        break
                    
                    rule_id = rule.get('id')
                    rule_name = rule.get('name', 'Unknown')
                    
                    if rule_id:
                        logger.debug(f"  📝 규칙 {i}/{len(rules)}: {rule_name} (ID: {rule_id}) 상세 정보 수집 중...")
                        
                        try:
                            detail = self.client.get_administrative_rule_detail(rule_id)
                            if detail:
                                detail['category'] = 'administrative_rule'
                                all_rules.append(detail)
                                page_success_count += 1
                                
                                # 원본 데이터 저장
                                self._save_raw_data(detail, f"administrative_rule_{rule_id}")
                                
                                logger.debug(f"  ✅ 규칙 {rule_name} 수집 완료")
                            else:
                                logger.warning(f"  ⚠️ 규칙 {rule_name} (ID: {rule_id}) 상세 정보 수집 실패")
                        except Exception as e:
                            logger.error(f"  ❌ 규칙 {rule_name} (ID: {rule_id}) 수집 중 오류: {e}")
                    else:
                        logger.warning(f"  ⚠️ 규칙 {i}/{len(rules)}: ID가 없습니다.")
                
                logger.info(f"📊 페이지 {page} 수집 결과: {page_success_count}건 성공")
                
                # 진행률 계산
                progress = (len(all_rules) / self.target_rules) * 100
                elapsed_time = datetime.now() - start_time
                estimated_total_time = elapsed_time * (self.target_rules / len(all_rules)) if all_rules else None
                remaining_time = estimated_total_time - elapsed_time if estimated_total_time else None
                
                logger.info(f"📈 진행률: {progress:.1f}% ({len(all_rules):,}/{self.target_rules:,}건)")
                logger.info(f"⏱️ 경과 시간: {elapsed_time}")
                if remaining_time:
                    logger.info(f"⏳ 예상 남은 시간: {remaining_time}")
                
            except Exception as e:
                logger.error(f"❌ 페이지 {page} 수집 중 예외 발생: {e}")
                failed_requests += 1
                if failed_requests >= 3:
                    logger.error("연속 3회 실패로 수집을 중단합니다.")
                    break
            
            page += 1
            
            # API 요청 제한 확인
            try:
                stats = self.client.get_request_stats()
                remaining = stats.get('remaining_requests', 0)
                logger.debug(f"🔢 API 요청 잔여량: {remaining}회")
                
                if remaining <= 10:
                    logger.warning("⚠️ API 요청 한도에 근접했습니다. 수집을 중단합니다.")
                    break
            except Exception as e:
                logger.warning(f"⚠️ API 요청 통계 확인 실패: {e}")
            
            # 요청 간 대기 (API 부하 방지)
            time.sleep(0.5)
        
        # 최종 수집 결과 로깅
        total_time = datetime.now() - start_time
        logger.info("=" * 60)
        logger.info("행정규칙 데이터 수집 완료")
        logger.info(f"📊 최종 수집 결과:")
        logger.info(f"  - 수집된 규칙 수: {len(all_rules):,}건")
        logger.info(f"  - 목표 대비 달성률: {(len(all_rules) / self.target_rules) * 100:.1f}%")
        logger.info(f"  - 성공한 API 요청: {successful_requests}회")
        logger.info(f"  - 실패한 API 요청: {failed_requests}회")
        logger.info(f"  - 총 소요 시간: {total_time}")
        logger.info(f"  - 평균 수집 속도: {len(all_rules) / total_time.total_seconds() * 60:.1f}건/분")
        logger.info("=" * 60)
        
        return all_rules
    
    def _save_raw_data(self, data: Dict[str, Any], filename: str):
        """원본 데이터 저장"""
        file_path = self.raw_data_dir / f"{filename}.json"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"💾 원본 데이터 저장 완료: {file_path}")
        except Exception as e:
            logger.error(f"❌ 원본 데이터 저장 실패 ({file_path}): {e}")
            raise
    
    def process_collected_data(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """수집된 데이터 전처리"""
        logger.info("=" * 60)
        logger.info("행정규칙 데이터 전처리 시작")
        logger.info(f"전처리 대상: {len(rules):,}건")
        logger.info("=" * 60)
        
        processed_rules = []
        failed_count = 0
        start_time = datetime.now()
        
        for i, rule in enumerate(rules, 1):
            try:
                # 데이터 정제 및 구조화
                processed_rule = self.data_processor.process_administrative_rule_data(rule)
                processed_rules.append(processed_rule)
                
                if i % 100 == 0:
                    logger.info(f"📊 전처리 진행률: {i:,}/{len(rules):,}건 ({i/len(rules)*100:.1f}%)")
                
            except Exception as e:
                logger.error(f"❌ 행정규칙 데이터 전처리 실패 (규칙 {i}): {e}")
                failed_count += 1
                continue
        
        # 전처리된 데이터 저장
        self._save_processed_data(processed_rules)
        
        total_time = datetime.now() - start_time
        success_rate = ((len(rules) - failed_count) / len(rules)) * 100 if rules else 0
        
        logger.info("=" * 60)
        logger.info("행정규칙 데이터 전처리 완료")
        logger.info(f"📊 전처리 결과:")
        logger.info(f"  - 성공: {len(processed_rules):,}건")
        logger.info(f"  - 실패: {failed_count:,}건")
        logger.info(f"  - 성공률: {success_rate:.1f}%")
        logger.info(f"  - 소요 시간: {total_time}")
        logger.info("=" * 60)
        
        return processed_rules
    
    def _save_processed_data(self, data: List[Dict[str, Any]]):
        """전처리된 데이터 저장"""
        file_path = self.processed_data_dir / "processed_administrative_rules.json"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB 단위
            logger.info(f"💾 전처리된 데이터 저장 완료: {file_path}")
            logger.info(f"📁 파일 크기: {file_size:.2f} MB")
        except Exception as e:
            logger.error(f"❌ 전처리된 데이터 저장 실패 ({file_path}): {e}")
            raise
    
    def generate_collection_report(self, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """수집 결과 보고서 생성"""
        logger.info("📋 수집 결과 보고서 생성 중...")
        
        try:
            api_stats = self.client.get_request_stats()
            api_requests_used = api_stats.get('request_count', 0)
        except Exception as e:
            logger.warning(f"⚠️ API 통계 조회 실패: {e}")
            api_requests_used = 0
        
        report = {
            "collection_date": datetime.now().isoformat(),
            "total_rules": len(rules),
            "api_requests_used": api_requests_used,
            "collection_summary": {
                "successful_collections": len([r for r in rules if r.get('status') == 'success']),
                "failed_collections": len([r for r in rules if r.get('status') == 'failed']),
            },
            "target_achievement": f"{len(rules)}/{self.target_rules}",
            "completion_rate": f"{(len(rules) / self.target_rules) * 100:.1f}%"
        }
        
        # 보고서 저장
        report_path = Path("docs/administrative_rule_collection_report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"# 행정규칙 데이터 수집 보고서\n\n")
                f.write(f"**수집 일시**: {report['collection_date']}\n")
                f.write(f"**수집된 규칙 수**: {report['total_rules']:,}건\n")
                f.write(f"**API 요청 수**: {report['api_requests_used']:,}회\n")
                f.write(f"**목표 달성률**: {report['completion_rate']}\n\n")
                f.write(f"## 수집 결과 요약\n")
                f.write(f"- 성공: {report['collection_summary']['successful_collections']:,}건\n")
                f.write(f"- 실패: {report['collection_summary']['failed_collections']:,}건\n")
                f.write(f"- 목표: {report['target_achievement']}\n\n")
                f.write(f"## 상세 통계\n")
                f.write(f"- 수집 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"- 수집 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"- 평균 수집 속도: {len(rules) / max(api_requests_used, 1) * 60:.1f}건/분\n")
            
            logger.info(f"📄 수집 보고서 생성 완료: {report_path}")
        except Exception as e:
            logger.error(f"❌ 수집 보고서 생성 실패: {e}")
        
        return report


def main():
    """메인 실행 함수"""
    logger.info("🚀 행정규칙 데이터 수집 스크립트 시작")
    logger.info(f"📅 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 환경변수 설정
        # if not os.getenv("LAW_OPEN_API_OC"):
        #     os.environ["LAW_OPEN_API_OC"] = "OC"
        #     logger.info("🔧 환경변수 LAW_OPEN_API_OC 설정: OC")
        
        # 수집기 초기화
        logger.info("🔧 수집기 초기화 중...")
        collector = AdministrativeRuleCollector()
        logger.info("✅ 수집기 초기화 완료")
        
        # 행정규칙 수집
        logger.info("📥 행정규칙 데이터 수집 시작")
        rules = collector.collect_administrative_rules()
        
        # 데이터 전처리
        logger.info("🔄 데이터 전처리 시작")
        processed_rules = collector.process_collected_data(rules)
        
        # 수집 보고서 생성
        logger.info("📊 수집 보고서 생성 시작")
        report = collector.generate_collection_report(processed_rules)
        
        # 최종 결과 출력
        logger.info("=" * 60)
        logger.info("🎉 행정규칙 데이터 수집 완료!")
        logger.info(f"📊 최종 결과:")
        logger.info(f"  - 수집된 규칙 수: {len(processed_rules):,}건")
        logger.info(f"  - API 요청 수: {report['api_requests_used']:,}회")
        logger.info(f"  - 목표 달성률: {report['completion_rate']}")
        logger.info(f"  - 원본 데이터 저장 위치: {collector.raw_data_dir}")
        logger.info(f"  - 전처리 데이터 저장 위치: {collector.processed_data_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ 행정규칙 데이터 수집 중 오류 발생: {e}")
        logger.error(f"오류 상세: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
