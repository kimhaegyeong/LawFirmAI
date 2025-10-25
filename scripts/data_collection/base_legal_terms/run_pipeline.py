"""
법령정보지식베이스 법령용어 수집 통합 파이프라인

이 스크립트는 데이터 수집부터 벡터 임베딩 생성까지의 전체 파이프라인을 실행합니다.
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# 설정 파일 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'base_legal_terms', 'config'))
from base_legal_term_collection_config import BaseLegalTermCollectionConfig as Config

# 수집기 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_collection', 'base_legal_terms'))
from base_legal_term_collector import BaseLegalTermCollector

# 처리기 import
sys.path.append(os.path.join(os.path.dirname(__file__), 'base_legal_terms'))
from process_terms import BaseLegalTermProcessor
from generate_embeddings import BaseLegalTermEmbeddingGenerator

# 로거 설정
from source.utils.logger import setup_logging, get_logger

# 로거 초기화
logger = get_logger(__name__)

class BaseLegalTermPipeline:
    """법령정보지식베이스 법령용어 수집 통합 파이프라인"""
    
    def __init__(self, config: Config):
        self.config = config
        self.collector = BaseLegalTermCollector(config)
        self.processor = BaseLegalTermProcessor(config)
        self.embedding_generator = BaseLegalTermEmbeddingGenerator(config)
        
        # 파이프라인 통계
        self.pipeline_stats = {
            "start_time": None,
            "end_time": None,
            "collection_success": False,
            "processing_success": False,
            "embedding_success": False,
            "total_duration": 0
        }
    
    async def run_collection_phase(self, args) -> bool:
        """수집 단계 실행"""
        try:
            logger.info("=== 수집 단계 시작 ===")
            self.pipeline_stats["start_time"] = datetime.now()
            
            if args.collect_alternating:
                logger.info("번갈아가면서 수집 실행")
                await self.collector.collect_alternating(
                    start_page=args.start_page,
                    end_page=args.end_page,
                    list_batch_size=args.batch_size,
                    detail_batch_size=args.detail_batch_size,
                    query=args.query,
                    homonym_yn=args.homonym_yn
                )
            elif args.collect_details:
                logger.info("상세 정보 수집 실행")
                await self.collector.collect_term_details(batch_size=args.detail_batch_size)
            elif args.collect_lists:
                logger.info("목록 수집 실행")
                await self.collector.collect_term_lists(
                    start_page=args.start_page,
                    end_page=args.end_page,
                    batch_size=args.batch_size,
                    query=args.query,
                    homonym_yn=args.homonym_yn
                )
            else:
                logger.info("기본 목록 수집 실행")
                await self.collector.collect_term_lists(
                    start_page=args.start_page,
                    end_page=args.end_page,
                    batch_size=args.batch_size,
                    query=args.query,
                    homonym_yn=args.homonym_yn
                )
            
            self.pipeline_stats["collection_success"] = True
            logger.info("=== 수집 단계 완료 ===")
            return True
            
        except Exception as e:
            logger.error(f"수집 단계 실패: {e}")
            return False
    
    def run_processing_phase(self) -> bool:
        """처리 단계 실행"""
        try:
            logger.info("=== 처리 단계 시작 ===")
            
            success = self.processor.process_all_terms()
            
            if success:
                self.pipeline_stats["processing_success"] = True
                logger.info("=== 처리 단계 완료 ===")
                return True
            else:
                logger.error("=== 처리 단계 실패 ===")
                return False
                
        except Exception as e:
            logger.error(f"처리 단계 실패: {e}")
            return False
    
    def run_embedding_phase(self) -> bool:
        """임베딩 단계 실행"""
        try:
            logger.info("=== 임베딩 단계 시작 ===")
            
            success = self.embedding_generator.generate_embeddings_pipeline()
            
            if success:
                self.pipeline_stats["embedding_success"] = True
                logger.info("=== 임베딩 단계 완료 ===")
                return True
            else:
                logger.error("=== 임베딩 단계 실패 ===")
                return False
                
        except Exception as e:
            logger.error(f"임베딩 단계 실패: {e}")
            return False
    
    def save_pipeline_report(self):
        """파이프라인 실행 보고서 저장"""
        try:
            self.pipeline_stats["end_time"] = datetime.now()
            
            if self.pipeline_stats["start_time"] and self.pipeline_stats["end_time"]:
                duration = self.pipeline_stats["end_time"] - self.pipeline_stats["start_time"]
                self.pipeline_stats["total_duration"] = duration.total_seconds()
            
            # 보고서 데이터 구성
            report_data = {
                "파이프라인실행보고서": {
                    "실행일시": datetime.now().isoformat(),
                    "시작시간": self.pipeline_stats["start_time"].isoformat() if self.pipeline_stats["start_time"] else None,
                    "종료시간": self.pipeline_stats["end_time"].isoformat() if self.pipeline_stats["end_time"] else None,
                    "총소요시간": f"{self.pipeline_stats['total_duration']:.2f}초",
                    "단계별성공여부": {
                        "수집단계": "성공" if self.pipeline_stats["collection_success"] else "실패",
                        "처리단계": "성공" if self.pipeline_stats["processing_success"] else "실패",
                        "임베딩단계": "성공" if self.pipeline_stats["embedding_success"] else "실패"
                    },
                    "전체성공여부": "성공" if all([
                        self.pipeline_stats["collection_success"],
                        self.pipeline_stats["processing_success"],
                        self.pipeline_stats["embedding_success"]
                    ]) else "실패"
                },
                "수집통계": self.collector.stats if hasattr(self.collector, 'stats') else {},
                "처리통계": self.processor.stats if hasattr(self.processor, 'stats') else {},
                "임베딩통계": self.embedding_generator.stats if hasattr(self.embedding_generator, 'stats') else {}
            }
            
            # 보고서 저장
            reports_dir = Path("data/base_legal_terms/reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = reports_dir / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            import json
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"파이프라인 보고서 저장: {report_file}")
            
        except Exception as e:
            logger.error(f"파이프라인 보고서 저장 실패: {e}")
    
    async def run_full_pipeline(self, args) -> bool:
        """전체 파이프라인 실행"""
        try:
            logger.info("=== 법령정보지식베이스 법령용어 수집 통합 파이프라인 시작 ===")
            
            # 1. 수집 단계
            if not args.skip_collection:
                collection_success = await self.run_collection_phase(args)
                if not collection_success:
                    logger.error("수집 단계 실패로 파이프라인 중단")
                    return False
            
            # 2. 처리 단계
            if not args.skip_processing:
                processing_success = self.run_processing_phase()
                if not processing_success:
                    logger.error("처리 단계 실패로 파이프라인 중단")
                    return False
            
            # 3. 임베딩 단계
            if not args.skip_embedding:
                embedding_success = self.run_embedding_phase()
                if not embedding_success:
                    logger.error("임베딩 단계 실패로 파이프라인 중단")
                    return False
            
            # 4. 보고서 저장
            self.save_pipeline_report()
            
            logger.info("=== 법령정보지식베이스 법령용어 수집 통합 파이프라인 완료 ===")
            return True
            
        except Exception as e:
            logger.error(f"파이프라인 실행 중 오류: {e}")
            return False


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='법령정보지식베이스 법령용어 수집 통합 파이프라인')
    
    # 파이프라인 단계 선택
    parser.add_argument('--skip-collection', action='store_true',
                       help='수집 단계 건너뛰기')
    parser.add_argument('--skip-processing', action='store_true',
                       help='처리 단계 건너뛰기')
    parser.add_argument('--skip-embedding', action='store_true',
                       help='임베딩 단계 건너뛰기')
    
    # 수집 옵션
    parser.add_argument('--collect-lists', action='store_true',
                       help='법령용어 목록만 수집')
    parser.add_argument('--collect-details', action='store_true',
                       help='법령용어 상세 정보만 수집')
    parser.add_argument('--collect-alternating', action='store_true',
                       help='목록 수집과 상세 수집을 번갈아가면서 진행')
    
    # 페이지 설정
    parser.add_argument('--start-page', type=int, default=1,
                       help='시작 페이지 (기본값: 1)')
    parser.add_argument('--end-page', type=int, default=None,
                       help='종료 페이지 (기본값: 무제한)')
    
    # 배치 설정
    parser.add_argument('--batch-size', type=int, default=20,
                       help='목록 수집 배치 크기 (기본값: 20)')
    parser.add_argument('--detail-batch-size', type=int, default=50,
                       help='상세 수집 배치 크기 (기본값: 50)')
    
    # 검색 설정
    parser.add_argument('--query', type=str, default='',
                       help='검색 쿼리')
    parser.add_argument('--homonym-yn', type=str, default='Y',
                       help='동음이의어 포함 여부 (Y/N)')
    
    # API 설정
    parser.add_argument('--display-count', type=int, default=100,
                       help='페이지당 결과 수 (기본값: 100)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='최대 재시도 횟수 (기본값: 3)')
    
    # 대기 시간 설정
    parser.add_argument('--rate-limit-delay', type=float, default=1.0,
                       help='요청 간 대기 시간(초) (기본값: 1.0)')
    parser.add_argument('--detail-delay', type=float, default=1.0,
                       help='상세 조회 간 대기 시간(초) (기본값: 1.0)')
    
    # 기타 옵션
    parser.add_argument('--config-file', type=str, default=None,
                       help='설정 파일 경로')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='상세 로그 출력')
    
    return parser.parse_args()


async def main():
    """메인 실행 함수"""
    args = parse_arguments()
    
    # 설정 로드
    if args.config_file:
        config = Config(args.config_file)
    else:
        config = Config()
    
    # 명령행 인수로 설정 오버라이드
    config_dict = config.get_all_config()
    config_dict['api']['display_count'] = args.display_count
    config_dict['api']['max_retries'] = args.max_retries
    config_dict['api']['rate_limit_delay'] = args.rate_limit_delay
    config_dict['collection']['start_page'] = args.start_page
    config_dict['collection']['end_page'] = args.end_page
    config_dict['collection']['query'] = args.query
    config_dict['collection']['homonym_yn'] = args.homonym_yn
    config_dict['collection']['list_batch_size'] = args.batch_size
    config_dict['collection']['detail_batch_size'] = args.detail_batch_size
    config_dict['collection']['detail_collection_delay'] = args.detail_delay
    
    # 로그 레벨 설정
    if args.verbose:
        config_dict['logging']['level'] = 'DEBUG'
    
    # 설정 업데이트
    config.update_config(config_dict)
    
    logger.info(f"=== 법령정보지식베이스 법령용어 수집 통합 파이프라인 ===")
    logger.info(f"수집 단계: {'건너뛰기' if args.skip_collection else '실행'}")
    logger.info(f"처리 단계: {'건너뛰기' if args.skip_processing else '실행'}")
    logger.info(f"임베딩 단계: {'건너뛰기' if args.skip_embedding else '실행'}")
    logger.info(f"시작 페이지: {args.start_page}")
    logger.info(f"종료 페이지: {args.end_page or '무제한'}")
    logger.info(f"검색 쿼리: '{args.query}'")
    logger.info(f"동음이의어 포함: {args.homonym_yn}")
    
    # 파이프라인 생성 및 실행
    pipeline = BaseLegalTermPipeline(config)
    
    try:
        success = await pipeline.run_full_pipeline(args)
        
        if success:
            logger.info("=== 통합 파이프라인 실행 완료 ===")
        else:
            logger.error("=== 통합 파이프라인 실행 실패 ===")
            
    except Exception as e:
        logger.error(f"파이프라인 실행 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
