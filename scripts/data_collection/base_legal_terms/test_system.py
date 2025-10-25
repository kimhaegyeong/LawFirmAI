"""
법령정보지식베이스 법령용어 수집 시스템 테스트

이 스크립트는 수집 시스템의 각 컴포넌트를 테스트합니다.
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# 설정 파일 import
from base_legal_term_collection_config import BaseLegalTermCollectionConfig as Config

# 수집기 import
collector_path = os.path.join(os.path.dirname(__file__), '..', 'data_collection', 'base_legal_terms')
sys.path.insert(0, collector_path)
from base_legal_term_collector import BaseLegalTermCollector

# 처리기 import
processor_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data_processing', 'base_legal_terms')
sys.path.insert(0, processor_path)
from process_terms import BaseLegalTermProcessor

# 로거 설정
from source.utils.logger import setup_logging, get_logger

# 로거 초기화
logger = get_logger(__name__)

class BaseLegalTermSystemTester:
    """법령정보지식베이스 법령용어 수집 시스템 테스터"""
    
    def __init__(self):
        self.config = Config()
        self.test_results = {
            "config_test": False,
            "collector_test": False,
            "processor_test": False,
            "api_test": False,
            "file_structure_test": False
        }
    
    def test_config(self) -> bool:
        """설정 파일 테스트"""
        try:
            logger.info("=== 설정 파일 테스트 ===")
            
            # 설정 유효성 검증
            if not self.config.validate_config():
                logger.error("설정 파일 유효성 검증 실패")
                return False
            
            # 주요 설정 확인
            api_config = self.config.get_api_config()
            collection_config = self.config.get_collection_config()
            file_storage_config = self.config.get_file_storage_config()
            
            logger.info(f"API 설정: {api_config.get('base_url')}")
            logger.info(f"수집 설정: 배치 크기 {collection_config.get('list_batch_size')}")
            logger.info(f"파일 저장 설정: {file_storage_config.get('base_dir')}")
            
            self.test_results["config_test"] = True
            logger.info("설정 파일 테스트 성공")
            return True
            
        except Exception as e:
            logger.error(f"설정 파일 테스트 실패: {e}")
            return False
    
    def test_file_structure(self) -> bool:
        """파일 구조 테스트"""
        try:
            logger.info("=== 파일 구조 테스트 ===")
            
            file_storage_config = self.config.get_file_storage_config()
            base_dir = Path(file_storage_config.get("base_dir", "data/base_legal_terms"))
            
            # 필요한 디렉토리 확인
            required_dirs = [
                base_dir / "raw" / "term_lists",
                base_dir / "raw" / "term_details",
                base_dir / "raw" / "term_relations",
                base_dir / "processed" / "cleaned_terms",
                base_dir / "processed" / "normalized_terms",
                base_dir / "processed" / "validated_terms",
                base_dir / "processed" / "integrated_terms",
                base_dir / "embeddings",
                base_dir / "database",
                base_dir / "logs",
                base_dir / "progress",
                base_dir / "reports",
                base_dir / "config"
            ]
            
            for directory in required_dirs:
                if not directory.exists():
                    logger.error(f"필수 디렉토리 누락: {directory}")
                    return False
                logger.info(f"디렉토리 확인: {directory}")
            
            self.test_results["file_structure_test"] = True
            logger.info("파일 구조 테스트 성공")
            return True
            
        except Exception as e:
            logger.error(f"파일 구조 테스트 실패: {e}")
            return False
    
    def test_collector_initialization(self) -> bool:
        """수집기 초기화 테스트"""
        try:
            logger.info("=== 수집기 초기화 테스트 ===")
            
            collector = BaseLegalTermCollector(self.config)
            
            # 기본 속성 확인
            if not hasattr(collector, 'base_url'):
                logger.error("수집기 base_url 속성 누락")
                return False
            
            if not hasattr(collector, 'oc_id'):
                logger.error("수집기 oc_id 속성 누락")
                return False
            
            logger.info(f"수집기 URL: {collector.base_url}")
            logger.info(f"수집기 OC ID: {collector.oc_id}")
            
            self.test_results["collector_test"] = True
            logger.info("수집기 초기화 테스트 성공")
            return True
            
        except Exception as e:
            logger.error(f"수집기 초기화 테스트 실패: {e}")
            return False
    
    def test_processor_initialization(self) -> bool:
        """처리기 초기화 테스트"""
        try:
            logger.info("=== 처리기 초기화 테스트 ===")
            
            processor = BaseLegalTermProcessor(self.config)
            
            # 기본 속성 확인
            if not hasattr(processor, 'base_dir'):
                logger.error("처리기 base_dir 속성 누락")
                return False
            
            if not hasattr(processor, 'stats'):
                logger.error("처리기 stats 속성 누락")
                return False
            
            logger.info(f"처리기 기본 디렉토리: {processor.base_dir}")
            logger.info(f"처리기 통계: {processor.stats}")
            
            self.test_results["processor_test"] = True
            logger.info("처리기 초기화 테스트 성공")
            return True
            
        except Exception as e:
            logger.error(f"처리기 초기화 테스트 실패: {e}")
            return False
    
    async def test_api_connection(self) -> bool:
        """API 연결 테스트"""
        try:
            logger.info("=== API 연결 테스트 ===")
            
            collector = BaseLegalTermCollector(self.config)
            
            # 간단한 API 요청 테스트
            async with collector.get_session():
                response = await collector.get_term_list(page=1, query="", homonym_yn="Y")
                
                if response is None:
                    logger.warning("API 응답이 None입니다. 네트워크 연결을 확인해주세요.")
                    return False
                
                logger.info(f"API 응답 수신: {type(response)}")
                
                # 응답 구조 확인
                if isinstance(response, dict):
                    logger.info(f"응답 키: {list(response.keys())}")
                    
                    if 'items' in response:
                        items = response['items']
                        logger.info(f"아이템 수: {len(items) if isinstance(items, list) else 'N/A'}")
                
                self.test_results["api_test"] = True
                logger.info("API 연결 테스트 성공")
                return True
                
        except Exception as e:
            logger.error(f"API 연결 테스트 실패: {e}")
            logger.info("API 테스트는 네트워크 연결이 필요합니다.")
            return False
    
    def create_sample_data(self):
        """샘플 데이터 생성"""
        try:
            logger.info("=== 샘플 데이터 생성 ===")
            
            file_storage_config = self.config.get_file_storage_config()
            term_lists_dir = Path(file_storage_config.get("term_lists_dir", "data/base_legal_terms/raw/term_lists"))
            
            # 샘플 용어 데이터
            sample_terms = [
                {
                    "법령용어ID": "test_001",
                    "법령용어명": "계약",
                    "동음이의어존재여부": "N",
                    "비고": "",
                    "용어간관계링크": "/test/link1",
                    "조문간관계링크": "/test/link2",
                    "수집일시": datetime.now().isoformat()
                },
                {
                    "법령용어ID": "test_002",
                    "법령용어명": "손해배상",
                    "동음이의어존재여부": "Y",
                    "비고": "손해배상(損害賠償)",
                    "용어간관계링크": "/test/link3",
                    "조문간관계링크": "/test/link4",
                    "수집일시": datetime.now().isoformat()
                },
                {
                    "법령용어ID": "test_003",
                    "법령용어명": "소유권",
                    "동음이의어존재여부": "N",
                    "비고": "",
                    "용어간관계링크": "/test/link5",
                    "조문간관계링크": "/test/link6",
                    "수집일시": datetime.now().isoformat()
                }
            ]
            
            # 샘플 파일 저장
            sample_file = term_lists_dir / f"sample_terms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(sample_file, 'w', encoding='utf-8') as f:
                json.dump(sample_terms, f, ensure_ascii=False, indent=2)
            
            logger.info(f"샘플 데이터 생성: {sample_file}")
            return True
            
        except Exception as e:
            logger.error(f"샘플 데이터 생성 실패: {e}")
            return False
    
    def test_data_processing(self) -> bool:
        """데이터 처리 테스트"""
        try:
            logger.info("=== 데이터 처리 테스트 ===")
            
            # 샘플 데이터 생성
            if not self.create_sample_data():
                return False
            
            # 처리기 생성 및 테스트
            processor = BaseLegalTermProcessor(self.config)
            
            # 샘플 데이터로 처리 테스트
            sample_term = {
                "법령용어ID": "test_001",
                "법령용어명": "계약",
                "법령용어정의": "당사자 일방이 상대방에 대하여 일정한 행위를 약속하고, 상대방이 그 약속에 대하여 대가를 지급할 것을 약속하는 법률행위",
                "동음이의어내용": "",
                "용어관계정보": [],
                "조문관계정보": []
            }
            
            processed_term = processor.process_term_data(sample_term)
            
            if processed_term is None:
                logger.error("용어 데이터 처리 실패")
                return False
            
            logger.info(f"처리된 용어: {processed_term.용어명}")
            logger.info(f"정규화된 용어명: {processed_term.정규화된용어명}")
            logger.info(f"카테고리: {processed_term.카테고리}")
            logger.info(f"품질 점수: {processed_term.품질점수}")
            logger.info(f"키워드: {processed_term.키워드}")
            
            logger.info("데이터 처리 테스트 성공")
            return True
            
        except Exception as e:
            logger.error(f"데이터 처리 테스트 실패: {e}")
            return False
    
    def save_test_report(self):
        """테스트 보고서 저장"""
        try:
            report_data = {
                "테스트보고서": {
                    "테스트일시": datetime.now().isoformat(),
                    "테스트결과": self.test_results,
                    "전체성공여부": "성공" if all(self.test_results.values()) else "실패",
                    "성공한테스트": [k for k, v in self.test_results.items() if v],
                    "실패한테스트": [k for k, v in self.test_results.items() if not v]
                }
            }
            
            reports_dir = Path("data/base_legal_terms/reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = reports_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"테스트 보고서 저장: {report_file}")
            
        except Exception as e:
            logger.error(f"테스트 보고서 저장 실패: {e}")
    
    async def run_all_tests(self) -> bool:
        """모든 테스트 실행"""
        try:
            logger.info("=== 법령정보지식베이스 법령용어 수집 시스템 테스트 시작 ===")
            
            # 1. 설정 파일 테스트
            self.test_config()
            
            # 2. 파일 구조 테스트
            self.test_file_structure()
            
            # 3. 수집기 초기화 테스트
            self.test_collector_initialization()
            
            # 4. 처리기 초기화 테스트
            self.test_processor_initialization()
            
            # 5. 데이터 처리 테스트
            self.test_data_processing()
            
            # 6. API 연결 테스트 (선택적)
            try:
                await self.test_api_connection()
            except Exception as e:
                logger.warning(f"API 연결 테스트 건너뛰기: {e}")
            
            # 7. 테스트 보고서 저장
            self.save_test_report()
            
            # 결과 출력
            logger.info("=== 테스트 결과 ===")
            for test_name, result in self.test_results.items():
                status = "성공" if result else "실패"
                logger.info(f"{test_name}: {status}")
            
            all_passed = all(self.test_results.values())
            logger.info(f"전체 테스트 결과: {'성공' if all_passed else '실패'}")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"테스트 실행 중 오류: {e}")
            return False


async def main():
    """메인 실행 함수"""
    try:
        tester = BaseLegalTermSystemTester()
        success = await tester.run_all_tests()
        
        if success:
            logger.info("=== 모든 테스트 통과 ===")
        else:
            logger.error("=== 일부 테스트 실패 ===")
            
    except Exception as e:
        logger.error(f"테스트 실행 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
