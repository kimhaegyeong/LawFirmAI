"""
법률 용어 수집 통합 관리 서비스

이 모듈은 법률 용어 수집, 데이터베이스 업데이트, 벡터스토어 업데이트를
통합적으로 관리하는 기능을 제공합니다.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from scripts.data_collection.law_open_api.legal_terms.legal_term_collector import LegalTermCollector
from scripts.data_collection.law_open_api.legal_terms.legal_term_vector_store import LegalTermVectorStore
from source.utils.config import Config
from source.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class CollectionConfig:
    """수집 설정"""
    # API 설정
    oc_id: str = "test"
    display_count: int = 100
    max_retries: int = 3
    
    # 수집 범위
    start_page: int = 1
    end_page: Optional[int] = None
    query: str = ""
    gana: str = ""
    
    # 배치 설정
    list_batch_size: int = 10
    detail_batch_size: int = 50
    vector_batch_size: int = 100
    
    # 재시도 설정
    retry_delays: List[int] = None
    
    def __post_init__(self):
        if self.retry_delays is None:
            self.retry_delays = [180, 300, 600]  # 3분, 5분, 10분

@dataclass
class CollectionStatus:
    """수집 상태"""
    phase: str = "idle"  # idle, collecting_list, collecting_details, updating_vectors, completed, error
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    current_page: int = 0
    total_pages: int = 0
    collected_terms: int = 0
    processed_details: int = 0
    vectorized_terms: int = 0
    error_message: Optional[str] = None
    progress_percentage: float = 0.0

class LegalTermCollectionManager:
    """법률 용어 수집 통합 관리자"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # 서비스 인스턴스
        self.collector: Optional[LegalTermCollector] = None
        self.vector_store: Optional[LegalTermVectorStore] = None
        
        # 상태 관리
        self.status = CollectionStatus()
        self.status_file = Path(config.get("DATA_DIR", "data")) / "collection_status.json"
        
        # 설정
        self.collection_config = CollectionConfig()
        
        # 로깅 설정
        self.logger = logger
        
    def _load_status(self) -> CollectionStatus:
        """상태 로드"""
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return CollectionStatus(**data)
        except Exception as e:
            self.logger.warning(f"상태 로드 실패: {e}")
        
        return CollectionStatus()
    
    def _save_status(self):
        """상태 저장"""
        try:
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.status), f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"상태 저장 실패: {e}")
    
    def _update_status(self, **kwargs):
        """상태 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.status, key):
                setattr(self.status, key, value)
        
        # 진행률 계산
        if self.status.total_pages > 0:
            self.status.progress_percentage = (self.status.current_page / self.status.total_pages) * 100
        
        self._save_status()
    
    async def _initialize_services(self):
        """서비스 초기화"""
        try:
            self.collector = LegalTermCollector(self.config)
            self.vector_store = LegalTermVectorStore(self.config)
            
            # 비동기 컨텍스트 매니저로 세션 관리
            await self.collector.__aenter__()
            
            self.logger.info("서비스 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"서비스 초기화 실패: {e}")
            raise
    
    async def _cleanup_services(self):
        """서비스 정리"""
        try:
            if self.collector:
                await self.collector.__aexit__(None, None, None)
            
            self.logger.info("서비스 정리 완료")
            
        except Exception as e:
            self.logger.error(f"서비스 정리 실패: {e}")
    
    async def collect_legal_terms(self, config: Optional[CollectionConfig] = None) -> bool:
        """법률 용어 수집 실행"""
        if config:
            self.collection_config = config
        
        try:
            # 상태 초기화
            self.status = CollectionStatus()
            self.status.phase = "collecting_list"
            self.status.start_time = str(datetime.now())
            self._save_status()
            
            # 서비스 초기화
            await self._initialize_services()
            
            self.logger.info("법률 용어 수집 시작")
            
            # 1단계: 용어 목록 수집
            await self._collect_term_list()
            
            # 2단계: 용어 상세 수집
            await self._collect_term_details()
            
            # 3단계: 벡터스토어 업데이트
            await self._update_vector_store()
            
            # 완료 상태 설정
            self._update_status(
                phase="completed",
                end_time=str(datetime.now()),
                progress_percentage=100.0
            )
            
            self.logger.info("법률 용어 수집 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"법률 용어 수집 실패: {e}")
            self._update_status(
                phase="error",
                end_time=str(datetime.now()),
                error_message=str(e)
            )
            return False
            
        finally:
            await self._cleanup_services()
    
    async def _collect_term_list(self):
        """용어 목록 수집"""
        try:
            self._update_status(phase="collecting_list")
            
            await self.collector.collect_term_list(
                start_page=self.collection_config.start_page,
                end_page=self.collection_config.end_page,
                batch_size=self.collection_config.list_batch_size,
                query=self.collection_config.query,
                gana=self.collection_config.gana
            )
            
            # 수집 통계 업데이트
            stats = self.collector.get_collection_stats()
            self._update_status(
                collected_terms=stats.get("total_terms", 0),
                total_pages=stats.get("progress", {}).get("total_pages", 0)
            )
            
            self.logger.info("용어 목록 수집 완료")
            
        except Exception as e:
            self.logger.error(f"용어 목록 수집 실패: {e}")
            raise
    
    async def _collect_term_details(self):
        """용어 상세 수집"""
        try:
            self._update_status(phase="collecting_details")
            
            await self.collector.collect_term_details(
                batch_size=self.collection_config.detail_batch_size
            )
            
            # 수집 통계 업데이트
            stats = self.collector.get_collection_stats()
            self._update_status(
                processed_details=stats.get("collected_details", 0)
            )
            
            self.logger.info("용어 상세 수집 완료")
            
        except Exception as e:
            self.logger.error(f"용어 상세 수집 실패: {e}")
            raise
    
    async def collect_term_details(self, term_names: List[str] = None, batch_size: int = None) -> Dict[str, Any]:
        """법률용어 상세 정보 수집"""
        logger.info(f"법률용어 상세 수집 시작: {len(term_names) if term_names else '전체'}개")
        
        if batch_size is None:
            batch_size = self.config.get("collection", {}).get("detail_batch_size", 50)
        
        collected_details = []
        failed_terms = []
        
        try:
            async with LegalTermCollector(self.config) as collector:
                # 수집할 용어 목록 가져오기
                if not term_names:
                    term_names = await self._get_terms_for_detail_collection()
                
                logger.info(f"상세 수집 대상 용어: {len(term_names)}개")
                
                # 배치 단위로 처리
                for i in range(0, len(term_names), batch_size):
                    batch_terms = term_names[i:i + batch_size]
                    batch_details = []
                    
                    logger.info(f"배치 {i//batch_size + 1} 처리 중: {len(batch_terms)}개")
                    
                    for term_name in batch_terms:
                        try:
                            # 상세 조회
                            response = await collector.get_term_detail(term_name)
                            
                            if response:
                                # 응답 파싱
                                detail = collector._parse_term_detail_response(response, term_name)
                                
                                if detail:
                                    batch_details.append(detail)
                                    
                                    # 원본 응답 파일 저장
                                    if self.config.get("file_storage", {}).get("save_raw_responses", True):
                                        await collector.save_term_detail_response(term_name, response, str(detail.법령용어일련번호))
                                    
                                    logger.info(f"상세 수집 성공: {term_name}")
                                else:
                                    logger.warning(f"상세 파싱 실패: {term_name}")
                                    failed_terms.append({"term": term_name, "reason": "파싱 실패"})
                            else:
                                logger.warning(f"상세 API 응답 실패: {term_name}")
                                failed_terms.append({"term": term_name, "reason": "API 응답 없음"})
                            
                            # 요청 간 대기
                            delay = self.config.get("collection", {}).get("detail_collection_delay", 1)
                            await asyncio.sleep(delay)
                            
                        except Exception as e:
                            logger.error(f"상세 수집 오류 {term_name}: {e}")
                            failed_terms.append({"term": term_name, "reason": str(e)})
                    
                    # 배치 데이터베이스 저장
                    if batch_details:
                        await collector.save_term_details(batch_details)
                        collected_details.extend([asdict(detail) for detail in batch_details])
                        
                        # 배치 파일 저장
                        if self.config.get("file_storage", {}).get("save_batch_files", True):
                            await collector.save_term_details_batch(batch_details, i//batch_size + 1)
                    
                    logger.info(f"배치 {i//batch_size + 1} 완료: {len(batch_details)}개")
                
                logger.info(f"상세 수집 완료: 성공 {len(collected_details)}개, 실패 {len(failed_terms)}개")
                
                return {
                    "collected_details": collected_details,
                    "failed_terms": failed_terms,
                    "success": len(failed_terms) == 0,
                    "total_collected": len(collected_details)
                }
                
        except Exception as e:
            logger.error(f"상세 수집 프로세스 오류: {e}")
            return {
                "collected_details": collected_details,
                "failed_terms": failed_terms,
                "success": False,
                "error": str(e)
            }
    
    async def _get_terms_for_detail_collection(self) -> List[str]:
        """상세 수집할 용어 목록 가져오기"""
        try:
            async with LegalTermCollector(self.config) as collector:
                with collector.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # 아직 상세 수집되지 않은 용어들 가져오기
                    skip_existing = self.config.get("collection", {}).get("skip_existing_details", True)
                    
                    if skip_existing:
                        cursor.execute("""
                            SELECT 법령용어명 FROM legal_term_list 
                            WHERE 법령용어ID NOT IN (
                                SELECT 법령용어ID FROM legal_term_details
                            )
                            ORDER BY collected_at
                        """)
                    else:
                        cursor.execute("""
                            SELECT 법령용어명 FROM legal_term_list 
                            ORDER BY collected_at
                        """)
                    
                    terms = [row[0] for row in cursor.fetchall()]
                    logger.info(f"상세 수집 대상 용어 조회: {len(terms)}개")
                    return terms
                    
        except Exception as e:
            logger.error(f"상세 수집 대상 용어 조회 실패: {e}")
            return []
    
    async def update_vector_store(self, terms: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """벡터스토어 업데이트"""
        logger.info("벡터스토어 업데이트 시작")
        
        try:
            if not terms:
                # 데이터베이스에서 벡터화되지 않은 상세 정보 가져오기
                terms = await self._get_unvectorized_terms()
            
            if not terms:
                logger.info("벡터화할 용어가 없습니다")
                return {"success": True, "updated_count": 0}
            
            logger.info(f"벡터화 대상 용어: {len(terms)}개")
            
            # 벡터스토어 업데이트
            success = await self.vector_store.update_legal_terms_vectors(terms)
            
            if success:
                logger.info(f"벡터스토어 업데이트 완료: {len(terms)}개")
                return {
                    "success": True,
                    "updated_count": len(terms),
                    "message": f"{len(terms)}개 용어 벡터화 완료"
                }
            else:
                logger.error("벡터스토어 업데이트 실패")
                return {
                    "success": False,
                    "updated_count": 0,
                    "error": "벡터스토어 업데이트 실패"
                }
                
        except Exception as e:
            logger.error(f"벡터스토어 업데이트 오류: {e}")
            return {
                "success": False,
                "updated_count": 0,
                "error": str(e)
            }
    
    async def _get_unvectorized_terms(self) -> List[Dict[str, Any]]:
        """벡터화되지 않은 용어 목록 가져오기"""
        try:
            async with LegalTermCollector(self.config) as collector:
                with collector.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        SELECT 법령용어ID, 법령용어명_한글, 법령용어명_한자,
                               법령용어코드, 법령용어코드명, 출처, 법령용어정의
                        FROM legal_term_details 
                        WHERE vectorized_at IS NULL
                        ORDER BY collected_at
                    """)
                    
                    terms = []
                    for row in cursor.fetchall():
                        terms.append({
                            "법령용어ID": row[0],
                            "법령용어명_한글": row[1],
                            "법령용어명_한자": row[2],
                            "법령용어코드": row[3],
                            "법령용어코드명": row[4],
                            "출처": row[5],
                            "법령용어정의": row[6]
                        })
                    
                    logger.info(f"벡터화 대상 용어 조회: {len(terms)}개")
                    return terms
                    
        except Exception as e:
            logger.error(f"벡터화 대상 용어 조회 실패: {e}")
            return []
    
    def resume_collection(self, from_page: int) -> bool:
        """수집 재개"""
        try:
            self.logger.info(f"수집 재개: 페이지 {from_page}부터")
            
            # 설정 업데이트
            self.collection_config.start_page = from_page
            
            # 상태 로드
            self.status = self._load_status()
            
            # 비동기 실행
            asyncio.create_task(self.collect_legal_terms())
            
            return True
            
        except Exception as e:
            self.logger.error(f"수집 재개 실패: {e}")
            return False
    
    def stop_collection(self) -> bool:
        """수집 중지"""
        try:
            self.logger.info("수집 중지 요청")
            
            # 현재 상태를 중지로 설정
            self._update_status(phase="stopped")
            
            return True
            
        except Exception as e:
            self.logger.error(f"수집 중지 실패: {e}")
            return False
    
    def get_collection_status(self) -> Dict[str, Any]:
        """수집 상태 조회"""
        try:
            # 현재 상태 로드
            current_status = self._load_status()
            
            # 추가 통계 정보 수집
            stats = {
                "status": asdict(current_status),
                "config": asdict(self.collection_config)
            }
            
            # 수집기 통계 (가능한 경우)
            if self.collector:
                try:
                    collector_stats = self.collector.get_collection_stats()
                    stats["collector_stats"] = collector_stats
                except:
                    pass
            
            # 벡터스토어 통계 (가능한 경우)
            if self.vector_store:
                try:
                    vector_stats = self.vector_store.get_vector_store_stats()
                    stats["vector_stats"] = vector_stats
                except:
                    pass
            
            return stats
            
        except Exception as e:
            self.logger.error(f"수집 상태 조회 실패: {e}")
            return {"error": str(e)}
    
    def reset_collection(self) -> bool:
        """수집 상태 초기화"""
        try:
            self.logger.info("수집 상태 초기화")
            
            # 상태 파일 삭제
            if self.status_file.exists():
                self.status_file.unlink()
            
            # 수집기 진행 상황 초기화
            if self.collector:
                self.collector.reset_collection_progress()
            
            # 벡터스토어 재구성 (선택사항)
            if self.vector_store:
                self.vector_store.rebuild_vector_store()
            
            # 상태 초기화
            self.status = CollectionStatus()
            
            self.logger.info("수집 상태 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"수집 상태 초기화 실패: {e}")
            return False
    
    def search_legal_terms(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """법률 용어 검색"""
        try:
            if not self.vector_store:
                self.vector_store = LegalTermVectorStore(self.config)
            
            return self.vector_store.search_similar_terms(query, top_k)
            
        except Exception as e:
            self.logger.error(f"법률 용어 검색 실패: {e}")
            return []
    
    def get_collection_report(self) -> Dict[str, Any]:
        """수집 보고서 생성"""
        try:
            report = {
                "collection_summary": {
                    "status": self.status.phase,
                    "start_time": self.status.start_time,
                    "end_time": self.status.end_time,
                    "duration": self._calculate_duration(),
                    "progress": self.status.progress_percentage
                },
                "data_statistics": {
                    "collected_terms": self.status.collected_terms,
                    "processed_details": self.status.processed_details,
                    "vectorized_terms": self.status.vectorized_terms
                },
                "configuration": asdict(self.collection_config),
                "error_info": {
                    "has_error": self.status.phase == "error",
                    "error_message": self.status.error_message
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"수집 보고서 생성 실패: {e}")
            return {"error": str(e)}
    
    def _calculate_duration(self) -> Optional[str]:
        """수집 소요 시간 계산"""
        try:
            if self.status.start_time and self.status.end_time:
                start = datetime.fromisoformat(self.status.start_time)
                end = datetime.fromisoformat(self.status.end_time)
                duration = end - start
                return str(duration)
        except:
            pass
        return None


# CLI 인터페이스
class LegalTermCollectionCLI:
    """법률 용어 수집 CLI 인터페이스"""
    
    def __init__(self, config: Config):
        self.manager = LegalTermCollectionManager(config)
        self.config = config
    
    async def run_interactive(self):
        """대화형 모드 실행"""
        print("=== 법률 용어 수집 시스템 ===")
        print("1. 전체 수집 실행")
        print("2. 용어 목록만 수집")
        print("3. 용어 상세만 수집")
        print("4. 벡터스토어만 업데이트")
        print("5. 수집 상태 조회")
        print("6. 수집 재개")
        print("7. 수집 중지")
        print("8. 수집 상태 초기화")
        print("9. 용어 검색")
        print("0. 종료")
        
        while True:
            try:
                choice = input("\n선택하세요 (0-9): ").strip()
                
                if choice == "0":
                    print("프로그램을 종료합니다.")
                    break
                elif choice == "1":
                    await self._run_full_collection()
                elif choice == "2":
                    await self._run_list_collection()
                elif choice == "3":
                    await self._run_detail_collection()
                elif choice == "4":
                    await self._run_vector_update()
                elif choice == "5":
                    self._show_status()
                elif choice == "6":
                    self._resume_collection()
                elif choice == "7":
                    self._stop_collection()
                elif choice == "8":
                    self._reset_collection()
                elif choice == "9":
                    self._search_terms()
                else:
                    print("잘못된 선택입니다.")
                    
            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"오류 발생: {e}")
    
    async def _run_full_collection(self):
        """전체 수집 실행"""
        print("전체 수집을 시작합니다...")
        
        # 설정 입력
        config = self._get_collection_config()
        
        success = await self.manager.collect_legal_terms(config)
        
        if success:
            print("전체 수집이 완료되었습니다.")
        else:
            print("전체 수집 중 오류가 발생했습니다.")
    
    async def _run_list_collection(self):
        """용어 목록 수집"""
        print("용어 목록 수집을 시작합니다...")
        
        config = self._get_collection_config()
        
        try:
            await self.manager._initialize_services()
            await self.manager._collect_term_list()
            print("용어 목록 수집이 완료되었습니다.")
        except Exception as e:
            print(f"용어 목록 수집 중 오류: {e}")
        finally:
            await self.manager._cleanup_services()
    
    async def _run_detail_collection(self):
        """용어 상세 수집"""
        print("용어 상세 수집을 시작합니다...")
        
        try:
            await self.manager._initialize_services()
            await self.manager._collect_term_details()
            print("용어 상세 수집이 완료되었습니다.")
        except Exception as e:
            print(f"용어 상세 수집 중 오류: {e}")
        finally:
            await self.manager._cleanup_services()
    
    async def _run_vector_update(self):
        """벡터스토어 업데이트"""
        print("벡터스토어 업데이트를 시작합니다...")
        
        try:
            await self.manager._initialize_services()
            await self.manager._update_vector_store()
            print("벡터스토어 업데이트가 완료되었습니다.")
        except Exception as e:
            print(f"벡터스토어 업데이트 중 오류: {e}")
        finally:
            await self.manager._cleanup_services()
    
    def _show_status(self):
        """상태 조회"""
        status = self.manager.get_collection_status()
        print("\n=== 수집 상태 ===")
        print(json.dumps(status, ensure_ascii=False, indent=2))
    
    def _resume_collection(self):
        """수집 재개"""
        try:
            page = int(input("재개할 페이지 번호를 입력하세요: "))
            success = self.manager.resume_collection(page)
            
            if success:
                print(f"페이지 {page}부터 수집을 재개합니다.")
            else:
                print("수집 재개에 실패했습니다.")
        except ValueError:
            print("올바른 페이지 번호를 입력하세요.")
    
    def _stop_collection(self):
        """수집 중지"""
        success = self.manager.stop_collection()
        
        if success:
            print("수집 중지 요청이 전송되었습니다.")
        else:
            print("수집 중지에 실패했습니다.")
    
    def _reset_collection(self):
        """수집 상태 초기화"""
        confirm = input("수집 상태를 초기화하시겠습니까? (y/N): ").strip().lower()
        
        if confirm == 'y':
            success = self.manager.reset_collection()
            
            if success:
                print("수집 상태가 초기화되었습니다.")
            else:
                print("수집 상태 초기화에 실패했습니다.")
        else:
            print("초기화가 취소되었습니다.")
    
    def _search_terms(self):
        """용어 검색"""
        query = input("검색할 용어를 입력하세요: ").strip()
        
        if query:
            results = self.manager.search_legal_terms(query, top_k=5)
            
            if results:
                print(f"\n'{query}' 검색 결과:")
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['법령용어명']} (유사도: {result['similarity_score']:.3f})")
                    print(f"   정의: {result['법령용어정의'][:100]}...")
            else:
                print("검색 결과가 없습니다.")
    
    def _get_collection_config(self) -> CollectionConfig:
        """수집 설정 입력"""
        config = CollectionConfig()
        
        try:
            start_page = input(f"시작 페이지 (기본값: {config.start_page}): ").strip()
            if start_page:
                config.start_page = int(start_page)
            
            end_page = input(f"종료 페이지 (기본값: 무제한): ").strip()
            if end_page:
                config.end_page = int(end_page)
            
            query = input("검색 쿼리 (기본값: 빈 문자열): ").strip()
            if query:
                config.query = query
            
            gana = input("가나다 검색 (기본값: 빈 문자열): ").strip()
            if gana:
                config.gana = gana
                
        except ValueError:
            print("올바른 값을 입력하세요. 기본값을 사용합니다.")
        
        return config


# 사용 예시
async def main():
    """메인 실행 함수"""
    config = Config()
    
    # CLI 모드 실행
    cli = LegalTermCollectionCLI(config)
    await cli.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
