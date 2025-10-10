#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
헌재결정례 수집기 클래스

국가법령정보센터 LAW OPEN API를 사용하여 헌재결정례를 수집합니다.
- 최근 5년간 헌재결정례 1,000건 수집
- 헌법재판소 결정례의 상세 내용 수집
- 결정유형별 분류 (위헌, 합헌, 각하, 기각 등)
- 향상된 에러 처리, 성능 최적화, 모니터링 기능
"""

import os
import sys
import json
import logging
import signal
import atexit
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIClient, LawOpenAPIConfig

# 헌법 관련 검색 키워드 (우선순위별)
CONSTITUTIONAL_KEYWORDS = [
    # 최고 우선순위 (각 100건)
    "헌법소원", "위헌법률심판", "탄핵심판", "권한쟁의심판", "정당해산심판",
    
    # 고우선순위 (각 50건)
    "생명권", "신체의 자유", "사생활의 자유", "양심의 자유", "종교의 자유",
    "언론의 자유", "출판의 자유", "집회의 자유", "결사의 자유", "재산권",
    "직업선택의 자유", "거주이전의 자유", "참정권", "교육을 받을 권리",
    "근로의 권리", "환경권", "보건권", "주거권", "문화를 향유할 권리",
    
    # 중우선순위 (각 30건)
    "학문의 자유", "예술의 자유", "생존권", "근로3권", "복지를 받을 권리",
    "법률에 의하지 아니하고는 처벌받지 아니할 권리", "무죄추정의 원칙",
    "진술거부권", "변호인의 조력을 받을 권리", "신속한 재판을 받을 권리",
    "공개재판을 받을 권리", "국회", "정부", "법원", "헌법재판소",
    "선거관리위원회", "감사원", "대통령", "국무총리", "국무위원",
    "국회의원", "대법원장", "헌법재판소장", "권리구제형 헌법소원",
    "규범통제형 헌법소원", "헌법재판소의 관할", "기본권 제한", "법률유보",
    "과잉금지의 원칙", "본질적 내용 침해 금지", "비례의 원칙",
    "명확성의 원칙", "적정성의 원칙", "국가의 의무", "기본권 보장 의무",
    "최소한의 생활 보장", "교육제도 확립", "근로조건의 기준", "환경보전",
    "문화진흥", "복지증진"
]

# 결정유형 분류 키워드
DECISION_TYPE_KEYWORDS = {
    "위헌": ["위헌", "위헌결정", "헌법에 위반"],
    "합헌": ["합헌", "합헌결정", "헌법에 합치"],
    "각하": ["각하", "각하결정", "각하판결"],
    "기각": ["기각", "기각결정", "기각판결"],
    "인용": ["인용", "인용결정", "인용판결"],
    "일부인용": ["일부인용", "일부인용결정"],
    "일부기각": ["일부기각", "일부기각결정"]
}

# 키워드별 우선순위 및 목표 건수
KEYWORD_PRIORITIES = {
    # 최고 우선순위 (100건)
    "헌법소원": 100, "위헌법률심판": 100, "탄핵심판": 100, 
    "권한쟁의심판": 100, "정당해산심판": 100,
    
    # 고우선순위 (50건)
    "생명권": 50, "신체의 자유": 50, "사생활의 자유": 50, "양심의 자유": 50,
    "종교의 자유": 50, "언론의 자유": 50, "출판의 자유": 50, "집회의 자유": 50,
    "결사의 자유": 50, "재산권": 50, "직업선택의 자유": 50, "거주이전의 자유": 50,
    "참정권": 50, "교육을 받을 권리": 50, "근로의 권리": 50, "환경권": 50,
    "보건권": 50, "주거권": 50, "문화를 향유할 권리": 50,
    
    # 중우선순위 (30건)
    "학문의 자유": 30, "예술의 자유": 30, "생존권": 30, "근로3권": 30,
    "복지를 받을 권리": 30, "법률에 의하지 아니하고는 처벌받지 아니할 권리": 30,
    "무죄추정의 원칙": 30, "진술거부권": 30, "변호인의 조력을 받을 권리": 30,
    "신속한 재판을 받을 권리": 30, "공개재판을 받을 권리": 30,
    "국회": 30, "정부": 30, "법원": 30, "헌법재판소": 30,
    "선거관리위원회": 30, "감사원": 30, "대통령": 30, "국무총리": 30,
    "국무위원": 30, "국회의원": 30, "대법원장": 30, "헌법재판소장": 30,
    "권리구제형 헌법소원": 30, "규범통제형 헌법소원": 30, "헌법재판소의 관할": 30,
    "기본권 제한": 30, "법률유보": 30, "과잉금지의 원칙": 30,
    "본질적 내용 침해 금지": 30, "비례의 원칙": 30, "명확성의 원칙": 30,
    "적정성의 원칙": 30, "국가의 의무": 30, "기본권 보장 의무": 30,
    "최소한의 생활 보장": 30, "교육제도 확립": 30, "근로조건의 기준": 30,
    "환경보전": 30, "문화진흥": 30, "복지증진": 30
}

# 기본 목표 건수 (우선순위가 없는 키워드)
DEFAULT_TARGET_COUNT = 15


class ConstitutionalDecisionCollector:
    """헌재결정례 수집 클래스"""
    
    def __init__(self, config: LawOpenAPIConfig):
        self.client = LawOpenAPIClient(config)
        self.output_dir = Path("data/raw/constitutional_decisions")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 수집 상태 관리
        self.collected_decisions = set()  # 중복 방지
        self.detailed_decisions = []
        self.current_batch = []
        self.batch_size = 50  # 배치 크기
        
        # 통계 정보
        self.stats = {
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'status': 'running',
            'target_count': 0,
            'collected_count': 0,
            'duplicate_count': 0,
            'failed_count': 0,
            'keywords_processed': 0,
            'total_keywords': len(CONSTITUTIONAL_KEYWORDS),
            'api_requests_made': 0,
            'api_errors': 0,
            'last_keyword_processed': None
        }
        
        # Graceful shutdown 관련 변수
        self.shutdown_requested = False
        self.checkpoint_file = None
        self.resume_info = {
            'progress_percentage': 0.0,
            'last_keyword_processed': None,
            'can_resume': False
        }
        
        # 시그널 핸들러 등록
        self._setup_signal_handlers()
        
        # 종료 시 정리 작업 등록
        atexit.register(self._cleanup_on_exit)
    
    def _setup_signal_handlers(self):
        """시그널 핸들러 설정"""
        def signal_handler(signum, frame):
            logger = logging.getLogger(__name__)
            logger.info(f"시그널 {signum} 수신. Graceful shutdown 시작...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # 종료 신호
    
    def _cleanup_on_exit(self):
        """프로그램 종료 시 정리 작업"""
        if self.detailed_decisions or self.current_batch:
            logger = logging.getLogger(__name__)
            logger.info("수집된 데이터를 저장 중...")
            self._save_checkpoint()
            logger.info(f"총 {len(self.detailed_decisions)}건의 데이터가 저장되었습니다.")
    
    def _save_checkpoint(self):
        """체크포인트 저장"""
        if not self.checkpoint_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.checkpoint_file = self.output_dir / f"collection_checkpoint_{timestamp}.json"
        
        # 현재 배치를 상세 데이터에 추가
        if self.current_batch:
            self.detailed_decisions.extend(self.current_batch)
            self.current_batch = []
        
        # 진행률 계산
        if self.stats['total_keywords'] > 0:
            self.resume_info['progress_percentage'] = (
                self.stats['keywords_processed'] / self.stats['total_keywords'] * 100
            )
        
        checkpoint_data = {
            'stats': self.stats,
            'resume_info': self.resume_info,
            'shutdown_info': {
                'graceful_shutdown_supported': True,
                'shutdown_requested': self.shutdown_requested,
                'shutdown_reason': 'User interrupt' if self.shutdown_requested else None
            },
            'detailed_decisions': self.detailed_decisions,
            'collected_decisions': list(self.collected_decisions)
        }
        
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            logger = logging.getLogger(__name__)
            logger.debug(f"체크포인트 저장: {self.checkpoint_file}")
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"체크포인트 저장 실패: {e}")
    
    def _load_checkpoint(self):
        """체크포인트 로드"""
        checkpoint_files = list(self.output_dir.glob("collection_checkpoint_*.json"))
        if not checkpoint_files:
            return False
        
        # 가장 최근 체크포인트 파일 로드
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # 상태 복원
            self.stats = checkpoint_data.get('stats', self.stats)
            self.resume_info = checkpoint_data.get('resume_info', self.resume_info)
            self.detailed_decisions = checkpoint_data.get('detailed_decisions', [])
            self.collected_decisions = set(checkpoint_data.get('collected_decisions', []))
            self.checkpoint_file = latest_checkpoint
            
            logger = logging.getLogger(__name__)
            logger.info(f"체크포인트 로드 완료: {len(self.detailed_decisions)}건의 상세 데이터 복원")
            return True
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"체크포인트 로드 실패: {e}")
            return False
    
    def _check_shutdown(self):
        """종료 요청 확인"""
        if self.shutdown_requested:
            logger = logging.getLogger(__name__)
            logger.info("종료 요청이 감지되었습니다. 현재 작업을 완료한 후 종료합니다.")
            return True
        return False
    
    def collect_decisions_by_keyword(self, keyword: str, max_count: int = 50) -> List[Dict[str, Any]]:
        """키워드로 헌재결정례 검색 및 수집"""
        logger = logging.getLogger(__name__)
        logger.info(f"키워드 '{keyword}'로 헌재결정례 검색 시작 (목표: {max_count}건)...")
        
        decisions = []
        page = 1
        
        while len(decisions) < max_count:
            # 종료 요청 확인
            if self._check_shutdown():
                break
                
            try:
                # 페이지별 진행률 표시
                page_progress = (page - 1) * 20
                logger.info(f"📄 페이지 {page} 요청 중... (현재 수집: {len(decisions)}/{max_count}건, 진행률: {len(decisions)/max_count*100:.1f}%)")
                
                # query가 있는 경우에만 검색, 없으면 전체 목록 조회 (선고일자 내림차순)
                if keyword and keyword.strip():
                    logger.debug(f"🔍 키워드 '{keyword}'로 검색 요청 (선고일자 내림차순)")
                    results = self.client.get_constitutional_list(
                        query=keyword,
                        display=20,  # 작은 배치 크기로 시작
                        page=page,
                        search=1  # 선고일자 내림차순 정렬 (최신순)
                    )
                else:
                    logger.debug("📋 전체 목록 조회 요청 (선고일자 내림차순)")
                    results = self.client.get_constitutional_list(
                        display=20,  # 작은 배치 크기로 시작
                        page=page,
                        search=1  # 선고일자 내림차순 정렬 (최신순)
                    )
                
                logger.info(f"📊 API 응답 결과: {len(results) if results else 0}건")
                
                if not results:
                    logger.info("더 이상 결과가 없어서 검색을 중단합니다.")
                    break
                
                new_decisions = 0
                for result in results:
                    # 종료 요청 확인
                    if self._check_shutdown():
                        break
                        
                    # 헌재결정례 ID 확인 (API 응답 구조에 따라)
                    decision_id = result.get('헌재결정례일련번호') or result.get('ID') or result.get('id')
                    if decision_id and decision_id not in self.collected_decisions:
                        decisions.append(result)
                        self.collected_decisions.add(decision_id)
                        self.stats['collected_count'] += 1
                        new_decisions += 1
                        
                        logger.info(f"✅ 새로운 헌재결정례 수집: {result.get('사건명', 'Unknown')} (ID: {decision_id})")
                        logger.info(f"   📈 현재 진행률: {len(decisions)}/{max_count}건 ({len(decisions)/max_count*100:.1f}%)")
                        
                        if len(decisions) >= max_count:
                            logger.info(f"🎯 목표 수량 {max_count}건에 도달했습니다!")
                            break
                    else:
                        self.stats['duplicate_count'] += 1
                        logger.debug(f"중복된 헌재결정례 건너뛰기: {decision_id}")
                
                logger.info(f"📄 페이지 {page} 완료: {new_decisions}건의 새로운 결정례 수집")
                logger.info(f"   📊 누적 수집: {len(decisions)}/{max_count}건 ({len(decisions)/max_count*100:.1f}%)")
                
                page += 1
                self.stats['api_requests_made'] += 1
                
                # API 요청 제한 확인
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 10:
                    logger.warning("API 요청 한도가 거의 소진되었습니다.")
                    break
                    
            except Exception as e:
                logger.error(f"키워드 '{keyword}' 검색 중 오류: {e}")
                self.stats['api_errors'] += 1
                break
        
        logger.info(f"키워드 '{keyword}'로 총 {len(decisions)}건 수집 완료")
        return decisions
    
    def collect_decisions_by_date_range(self, start_date: str, end_date: str, max_count: int = 1000) -> List[Dict[str, Any]]:
        """날짜 범위로 헌재결정례 검색 및 수집"""
        logger = logging.getLogger(__name__)
        logger.info(f"📅 날짜 범위 {start_date} ~ {end_date}로 헌재결정례 검색 시작 (목표: {max_count:,}건)...")
        
        decisions = []
        page = 1
        total_pages_estimated = max_count // 100 + 1  # 대략적인 페이지 수 추정
        
        while len(decisions) < max_count:
            # 종료 요청 확인
            if self._check_shutdown():
                break
                
            try:
                # 페이지별 진행률 표시
                progress = (page - 1) / total_pages_estimated * 100 if total_pages_estimated > 0 else 0
                logger.info(f"📄 페이지 {page} 요청 중... (현재 수집: {len(decisions):,}/{max_count:,}건, 진행률: {len(decisions)/max_count*100:.1f}%)")
                
                results = self.client.get_constitutional_list(
                    display=100,
                    page=page,
                    from_date=start_date,
                    to_date=end_date,
                    search=1  # 사건명 검색
                )
                
                logger.info(f"📊 API 응답 결과: {len(results) if results else 0}건")
                
                if not results:
                    logger.info("더 이상 결과가 없어서 검색을 중단합니다.")
                    break
                
                new_decisions = 0
                for result in results:
                    # 종료 요청 확인
                    if self._check_shutdown():
                        break
                        
                    # 헌재결정례 ID 확인 (API 응답 구조에 따라)
                    decision_id = result.get('헌재결정례일련번호') or result.get('ID') or result.get('id')
                    if decision_id and decision_id not in self.collected_decisions:
                        decisions.append(result)
                        self.collected_decisions.add(decision_id)
                        self.stats['collected_count'] += 1
                        new_decisions += 1
                        
                        # 5건마다 수집 현황 표시 (더 자주)
                        if len(decisions) % 5 == 0:
                            logger.info(f"✅ {len(decisions):,}건 수집 완료 (진행률: {len(decisions)/max_count*100:.1f}%)")
                        
                        if len(decisions) >= max_count:
                            logger.info(f"🎯 목표 수량 {max_count:,}건에 도달했습니다!")
                            break
                    else:
                        self.stats['duplicate_count'] += 1
                
                logger.info(f"📄 페이지 {page} 완료: {new_decisions}건의 새로운 결정례 수집")
                logger.info(f"   📊 누적 수집: {len(decisions):,}/{max_count:,}건 ({len(decisions)/max_count*100:.1f}%)")
                logger.info(f"   ⏱️  예상 남은 페이지: {max(0, (max_count - len(decisions)) // 100)}페이지")
                
                page += 1
                self.stats['api_requests_made'] += 1
                
                # API 요청 제한 확인
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 10:
                    logger.warning("API 요청 한도가 거의 소진되었습니다.")
                    break
                    
            except Exception as e:
                logger.error(f"날짜 범위 검색 중 오류: {e}")
                self.stats['api_errors'] += 1
                break
        
        logger.info("=" * 60)
        logger.info(f"📅 날짜 범위 수집 완료!")
        logger.info(f"📊 최종 수집 결과: {len(decisions):,}건")
        logger.info(f"📄 처리된 페이지: {page-1}페이지")
        logger.info(f"🌐 API 요청 수: {self.stats['api_requests_made']:,}회")
        logger.info(f"❌ 중복 제외: {self.stats['duplicate_count']:,}건")
        logger.info("=" * 60)
        return decisions
    
    def collect_decision_details(self, decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """헌재결정례 상세 정보 수집"""
        # 헌재결정례 ID 확인 (API 응답 구조에 따라)
        decision_id = decision.get('헌재결정례일련번호') or decision.get('ID') or decision.get('id')
        if not decision_id:
            logger = logging.getLogger(__name__)
            logger.warning(f"헌재결정례 ID를 찾을 수 없습니다: {decision}")
            return None
        
        try:
            detail = self.client.get_constitutional_detail(constitutional_id=decision_id)
            if detail:
                # 기본 정보와 상세 정보 결합
                combined_data = {
                    'basic_info': decision,
                    'detail_info': detail,
                    'collected_at': datetime.now().isoformat()
                }
                return combined_data
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"헌재결정례 {decision_id} 상세 정보 수집 실패: {e}")
            self.stats['api_errors'] += 1
        
        return None
    
    def classify_decision_type(self, decision: Dict[str, Any]) -> str:
        """헌재결정례 유형 분류"""
        case_name = decision.get('사건명', '').lower()
        decision_text = decision.get('판시사항', '') + ' ' + decision.get('판결요지', '')
        decision_text = decision_text.lower()
        
        # 결정유형별 키워드 매칭
        for decision_type, keywords in DECISION_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in case_name or keyword in decision_text:
                    return decision_type
        
        # 기본권별 분류
        if any(keyword in case_name for keyword in ["생명권", "신체의 자유"]):
            return "기본권_생명신체"
        elif any(keyword in case_name for keyword in ["사생활", "양심", "종교"]):
            return "기본권_사생활"
        elif any(keyword in case_name for keyword in ["언론", "출판", "집회", "결사"]):
            return "기본권_표현"
        elif any(keyword in case_name for keyword in ["재산권", "직업선택"]):
            return "기본권_경제"
        elif any(keyword in case_name for keyword in ["교육", "근로", "환경"]):
            return "기본권_사회"
        elif any(keyword in case_name for keyword in ["헌법소원", "위헌법률"]):
            return "헌법재판"
        else:
            return "기타"
    
    def save_batch_data(self, batch_data: List[Dict[str, Any]], category: str):
        """배치 데이터를 파일로 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"batch_{category}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        batch_info = {
            'metadata': {
                'category': category,
                'count': len(batch_data),
                'timestamp': timestamp,
                'batch_size': self.batch_size
            },
            'data': batch_data
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(batch_info, f, ensure_ascii=False, indent=2)
            logger = logging.getLogger(__name__)
            logger.debug(f"배치 데이터 저장: {filepath}")
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"배치 데이터 저장 실패: {e}")
    
    def collect_all_decisions(self, target_count: int = 1000, resume: bool = True, keyword_mode: bool = True):
        """모든 헌재결정례 수집"""
        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info(f"🚀 헌재결정례 수집 시작!")
        logger.info(f"🎯 목표 수량: {target_count:,}건")
        if keyword_mode:
            logger.info(f"📝 검색 방식: 키워드 기반 ({len(CONSTITUTIONAL_KEYWORDS)}개 키워드)")
        else:
            logger.info(f"📝 검색 방식: 전체 데이터 수집 (키워드 무관)")
        logger.info(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        self.stats['target_count'] = target_count
        
        # 기존 체크포인트 복원
        if resume:
            if self._load_checkpoint():
                logger.info(f"기존 진행 상황 복원: {len(self.detailed_decisions)}건")
                self.resume_info['can_resume'] = True
        
        all_decisions = []
        
        if keyword_mode:
            # 1. 우선순위 키워드별 검색
            logger.info(f"총 {len(CONSTITUTIONAL_KEYWORDS)}개 키워드로 검색 시작")
            
            for i, keyword in enumerate(CONSTITUTIONAL_KEYWORDS):
                # 이전에 완료된 키워드는 건너뛰기
                if resume and i < self.stats['keywords_processed']:
                    logger.info(f"키워드 '{keyword}' 건너뛰기 (이미 처리됨)")
                    continue
                    
                if len(all_decisions) >= target_count:
                    logger.info(f"목표 수량 {target_count}건에 도달하여 키워드 검색을 중단합니다.")
                    break
                
                # 종료 요청 확인
                if self._check_shutdown():
                    break
                    
                try:
                    # 키워드별 목표 건수 설정
                    max_count = KEYWORD_PRIORITIES.get(keyword, DEFAULT_TARGET_COUNT)
                    max_count = min(max_count, target_count - len(all_decisions))
                    
                    # 진행률 계산
                    progress = (i + 1) / len(CONSTITUTIONAL_KEYWORDS) * 100
                    remaining_keywords = len(CONSTITUTIONAL_KEYWORDS) - i - 1
                    
                    logger.info(f"📊 진행률: {progress:.1f}% ({i+1}/{len(CONSTITUTIONAL_KEYWORDS)}) - 키워드 '{keyword}' 처리 시작")
                    logger.info(f"🎯 목표: {max_count}건, 우선순위: {KEYWORD_PRIORITIES.get(keyword, '기본')}, 남은 키워드: {remaining_keywords}개")
                    
                    decisions = self.collect_decisions_by_keyword(keyword, max_count)
                    all_decisions.extend(decisions)
                    
                    self.stats['keywords_processed'] += 1
                    self.stats['last_keyword_processed'] = keyword
                    
                    # 상세한 완료 정보
                    completion_rate = len(all_decisions) / target_count * 100 if target_count > 0 else 0
                    logger.info(f"✅ 키워드 '{keyword}' 완료!")
                    logger.info(f"   📈 수집: {len(decisions)}건 | 누적: {len(all_decisions)}건 | 목표 대비: {completion_rate:.1f}%")
                    logger.info(f"   ⏱️  남은 키워드: {remaining_keywords}개")
                    
                    # 진행 상황 저장 (10개 키워드마다)
                    if self.stats['keywords_processed'] % 10 == 0:
                        logger.info("진행 상황을 체크포인트에 저장합니다.")
                        self._save_checkpoint()
                    
                    # API 요청 제한 확인
                    stats = self.client.get_request_stats()
                    if stats['remaining_requests'] < 100:
                        logger.warning("API 요청 한도가 부족합니다.")
                        break
                        
                except Exception as e:
                    logger.error(f"키워드 '{keyword}' 검색 실패: {e}")
                    self.stats['api_errors'] += 1
                    continue
        else:
            # 키워드 없이 전체 데이터 수집
            logger.info("🔍 키워드 없이 전체 헌재결정례 수집 시작")
            logger.info(f"📅 수집 기간: 최근 5년 (2020년 ~ 현재)")
            logger.info(f"🎯 목표 수량: {target_count:,}건")
            
            # 최근 5년간 데이터 수집
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y%m%d')
            
            logger.info(f"📊 수집 기간: {start_date} ~ {end_date}")
            all_decisions = self.collect_decisions_by_date_range(
                start_date, end_date, target_count
            )
        
        # 2. 날짜 범위별 검색 (최근 5년) - 키워드 모드에서만
        if keyword_mode and len(all_decisions) < target_count and not self._check_shutdown():
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y%m%d')
            
            remaining_count = target_count - len(all_decisions)
            date_decisions = self.collect_decisions_by_date_range(
                start_date, end_date, remaining_count
            )
            all_decisions.extend(date_decisions)
        
        logger.info("=" * 60)
        logger.info(f"📋 1단계 완료: 총 {len(all_decisions)}건의 헌재결정례 목록 수집 완료")
        logger.info("=" * 60)
        
        # 3. 각 헌재결정례의 상세 정보 수집
        logger.info(f"🔍 2단계 시작: 상세 정보 수집 ({len(all_decisions)}건)")
        for i, decision in enumerate(all_decisions):
            if i >= target_count:
                break
            
            # 종료 요청 확인
            if self._check_shutdown():
                break
                
            try:
                detail = self.collect_decision_details(decision)
                if detail:
                    # 결정유형 분류
                    decision_type = self.classify_decision_type(decision)
                    detail['decision_type'] = decision_type
                    
                    self.current_batch.append(detail)
                    
                    # 배치 크기에 도달하면 저장
                    if len(self.current_batch) >= self.batch_size:
                        self.save_batch_data(self.current_batch, f"constitutional_decisions_{i//self.batch_size}")
                        self.detailed_decisions.extend(self.current_batch)
                        self.current_batch = []
                
                # 진행률 로그 및 체크포인트 저장
                if (i + 1) % 50 == 0:  # 50건마다 체크포인트 저장
                    progress = (i + 1) / len(all_decisions) * 100
                    logger.info(f"📊 상세 정보 수집 진행률: {i + 1:,}/{len(all_decisions):,} ({progress:.1f}%)")
                    self._save_checkpoint()
                elif (i + 1) % 10 == 0:  # 10건마다 간단한 진행률 표시
                    progress = (i + 1) / len(all_decisions) * 100
                    logger.info(f"⏳ 상세 정보 수집 진행률: {i + 1:,}/{len(all_decisions):,} ({progress:.1f}%)")
                elif (i + 1) % 5 == 0:  # 5건마다 간단한 진행률 표시 (키워드 없이 수집 시)
                    progress = (i + 1) / len(all_decisions) * 100
                    logger.info(f"🔍 상세 정보 수집: {i + 1:,}/{len(all_decisions):,} ({progress:.1f}%)")
                elif (i + 1) % 2 == 0:  # 2건마다 간단한 진행률 표시 (매우 자주)
                    progress = (i + 1) / len(all_decisions) * 100
                    logger.info(f"⚡ 상세 정보 수집: {i + 1:,}/{len(all_decisions):,} ({progress:.1f}%)")
                
                # API 요청 제한 확인
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 10:
                    logger.warning("API 요청 한도가 거의 소진되었습니다.")
                    break
                    
            except Exception as e:
                logger.error(f"헌재결정례 {i} 상세 정보 수집 실패: {e}")
                self.stats['failed_count'] += 1
                continue
        
        # 마지막 배치 저장
        if self.current_batch:
            self.save_batch_data(self.current_batch, f"constitutional_decisions_final")
            self.detailed_decisions.extend(self.current_batch)
            self.current_batch = []
        
        # 종료 요청이 있었는지 확인
        if self._check_shutdown():
            logger.info("=" * 60)
            logger.info("⚠️ 사용자 요청에 의해 수집이 중단되었습니다.")
            logger.info(f"📊 현재까지 {len(self.detailed_decisions)}건의 상세 데이터가 수집되었습니다.")
            logger.info("=" * 60)
            self.stats['status'] = 'interrupted'
        else:
            logger.info("=" * 60)
            logger.info("🎉 헌재결정례 수집이 성공적으로 완료되었습니다!")
            logger.info(f"📊 최종 수집 결과: {len(self.detailed_decisions)}건")
            logger.info(f"⏰ 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 60)
            self.stats['status'] = 'completed'
        
        # 최종 통계 업데이트
        self.stats['end_time'] = datetime.now().isoformat()
        self.stats['collected_count'] = len(self.detailed_decisions)
        
        # 4. 수집 결과 요약 생성
        self.generate_collection_summary()
        
        # 완료 후 체크포인트 파일 정리
        self._cleanup_checkpoint_files()
    
    def _cleanup_checkpoint_files(self):
        """체크포인트 파일 정리"""
        try:
            if self.checkpoint_file and self.checkpoint_file.exists():
                # 완료된 경우에만 체크포인트 파일 삭제
                if self.stats['status'] == 'completed':
                    self.checkpoint_file.unlink()
                    logger = logging.getLogger(__name__)
                    logger.debug("완료 후 체크포인트 파일 삭제")
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"체크포인트 파일 정리 실패: {e}")
    
    def reset_collection(self):
        """수집 상태 초기화"""
        try:
            # 체크포인트 파일들 삭제
            checkpoint_files = list(self.output_dir.glob("collection_checkpoint_*.json"))
            for file_path in checkpoint_files:
                file_path.unlink()
            
            # 상태 초기화
            self.detailed_decisions = []
            self.current_batch = []
            self.collected_decisions = set()
            self.stats = {
                'start_time': datetime.now().isoformat(),
                'end_time': None,
                'status': 'running',
                'target_count': 0,
                'collected_count': 0,
                'duplicate_count': 0,
                'failed_count': 0,
                'keywords_processed': 0,
                'total_keywords': len(CONSTITUTIONAL_KEYWORDS),
                'api_requests_made': 0,
                'api_errors': 0,
                'last_keyword_processed': None
            }
            self.resume_info = {
                'progress_percentage': 0.0,
                'last_keyword_processed': None,
                'can_resume': False
            }
            
            logger = logging.getLogger(__name__)
            logger.info("수집 상태가 초기화되었습니다.")
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"수집 상태 초기화 실패: {e}")
    
    def get_collection_status(self):
        """현재 수집 상태 반환"""
        return {
            'stats': self.stats,
            'resume_info': self.resume_info,
            'checkpoint_file': str(self.checkpoint_file) if self.checkpoint_file else None,
            'output_directory': str(self.output_dir)
        }
    
    def generate_collection_summary(self):
        """수집 결과 요약 생성"""
        # 결정유형별 통계
        decision_type_stats = {}
        
        for decision in self.detailed_decisions:
            decision_type = decision.get('decision_type', '기타')
            decision_type_stats[decision_type] = decision_type_stats.get(decision_type, 0) + 1
        
        summary = {
            'collection_date': datetime.now().isoformat(),
            'total_decisions': len(self.detailed_decisions),
            'decision_type_distribution': decision_type_stats,
            'api_stats': self.client.get_request_stats(),
            'collection_stats': self.stats
        }
        
        summary_file = self.output_dir / f"collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger = logging.getLogger(__name__)
            logger.info(f"수집 결과 요약 저장: {summary_file}")
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"수집 결과 요약 저장 실패: {e}")
