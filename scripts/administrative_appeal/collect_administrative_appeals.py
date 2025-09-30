#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
행정심판례 수집 스크립트

국가법령정보센터 LAW OPEN API를 사용하여 행정심판례를 수집합니다.
- 최근 3년간 행정심판례 1,000건 수집
- 심판 유형별 분류 및 메타데이터 정제
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIClient, LawOpenAPIConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/collect_administrative_appeals.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 행정심판 관련 검색 키워드
ADMINISTRATIVE_APPEAL_KEYWORDS = [
    # 행정처분 관련
    "행정처분", "허가", "인가", "신고", "신청", "이의신청", "취소처분", "정지처분",
    "과태료", "과징금", "부과처분", "징계처분", "면허취소", "허가취소",
    
    # 국세 관련
    "국세", "지방세", "세무조사", "가산세", "가산금", "체납처분", "압류처분",
    "세무서", "국세청", "지방세청", "세무조정", "세무심사",
    
    # 건축 관련
    "건축허가", "건축신고", "건축법", "건축물", "건축계획", "건축심의",
    "용도변경", "증축", "개축", "재건축", "철거명령",
    
    # 환경 관련
    "환경영향평가", "환경오염", "대기오염", "수질오염", "소음진동", "악취",
    "폐기물", "폐기물처리", "환경영향평가서", "환경영향평가심의",
    
    # 도시계획 관련
    "도시계획", "도시계획시설", "도시계획사업", "도시계획변경", "도시계획결정",
    "개발행위허가", "개발행위신고", "개발제한구역", "도시계획구역",
    
    # 교통 관련
    "교통", "교통영향평가", "교통계획", "교통시설", "도로", "교량", "터널",
    "교통사고", "교통위반", "교통정리", "교통신호",
    
    # 보건복지 관련
    "보건", "복지", "의료", "의료기관", "의료기기", "의료인", "의료법",
    "사회보장", "국민연금", "건강보험", "산업재해보상보험",
    
    # 교육 관련
    "교육", "학교", "교육기관", "교육법", "교육과정", "교육시설", "교육시설기준",
    "사립학교", "사립학교법", "교육감", "교육위원회",
    
    # 노동 관련
    "노동", "고용", "근로", "근로기준법", "산업안전보건법", "산업재해",
    "노동조합", "단체교섭", "파업", "파견근로", "기간제근로",
    
    # 금융 관련
    "금융", "금융감독", "금융기관", "금융상품", "금융거래", "금융투자",
    "은행", "보험", "증권", "자본시장", "금융투자업법"
]

# 심판 유형별 분류 키워드
APPEAL_TYPE_KEYWORDS = {
    "허가인가": ["허가", "인가", "면허", "등록", "신고"],
    "처분취소": ["처분", "취소", "정지", "철회", "무효"],
    "부과처분": ["부과", "과태료", "과징금", "가산세", "가산금"],
    "징계처분": ["징계", "해임", "파면", "정직", "감봉"],
    "세무처분": ["국세", "지방세", "세무조사", "체납처분", "압류"],
    "건축처분": ["건축", "건축허가", "건축신고", "용도변경", "철거명령"],
    "환경처분": ["환경", "환경영향평가", "환경오염", "폐기물", "소음진동"],
    "도시계획": ["도시계획", "개발행위", "개발제한구역", "도시계획시설"],
    "교통처분": ["교통", "교통영향평가", "교통사고", "교통위반", "교통정리"],
    "보건복지": ["보건", "복지", "의료", "사회보장", "국민연금"],
    "교육처분": ["교육", "학교", "교육기관", "사립학교", "교육법"],
    "노동처분": ["노동", "고용", "근로", "산업재해", "노동조합"],
    "금융처분": ["금융", "금융감독", "금융기관", "금융상품", "금융거래"]
}


class AdministrativeAppealCollector:
    """행정심판례 수집 클래스"""
    
    def __init__(self, config: LawOpenAPIConfig):
        self.client = LawOpenAPIClient(config)
        self.output_dir = Path("data/raw/administrative_appeals")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collected_appeals = set()  # 중복 방지
        
    def collect_appeals_by_keyword(self, keyword: str, max_count: int = 50) -> List[Dict[str, Any]]:
        """키워드로 행정심판례 검색 및 수집"""
        logger.info(f"키워드 '{keyword}'로 행정심판례 검색 시작...")
        
        appeals = []
        page = 1
        
        while len(appeals) < max_count:
            try:
                results = self.client.get_administrative_appeal_list(
                    query=keyword,
                    display=100,
                    page=page
                )
                
                if not results:
                    break
                
                for result in results:
                    appeal_id = result.get('판례일련번호')
                    if appeal_id and appeal_id not in self.collected_appeals:
                        appeals.append(result)
                        self.collected_appeals.add(appeal_id)
                        
                        if len(appeals) >= max_count:
                            break
                
                page += 1
                
                # API 요청 제한 확인
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 10:
                    logger.warning("API 요청 한도가 거의 소진되었습니다.")
                    break
                    
            except Exception as e:
                logger.error(f"키워드 '{keyword}' 검색 중 오류: {e}")
                break
        
        logger.info(f"키워드 '{keyword}'로 {len(appeals)}건 수집")
        return appeals
    
    def collect_appeals_by_date_range(self, start_date: str, end_date: str, max_count: int = 1000) -> List[Dict[str, Any]]:
        """날짜 범위로 행정심판례 검색 및 수집"""
        logger.info(f"날짜 범위 {start_date} ~ {end_date}로 행정심판례 검색 시작...")
        
        appeals = []
        page = 1
        
        while len(appeals) < max_count:
            try:
                results = self.client.get_administrative_appeal_list(
                    display=100,
                    page=page,
                    from_date=start_date,
                    to_date=end_date
                )
                
                if not results:
                    break
                
                for result in results:
                    appeal_id = result.get('판례일련번호')
                    if appeal_id and appeal_id not in self.collected_appeals:
                        appeals.append(result)
                        self.collected_appeals.add(appeal_id)
                        
                        if len(appeals) >= max_count:
                            break
                
                page += 1
                
                # API 요청 제한 확인
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 10:
                    logger.warning("API 요청 한도가 거의 소진되었습니다.")
                    break
                    
            except Exception as e:
                logger.error(f"날짜 범위 검색 중 오류: {e}")
                break
        
        logger.info(f"날짜 범위로 {len(appeals)}건 수집")
        return appeals
    
    def collect_appeal_details(self, appeal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """행정심판례 상세 정보 수집"""
        appeal_id = appeal.get('판례일련번호')
        if not appeal_id:
            return None
        
        try:
            detail = self.client.get_administrative_appeal_detail(appeal_id=appeal_id)
            if detail:
                # 기본 정보와 상세 정보 결합
                combined_data = {
                    'basic_info': appeal,
                    'detail_info': detail,
                    'collected_at': datetime.now().isoformat()
                }
                return combined_data
        except Exception as e:
            logger.error(f"행정심판례 {appeal_id} 상세 정보 수집 실패: {e}")
        
        return None
    
    def classify_appeal_type(self, appeal: Dict[str, Any]) -> str:
        """행정심판례 유형 분류"""
        case_name = appeal.get('사건명', '').lower()
        case_content = appeal.get('판시사항', '') + ' ' + appeal.get('판결요지', '')
        case_content = case_content.lower()
        
        # 심판 유형별 키워드 매칭
        for appeal_type, keywords in APPEAL_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in case_name or keyword in case_content:
                    return appeal_type
        
        return "기타"
    
    def save_appeal_data(self, appeal_data: Dict[str, Any], filename: str):
        """행정심판례 데이터를 파일로 저장"""
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(appeal_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"행정심판례 데이터 저장: {filepath}")
        except Exception as e:
            logger.error(f"행정심판례 데이터 저장 실패: {e}")
    
    def collect_all_appeals(self, target_count: int = 1000):
        """모든 행정심판례 수집"""
        logger.info(f"행정심판례 수집 시작 (목표: {target_count}건)...")
        
        all_appeals = []
        
        # 1. 키워드별 검색 (각 키워드당 최대 30건)
        max_per_keyword = min(30, target_count // len(ADMINISTRATIVE_APPEAL_KEYWORDS))
        
        for i, keyword in enumerate(ADMINISTRATIVE_APPEAL_KEYWORDS):
            if len(all_appeals) >= target_count:
                break
                
            try:
                appeals = self.collect_appeals_by_keyword(keyword, max_per_keyword)
                all_appeals.extend(appeals)
                logger.info(f"키워드 '{keyword}' 완료. 누적: {len(all_appeals)}건")
                
                # API 요청 제한 확인
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 100:
                    logger.warning("API 요청 한도가 부족합니다.")
                    break
                    
            except Exception as e:
                logger.error(f"키워드 '{keyword}' 검색 실패: {e}")
                continue
        
        # 2. 날짜 범위별 검색 (최근 3년)
        if len(all_appeals) < target_count:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y%m%d')
            
            remaining_count = target_count - len(all_appeals)
            date_appeals = self.collect_appeals_by_date_range(
                start_date, end_date, remaining_count
            )
            all_appeals.extend(date_appeals)
        
        logger.info(f"총 {len(all_appeals)}건의 행정심판례 목록 수집 완료")
        
        # 3. 각 행정심판례의 상세 정보 수집
        detailed_appeals = []
        for i, appeal in enumerate(all_appeals):
            if i >= target_count:
                break
                
            try:
                detail = self.collect_appeal_details(appeal)
                if detail:
                    # 심판 유형 분류
                    appeal_type = self.classify_appeal_type(appeal)
                    detail['appeal_type'] = appeal_type
                    
                    detailed_appeals.append(detail)
                    
                    # 개별 파일로 저장
                    appeal_id = appeal.get('판례일련번호', f'unknown_{i}')
                    filename = f"administrative_appeal_{appeal_id}_{datetime.now().strftime('%Y%m%d')}.json"
                    self.save_appeal_data(detail, filename)
                
                # 진행률 로그
                if (i + 1) % 100 == 0:
                    logger.info(f"상세 정보 수집 진행률: {i + 1}/{len(all_appeals)}")
                
                # API 요청 제한 확인
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 10:
                    logger.warning("API 요청 한도가 거의 소진되었습니다.")
                    break
                    
            except Exception as e:
                logger.error(f"행정심판례 {i} 상세 정보 수집 실패: {e}")
                continue
        
        logger.info(f"행정심판례 상세 정보 수집 완료: {len(detailed_appeals)}건")
        
        # 수집 결과 요약 생성
        self.generate_collection_summary(detailed_appeals)
    
    def generate_collection_summary(self, appeals: List[Dict[str, Any]]):
        """수집 결과 요약 생성"""
        # 심판 유형별 통계
        appeal_type_stats = {}
        
        for appeal in appeals:
            appeal_type = appeal.get('appeal_type', '기타')
            appeal_type_stats[appeal_type] = appeal_type_stats.get(appeal_type, 0) + 1
        
        summary = {
            'collection_date': datetime.now().isoformat(),
            'total_appeals': len(appeals),
            'appeal_type_distribution': appeal_type_stats,
            'api_stats': self.client.get_request_stats()
        }
        
        summary_file = self.output_dir / f"collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info(f"수집 결과 요약 저장: {summary_file}")
        except Exception as e:
            logger.error(f"수집 결과 요약 저장 실패: {e}")


def main():
    """메인 함수"""
    # 환경변수 확인
    oc = os.getenv("LAW_OPEN_API_OC")
    if not oc:
        logger.error("LAW_OPEN_API_OC 환경변수가 설정되지 않았습니다.")
        logger.info("사용법: LAW_OPEN_API_OC=your_email_id python collect_administrative_appeals.py")
        return
    
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # API 설정
    config = LawOpenAPIConfig(oc=oc)
    
    # 행정심판례 수집 실행
    collector = AdministrativeAppealCollector(config)
    collector.collect_all_appeals(target_count=1000)


if __name__ == "__main__":
    main()
