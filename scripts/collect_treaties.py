#!/usr/bin/env python3
"""
조약 수집 스크립트

국가법령정보센터 LAW OPEN API를 사용하여 조약을 수집합니다.
- 주요 조약 100건 수집
- 조약 유형별 분류 및 메타데이터 정제
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
        logging.FileHandler('logs/collect_treaties.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 조약 관련 검색 키워드
TREATY_KEYWORDS = [
    # 경제통상 관련
    "자유무역협정", "FTA", "경제협력", "투자보장", "이중과세방지", "관세",
    "무역", "수출입", "경제통상", "상호원조", "경제협력협정",
    
    # 외교안보 관련
    "외교", "안보", "방위", "군사", "군사협력", "정보교환", "범죄인인도",
    "사법공조", "형사사법공조", "마약", "테러", "국제범죄",
    
    # 환경 관련
    "환경", "기후변화", "온실가스", "오존층", "생물다양성", "해양환경",
    "대기오염", "수질오염", "폐기물", "환경보호", "지속가능발전",
    
    # 인권 관련
    "인권", "인권보호", "아동권리", "여성권리", "장애인권리", "난민",
    "이주", "인신매매", "강제노동", "차별금지", "평등권",
    
    # 교육문화 관련
    "교육", "문화", "과학기술", "연구개발", "학술교류", "문화교류",
    "교육협력", "과학기술협력", "연구협력", "기술이전", "지적재산권",
    
    # 보건의료 관련
    "보건", "의료", "공중보건", "질병예방", "의료기술", "의료협력",
    "보건협력", "의료진", "의료기기", "의약품", "의료정보",
    
    # 교통통신 관련
    "교통", "통신", "항공", "해운", "육상교통", "전자통신", "정보통신",
    "항공협정", "해운협정", "교통협력", "통신협력", "디지털협력",
    
    # 농업식품 관련
    "농업", "식품", "축산", "수산", "농업협력", "식품안전", "농산물",
    "축산물", "수산물", "농업기술", "식품기술", "농업교역",
    
    # 에너지자원 관련
    "에너지", "자원", "석유", "가스", "원자력", "재생에너지", "에너지협력",
    "자원협력", "에너지안보", "자원안보", "에너지효율", "신재생에너지",
    
    # 사회보장 관련
    "사회보장", "복지", "노동", "고용", "사회보험", "국민연금", "건강보험",
    "산업재해보상보험", "고용보험", "사회보장협력", "복지협력", "노동협력"
]

# 조약 유형별 분류 키워드
TREATY_TYPE_KEYWORDS = {
    "경제통상": ["자유무역협정", "FTA", "경제협력", "투자보장", "이중과세방지", "관세", "무역"],
    "외교안보": ["외교", "안보", "방위", "군사", "군사협력", "정보교환", "범죄인인도", "사법공조"],
    "환경": ["환경", "기후변화", "온실가스", "오존층", "생물다양성", "해양환경", "대기오염"],
    "인권": ["인권", "인권보호", "아동권리", "여성권리", "장애인권리", "난민", "이주", "인신매매"],
    "교육문화": ["교육", "문화", "과학기술", "연구개발", "학술교류", "문화교류", "교육협력"],
    "보건의료": ["보건", "의료", "공중보건", "질병예방", "의료기술", "의료협력", "보건협력"],
    "교통통신": ["교통", "통신", "항공", "해운", "육상교통", "전자통신", "정보통신"],
    "농업식품": ["농업", "식품", "축산", "수산", "농업협력", "식품안전", "농산물"],
    "에너지자원": ["에너지", "자원", "석유", "가스", "원자력", "재생에너지", "에너지협력"],
    "사회보장": ["사회보장", "복지", "노동", "고용", "사회보험", "국민연금", "건강보험"]
}


class TreatyCollector:
    """조약 수집 클래스"""
    
    def __init__(self, config: LawOpenAPIConfig):
        self.client = LawOpenAPIClient(config)
        self.output_dir = Path("data/raw/treaties")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collected_treaties = set()  # 중복 방지
        
    def collect_treaties_by_keyword(self, keyword: str, max_count: int = 10) -> List[Dict[str, Any]]:
        """키워드로 조약 검색 및 수집"""
        logger.info(f"키워드 '{keyword}'로 조약 검색 시작...")
        
        treaties = []
        page = 1
        
        while len(treaties) < max_count:
            try:
                results = self.client.get_treaty_list(
                    query=keyword,
                    display=100,
                    page=page
                )
                
                if not results:
                    break
                
                for result in results:
                    treaty_id = result.get('판례일련번호')
                    if treaty_id and treaty_id not in self.collected_treaties:
                        treaties.append(result)
                        self.collected_treaties.add(treaty_id)
                        
                        if len(treaties) >= max_count:
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
        
        logger.info(f"키워드 '{keyword}'로 {len(treaties)}건 수집")
        return treaties
    
    def collect_treaties_by_date_range(self, start_date: str, end_date: str, max_count: int = 100) -> List[Dict[str, Any]]:
        """날짜 범위로 조약 검색 및 수집"""
        logger.info(f"날짜 범위 {start_date} ~ {end_date}로 조약 검색 시작...")
        
        treaties = []
        page = 1
        
        while len(treaties) < max_count:
            try:
                results = self.client.get_treaty_list(
                    display=100,
                    page=page,
                    from_date=start_date,
                    to_date=end_date
                )
                
                if not results:
                    break
                
                for result in results:
                    treaty_id = result.get('판례일련번호')
                    if treaty_id and treaty_id not in self.collected_treaties:
                        treaties.append(result)
                        self.collected_treaties.add(treaty_id)
                        
                        if len(treaties) >= max_count:
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
        
        logger.info(f"날짜 범위로 {len(treaties)}건 수집")
        return treaties
    
    def collect_treaty_details(self, treaty: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """조약 상세 정보 수집"""
        treaty_id = treaty.get('판례일련번호')
        if not treaty_id:
            return None
        
        try:
            detail = self.client.get_treaty_detail(treaty_id=treaty_id)
            if detail:
                # 기본 정보와 상세 정보 결합
                combined_data = {
                    'basic_info': treaty,
                    'detail_info': detail,
                    'collected_at': datetime.now().isoformat()
                }
                return combined_data
        except Exception as e:
            logger.error(f"조약 {treaty_id} 상세 정보 수집 실패: {e}")
        
        return None
    
    def classify_treaty_type(self, treaty: Dict[str, Any]) -> str:
        """조약 유형 분류"""
        case_name = treaty.get('사건명', '').lower()
        case_content = treaty.get('판시사항', '') + ' ' + treaty.get('판결요지', '')
        case_content = case_content.lower()
        
        # 조약 유형별 키워드 매칭
        for treaty_type, keywords in TREATY_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in case_name or keyword in case_content:
                    return treaty_type
        
        return "기타"
    
    def save_treaty_data(self, treaty_data: Dict[str, Any], filename: str):
        """조약 데이터를 파일로 저장"""
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(treaty_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"조약 데이터 저장: {filepath}")
        except Exception as e:
            logger.error(f"조약 데이터 저장 실패: {e}")
    
    def collect_all_treaties(self, target_count: int = 100):
        """모든 조약 수집"""
        logger.info(f"조약 수집 시작 (목표: {target_count}건)...")
        
        all_treaties = []
        
        # 1. 키워드별 검색 (각 키워드당 최대 5건)
        max_per_keyword = min(5, target_count // len(TREATY_KEYWORDS))
        
        for i, keyword in enumerate(TREATY_KEYWORDS):
            if len(all_treaties) >= target_count:
                break
                
            try:
                treaties = self.collect_treaties_by_keyword(keyword, max_per_keyword)
                all_treaties.extend(treaties)
                logger.info(f"키워드 '{keyword}' 완료. 누적: {len(all_treaties)}건")
                
                # API 요청 제한 확인
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 100:
                    logger.warning("API 요청 한도가 부족합니다.")
                    break
                    
            except Exception as e:
                logger.error(f"키워드 '{keyword}' 검색 실패: {e}")
                continue
        
        # 2. 날짜 범위별 검색 (최근 10년)
        if len(all_treaties) < target_count:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y%m%d')
            
            remaining_count = target_count - len(all_treaties)
            date_treaties = self.collect_treaties_by_date_range(
                start_date, end_date, remaining_count
            )
            all_treaties.extend(date_treaties)
        
        logger.info(f"총 {len(all_treaties)}건의 조약 목록 수집 완료")
        
        # 3. 각 조약의 상세 정보 수집
        detailed_treaties = []
        for i, treaty in enumerate(all_treaties):
            if i >= target_count:
                break
                
            try:
                detail = self.collect_treaty_details(treaty)
                if detail:
                    # 조약 유형 분류
                    treaty_type = self.classify_treaty_type(treaty)
                    detail['treaty_type'] = treaty_type
                    
                    detailed_treaties.append(detail)
                    
                    # 개별 파일로 저장
                    treaty_id = treaty.get('판례일련번호', f'unknown_{i}')
                    filename = f"treaty_{treaty_id}_{datetime.now().strftime('%Y%m%d')}.json"
                    self.save_treaty_data(detail, filename)
                
                # 진행률 로그
                if (i + 1) % 10 == 0:
                    logger.info(f"상세 정보 수집 진행률: {i + 1}/{len(all_treaties)}")
                
                # API 요청 제한 확인
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 10:
                    logger.warning("API 요청 한도가 거의 소진되었습니다.")
                    break
                    
            except Exception as e:
                logger.error(f"조약 {i} 상세 정보 수집 실패: {e}")
                continue
        
        logger.info(f"조약 상세 정보 수집 완료: {len(detailed_treaties)}건")
        
        # 수집 결과 요약 생성
        self.generate_collection_summary(detailed_treaties)
    
    def generate_collection_summary(self, treaties: List[Dict[str, Any]]):
        """수집 결과 요약 생성"""
        # 조약 유형별 통계
        treaty_type_stats = {}
        
        for treaty in treaties:
            treaty_type = treaty.get('treaty_type', '기타')
            treaty_type_stats[treaty_type] = treaty_type_stats.get(treaty_type, 0) + 1
        
        summary = {
            'collection_date': datetime.now().isoformat(),
            'total_treaties': len(treaties),
            'treaty_type_distribution': treaty_type_stats,
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
        logger.info("사용법: LAW_OPEN_API_OC=your_email_id python collect_treaties.py")
        return
    
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # API 설정
    config = LawOpenAPIConfig(oc=oc)
    
    # 조약 수집 실행
    collector = TreatyCollector(config)
    collector.collect_all_treaties(target_count=100)


if __name__ == "__main__":
    main()
