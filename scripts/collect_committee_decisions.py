#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
위원회결정문 수집 스크립트

국가법령정보센터 LAW OPEN API를 사용하여 위원회결정문을 수집합니다.
- 주요 위원회별 결정문 500건 수집
- 위원회별 분류 및 메타데이터 정제
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
        logging.FileHandler('logs/collect_committee_decisions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 위원회별 키워드
COMMITTEE_KEYWORDS = {
    "국정감사위원회": ["국정감사", "감사", "정부감사", "국정감사위원회"],
    "예산결산특별위원회": ["예산", "결산", "예산안", "결산안", "예산결산"],
    "법제사법위원회": ["법제", "사법", "법률", "법안", "법제사법"],
    "기획재정위원회": ["기획", "재정", "경제", "정책", "기획재정"],
    "과학기술정보통신위원회": ["과학", "기술", "정보통신", "디지털", "ICT"],
    "행정안전위원회": ["행정", "안전", "지방자치", "공무원", "행정안전"],
    "문화체육관광위원회": ["문화", "체육", "관광", "예술", "스포츠"],
    "농림축산식품해양수산위원회": ["농업", "축산", "식품", "해양", "수산"],
    "산업통상자원중소벤처기업위원회": ["산업", "통상", "자원", "중소기업", "벤처"],
    "보건복지위원회": ["보건", "복지", "의료", "건강", "사회보장"],
    "환경노동위원회": ["환경", "노동", "고용", "산업안전", "환경노동"]
}

# 위원회 코드 매핑
COMMITTEE_CODES = {
    "국정감사위원회": "audit",
    "예산결산특별위원회": "budget", 
    "법제사법위원회": "legis",
    "기획재정위원회": "plan",
    "과학기술정보통신위원회": "scitech",
    "행정안전위원회": "admin",
    "문화체육관광위원회": "culture",
    "농림축산식품해양수산위원회": "agri",
    "산업통상자원중소벤처기업위원회": "industry",
    "보건복지위원회": "welfare",
    "환경노동위원회": "envlabor"
}


class CommitteeDecisionCollector:
    """위원회결정문 수집 클래스"""
    
    def __init__(self, config: LawOpenAPIConfig):
        self.client = LawOpenAPIClient(config)
        self.output_dir = Path("data/raw/committee_decisions")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collected_decisions = set()  # 중복 방지
        
    def collect_decisions_by_committee(self, committee: str, max_count: int = 50) -> List[Dict[str, Any]]:
        """위원회별 결정문 검색 및 수집"""
        logger.info(f"위원회 '{committee}' 결정문 검색 시작...")
        
        committee_code = COMMITTEE_CODES.get(committee)
        if not committee_code:
            logger.error(f"지원하지 않는 위원회: {committee}")
            return []
        
        decisions = []
        page = 1
        
        while len(decisions) < max_count:
            try:
                results = self.client.get_committee_decision_list(
                    committee=committee_code,
                    display=100,
                    page=page
                )
                
                if not results:
                    break
                
                for result in results:
                    decision_id = result.get('판례일련번호')
                    if decision_id and decision_id not in self.collected_decisions:
                        decisions.append(result)
                        self.collected_decisions.add(decision_id)
                        
                        if len(decisions) >= max_count:
                            break
                
                page += 1
                
                # API 요청 제한 확인
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 10:
                    logger.warning("API 요청 한도가 거의 소진되었습니다.")
                    break
                    
            except Exception as e:
                logger.error(f"위원회 '{committee}' 검색 중 오류: {e}")
                break
        
        logger.info(f"위원회 '{committee}'로 {len(decisions)}건 수집")
        return decisions
    
    def collect_decision_details(self, decision: Dict[str, Any], committee: str) -> Optional[Dict[str, Any]]:
        """위원회결정문 상세 정보 수집"""
        decision_id = decision.get('판례일련번호')
        if not decision_id:
            return None
        
        committee_code = COMMITTEE_CODES.get(committee)
        if not committee_code:
            return None
        
        try:
            detail = self.client.get_committee_decision_detail(
                committee=committee_code, 
                decision_id=decision_id
            )
            if detail:
                # 기본 정보와 상세 정보 결합
                combined_data = {
                    'basic_info': decision,
                    'detail_info': detail,
                    'committee': committee,
                    'collected_at': datetime.now().isoformat()
                }
                return combined_data
        except Exception as e:
            logger.error(f"위원회결정문 {decision_id} 상세 정보 수집 실패: {e}")
        
        return None
    
    def save_decision_data(self, decision_data: Dict[str, Any], filename: str):
        """위원회결정문 데이터를 파일로 저장"""
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(decision_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"위원회결정문 데이터 저장: {filepath}")
        except Exception as e:
            logger.error(f"위원회결정문 데이터 저장 실패: {e}")
    
    def collect_all_committee_decisions(self, target_count: int = 500):
        """모든 위원회결정문 수집"""
        logger.info(f"위원회결정문 수집 시작 (목표: {target_count}건)...")
        
        all_decisions = []
        max_per_committee = target_count // len(COMMITTEE_KEYWORDS)
        
        for committee in COMMITTEE_KEYWORDS.keys():
            if len(all_decisions) >= target_count:
                break
                
            try:
                decisions = self.collect_decisions_by_committee(committee, max_per_committee)
                all_decisions.extend(decisions)
                logger.info(f"위원회 '{committee}' 완료. 누적: {len(all_decisions)}건")
                
                # API 요청 제한 확인
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 100:
                    logger.warning("API 요청 한도가 부족합니다.")
                    break
                    
            except Exception as e:
                logger.error(f"위원회 '{committee}' 검색 실패: {e}")
                continue
        
        logger.info(f"총 {len(all_decisions)}건의 위원회결정문 목록 수집 완료")
        
        # 각 위원회결정문의 상세 정보 수집
        detailed_decisions = []
        for i, decision in enumerate(all_decisions):
            if i >= target_count:
                break
                
            try:
                # 위원회 정보 추출 (기본 정보에서)
                committee = decision.get('소관부처명', '기타')
                if committee not in COMMITTEE_KEYWORDS:
                    committee = '기타'
                
                detail = self.collect_decision_details(decision, committee)
                if detail:
                    detailed_decisions.append(detail)
                    
                    # 개별 파일로 저장
                    decision_id = decision.get('판례일련번호', f'unknown_{i}')
                    filename = f"committee_decision_{decision_id}_{datetime.now().strftime('%Y%m%d')}.json"
                    self.save_decision_data(detail, filename)
                
                # 진행률 로그
                if (i + 1) % 50 == 0:
                    logger.info(f"상세 정보 수집 진행률: {i + 1}/{len(all_decisions)}")
                
                # API 요청 제한 확인
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 10:
                    logger.warning("API 요청 한도가 거의 소진되었습니다.")
                    break
                    
            except Exception as e:
                logger.error(f"위원회결정문 {i} 상세 정보 수집 실패: {e}")
                continue
        
        logger.info(f"위원회결정문 상세 정보 수집 완료: {len(detailed_decisions)}건")
        
        # 수집 결과 요약 생성
        self.generate_collection_summary(detailed_decisions)
    
    def generate_collection_summary(self, decisions: List[Dict[str, Any]]):
        """수집 결과 요약 생성"""
        # 위원회별 통계
        committee_stats = {}
        
        for decision in decisions:
            committee = decision.get('committee', '기타')
            committee_stats[committee] = committee_stats.get(committee, 0) + 1
        
        summary = {
            'collection_date': datetime.now().isoformat(),
            'total_decisions': len(decisions),
            'committee_distribution': committee_stats,
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
        logger.info("사용법: LAW_OPEN_API_OC=your_email_id python collect_committee_decisions.py")
        return
    
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # API 설정
    config = LawOpenAPIConfig(oc=oc)
    
    # 위원회결정문 수집 실행
    collector = CommitteeDecisionCollector(config)
    collector.collect_all_committee_decisions(target_count=500)


if __name__ == "__main__":
    main()
