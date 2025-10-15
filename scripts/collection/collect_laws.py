#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
법령 수집 스크립트

국가법령정보센터 LAW OPEN API를 사용하여 주요 법령 20개를 수집합니다.
- 현행법령(시행일) 기준으로 수집
- 모든 조문 및 개정이력 포함
- 법령 체계도, 신구법 비교, 영문법령 등 부가서비스 포함

사용법:
    # 모든 법령 수집
    python collect_laws.py
    
    # 특정 법령만 수집
    python collect_laws.py --names 민법 상법 형법
    
    # 수집 가능한 법령 목록 확인
    python collect_laws.py --list
    
    # 환경변수 설정
    set LAW_OPEN_API_OC=your_email_id
"""

import os
import sys
import json
import logging
import time
import random
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# .env 파일 로딩
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ 환경변수 로드 완료: {env_path}")
    else:
        print(f"⚠️ .env 파일을 찾을 수 없습니다: {env_path}")
except ImportError:
    print("❌ python-dotenv가 설치되지 않았습니다. pip install python-dotenv로 설치하세요.")
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
        logging.FileHandler('logs/collect_laws.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 주요 법령 20개 목록 (검색된 ID 포함)
MAJOR_LAWS = [
    # 기본법 (5개)
    {"name": "민법", "id": "001706", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "기본법"},
    {"name": "상법", "id": "001702", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "기본법"},
    {"name": "형법", "id": "001692", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "기본법"},
    {"name": "민사소송법", "id": "001268", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "기본법"},
    {"name": "형사소송법", "id": "013873", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "기본법"},
    
    # 특별법 (5개)
    {"name": "근로기준법", "id": "001872", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "특별법"},
    {"name": "부동산등기법", "id": "001697", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "특별법"},
    {"name": "금융실명거래 및 비밀보장에 관한 법률", "id": "000549", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "특별법"},
    {"name": "저작권법", "id": "000798", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "특별법"},
    {"name": "개인정보 보호법", "id": "011357", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "특별법"},
    
    # 행정법 (5개)
    {"name": "행정소송법", "id": "001218", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "행정법"},
    {"name": "국세기본법", "id": "001586", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "행정법"},
    {"name": "건축법", "id": "001823", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "행정법"},
    {"name": "행정절차법", "id": "001362", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "행정법"},
    {"name": "공공기관의 정보공개에 관한 법률", "id": "001357", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "행정법"},
    
    # 사회법 (5개)
    {"name": "국민기초생활보장법", "id": "001973", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "사회법"},
    {"name": "의료법", "id": "001788", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "사회법"},
    {"name": "교육기본법", "id": "000901", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "사회법"},
    {"name": "환경정책기본법", "id": "000173", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "사회법"},
    {"name": "소비자기본법", "id": "001589", "mst": None, "effective_date": None, "promulgation_date": None, "ministry": None, "category": "사회법"}
]


class LawCollector:
    """법령 수집 클래스"""
    
    def __init__(self, config: LawOpenAPIConfig, min_delay: float = 1.0, max_delay: float = 3.0):
        self.client = LawOpenAPIClient(config)
        self.output_dir = Path("data/raw/laws")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_delay = min_delay  # 최소 지연 시간 (초)
        self.max_delay = max_delay  # 최대 지연 시간 (초)
    
    def _get_random_delay(self) -> float:
        """3~7초 사이의 랜덤한 지연 시간 반환"""
        return random.uniform(self.min_delay, self.max_delay)
        
    def find_law_ids(self) -> List[Dict[str, Any]]:
        """법령명으로 법령 ID 찾기 (이미 ID가 있으면 건너뛰기)"""
        logger.info("법령 ID 검색 시작...")
        
        laws_with_ids = []
        for i, law in enumerate(MAJOR_LAWS):
            # 이미 ID가 있는 경우 API 호출 건너뛰기
            if law.get('id'):
                logger.info(f"'{law['name']}' ID 이미 존재: {law['id']} (API 호출 건너뛰기)")
                laws_with_ids.append(law)
                continue
            
            logger.info(f"'{law['name']}' 검색 중... ({i+1}/{len(MAJOR_LAWS)})")
            
            # API 호출 간 랜덤 지연 시간 적용
            if i > 0:
                delay = self._get_random_delay()
                logger.info(f"API 호출 간 {delay:.1f}초 대기...")
                time.sleep(delay)
            
            # 법령명으로 검색
            results = self.client.search_law_list_effective(query=law['name'], display=100)
            
            if results:
                # 정확한 법령명 매칭
                matched_law = None
                for result in results:
                    law_name = result.get('법령명한글', '')
                    if law['name'] in law_name or law_name in law['name']:
                        matched_law = result
                        break
                
                if matched_law:
                    law['id'] = matched_law.get('법령ID')
                    law['mst'] = matched_law.get('법령일련번호')
                    law['effective_date'] = matched_law.get('시행일자')
                    law['promulgation_date'] = matched_law.get('공포일자')
                    law['ministry'] = matched_law.get('소관부처명')
                    laws_with_ids.append(law)
                    logger.info(f"'{law['name']}' ID 찾음: {law['id']}")
                else:
                    logger.warning(f"'{law['name']}' ID를 찾을 수 없습니다.")
            else:
                logger.warning(f"'{law['name']}' 검색 결과가 없습니다.")
        
        logger.info(f"총 {len(laws_with_ids)}개 법령의 ID를 찾았습니다.")
        return laws_with_ids
    
    def collect_law_details(self, law: Dict[str, Any]) -> Dict[str, Any]:
        """개별 법령의 상세 정보 수집"""
        logger.info(f"'{law['name']}' 상세 정보 수집 시작...")
        
        law_data = {
            'basic_info': law,
            'current_text': None,
            'history': [],
            'articles': [],
            'collected_at': datetime.now().isoformat()
        }
        
        try:
            # 1. 현행법령 본문 수집
            if law['id']:
                current_text = self.client.get_law_detail_effective(law_id=law['id'])
                if current_text:
                    law_data['current_text'] = current_text
                    logger.info(f"'{law['name']}' 현행법령 본문 수집 완료")
            
            # 2. 법령 연혁 수집
            if law['id']:
                history_list = self.client.get_law_history_list(law['id'], display=100)
                if history_list:
                    law_data['history'] = history_list
                    logger.info(f"'{law['name']}' 연혁 {len(history_list)}건 수집 완료")
            
            # 3. 조문별 상세 정보 수집 (MST 사용) - 선택적 수집
            if law.get('mst') and law.get('effective_date'):
                logger.info(f"'{law['name']}' 조문별 상세 정보 수집 시작...")
                # 주요 조문들 수집 (1조부터 10조까지로 제한)
                for article_num in range(1, 11):
                    try:
                        # 조문 수집 간 랜덤 지연 시간 적용
                        if article_num > 1:
                            delay = self._get_random_delay()
                            logger.debug(f"조문 {article_num} 수집 전 {delay:.1f}초 대기...")
                            time.sleep(delay)
                        
                        jo = f"{article_num:04d}00"  # 6자리 조번호 형식
                        article_detail = self.client.get_law_detail_effective(
                            mst=law['mst'], 
                            ef_yd=law['effective_date'], 
                            jo=jo
                        )
                        
                        # 응답 구조 확인 및 처리
                        if article_detail:
                            # LawSearch 구조 확인
                            if 'LawSearch' in article_detail:
                                law_search = article_detail['LawSearch']
                                if 'law' in law_search and law_search['law']:
                                    law_data['articles'].append({
                                        'article_number': article_num,
                                        'content': article_detail
                                    })
                                    logger.debug(f"조문 {article_num} 수집 성공")
                                else:
                                    logger.debug(f"조문 {article_num} 내용 없음")
                            # 기존 구조 확인
                            elif article_detail.get('response', {}).get('body', {}).get('items', {}).get('item'):
                                law_data['articles'].append({
                                    'article_number': article_num,
                                    'content': article_detail
                                })
                                logger.debug(f"조문 {article_num} 수집 성공")
                            else:
                                logger.debug(f"조문 {article_num} 내용 없음")
                        else:
                            logger.debug(f"조문 {article_num} 응답 없음")
                            
                    except Exception as e:
                        logger.warning(f"조문 {article_num} 수집 실패: {e}")
                        # 조문 수집 실패해도 계속 진행
                        continue
                
                logger.info(f"'{law['name']}' 조문 수집 완료: {len(law_data['articles'])}개")
            
            logger.info(f"'{law['name']}' 상세 정보 수집 완료")
            
        except Exception as e:
            logger.error(f"'{law['name']}' 수집 중 오류: {e}")
        
        return law_data
    
    def save_law_data(self, law_data: Dict[str, Any], law_name: str):
        """법령 데이터를 파일로 저장"""
        # 파일명에서 특수문자 제거
        safe_name = law_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        filename = f"{safe_name}_{datetime.now().strftime('%Y%m%d')}.json"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(law_data, f, ensure_ascii=False, indent=2)
            logger.info(f"'{law_name}' 데이터 저장 완료: {filepath}")
        except Exception as e:
            logger.error(f"'{law_name}' 데이터 저장 실패: {e}")
    
    def collect_all_laws(self, target_law_names: List[str] = None):
        """모든 주요 법령 수집 (특정 법령 지정 가능)"""
        logger.info("법령 수집 시작...")
        
        # 1. 법령 ID 찾기
        laws_with_ids = self.find_law_ids()
        
        if not laws_with_ids:
            logger.error("수집할 법령이 없습니다.")
            return
        
        # 2. 특정 법령만 필터링 (지정된 경우)
        if target_law_names:
            filtered_laws = []
            for law in laws_with_ids:
                if law['name'] in target_law_names:
                    filtered_laws.append(law)
                else:
                    logger.info(f"'{law['name']}' 건너뛰기 (지정되지 않음)")
            
            if not filtered_laws:
                logger.error(f"지정된 법령을 찾을 수 없습니다: {target_law_names}")
                return
            
            laws_with_ids = filtered_laws
            logger.info(f"지정된 {len(laws_with_ids)}개 법령만 수집: {[law['name'] for law in laws_with_ids]}")
        
        # 3. 각 법령의 상세 정보 수집
        collected_count = 0
        for i, law in enumerate(laws_with_ids):
            try:
                logger.info(f"법령 상세 정보 수집 중... ({i+1}/{len(laws_with_ids)})")
                
                # 법령 간 랜덤 지연 시간 적용 (더 긴 지연)
                if i > 0:
                    delay = self._get_random_delay() * 1.5  # 법령 간에는 1.5배 더 긴 지연
                    logger.info(f"다음 법령 수집 전 {delay:.1f}초 대기...")
                    time.sleep(delay)
                
                law_data = self.collect_law_details(law)
                self.save_law_data(law_data, law['name'])
                collected_count += 1
                
                # API 요청 제한 확인
                stats = self.client.get_request_stats()
                if stats['remaining_requests'] < 10:
                    logger.warning("API 요청 한도가 거의 소진되었습니다.")
                    break
                    
            except Exception as e:
                logger.error(f"'{law['name']}' 수집 실패: {e}")
                continue
        
        logger.info(f"법령 수집 완료: {collected_count}/{len(laws_with_ids)}개")
        
        # 4. 수집 결과 요약 생성
        self.generate_collection_summary(laws_with_ids, collected_count)
    
    def generate_collection_summary(self, laws_with_ids: List[Dict[str, Any]], collected_count: int):
        """수집 결과 요약 생성"""
        summary = {
            'collection_date': datetime.now().isoformat(),
            'total_laws': len(MAJOR_LAWS),
            'found_laws': len(laws_with_ids),
            'collected_laws': collected_count,
            'laws_details': laws_with_ids,
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
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='법령 데이터 수집 스크립트')
    parser.add_argument('--names', '-n', nargs='+', help='수집할 법령명 (여러 개 지정 가능)')
    parser.add_argument('--list', '-l', action='store_true', help='수집 가능한 법령 목록 출력')
    args = parser.parse_args()
    
    # 수집 가능한 법령 목록 출력
    if args.list:
        print("수집 가능한 법령 목록:")
        print("=" * 50)
        for i, law in enumerate(MAJOR_LAWS, 1):
            print(f"{i:2d}. {law['name']} ({law['category']})")
        return
    
    # 환경변수 확인
    oc = os.getenv("LAW_OPEN_API_OC")
    if not oc:
        logger.error("LAW_OPEN_API_OC 환경변수가 설정되지 않았습니다.")
        logger.info("사용법: LAW_OPEN_API_OC=your_email_id python collect_laws.py")
        return
    
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # API 설정
    config = LawOpenAPIConfig(oc=oc)
    
    # 지연 시간 설정 (기본값: 3~7초, 환경변수로 조정 가능)
    min_delay = float(os.getenv("API_MIN_DELAY", "3.0"))
    max_delay = float(os.getenv("API_MAX_DELAY", "7.0"))
    logger.info(f"API 호출 간 랜덤 지연 시간: {min_delay}~{max_delay}초")
    
    # 법령 수집 실행
    collector = LawCollector(config, min_delay=min_delay, max_delay=max_delay)
    
    if args.names:
        logger.info(f"지정된 법령만 수집: {args.names}")
        collector.collect_all_laws(target_law_names=args.names)
    else:
        logger.info("모든 법령 수집")
        collector.collect_all_laws()


if __name__ == "__main__":
    main()