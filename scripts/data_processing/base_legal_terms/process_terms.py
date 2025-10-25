"""
법령정보지식베이스 법령용어 데이터 처리 및 정제 파이프라인

이 모듈은 수집된 법령용어 데이터를 정제, 정규화, 검증하고
벡터 임베딩을 생성하는 기능을 제공합니다.
"""

import json
import logging
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import re
import sys
import os
from datetime import datetime
import hashlib

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# 설정 파일 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'base_legal_terms', 'config'))
from base_legal_term_collection_config import BaseLegalTermCollectionConfig as Config

# 로거 설정
from source.utils.logger import setup_logging, get_logger

# 로거 초기화
logger = get_logger(__name__)

@dataclass
class ProcessedLegalTerm:
    """처리된 법령용어"""
    원본ID: str
    용어명: str
    용어정의: str
    동음이의어: str
    용어관계: List[Dict]
    조문관계: List[Dict]
    정규화된용어명: str
    키워드: List[str]
    카테고리: str
    품질점수: float
    처리일시: str

class BaseLegalTermProcessor:
    """법령정보지식베이스 법령용어 데이터 처리기"""
    
    def __init__(self, config: Config):
        self.config = config
        self.file_storage_config = config.get_file_storage_config()
        self.processing_config = config.get_processing_config()
        
        # 디렉토리 설정
        self.base_dir = Path(self.file_storage_config.get("base_dir", "data/base_legal_terms"))
        self.raw_dir = Path(self.file_storage_config.get("raw_data_dir", "data/base_legal_terms/raw"))
        self.processed_dir = Path(self.file_storage_config.get("processed_data_dir", "data/base_legal_terms/processed"))
        
        # 세부 디렉토리
        self.cleaned_dir = Path(self.file_storage_config.get("cleaned_terms_dir", "data/base_legal_terms/processed/cleaned_terms"))
        self.normalized_dir = Path(self.file_storage_config.get("normalized_terms_dir", "data/base_legal_terms/processed/normalized_terms"))
        self.validated_dir = Path(self.file_storage_config.get("validated_terms_dir", "data/base_legal_terms/processed/validated_terms"))
        self.integrated_dir = Path(self.file_storage_config.get("integrated_terms_dir", "data/base_legal_terms/processed/integrated_terms"))
        
        # 디렉토리 생성
        self._create_directories()
        
        # 처리 통계
        self.stats = {
            "total_processed": 0,
            "cleaned": 0,
            "normalized": 0,
            "validated": 0,
            "integrated": 0,
            "errors": 0
        }
    
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            self.processed_dir,
            self.cleaned_dir,
            self.normalized_dir,
            self.validated_dir,
            self.integrated_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def clean_term_data(self, term_data: Dict) -> Optional[Dict]:
        """용어 데이터 정제"""
        try:
            # 필수 필드 확인
            required_fields = ['법령용어ID', '법령용어명', '법령용어정의']
            for field in required_fields:
                if field not in term_data or not term_data[field]:
                    logger.warning(f"필수 필드 누락: {field}")
                    return None
            
            # 텍스트 정제
            cleaned_data = {}
            for key, value in term_data.items():
                if isinstance(value, str):
                    # 공백 정리
                    cleaned_value = re.sub(r'\s+', ' ', value.strip())
                    # 특수문자 정리
                    cleaned_value = re.sub(r'[^\w\s가-힣()]', '', cleaned_value)
                    cleaned_data[key] = cleaned_value
                else:
                    cleaned_data[key] = value
            
            # 용어명 길이 검증
            term_name = cleaned_data.get('법령용어명', '')
            min_length = self.processing_config.get('min_term_length', 2)
            max_length = self.processing_config.get('max_term_length', 100)
            
            if len(term_name) < min_length or len(term_name) > max_length:
                logger.warning(f"용어명 길이 부적절: {term_name} ({len(term_name)}자)")
                return None
            
            # 정의 길이 검증
            definition = cleaned_data.get('법령용어정의', '')
            if len(definition) < 10:  # 최소 정의 길이
                logger.warning(f"정의가 너무 짧음: {definition}")
                return None
            
            cleaned_data['정제일시'] = datetime.now().isoformat()
            self.stats['cleaned'] += 1
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"데이터 정제 실패: {e}")
            self.stats['errors'] += 1
            return None
    
    def normalize_term_name(self, term_name: str) -> str:
        """용어명 정규화"""
        try:
            # 기본 정규화
            normalized = term_name.strip()
            
            # 괄호 내용 정리
            normalized = re.sub(r'\([^)]*\)', '', normalized)
            
            # 숫자 패턴 정리
            normalized = re.sub(r'\d+', '', normalized)
            
            # 공백 정리
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            
            # 소문자 변환 (영문인 경우)
            normalized = normalized.lower()
            
            return normalized
            
        except Exception as e:
            logger.error(f"용어명 정규화 실패: {e}")
            return term_name
    
    def extract_keywords(self, term_name: str, definition: str) -> List[str]:
        """키워드 추출"""
        try:
            keywords = []
            
            # 용어명에서 키워드 추출
            words = re.findall(r'[가-힣]+', term_name)
            keywords.extend(words)
            
            # 정의에서 중요한 키워드 추출
            definition_words = re.findall(r'[가-힣]{2,}', definition)
            
            # 빈도 기반 키워드 선택
            word_freq = {}
            for word in definition_words:
                if len(word) >= 2:  # 최소 2글자
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # 상위 키워드 선택
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            for word, freq in sorted_words[:10]:  # 상위 10개
                if word not in keywords:
                    keywords.append(word)
            
            return keywords[:20]  # 최대 20개
            
        except Exception as e:
            logger.error(f"키워드 추출 실패: {e}")
            return []
    
    def categorize_term(self, term_name: str, definition: str) -> str:
        """용어 카테고리 분류"""
        try:
            # 법률 분야별 키워드 매핑
            categories = {
                "민사법": ["계약", "손해", "배상", "소유", "물권", "채권", "가족", "상속"],
                "형사법": ["범죄", "형벌", "처벌", "구금", "수사", "기소", "재판"],
                "행정법": ["행정", "허가", "인가", "신고", "신청", "처분", "행정행위"],
                "상법": ["회사", "주식", "이사", "주주", "상장", "합병", "분할"],
                "노동법": ["근로", "임금", "근로자", "사용자", "노동조합", "파업"],
                "지적재산권법": ["특허", "상표", "저작권", "디자인", "영업비밀"],
                "환경법": ["환경", "오염", "폐기물", "대기", "수질", "토양"],
                "의료법": ["의료", "의사", "병원", "진료", "처방", "의료기관"],
                "교육법": ["교육", "학교", "교사", "학생", "교육과정", "교육시설"],
                "건설법": ["건설", "건축", "공사", "시공", "건축물", "건축법"]
            }
            
            # 용어명과 정의에서 카테고리 키워드 검색
            text = f"{term_name} {definition}"
            category_scores = {}
            
            for category, keywords in categories.items():
                score = 0
                for keyword in keywords:
                    if keyword in text:
                        score += 1
                category_scores[category] = score
            
            # 가장 높은 점수의 카테고리 반환
            if category_scores:
                best_category = max(category_scores.items(), key=lambda x: x[1])
                if best_category[1] > 0:
                    return best_category[0]
            
            return "기타"
            
        except Exception as e:
            logger.error(f"카테고리 분류 실패: {e}")
            return "기타"
    
    def calculate_quality_score(self, term_data: Dict) -> float:
        """품질 점수 계산"""
        try:
            score = 0.0
            
            # 용어명 품질 (30점)
            term_name = term_data.get('법령용어명', '')
            if len(term_name) >= 2:
                score += 30
            
            # 정의 품질 (40점)
            definition = term_data.get('법령용어정의', '')
            if len(definition) >= 20:
                score += 40
            elif len(definition) >= 10:
                score += 20
            
            # 동음이의어 정보 (10점)
            if term_data.get('동음이의어존재여부') == 'Y':
                score += 10
            
            # 관계 정보 (20점)
            if term_data.get('용어간관계링크'):
                score += 10
            if term_data.get('조문간관계링크'):
                score += 10
            
            return min(score, 100.0)
            
        except Exception as e:
            logger.error(f"품질 점수 계산 실패: {e}")
            return 0.0
    
    def process_term_data(self, term_data: Dict) -> Optional[ProcessedLegalTerm]:
        """용어 데이터 전체 처리"""
        try:
            # 1. 데이터 정제
            cleaned_data = self.clean_term_data(term_data)
            if not cleaned_data:
                return None
            
            # 2. 용어명 정규화
            normalized_name = self.normalize_term_name(cleaned_data.get('법령용어명', ''))
            
            # 3. 키워드 추출
            keywords = self.extract_keywords(
                cleaned_data.get('법령용어명', ''),
                cleaned_data.get('법령용어정의', '')
            )
            
            # 4. 카테고리 분류
            category = self.categorize_term(
                cleaned_data.get('법령용어명', ''),
                cleaned_data.get('법령용어정의', '')
            )
            
            # 5. 품질 점수 계산
            quality_score = self.calculate_quality_score(cleaned_data)
            
            # 6. 처리된 데이터 생성
            processed_term = ProcessedLegalTerm(
                원본ID=cleaned_data.get('법령용어ID', ''),
                용어명=cleaned_data.get('법령용어명', ''),
                용어정의=cleaned_data.get('법령용어정의', ''),
                동음이의어=cleaned_data.get('동음이의어내용', ''),
                용어관계=cleaned_data.get('용어관계정보', []),
                조문관계=cleaned_data.get('조문관계정보', []),
                정규화된용어명=normalized_name,
                키워드=keywords,
                카테고리=category,
                품질점수=quality_score,
                처리일시=datetime.now().isoformat()
            )
            
            self.stats['total_processed'] += 1
            return processed_term
            
        except Exception as e:
            logger.error(f"용어 데이터 처리 실패: {e}")
            self.stats['errors'] += 1
            return None
    
    def process_all_terms(self) -> bool:
        """모든 용어 데이터 처리"""
        try:
            logger.info("법령용어 데이터 처리 시작")
            
            # 목록 파일들 로드
            list_files = list(self.raw_dir.glob("term_lists/*.json"))
            detail_files = list(self.raw_dir.glob("term_details/*.json"))
            
            logger.info(f"목록 파일: {len(list_files)}개, 상세 파일: {len(detail_files)}개")
            
            processed_terms = []
            
            # 목록 데이터 처리
            for list_file in list_files:
                logger.info(f"목록 파일 처리: {list_file.name}")
                
                with open(list_file, 'r', encoding='utf-8') as f:
                    terms = json.load(f)
                
                for term in terms:
                    processed_term = self.process_term_data(term)
                    if processed_term:
                        processed_terms.append(asdict(processed_term))
            
            # 상세 데이터 처리
            for detail_file in detail_files:
                logger.info(f"상세 파일 처리: {detail_file.name}")
                
                with open(detail_file, 'r', encoding='utf-8') as f:
                    details = json.load(f)
                
                for detail in details:
                    processed_term = self.process_term_data(detail)
                    if processed_term:
                        processed_terms.append(asdict(processed_term))
            
            # 중복 제거
            unique_terms = self._remove_duplicates(processed_terms)
            logger.info(f"중복 제거 후: {len(unique_terms)}개 용어")
            
            # 품질 필터링
            quality_threshold = self.processing_config.get('quality_threshold', 0.8)
            high_quality_terms = [
                term for term in unique_terms 
                if term['품질점수'] >= quality_threshold * 100
            ]
            logger.info(f"품질 필터링 후: {len(high_quality_terms)}개 용어")
            
            # 처리된 데이터 저장
            self._save_processed_data(high_quality_terms)
            
            # 통계 저장
            self._save_statistics()
            
            logger.info("법령용어 데이터 처리 완료")
            return True
            
        except Exception as e:
            logger.error(f"데이터 처리 중 오류: {e}")
            return False
    
    def _remove_duplicates(self, terms: List[Dict]) -> List[Dict]:
        """중복 제거"""
        try:
            seen_ids = set()
            unique_terms = []
            
            for term in terms:
                term_id = term.get('원본ID', '')
                if term_id and term_id not in seen_ids:
                    seen_ids.add(term_id)
                    unique_terms.append(term)
            
            return unique_terms
            
        except Exception as e:
            logger.error(f"중복 제거 실패: {e}")
            return terms
    
    def _save_processed_data(self, processed_terms: List[Dict]):
        """처리된 데이터 저장"""
        try:
            # 통합 데이터 저장
            integrated_file = self.integrated_dir / f"integrated_terms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(integrated_file, 'w', encoding='utf-8') as f:
                json.dump(processed_terms, f, ensure_ascii=False, indent=2)
            
            logger.info(f"통합 데이터 저장: {integrated_file}")
            
            # 카테고리별 분류 저장
            category_files = {}
            for term in processed_terms:
                category = term.get('카테고리', '기타')
                if category not in category_files:
                    category_files[category] = []
                category_files[category].append(term)
            
            for category, terms in category_files.items():
                category_file = self.integrated_dir / f"category_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(category_file, 'w', encoding='utf-8') as f:
                    json.dump(terms, f, ensure_ascii=False, indent=2)
                logger.info(f"카테고리 {category} 데이터 저장: {len(terms)}개")
            
        except Exception as e:
            logger.error(f"처리된 데이터 저장 실패: {e}")
    
    def _save_statistics(self):
        """처리 통계 저장"""
        try:
            stats_file = self.processed_dir / f"processing_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            stats_data = {
                "처리통계": self.stats,
                "처리일시": datetime.now().isoformat(),
                "설정": {
                    "품질임계값": self.processing_config.get('quality_threshold', 0.8),
                    "최소용어길이": self.processing_config.get('min_term_length', 2),
                    "최대용어길이": self.processing_config.get('max_term_length', 100)
                }
            }
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"처리 통계 저장: {stats_file}")
            
        except Exception as e:
            logger.error(f"통계 저장 실패: {e}")


def main():
    """메인 실행 함수"""
    try:
        # 설정 로드
        config = Config()
        
        # 처리기 생성 및 실행
        processor = BaseLegalTermProcessor(config)
        
        success = processor.process_all_terms()
        
        if success:
            logger.info("=== 데이터 처리 완료 ===")
            logger.info(f"처리 통계: {processor.stats}")
        else:
            logger.error("=== 데이터 처리 실패 ===")
            
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
