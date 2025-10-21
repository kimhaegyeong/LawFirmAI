#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
향상된 Raw 데이터 전처리 파이프라인
Raw 데이터를 고품질로 전처리하여 데이터베이스 적재에 최적화된 형태로 변환합니다.
"""

import os
import sys
import json
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_preprocessing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """전처리 결과 데이터 클래스"""
    processed_files: int = 0
    total_laws: int = 0
    total_articles: int = 0
    quality_scores: List[float] = None
    errors: List[str] = None
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.quality_scores is None:
            self.quality_scores = []
        if self.errors is None:
            self.errors = []


class DataQualityValidator:
    """데이터 품질 검증기"""
    
    def __init__(self):
        self.min_content_length = 10
        self.min_article_count = 1
        self.max_content_length = 100000
        
    def calculate_quality_score(self, law_data: Dict[str, Any]) -> float:
        """
        법률 데이터 품질 점수 계산 (0.0 ~ 1.0)
        
        Args:
            law_data (Dict[str, Any]): 법률 데이터
            
        Returns:
            float: 품질 점수
        """
        score = 0.0
        max_score = 0.0
        
        # 기본 정보 완성도 (30%)
        max_score += 30
        basic_info_score = self._check_basic_info(law_data)
        score += basic_info_score * 30
        
        # 조문 구조 완성도 (40%)
        max_score += 40
        article_structure_score = self._check_article_structure(law_data)
        score += article_structure_score * 40
        
        # 내용 품질 (20%)
        max_score += 20
        content_quality_score = self._check_content_quality(law_data)
        score += content_quality_score * 20
        
        # 일관성 (10%)
        max_score += 10
        consistency_score = self._check_consistency(law_data)
        score += consistency_score * 10
        
        return score / max_score if max_score > 0 else 0.0
    
    def _check_basic_info(self, law_data: Dict[str, Any]) -> float:
        """기본 정보 완성도 검사"""
        required_fields = ['law_name', 'law_type', 'promulgation_date']
        present_fields = sum(1 for field in required_fields if law_data.get(field))
        return present_fields / len(required_fields)
    
    def _check_article_structure(self, law_data: Dict[str, Any]) -> float:
        """조문 구조 완성도 검사"""
        articles = law_data.get('articles', [])
        if not articles:
            return 0.0
        
        valid_articles = 0
        for article in articles:
            if (article.get('article_number') and 
                article.get('article_content') and 
                len(article.get('article_content', '')) > self.min_content_length):
                valid_articles += 1
        
        return valid_articles / len(articles) if articles else 0.0
    
    def _check_content_quality(self, law_data: Dict[str, Any]) -> float:
        """내용 품질 검사"""
        full_text = law_data.get('full_text', '')
        if not full_text:
            return 0.0
        
        # 길이 적절성
        length_score = min(len(full_text) / 1000, 1.0)  # 1000자 이상이면 만점
        
        # HTML 태그 제거 확인
        html_tags = re.findall(r'<[^>]+>', full_text)
        html_score = 1.0 - min(len(html_tags) / 100, 1.0)  # HTML 태그가 적을수록 좋음
        
        # 특수문자 비율
        special_chars = re.findall(r'[^\w\s가-힣]', full_text)
        special_score = 1.0 - min(len(special_chars) / len(full_text), 0.3)  # 특수문자 30% 이하
        
        return (length_score + html_score + special_score) / 3
    
    def _check_consistency(self, law_data: Dict[str, Any]) -> float:
        """일관성 검사"""
        articles = law_data.get('articles', [])
        if not articles:
            return 0.0
        
        # 조문 번호 연속성
        article_numbers = []
        for article in articles:
            try:
                num = int(re.findall(r'\d+', article.get('article_number', '0'))[0])
                article_numbers.append(num)
            except (ValueError, IndexError):
                continue
        
        if not article_numbers:
            return 0.0
        
        # 연속성 점수 계산
        sorted_numbers = sorted(article_numbers)
        expected_range = list(range(min(sorted_numbers), max(sorted_numbers) + 1))
        continuity_score = len(set(sorted_numbers) & set(expected_range)) / len(expected_range)
        
        return continuity_score


class HybridArticleParser:
    """하이브리드 조문 파서 (규칙 기반 + ML)"""
    
    def __init__(self):
        # 조문 패턴
        self.article_patterns = [
            re.compile(r'제(\d+)조\s*\(([^)]+)\)'),  # 제1조(목적)
            re.compile(r'제(\d+)조'),  # 제1조
            re.compile(r'제(\d+)조\s*의\s*(\d+)'),  # 제1조의2
        ]
        
        # 항 패턴
        self.paragraph_patterns = [
            re.compile(r'①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩|⑪|⑫|⑬|⑭|⑮|⑯|⑰|⑱|⑲|⑳'),
            re.compile(r'(\d+)\s*항'),
            re.compile(r'제(\d+)\s*항')
        ]
        
        # 호 패턴
        self.subparagraph_patterns = [
            re.compile(r'(\d+)\s*\.'),
            re.compile(r'제(\d+)\s*호')
        ]
    
    def parse_with_validation(self, law_content: str) -> Dict[str, Any]:
        """
        하이브리드 파싱으로 조문 추출
        
        Args:
            law_content (str): 법률 내용
            
        Returns:
            Dict[str, Any]: 파싱 결과
        """
        try:
            # 규칙 기반 파싱
            rule_based_result = self._rule_based_parsing(law_content)
            
            # 품질 점수 계산
            quality_score = self._calculate_parsing_quality(rule_based_result)
            
            return {
                'articles': rule_based_result['articles'],
                'quality_score': quality_score,
                'parsing_method': 'hybrid',
                'ml_confidence': quality_score,
                'metadata': {
                    'total_articles': len(rule_based_result['articles']),
                    'parsing_time': rule_based_result.get('parsing_time', 0),
                    'errors': rule_based_result.get('errors', [])
                }
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid parsing: {e}")
            return {
                'articles': [],
                'quality_score': 0.0,
                'parsing_method': 'hybrid',
                'ml_confidence': 0.0,
                'metadata': {
                    'total_articles': 0,
                    'parsing_time': 0,
                    'errors': [str(e)]
                }
            }
    
    def _rule_based_parsing(self, content: str) -> Dict[str, Any]:
        """규칙 기반 파싱"""
        articles = []
        errors = []
        
        try:
            # 조문 추출
            for pattern in self.article_patterns:
                matches = pattern.findall(content)
                for match in matches:
                    if len(match) == 2:  # 제1조(목적) 형태
                        article_num, title = match
                        article_content = self._extract_article_content(content, f"제{article_num}조")
                    else:  # 제1조 형태
                        article_num = match[0] if isinstance(match, tuple) else match
                        title = ""
                        article_content = self._extract_article_content(content, f"제{article_num}조")
                    
                    if article_content:
                        article = {
                            'article_number': f"제{article_num}조",
                            'article_title': title,
                            'article_content': article_content.strip(),
                            'word_count': len(article_content.split()),
                            'char_count': len(article_content),
                            'is_supplementary': False,
                            'parsing_method': 'rule_based'
                        }
                        articles.append(article)
            
            # 중복 제거 및 정렬
            articles = self._remove_duplicates_and_sort(articles)
            
            return {
                'articles': articles,
                'parsing_time': 0,
                'errors': errors
            }
            
        except Exception as e:
            errors.append(f"Rule-based parsing error: {e}")
            return {
                'articles': [],
                'parsing_time': 0,
                'errors': errors
            }
    
    def _extract_article_content(self, content: str, article_ref: str) -> str:
        """조문 내용 추출"""
        try:
            # 조문 시작 위치 찾기
            start_pattern = re.compile(f'{re.escape(article_ref)}(?:\s*\([^)]+\))?')
            start_match = start_pattern.search(content)
            
            if not start_match:
                return ""
            
            start_pos = start_match.end()
            
            # 다음 조문 또는 끝까지 추출
            next_article_pattern = re.compile(r'제\d+조')
            remaining_content = content[start_pos:]
            
            # 다음 조문 찾기
            next_match = next_article_pattern.search(remaining_content)
            if next_match:
                end_pos = start_pos + next_match.start()
                return content[start_pos:end_pos]
            else:
                return content[start_pos:]
                
        except Exception as e:
            logger.error(f"Error extracting article content: {e}")
            return ""
    
    def _remove_duplicates_and_sort(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """중복 제거 및 정렬"""
        # 중복 제거 (조문 번호 기준)
        seen_numbers = set()
        unique_articles = []
        
        for article in articles:
            article_num = article.get('article_number', '')
            if article_num not in seen_numbers:
                seen_numbers.add(article_num)
                unique_articles.append(article)
        
        # 조문 번호로 정렬
        def sort_key(article):
            try:
                num = int(re.findall(r'\d+', article.get('article_number', '0'))[0])
                return num
            except (ValueError, IndexError):
                return 999999
        
        unique_articles.sort(key=sort_key)
        return unique_articles
    
    def _calculate_parsing_quality(self, parsing_result: Dict[str, Any]) -> float:
        """파싱 품질 점수 계산"""
        articles = parsing_result.get('articles', [])
        errors = parsing_result.get('errors', [])
        
        if not articles:
            return 0.0
        
        # 기본 점수
        base_score = 0.5
        
        # 조문 수에 따른 보너스
        article_count_score = min(len(articles) / 10, 0.3)  # 10개 이상이면 만점
        
        # 에러 수에 따른 페널티
        error_penalty = min(len(errors) * 0.1, 0.3)
        
        # 조문 내용 품질
        content_quality = 0.0
        for article in articles:
            content = article.get('article_content', '')
            if len(content) > 50:  # 50자 이상
                content_quality += 1
        content_quality = content_quality / len(articles) * 0.2 if articles else 0
        
        final_score = base_score + article_count_score - error_penalty + content_quality
        return max(0.0, min(1.0, final_score))


class EnhancedRawPreprocessor:
    """향상된 Raw 데이터 전처리기"""
    
    def __init__(self, 
                 raw_data_base_path: str = "data/raw",
                 processed_data_base_path: str = "data/processed"):
        self.raw_data_base_path = Path(raw_data_base_path)
        self.processed_data_base_path = Path(processed_data_base_path)
        
        # 컴포넌트 초기화
        self.quality_validator = DataQualityValidator()
        self.hybrid_parser = HybridArticleParser()
        
        # 처리 통계
        self.stats = {
            'processed_files': 0,
            'total_laws': 0,
            'total_articles': 0,
            'quality_scores': [],
            'errors': [],
            'processing_time': 0.0
        }
    
    def process_law_only_data(self, raw_dir: str) -> ProcessingResult:
        """
        법률 전용 데이터 전처리
        
        Args:
            raw_dir (str): Raw 데이터 디렉토리
            
        Returns:
            ProcessingResult: 처리 결과
        """
        logger.info(f"🔄 Processing law-only data from: {raw_dir}")
        start_time = datetime.now()
        
        result = ProcessingResult()
        raw_path = Path(raw_dir)
        
        if not raw_path.exists():
            error_msg = f"Raw directory not found: {raw_dir}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            return result
        
        # JSON 파일 처리
        json_files = list(raw_path.glob("**/*.json"))
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        for file_path in json_files:
            try:
                file_result = self._process_single_law_file(file_path)
                result.processed_files += 1
                result.total_laws += file_result['total_laws']
                result.total_articles += file_result['total_articles']
                result.quality_scores.extend(file_result['quality_scores'])
                result.errors.extend(file_result['errors'])
                
                # 전처리된 데이터 저장
                self._save_processed_data(file_result['processed_data'], file_path)
                
            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)
        
        # 처리 시간 계산
        end_time = datetime.now()
        result.processing_time = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ Processing completed:")
        logger.info(f"  - Processed files: {result.processed_files}")
        logger.info(f"  - Total laws: {result.total_laws}")
        logger.info(f"  - Total articles: {result.total_articles}")
        logger.info(f"  - Average quality: {sum(result.quality_scores)/len(result.quality_scores):.3f}" if result.quality_scores else "N/A")
        logger.info(f"  - Processing time: {result.processing_time:.2f} seconds")
        
        return result
    
    def _process_single_law_file(self, file_path: Path) -> Dict[str, Any]:
        """단일 법률 파일 처리"""
        try:
            # Raw 데이터 로드
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            processed_laws = []
            quality_scores = []
            errors = []
            
            # 법률 데이터 처리
            laws = raw_data.get('laws', [])
            for law_item in laws:
                try:
                    # 하이브리드 파싱 적용
                    parsed_result = self.hybrid_parser.parse_with_validation(
                        law_item.get('law_content', '')
                    )
                    
                    # 법률 데이터 구성
                    law_data = {
                        'law_id': law_item.get('law_id', ''),
                        'law_name': law_item.get('law_name', ''),
                        'law_type': law_item.get('law_type', ''),
                        'promulgation_date': law_item.get('promulgation_date', ''),
                        'enforcement_date': law_item.get('enforcement_date', ''),
                        'ministry': law_item.get('ministry', ''),
                        'full_text': law_item.get('law_content', ''),
                        'articles': parsed_result.get('articles', []),
                        'parsing_method': parsed_result.get('parsing_method', 'hybrid'),
                        'parsing_quality_score': parsed_result.get('quality_score', 0.0),
                        'ml_confidence_score': parsed_result.get('ml_confidence', 0.0),
                        'article_count': len(parsed_result.get('articles', [])),
                        'created_at': datetime.now().isoformat()
                    }
                    
                    # 품질 검증
                    quality_score = self.quality_validator.calculate_quality_score(law_data)
                    law_data['parsing_quality_score'] = quality_score
                    quality_scores.append(quality_score)
                    
                    processed_laws.append(law_data)
                    
                except Exception as e:
                    error_msg = f"Error processing law {law_item.get('law_name', 'Unknown')}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            return {
                'processed_data': {'laws': processed_laws},
                'total_laws': len(processed_laws),
                'total_articles': sum(len(law.get('articles', [])) for law in processed_laws),
                'quality_scores': quality_scores,
                'errors': errors
            }
            
        except Exception as e:
            error_msg = f"Error loading file {file_path}: {e}"
            logger.error(error_msg)
            return {
                'processed_data': {'laws': []},
                'total_laws': 0,
                'total_articles': 0,
                'quality_scores': [],
                'errors': [error_msg]
            }
    
    def _save_processed_data(self, processed_data: Dict[str, Any], original_file_path: Path):
        """전처리된 데이터 저장"""
        try:
            # 출력 경로 설정
            relative_path = original_file_path.relative_to(self.raw_data_base_path)
            output_subdir = self.processed_data_base_path / relative_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # 파일명 생성
            output_file_name = f"enhanced_{original_file_path.stem}.json"
            output_file_path = output_subdir / output_file_name
            
            # 데이터 저장
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Saved processed data to: {output_file_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
    
    def process_precedent_data(self, raw_dir: str) -> ProcessingResult:
        """
        판례 데이터 전처리
        
        Args:
            raw_dir (str): Raw 데이터 디렉토리
            
        Returns:
            ProcessingResult: 처리 결과
        """
        logger.info(f"🔄 Processing precedent data from: {raw_dir}")
        start_time = datetime.now()
        
        result = ProcessingResult()
        raw_path = Path(raw_dir)
        
        if not raw_path.exists():
            error_msg = f"Raw directory not found: {raw_dir}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            return result
        
        # 카테고리별 처리
        categories = ['civil', 'criminal', 'family', 'administrative']
        
        for category in categories:
            category_path = raw_path / category
            if category_path.exists():
                category_result = self._process_precedent_category(category_path, category)
                result.processed_files += category_result.processed_files
                result.total_laws += category_result.total_laws  # 판례는 cases로 처리
                result.errors.extend(category_result.errors)
        
        # 처리 시간 계산
        end_time = datetime.now()
        result.processing_time = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ Precedent processing completed:")
        logger.info(f"  - Processed files: {result.processed_files}")
        logger.info(f"  - Total cases: {result.total_laws}")
        logger.info(f"  - Processing time: {result.processing_time:.2f} seconds")
        
        return result
    
    def _process_precedent_category(self, category_path: Path, category: str) -> ProcessingResult:
        """카테고리별 판례 처리"""
        result = ProcessingResult()
        
        json_files = list(category_path.glob("**/*.json"))
        logger.info(f"Processing {len(json_files)} files for category: {category}")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                # 판례 데이터 처리 로직 (간단한 구현)
                cases = raw_data.get('cases', [])
                processed_cases = []
                
                for case in cases:
                    processed_case = {
                        'case_id': case.get('case_id', ''),
                        'case_name': case.get('case_name', ''),
                        'case_number': case.get('case_number', ''),
                        'field': category,
                        'court': case.get('court', ''),
                        'decision_date': case.get('decision_date', ''),
                        'full_text': case.get('full_text', ''),
                        'created_at': datetime.now().isoformat()
                    }
                    processed_cases.append(processed_case)
                
                # 전처리된 데이터 저장
                self._save_processed_precedent_data(processed_cases, file_path, category)
                
                result.processed_files += 1
                result.total_laws += len(processed_cases)
                
            except Exception as e:
                error_msg = f"Error processing precedent file {file_path}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)
        
        return result
    
    def _save_processed_precedent_data(self, processed_cases: List[Dict[str, Any]], 
                                     original_file_path: Path, category: str):
        """전처리된 판례 데이터 저장"""
        try:
            # 출력 경로 설정
            relative_path = original_file_path.relative_to(self.raw_data_base_path)
            output_subdir = self.processed_data_base_path / "precedent" / category / relative_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # 파일명 생성
            output_file_name = f"enhanced_{original_file_path.stem}.json"
            output_file_path = output_subdir / output_file_name
            
            # 데이터 저장
            processed_data = {'cases': processed_cases}
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Saved processed precedent data to: {output_file_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed precedent data: {e}")


def main():
    """메인 함수"""
    logger.info("🚀 Starting enhanced raw data preprocessing...")
    
    # 전처리기 초기화
    preprocessor = EnhancedRawPreprocessor()
    
    # 법률 데이터 전처리
    logger.info("\n📋 Phase 1: Processing law-only data...")
    law_result = preprocessor.process_law_only_data("data/raw/assembly/law_only")
    
    # 판례 데이터 전처리
    logger.info("\n📋 Phase 2: Processing precedent data...")
    precedent_result = preprocessor.process_precedent_data("data/raw/assembly/precedent")
    
    # 결과 리포트 생성
    total_result = ProcessingResult(
        processed_files=law_result.processed_files + precedent_result.processed_files,
        total_laws=law_result.total_laws + precedent_result.total_laws,
        total_articles=law_result.total_articles,
        quality_scores=law_result.quality_scores,
        errors=law_result.errors + precedent_result.errors,
        processing_time=law_result.processing_time + precedent_result.processing_time
    )
    
    # 결과 저장
    result_data = {
        'law_processing': {
            'processed_files': law_result.processed_files,
            'total_laws': law_result.total_laws,
            'total_articles': law_result.total_articles,
            'average_quality': sum(law_result.quality_scores)/len(law_result.quality_scores) if law_result.quality_scores else 0,
            'errors': law_result.errors
        },
        'precedent_processing': {
            'processed_files': precedent_result.processed_files,
            'total_cases': precedent_result.total_laws,
            'errors': precedent_result.errors
        },
        'total_processing': {
            'processed_files': total_result.processed_files,
            'total_laws': total_result.total_laws,
            'total_articles': total_result.total_articles,
            'processing_time': total_result.processing_time,
            'total_errors': len(total_result.errors)
        }
    }
    
    # 리포트 저장
    with open("data/preprocessing_report.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n📊 Detailed report saved to: data/preprocessing_report.json")
    logger.info("✅ Enhanced preprocessing completed successfully!")
    
    return total_result


if __name__ == "__main__":
    result = main()
    if result.errors:
        print(f"\n⚠️ Processing completed with {len(result.errors)} errors")
        print("Check logs for details.")
    else:
        print("\n🎉 Preprocessing completed successfully!")
        print("You can now proceed with database import.")
