#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML 강화 조문 파서
훈련된 머신러닝 모델을 사용하여 조문 경계를 더 정확하게 식별
"""

import re
import joblib
import numpy as np
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

# Windows 콘솔에서 UTF-8 인코딩 설정
if os.name == 'nt':  # Windows
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Try to import parsers module
try:
    from parsers.improved_article_parser import ImprovedArticleParser
    PARSERS_AVAILABLE = True
except ImportError:
    PARSERS_AVAILABLE = False
    logging.warning("parsers module not available. Some features will be limited.")

# Import quality validator
try:
    sys.path.append(str(Path(__file__).parent.parent.parent / 'data_processing' / 'quality'))
    from data_quality_validator import DataQualityValidator, QualityReport
    QUALITY_VALIDATOR_AVAILABLE = True
except ImportError:
    QUALITY_VALIDATOR_AVAILABLE = False
    logger.warning("Quality validator not available. Validation features disabled.")

logger = logging.getLogger(__name__)

class MLEnhancedArticleParser:
    """ML 모델이 강화된 조문 파서"""
    
    def __init__(self, ml_model_path: str = "models/article_classifier.pkl"):
        """
        초기화
        
        Args:
            ml_model_path: 훈련된 ML 모델 경로
        """
        # Initialize base parser if available
        self.base_parser = None
        if PARSERS_AVAILABLE:
            try:
                self.base_parser = ImprovedArticleParser()
            except Exception as e:
                logging.warning(f"Could not initialize base parser: {e}")
                self.base_parser = None
        
        # ML 모델 로드
        self.ml_model = None
        self.vectorizer = None
        self.feature_names = None
        self.ml_threshold = 0.4  # ML 예측 임계값 (0.5 → 0.4로 낮춤, 더 나은 recall)
        
        # Quality validation
        self.quality_validator = None
        if QUALITY_VALIDATOR_AVAILABLE:
            self.quality_validator = DataQualityValidator()
            logger.info("Quality validator initialized")
        
        if Path(ml_model_path).exists():
            self._load_ml_model(ml_model_path)
            logger.info(f"ML model loaded from {ml_model_path}")
        else:
            logger.warning(f"ML model not found at {ml_model_path}. Using rule-based parser only.")
    
    def _load_ml_model(self, model_path: str) -> None:
        """ML 모델 로드"""
        try:
            model_data = joblib.load(model_path)
            self.ml_model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.feature_names = model_data['feature_names']
            logger.info("ML model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            self.ml_model = None
    
    def _extract_ml_features(self, content: str, position: int, article_number: str, has_title: bool) -> np.ndarray:
        """ML 모델을 위한 특성 추출"""
        # 수치형 특성 추출
        context_before = content[max(0, position - 200):position]
        context_after = content[position:min(len(content), position + 200)]
        
        features = [
            position / len(content) if len(content) > 0 else 0,  # position_ratio
            1 if position < 200 else 0,  # is_at_start
            1 if position > len(content) * 0.8 else 0,  # is_at_end
            1 if re.search(r'[.!?]\s*$', context_before) else 0,  # has_sentence_end
            1 if self._has_reference_pattern(context_before) else 0,  # has_reference_pattern
            int(re.search(r'\d+', article_number).group()) if re.search(r'\d+', article_number) else 0,  # article_number
            1 if '부칙' in article_number else 0,  # is_supplementary
            len(context_before),  # context_before_length
            len(context_after),  # context_after_length
            1 if has_title else 0,  # has_title
            1 if '(' in context_after[:50] else 0,  # has_parentheses
            1 if '"' in context_after[:50] or "'" in context_after[:50] else 0,  # has_quotes
            self._count_legal_terms(context_after[:100]),  # legal_term_count
            len(re.findall(r'\d+', context_after[:100])),  # number_count
            self._get_article_length(content, position),  # article_length
            self._calculate_reference_density(context_before)  # reference_density
        ]
        
        # 텍스트 특성
        context_text = f"{article_number} {self._extract_title_from_context(context_after)}"
        
        # TF-IDF 변환
        if self.vectorizer:
            text_features = self.vectorizer.transform([context_text]).toarray()[0]
            features.extend(text_features)
        
        return np.array(features)
    
    def _has_reference_pattern(self, context: str) -> bool:
        """조문 참조 패턴 확인"""
        reference_patterns = [
            r'제\d+조에\s*따라',
            r'제\d+조제\d+항',
            r'제\d+조의\d+',
            r'제\d+조.*?에\s*의하여',
            r'제\d+조.*?에\s*따라',
        ]
        
        for pattern in reference_patterns:
            if re.search(pattern, context):
                return True
        return False
    
    def _count_legal_terms(self, text: str) -> int:
        """법률 용어 개수 계산"""
        legal_terms = [
            '법률', '법령', '규정', '조항', '항', '호', '목',
            '시행', '공포', '개정', '폐지', '제정'
        ]
        return sum(1 for term in legal_terms if term in text)
    
    def _get_article_length(self, content: str, position: int) -> int:
        """조문 길이 계산"""
        next_article_match = re.search(r'제\d+조', content[position + 1:])
        if next_article_match:
            return next_article_match.start()
        else:
            return len(content) - position
    
    def _calculate_reference_density(self, context: str) -> float:
        """조문 참조 밀도 계산"""
        article_refs = len(re.findall(r'제\d+조', context))
        return article_refs / max(len(context), 1) * 1000
    
    def _extract_title_from_context(self, context: str) -> str:
        """문맥에서 제목 추출"""
        title_match = re.search(r'제\d+조\s*\(([^)]+)\)', context)
        return title_match.group(1) if title_match else ""
    
    def _ml_predict_article_boundary(self, content: str, position: int, article_number: str, has_title: bool) -> float:
        """ML 모델을 사용한 조문 경계 예측"""
        if not self.ml_model:
            return 0.5  # 모델이 없으면 중립값 반환
        
        try:
            features = self._extract_ml_features(content, position, article_number, has_title)
            
            # 특성 벡터를 올바른 형태로 변환
            if len(features) != len(self.feature_names):
                logger.warning(f"Feature dimension mismatch: {len(features)} vs {len(self.feature_names)}")
                return 0.5
            
            # 예측 확률 계산
            prediction_proba = self.ml_model.predict_proba([features])[0]
            return prediction_proba[1]  # Real article 확률
            
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return 0.5
    
    def _parse_articles_from_text(self, content: str) -> List[Dict[str, Any]]:
        """
        ML 강화된 조문 파싱
        기존 규칙 기반 방법과 ML 모델을 결합하여 더 정확한 조문 경계 식별
        """
        articles = []
        
        # 조문 패턴으로 모든 후보 찾기 (제2조의13 형태도 올바르게 인식, \xa0 등 공백 허용)
        article_pattern = re.compile(r'제(\d+)조(?:\s*의\s*(\d+))?(?:\s*\(([^)]+)\))?')
        matches = list(article_pattern.finditer(content))
        
        if not matches:
            return articles
        
        logger.info(f"Found {len(matches)} potential article boundaries")
        
        # ML 모델을 사용한 필터링
        valid_matches = []
        
        for i, match in enumerate(matches):
            # 조문 번호 조합 (제2조의13 형태 처리)
            article_num = match.group(1)
            article_sub = match.group(2) if match.group(2) else ""
            article_number = f"제{article_num}조" if not article_sub else f"제{article_num}조의{article_sub}"
            article_title = match.group(3) if match.group(3) else ""
            position = match.start()
            
            # ML 예측
            ml_score = self._ml_predict_article_boundary(content, position, article_number, bool(article_title))
            
            # 하이브리드 점수 계산 (규칙 기반 + ML)
            rule_score = self._calculate_rule_based_score(content, match, valid_matches)
            
            # 가중 평균 점수 (ML 50%, 규칙 50%)
            hybrid_score = 0.5 * ml_score + 0.5 * rule_score
            
            logger.debug(f"Article {article_number}: ML={ml_score:.3f}, Rule={rule_score:.3f}, Hybrid={hybrid_score:.3f}")
            
            # 임계값 이상이면 유효한 조문으로 판단
            if hybrid_score >= self.ml_threshold:
                valid_matches.append(match)
                logger.debug(f"✓ Accepted {article_number} (score: {hybrid_score:.3f})")
            else:
                logger.debug(f"✗ Rejected {article_number} (score: {hybrid_score:.3f})")
        
        logger.info(f"ML filtering: {len(matches)} → {len(valid_matches)} articles")
        
        # 유효한 조문들 처리
        for i, match in enumerate(valid_matches):
            # 조문 번호 조합 (제2조의13 형태 처리)
            article_num = match.group(1)
            article_sub = match.group(2) if match.group(2) else ""
            article_number = f"제{article_num}조" if not article_sub else f"제{article_num}조의{article_sub}"
            article_title = match.group(3) if match.group(3) else ""
            
            # 조문 내용 추출
            if i + 1 < len(valid_matches):
                next_match = valid_matches[i + 1]
                article_content = content[match.start():next_match.start()].strip()
            else:
                article_content = content[match.start():].strip()
            
            # 조문 헤더 제거
            article_header = match.group(0)
            if article_content.startswith(article_header):
                article_content = article_content[len(article_header):].strip()
            
            # 조문 파싱
            parsed_article = self._parse_single_article(
                article_number, article_title, article_content
            )
            
            if parsed_article:
                articles.append(parsed_article)
        
        return articles
    
    def _calculate_rule_based_score(self, content: str, match, valid_matches: List) -> float:
        """규칙 기반 점수 계산 (기존 ImprovedArticleParser의 로직 활용)"""
        score = 0.0
        
        article_start = match.start()
        article_number = int(match.group(1)) if match.group(1).isdigit() else 0
        
        # 위치 기반 점수
        if self._is_at_article_boundary(content, article_start):
            score += 0.2
        
        # 문맥 기반 점수
        context_score = self._analyze_context(content, article_start, str(article_number))
        score += context_score * 0.3
        
        # 순서 기반 점수
        if self._follows_article_sequence(article_number, valid_matches):
            score += 0.2
        
        # 길이 기반 점수
        if self._has_reasonable_length(content, match):
            score += 0.1
        
        # 조문 제목 유무 점수
        if match.group(2):
            score += 0.1
        
        # 조문 내용 품질 점수
        content_quality = self._assess_content_quality(content, match)
        score += content_quality * 0.1
        
        return max(0.0, min(1.0, score))
    
    def _separate_main_and_supplementary(self, content: str) -> Tuple[str, str]:
        """본칙과 부칙을 분리"""
        # 부칙 시작 패턴들
        supplementary_patterns = [
            r'부칙\s*<[^>]*>펼치기접기\s*(.*?)$',
            r'부칙\s*<[^>]*>\s*(.*?)$',
            r'부칙\s*펼치기접기\s*(.*?)$',
            r'부칙\s*(.*?)$'
        ]
        
        for pattern in supplementary_patterns:
            match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
            if match:
                main_content = content[:match.start()].strip()
                supplementary_content = match.group(1).strip()
                return main_content, supplementary_content
        
        return content, ""
    
    def _parse_supplementary_articles(self, supplementary_content: str) -> List[Dict[str, Any]]:
        """부칙 조문 파싱"""
        articles = []
        
        if not supplementary_content.strip():
            return articles
        
        # 부칙 조문 패턴 (제1조(시행일) 형태)
        article_pattern = r'제(\d+)조\s*\(([^)]*)\)\s*(.*?)(?=제\d+조\s*\(|$)'
        matches = re.finditer(article_pattern, supplementary_content, re.DOTALL)
        
        for match in matches:
            article_number = f"부칙제{match.group(1)}조"
            article_title = match.group(2).strip()
            article_content = match.group(3).strip()
            
            # 내용 정리
            article_content = self._clean_content(article_content)
            
            if article_content:
                articles.append({
                    'article_number': article_number,
                    'article_title': article_title,
                    'article_content': article_content,
                    'sub_articles': [],
                    'references': [],
                    'word_count': len(article_content.split()),
                    'char_count': len(article_content),
                    'is_supplementary': True
                })
        
        # 조문이 없는 단순 부칙 처리
        if not articles and supplementary_content.strip():
            # 시행일만 있는 경우
            if re.search(r'시행한다', supplementary_content):
                articles.append({
                    'article_number': '부칙',
                    'article_title': '',
                    'article_content': supplementary_content.strip(),
                    'sub_articles': [],
                    'references': [],
                    'word_count': len(supplementary_content.split()),
                    'char_count': len(supplementary_content),
                    'is_supplementary': True
                })
        
        return articles

    def _basic_parse_law(self, law_content: str) -> Dict[str, Any]:
        """
        기본 파싱 (parsers 모듈이 없을 때 사용)
        
        Args:
            law_content: 법률 문서 내용
            
        Returns:
            Dict[str, Any]: 기본 파싱 결과
        """
        # 간단한 정규식 기반 파싱
        articles = []
        
        # 조문 패턴 찾기
        article_pattern = r'제(\d+)조\s*\([^)]*\)\s*([^\n]+(?:\n(?!제\d+조)[^\n]*)*)'
        matches = re.findall(article_pattern, law_content, re.MULTILINE)
        
        for match in matches:
            article_number = match[0]
            content = match[1].strip()
            
            articles.append({
                'article_number': article_number,
                'content': content,
                'title': f"제{article_number}조"
            })
        
        return {
            'articles': articles,
            'parsing_method': 'basic_regex',
            'parsing_timestamp': datetime.now().isoformat(),
            'quality_score': 0.5,  # 기본 점수
            'auto_corrected': False,
            'manual_review_required': True
        }
    
    def parse_law_document(self, law_content: str) -> Dict[str, Any]:
        """
        법률 문서 파싱 (ML 강화 버전 + 부칙 파싱)
        
        Args:
            law_content: 법률 문서 내용
            
        Returns:
            Dict[str, Any]: 파싱 결과
        """
        logger.info("Starting ML-enhanced law document parsing with supplementary parsing")
        
        # 기본 전처리
        cleaned_content = self._clean_content(law_content)
        
        # 본칙과 부칙 분리
        main_content, supplementary_content = self._separate_main_and_supplementary(cleaned_content)
        
        # 본칙 조문 파싱
        main_articles = self._parse_articles_from_text(main_content)
        
        # 부칙 조문 파싱
        supplementary_articles = self._parse_supplementary_articles(supplementary_content)
        
        # 모든 조문 합치기
        all_articles = main_articles + supplementary_articles
        
        result = {
            'main_articles': main_articles,
            'supplementary_articles': supplementary_articles,
            'all_articles': all_articles,
            'total_articles': len(all_articles),
            'parsing_status': 'success',
            'ml_enhanced': self.ml_model is not None
        }
        
        logger.info(f"ML-enhanced parsing completed: {len(main_articles)} main articles, {len(supplementary_articles)} supplementary articles")
        
        return result


    def validate_parsed_result(self, parsed_result: Dict[str, Any]) -> QualityReport:
        """
        Validate parsed result using quality validator
        
        Args:
            parsed_result: Parsed law data dictionary
            
        Returns:
            QualityReport: Quality validation report
        """
        if not self.quality_validator:
            logger.warning("Quality validator not available")
            return None
        
        try:
            report = self.quality_validator.validate_parsing_quality(parsed_result)
            logger.info(f"Validation completed. Quality score: {report.overall_score:.3f}")
            return report
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            return None
    
    def fix_missing_articles(self, parsed_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to recover missing articles from parsing result
        
        Args:
            parsed_result: Parsed law data dictionary
            
        Returns:
            Dict[str, Any]: Improved parsing result with recovered articles
        """
        try:
            articles = parsed_result.get('all_articles', [])
            if not articles:
                logger.warning("No articles found to fix")
                return parsed_result
            
            # Extract article numbers
            article_numbers = []
            for article in articles:
                try:
                    number_text = str(article.get('article_number', ''))
                    number_match = re.search(r'\d+', number_text)
                    if number_match:
                        article_numbers.append(int(number_match.group()))
                except (ValueError, TypeError):
                    continue
            
            if not article_numbers:
                logger.warning("No article numbers found")
                return parsed_result
            
            # Find missing numbers in sequence
            article_numbers.sort()
            missing_numbers = []
            for i in range(1, len(article_numbers)):
                gap = article_numbers[i] - article_numbers[i-1]
                if gap > 1:
                    missing_numbers.extend(range(article_numbers[i-1] + 1, article_numbers[i]))
            
            if missing_numbers:
                logger.info(f"Found {len(missing_numbers)} missing article numbers: {missing_numbers}")
                
                # Try to recover missing articles from original content
                original_content = parsed_result.get('original_content', '')
                if original_content:
                    recovered_articles = self._recover_missing_articles(original_content, missing_numbers)
                    if recovered_articles:
                        parsed_result['all_articles'].extend(recovered_articles)
                        parsed_result['recovered_articles'] = recovered_articles
                        logger.info(f"Recovered {len(recovered_articles)} missing articles")
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error fixing missing articles: {e}")
            return parsed_result
    
    def remove_duplicate_articles(self, parsed_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove duplicate articles from parsing result
        
        Args:
            parsed_result: Parsed law data dictionary
            
        Returns:
            Dict[str, Any]: Parsing result with duplicates removed
        """
        try:
            articles = parsed_result.get('all_articles', [])
            if not articles:
                return parsed_result
            
            # Group articles by content similarity
            unique_articles = []
            seen_contents = set()
            duplicates_removed = 0
            
            for article in articles:
                content = str(article.get('content', '')) + str(article.get('text', ''))
                content_hash = hash(content.strip())
                
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    unique_articles.append(article)
                else:
                    duplicates_removed += 1
                    logger.debug(f"Removed duplicate article: {article.get('article_number', 'Unknown')}")
            
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate articles")
                parsed_result['all_articles'] = unique_articles
                parsed_result['duplicates_removed'] = duplicates_removed
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error removing duplicate articles: {e}")
            return parsed_result
    
    def _recover_missing_articles(self, content: str, missing_numbers: List[int]) -> List[Dict[str, Any]]:
        """
        Attempt to recover missing articles from original content
        
        Args:
            content: Original law content
            missing_numbers: List of missing article numbers
            
        Returns:
            List[Dict[str, Any]]: Recovered articles
        """
        recovered_articles = []
        
        for missing_number in missing_numbers:
            # Look for article pattern in content
            pattern = rf'제\s*{missing_number}\s*조[^제]*?(?=제\s*\d+\s*조|$)'
            match = re.search(pattern, content, re.DOTALL)
            
            if match:
                article_text = match.group(0).strip()
                
                # Extract title if present
                title_match = re.search(r'제\s*\d+\s*조\s*\(([^)]+)\)', article_text)
                title = title_match.group(1) if title_match else f"제{missing_number}조"
                
                recovered_article = {
                    'article_number': f'제{missing_number}조',
                    'article_title': title,
                    'content': article_text,
                    'text': article_text,
                    'is_recovered': True,
                    'recovery_method': 'pattern_matching'
                }
                
                recovered_articles.append(recovered_article)
                logger.debug(f"Recovered article {missing_number}: {title}")
        
        return recovered_articles
    
    def parse_law_with_validation(self, law_content: str) -> Dict[str, Any]:
        """
        Parse law content with validation and auto-correction
        
        Args:
            law_content: Law content to parse
            
        Returns:
            Dict[str, Any]: Parsed result with quality information
        """
        try:
            # Initial parsing - use fallback if base parser not available
            if self.base_parser:
                parsed_result = self.parse_law_document(law_content)
            else:
                logger.warning("Base parser not available, using basic parsing")
                parsed_result = self._basic_parse_law(law_content)
            
            parsed_result['original_content'] = law_content
            
            # Validate result
            if self.quality_validator:
                quality_report = self.validate_parsed_result(parsed_result)
                if quality_report:
                    parsed_result['quality_report'] = quality_report
                    parsed_result['quality_score'] = quality_report.overall_score
                    
                    # Auto-correction based on quality
                    if quality_report.overall_score < 0.7:
                        logger.info("Low quality detected, applying auto-corrections")
                        
                        # Fix missing articles
                        parsed_result = self.fix_missing_articles(parsed_result)
                        
                        # Remove duplicates
                        parsed_result = self.remove_duplicate_articles(parsed_result)
                        
                        # Re-validate after corrections
                        updated_report = self.validate_parsed_result(parsed_result)
                        if updated_report:
                            parsed_result['quality_report'] = updated_report
                            parsed_result['quality_score'] = updated_report.overall_score
                            parsed_result['auto_corrected'] = True
                            
                            logger.info(f"Auto-correction completed. New quality score: {updated_report.overall_score:.3f}")
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error in parse_law_with_validation: {e}")
            return {
                'all_articles': [],
                'main_articles': [],
                'supplementary_articles': [],
                'total_articles': 0,
                'ml_enhanced': False,
                'error': str(e),
                'quality_score': 0.0
            }


def main():
    """테스트 함수"""
    # 테스트용 법률 내용
    test_content = """
    제1조(목적) 이 법은 공공기관의 소방안전관리에 관한 사항을 규정함을 목적으로 한다.
    
    제2조(적용 범위) 이 법은 다음 각 호의 어느 하나에 해당하는 공공기관에 적용한다.
    1. 국가기관
    2. 지방자치단체
    3. 공공기관의 운영에 관한 법률 제4조에 따른 공공기관
    
    제3조(기관장의 책임) 제2조에 따른 공공기관의 장은 소방안전관리에 대한 책임을 진다.
    
    부칙제1조(시행일) 이 법은 공포한 날부터 시행한다.
    """
    
    # ML 강화 파서 테스트
    parser = MLEnhancedArticleParser()
    result = parser.parse_law_document(test_content)
    
    print("=== ML-Enhanced Parsing Result ===")
    print(f"Total articles: {result['total_articles']}")
    print(f"Main articles: {len(result['main_articles'])}")
    print(f"Supplementary articles: {len(result['supplementary_articles'])}")
    print(f"ML enhanced: {result['ml_enhanced']}")
    
    for article in result['all_articles']:
        print(f"- {article['article_number']}: {article['article_title']}")


if __name__ == "__main__":
    main()