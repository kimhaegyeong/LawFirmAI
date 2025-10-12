#!/usr/bin/env python3
"""
법령 데이터 HTML 정제 스크립트

이 스크립트는 law_page_001_181503.json 파일의 content_html을 사용하여
모든 조문을 추출하고 정제된 데이터를 생성합니다.
"""

import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.data.assembly_playwright_client import AssemblyPlaywrightClient
from scripts.test_improved_html_parser import ImprovedLawHTMLParser

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/refine_law_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LawDataRefiner:
    """법령 데이터 HTML 정제 클래스"""
    
    def __init__(self):
        """초기화"""
        self.html_parser = ImprovedLawHTMLParser()
        self.assembly_client = AssemblyPlaywrightClient()
    
    def refine_law_file(self, input_file: Path, output_file: Path) -> Dict[str, Any]:
        """
        법령 파일을 HTML을 사용하여 정제
        
        Args:
            input_file (Path): 입력 파일 경로
            output_file (Path): 출력 파일 경로
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            logger.info(f"법령 파일 정제 시작: {input_file}")
            
            # 원본 데이터 로드
            with open(input_file, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            refined_laws = []
            processing_stats = {
                'total_laws': len(original_data.get('laws', [])),
                'successful_refinements': 0,
                'failed_refinements': 0,
                'errors': []
            }
            
            # 각 법령 처리
            for i, law_data in enumerate(original_data.get('laws', [])):
                try:
                    logger.info(f"법령 {i+1}/{processing_stats['total_laws']} 처리 중: {law_data.get('law_name', 'Unknown')}")
                    
                    refined_law = self._refine_single_law(law_data)
                    if refined_law:
                        refined_laws.append(refined_law)
                        processing_stats['successful_refinements'] += 1
                    else:
                        processing_stats['failed_refinements'] += 1
                        
                except Exception as e:
                    error_msg = f"법령 {i+1} 처리 중 오류: {str(e)}"
                    logger.error(error_msg)
                    processing_stats['errors'].append(error_msg)
                    processing_stats['failed_refinements'] += 1
            
            # 정제된 데이터 구성
            refined_data = {
                'page_number': original_data.get('page_number', 1),
                'total_pages': original_data.get('total_pages', 1),
                'laws_count': len(refined_laws),
                'collected_at': original_data.get('collected_at', ''),
                'refined_at': datetime.now().isoformat(),
                'refinement_version': '1.0',
                'processing_stats': processing_stats,
                'laws': refined_laws
            }
            
            # 정제된 데이터 저장
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(refined_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"정제 완료: {output_file}")
            logger.info(f"처리 통계: {processing_stats}")
            
            return {
                'success': True,
                'output_file': str(output_file),
                'processing_stats': processing_stats
            }
            
        except Exception as e:
            error_msg = f"파일 정제 중 오류: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def _refine_single_law(self, law_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        단일 법령 데이터 정제
        
        Args:
            law_data (Dict[str, Any]): 원본 법령 데이터
            
        Returns:
            Optional[Dict[str, Any]]: 정제된 법령 데이터
        """
        try:
            law_name = law_data.get('law_name', '')
            content_html = law_data.get('content_html', '')
            original_content = law_data.get('law_content', '')
            
            if not content_html:
                logger.warning(f"HTML 내용이 없음: {law_name}")
                return None
            
            # HTML 파싱
            parsed_html = self.html_parser.parse_html(content_html)
            
            # 정제된 법령 데이터 구성
            refined_law = {
                # 기본 정보
                'law_id': f"assembly_law_{law_data.get('row_number', 'unknown')}",
                'law_name': law_name,
                'law_type': law_data.get('law_type', ''),
                'category': law_data.get('category', ''),
                'row_number': law_data.get('row_number', ''),
                
                # 공포 정보
                'promulgation_info': {
                    'number': law_data.get('promulgation_number', ''),
                    'date': law_data.get('promulgation_date', ''),
                    'enforcement_date': law_data.get('enforcement_date', ''),
                    'amendment_type': law_data.get('amendment_type', '')
                },
                
                # 수집 정보
                'collection_info': {
                    'cont_id': law_data.get('cont_id', ''),
                    'cont_sid': law_data.get('cont_sid', ''),
                    'detail_url': law_data.get('detail_url', ''),
                    'collected_at': law_data.get('collected_at', '')
                },
                
                # 정제된 내용
                'refined_content': {
                    'full_text': parsed_html.get('clean_text', ''),
                    'articles': parsed_html.get('articles', []),
                    'html_metadata': parsed_html.get('metadata', {})
                },
                
                # 원본 데이터 (참조용)
                'original_content': original_content,
                'content_html': content_html,
                
                # 처리 메타데이터
                'refined_at': datetime.now().isoformat(),
                'refinement_version': '1.0',
                'data_quality': self._calculate_data_quality(parsed_html, original_content)
            }
            
            return refined_law
            
        except Exception as e:
            logger.error(f"단일 법령 정제 중 오류: {str(e)}")
            return None
    
    def _calculate_data_quality(self, parsed_html: Dict[str, Any], original_content: str) -> Dict[str, Any]:
        """
        데이터 품질 계산
        
        Args:
            parsed_html (Dict[str, Any]): 파싱된 HTML 데이터
            original_content (str): 원본 내용
            
        Returns:
            Dict[str, Any]: 데이터 품질 정보
        """
        try:
            clean_text = parsed_html.get('clean_text', '')
            articles = parsed_html.get('articles', [])
            
            # 기본 통계
            stats = {
                'original_content_length': len(original_content),
                'clean_text_length': len(clean_text),
                'articles_count': len(articles),
                'improvement_ratio': len(clean_text) / len(original_content) if original_content else 0,
                'has_articles': len(articles) > 0,
                'quality_score': 0.0
            }
            
            # 품질 점수 계산
            quality_score = 0.0
            
            # 텍스트 길이 개선 (최대 30점)
            if stats['improvement_ratio'] > 1.5:
                quality_score += 30
            elif stats['improvement_ratio'] > 1.0:
                quality_score += 20
            elif stats['improvement_ratio'] > 0.5:
                quality_score += 10
            
            # 조문 추출 성공 (최대 40점)
            if stats['articles_count'] > 0:
                quality_score += min(40, stats['articles_count'] * 5)
            
            # 조문 내용 품질 (최대 30점)
            if articles:
                avg_article_length = sum(len(article.get('article_content', '')) for article in articles) / len(articles)
                if avg_article_length > 100:
                    quality_score += 30
                elif avg_article_length > 50:
                    quality_score += 20
                elif avg_article_length > 20:
                    quality_score += 10
            
            stats['quality_score'] = min(100.0, quality_score)
            
            return stats
            
        except Exception as e:
            logger.error(f"데이터 품질 계산 중 오류: {str(e)}")
            return {'quality_score': 0.0, 'error': str(e)}


def main():
    """메인 함수"""
    try:
        # 파일 경로 설정
        input_file = Path("data/raw/assembly/law/20251010/law_page_001_181503.json")
        output_file = Path("data/processed/assembly/law/20251011/refined_law_page_001_181503.json")
        
        # 파일 존재 확인
        if not input_file.exists():
            logger.error(f"입력 파일이 존재하지 않음: {input_file}")
            return
        
        # 정제 실행
        refiner = LawDataRefiner()
        result = refiner.refine_law_file(input_file, output_file)
        
        if result['success']:
            logger.info("법령 데이터 정제 완료!")
            logger.info(f"출력 파일: {result['output_file']}")
            logger.info(f"처리 통계: {result['processing_stats']}")
        else:
            logger.error(f"정제 실패: {result['error']}")
            
    except Exception as e:
        logger.error(f"메인 함수 실행 중 오류: {str(e)}")


if __name__ == "__main__":
    main()
