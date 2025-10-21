#!/usr/bin/env python3
"""
Raw 데이터 파서 - 실제 Raw JSON 파일 구조에 맞춘 파서
"""

import json
import re
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import html
from bs4 import BeautifulSoup

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/raw_data_parser.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RawDataParser:
    """Raw 데이터 파서 클래스"""
    
    def __init__(self):
        self.processed_count = 0
        self.error_count = 0
        
    def parse_raw_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Raw JSON 파일을 파싱하여 법률 데이터 추출"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            laws = []
            
            # metadata 확인
            metadata = data.get('metadata', {})
            logger.info(f"Processing file: {file_path.name}, Items: {len(data.get('items', []))}")
            
            # items 배열 처리
            for item in data.get('items', []):
                try:
                    law_data = self._parse_law_item(item, metadata)
                    if law_data:
                        laws.append(law_data)
                except Exception as e:
                    logger.error(f"Error parsing item {item.get('cont_id', 'unknown')}: {e}")
                    self.error_count += 1
            
            self.processed_count += len(laws)
            logger.info(f"Successfully parsed {len(laws)} laws from {file_path.name}")
            return laws
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            self.error_count += 1
            return []
    
    def _parse_law_item(self, item: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """개별 법률 항목 파싱"""
        try:
            # 기본 정보 추출
            cont_id = item.get('cont_id', '')
            cont_sid = item.get('cont_sid', '')
            law_name = item.get('law_name', '')
            law_content = item.get('law_content', '')
            content_html = item.get('content_html', '')
            collected_at = item.get('collected_at', '')
            
            # HTML에서 텍스트 추출
            clean_text = self._extract_text_from_html(content_html)
            
            # 법률 정보 파싱
            law_info = self._parse_law_info(law_content, law_name)
            
            # 조문 추출
            articles = self._extract_articles(content_html)
            
            # 데이터 품질 평가
            quality_score = self._calculate_quality_score(item, clean_text, articles)
            
            # 데이터베이스에 저장할 형태로 변환
            law_data = {
                # 기본 식별자
                'law_id': cont_id,
                'cont_id': cont_id,
                'cont_sid': cont_sid,
                'source': 'assembly_raw',
                
                # 법률 기본 정보
                'law_name': law_name,
                'law_type': law_info.get('law_type', '법률'),
                'category': metadata.get('category'),
                'row_number': cont_sid,
                
                # 공포/시행 정보
                'promulgation_number': law_info.get('promulgation_number', ''),
                'promulgation_date': law_info.get('promulgation_date', ''),
                'enforcement_date': law_info.get('enforcement_date', ''),
                'amendment_type': law_info.get('amendment_type', '제정'),
                'ministry': law_info.get('ministry', ''),
                
                # 관련 법률
                'parent_law': '',
                'related_laws': '',
                
                # 텍스트 내용
                'full_text': clean_text,
                'searchable_text': clean_text,
                'keywords': self._extract_keywords(clean_text),
                'summary': self._generate_summary(clean_text),
                'html_clean_text': clean_text,
                'content_html': content_html,
                'raw_content': law_content,
                
                # URL 및 메타데이터
                'detail_url': '',
                'collected_at': collected_at,
                'processed_at': datetime.now().isoformat(),
                'processing_version': '1.0',
                'data_quality': 'good' if quality_score > 0.7 else 'poor',
                
                # 품질 관련
                'ml_enhanced': False,
                'parsing_quality_score': quality_score,
                'article_count': len(articles),
                'supplementary_count': 0,
                'control_characters_removed': True,
                'law_name_hash': hash(law_name),
                'content_hash': hash(clean_text),
                'quality_score': quality_score,
                
                # 중복 처리
                'duplicate_group_id': '',
                'is_primary_version': True,
                'version_number': 1,
                'parsing_method': 'raw_data_parser',
                'auto_corrected': False,
                'manual_review_required': quality_score < 0.5,
                'migration_timestamp': datetime.now().isoformat(),
                
                # 타임스탬프
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            return law_data
            
        except Exception as e:
            logger.error(f"Error parsing law item: {e}")
            return None
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """HTML에서 깨끗한 텍스트 추출"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 불필요한 태그 제거
            for tag in soup(['script', 'style', 'img', 'br']):
                tag.decompose()
            
            # 텍스트 추출
            text = soup.get_text()
            
            # 정리
            text = re.sub(r'\s+', ' ', text)  # 여러 공백을 하나로
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return html_content
    
    def _parse_law_info(self, law_content: str, law_name: str) -> Dict[str, str]:
        """법률 내용에서 정보 추출"""
        info = {
            'law_type': '법률',
            'promulgation_number': '',
            'promulgation_date': '',
            'enforcement_date': '',
            'amendment_type': '제정',
            'ministry': ''
        }
        
        try:
            # 시행일 추출
            enforcement_match = re.search(r'\[시행\s+(\d{4}\.\d{1,2}\.\d{1,2})\.\]', law_content)
            if enforcement_match:
                info['enforcement_date'] = enforcement_match.group(1)
            
            # 공포일 및 번호 추출
            promulgation_match = re.search(r'\[([^,]+),\s*(\d{4}\.\d{1,2}\.\d{1,2})\.\]', law_content)
            if promulgation_match:
                info['promulgation_number'] = promulgation_match.group(1)
                info['promulgation_date'] = promulgation_match.group(2)
            
            # 법률 유형 추출
            if '포고' in law_name:
                info['law_type'] = '포고'
            elif '령' in law_name:
                info['law_type'] = '령'
            elif '규칙' in law_name:
                info['law_type'] = '규칙'
            
        except Exception as e:
            logger.error(f"Error parsing law info: {e}")
        
        return info
    
    def _extract_articles(self, html_content: str) -> List[Dict[str, Any]]:
        """HTML에서 조문 추출"""
        articles = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 조문 패턴 찾기
            # 실제 HTML 구조에 따라 조정 필요
            spans = soup.find_all('span')
            for span in spans:
                text = span.get_text().strip()
                if text and len(text) > 10:  # 의미있는 텍스트만
                    articles.append({
                        'article_text': text,
                        'article_number': '',
                        'article_title': ''
                    })
            
        except Exception as e:
            logger.error(f"Error extracting articles: {e}")
        
        return articles
    
    def _extract_keywords(self, text: str) -> str:
        """텍스트에서 키워드 추출"""
        try:
            # 간단한 키워드 추출 (실제로는 더 정교한 방법 사용)
            words = re.findall(r'[가-힣]{2,}', text)
            keywords = list(set(words))[:10]  # 상위 10개
            return ', '.join(keywords)
        except:
            return ''
    
    def _generate_summary(self, text: str) -> str:
        """텍스트 요약 생성"""
        try:
            # 간단한 요약 (실제로는 더 정교한 방법 사용)
            if len(text) > 200:
                return text[:200] + '...'
            return text
        except:
            return ''
    
    def _calculate_quality_score(self, item: Dict[str, Any], clean_text: str, articles: List[Dict[str, Any]]) -> float:
        """데이터 품질 점수 계산"""
        score = 0.0
        
        try:
            # 기본 점수
            if item.get('cont_id'):
                score += 0.2
            if item.get('law_name'):
                score += 0.2
            if clean_text and len(clean_text) > 50:
                score += 0.3
            if articles:
                score += 0.2
            if item.get('collected_at'):
                score += 0.1
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
        
        return min(score, 1.0)

def main():
    """메인 함수"""
    parser = RawDataParser()
    
    # 테스트용 파일 처리
    test_file = Path("data/raw/assembly/law/20251016/law_only_page_019_20251016_191650_5.json")
    
    if test_file.exists():
        logger.info(f"Processing test file: {test_file}")
        laws = parser.parse_raw_file(test_file)
        
        logger.info(f"Parsed {len(laws)} laws")
        for i, law in enumerate(laws[:3]):  # 처음 3개만 출력
            logger.info(f"Law {i+1}: {law['law_name']} (Quality: {law['quality_score']:.2f})")
    
    else:
        logger.error(f"Test file not found: {test_file}")

if __name__ == "__main__":
    main()
