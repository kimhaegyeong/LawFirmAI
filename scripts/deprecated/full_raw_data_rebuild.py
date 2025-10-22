#!/usr/bin/env python3
"""
조문 처리 기능이 포함된 Raw 데이터 재구축 스크립트
"""

import sys
import os
from pathlib import Path
import json
import logging
from datetime import datetime
import sqlite3
from typing import List, Dict, Any, Optional
import re

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/full_raw_data_rebuild.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Raw 데이터 파서 클래스 (인라인)
import html
from bs4 import BeautifulSoup

class RawDataParser:
    """Raw 데이터 파서 클래스 (조문 처리 포함)"""
    
    def __init__(self):
        self.processed_count = 0
        self.error_count = 0
        
    def parse_raw_file(self, file_path: Path) -> Dict[str, List[Dict[str, Any]]]:
        """Raw JSON 파일을 파싱하여 법률 및 조문 데이터 추출"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            laws = []
            articles = []
            
            # metadata 확인
            metadata = data.get('metadata', {})
            logger.info(f"Processing file: {file_path.name}, Items: {len(data.get('items', []))}")
            
            # items 배열 처리
            for item in data.get('items', []):
                try:
                    law_data = self._parse_law_item(item, metadata)
                    if law_data:
                        laws.append(law_data)
                        
                        # 조문 추출
                        law_articles = self._extract_articles_from_law(item, law_data['law_id'])
                        articles.extend(law_articles)
                        
                except Exception as e:
                    logger.error(f"Error parsing item {item.get('cont_id', 'unknown')}: {e}")
                    self.error_count += 1
            
            self.processed_count += len(laws)
            logger.info(f"Successfully parsed {len(laws)} laws and {len(articles)} articles from {file_path.name}")
            
            return {
                'laws': laws,
                'articles': articles
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            self.error_count += 1
            return {'laws': [], 'articles': []}
    
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
            
            # 데이터 품질 평가
            quality_score = self._calculate_quality_score(item, clean_text, [])
            
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
                'article_count': 0,  # 나중에 업데이트됨
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
    
    def _extract_articles_from_law(self, item: Dict[str, Any], law_id: str) -> List[Dict[str, Any]]:
        """법률에서 조문 추출"""
        articles = []
        
        try:
            content_html = item.get('content_html', '')
            if not content_html:
                return articles
            
            soup = BeautifulSoup(content_html, 'html.parser')
            
            # 조문 패턴 찾기 - 다양한 패턴 시도
            article_patterns = [
                # 제1조, 제2조 등의 패턴
                r'제\s*(\d+)\s*조\s*([^제]*?)(?=제\s*\d+\s*조|$)',
                # 제1조(목적), 제2조(정의) 등의 패턴
                r'제\s*(\d+)\s*조\s*\(([^)]+)\)\s*([^제]*?)(?=제\s*\d+\s*조|$)',
                # 제1조 목적 등의 패턴
                r'제\s*(\d+)\s*조\s*([가-힣]+)\s*([^제]*?)(?=제\s*\d+\s*조|$)',
            ]
            
            # HTML에서 텍스트 추출
            text_content = soup.get_text()
            
            for pattern in article_patterns:
                matches = re.finditer(pattern, text_content, re.DOTALL | re.MULTILINE)
                
                for match in matches:
                    try:
                        article_number = match.group(1).strip()
                        article_title = match.group(2).strip() if len(match.groups()) > 1 else ''
                        article_content = match.group(3).strip() if len(match.groups()) > 2 else match.group(2).strip()
                        
                        # 조문 내용이 너무 짧으면 스킵
                        if len(article_content) < 10:
                            continue
                        
                        # 조문 데이터 생성
                        article_data = {
                            'law_id': law_id,
                            'article_number': article_number,
                            'article_title': article_title,
                            'article_content': article_content,
                            'sub_articles': '',
                            'law_references': '',
                            'word_count': len(article_content.split()),
                            'char_count': len(article_content),
                            'created_at': datetime.now().isoformat(),
                            'updated_at': datetime.now().isoformat(),
                            'is_supplementary': False,
                            'ml_confidence_score': 0.8,  # 기본 신뢰도
                            'parsing_method': 'regex_pattern',
                            'article_type': 'main',
                            'parsing_quality_score': self._calculate_article_quality(article_content, article_title)
                        }
                        
                        articles.append(article_data)
                        
                    except Exception as e:
                        logger.debug(f"Error parsing article match: {e}")
                        continue
            
            # 추가 조문 패턴 - span 태그에서 추출
            spans = soup.find_all('span')
            for span in spans:
                text = span.get_text().strip()
                if text and len(text) > 20:
                    # 조문 번호 패턴 확인
                    article_match = re.match(r'제\s*(\d+)\s*조', text)
                    if article_match:
                        article_number = article_match.group(1)
                        article_content = text
                        
                        article_data = {
                            'law_id': law_id,
                            'article_number': article_number,
                            'article_title': '',
                            'article_content': article_content,
                            'sub_articles': '',
                            'law_references': '',
                            'word_count': len(article_content.split()),
                            'char_count': len(article_content),
                            'created_at': datetime.now().isoformat(),
                            'updated_at': datetime.now().isoformat(),
                            'is_supplementary': False,
                            'ml_confidence_score': 0.7,
                            'parsing_method': 'span_extraction',
                            'article_type': 'main',
                            'parsing_quality_score': self._calculate_article_quality(article_content, '')
                        }
                        
                        articles.append(article_data)
            
            # 중복 제거 (같은 조문 번호)
            seen_numbers = set()
            unique_articles = []
            for article in articles:
                if article['article_number'] not in seen_numbers:
                    seen_numbers.add(article['article_number'])
                    unique_articles.append(article)
            
            logger.debug(f"Extracted {len(unique_articles)} articles from law {law_id}")
            return unique_articles
            
        except Exception as e:
            logger.error(f"Error extracting articles from law {law_id}: {e}")
            return []
    
    def _calculate_article_quality(self, content: str, title: str) -> float:
        """조문 품질 점수 계산"""
        score = 0.0
        
        try:
            # 기본 점수
            if content and len(content) > 20:
                score += 0.3
            if title and len(title) > 0:
                score += 0.2
            if '조' in content:
                score += 0.2
            if len(content.split()) > 5:
                score += 0.2
            if any(keyword in content for keyword in ['법률', '규정', '의무', '권리', '금지', '허용']):
                score += 0.1
            
        except Exception as e:
            logger.error(f"Error calculating article quality: {e}")
        
        return min(score, 1.0)
    
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
    
    def _extract_keywords(self, text: str) -> str:
        """텍스트에서 키워드 추출"""
        try:
            words = re.findall(r'[가-힣]{2,}', text)
            keywords = list(set(words))[:10]  # 상위 10개
            return ', '.join(keywords)
        except:
            return ''
    
    def _generate_summary(self, text: str) -> str:
        """텍스트 요약 생성"""
        try:
            if len(text) > 200:
                return text[:200] + '...'
            return text
        except:
            return ''
    
    def _calculate_quality_score(self, item: Dict[str, Any], clean_text: str, articles: List[Dict[str, Any]]) -> float:
        """데이터 품질 점수 계산"""
        score = 0.0
        
        try:
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

class DatabaseManager:
    """데이터베이스 관리 클래스 (조문 처리 포함)"""
    
    def __init__(self, db_path: str = "data/lawfirm.db"):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """데이터베이스 연결"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.execute("PRAGMA foreign_keys = ON")
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def disconnect(self):
        """데이터베이스 연결 해제"""
        if self.conn:
            self.conn.close()
            logger.info("Database disconnected")
    
    def clear_existing_data(self):
        """기존 데이터 정리"""
        try:
            cursor = self.conn.cursor()
            
            # 기존 데이터 삭제
            cursor.execute("DELETE FROM assembly_laws")
            cursor.execute("DELETE FROM assembly_articles")
            
            # 시퀀스 리셋
            cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('assembly_laws', 'assembly_articles')")
            
            self.conn.commit()
            logger.info("Existing data cleared")
            
        except Exception as e:
            logger.error(f"Error clearing existing data: {e}")
            raise
    
    def insert_law(self, law_data: Dict[str, Any]) -> int:
        """법률 데이터 삽입"""
        try:
            cursor = self.conn.cursor()
            
            # 필요한 컬럼만 추출 (45개 컬럼에 맞춤)
            columns = [
                'law_id', 'source', 'law_name', 'law_type', 'category', 'row_number',
                'promulgation_number', 'promulgation_date', 'enforcement_date', 'amendment_type',
                'ministry', 'parent_law', 'related_laws', 'full_text', 'searchable_text',
                'keywords', 'summary', 'html_clean_text', 'content_html', 'raw_content',
                'detail_url', 'cont_id', 'cont_sid', 'collected_at', 'processed_at',
                'processing_version', 'data_quality', 'created_at', 'updated_at',
                'ml_enhanced', 'parsing_quality_score', 'article_count', 'supplementary_count',
                'control_characters_removed', 'law_name_hash', 'content_hash', 'quality_score',
                'duplicate_group_id', 'is_primary_version', 'version_number', 'parsing_method',
                'auto_corrected', 'manual_review_required', 'migration_timestamp'
            ]
            
            # 값 추출
            values = []
            for col in columns:
                value = law_data.get(col)
                if value is None:
                    if col in ['ml_enhanced', 'control_characters_removed', 'is_primary_version', 'auto_corrected', 'manual_review_required']:
                        value = False
                    elif col in ['parsing_quality_score', 'quality_score']:
                        value = 0.0
                    elif col in ['article_count', 'supplementary_count', 'version_number']:
                        value = 0
                    else:
                        value = ''
                values.append(value)
            
            # INSERT 쿼리 실행
            placeholders = ','.join(['?' for _ in columns])
            query = f"INSERT INTO assembly_laws ({','.join(columns)}) VALUES ({placeholders})"
            
            cursor.execute(query, values)
            law_db_id = cursor.lastrowid
            
            self.conn.commit()
            return law_db_id
            
        except Exception as e:
            logger.error(f"Error inserting law {law_data.get('law_name', 'unknown')}: {e}")
            raise
    
    def insert_article(self, article_data: Dict[str, Any]) -> int:
        """조문 데이터 삽입"""
        try:
            cursor = self.conn.cursor()
            
            # 필요한 컬럼만 추출
            columns = [
                'law_id', 'article_number', 'article_title', 'article_content',
                'sub_articles', 'law_references', 'word_count', 'char_count',
                'created_at', 'updated_at', 'is_supplementary', 'ml_confidence_score',
                'parsing_method', 'article_type', 'parsing_quality_score'
            ]
            
            # 값 추출
            values = []
            for col in columns:
                value = article_data.get(col)
                if value is None:
                    if col in ['is_supplementary']:
                        value = False
                    elif col in ['ml_confidence_score', 'parsing_quality_score']:
                        value = 0.0
                    elif col in ['word_count', 'char_count']:
                        value = 0
                    else:
                        value = ''
                values.append(value)
            
            # INSERT 쿼리 실행
            placeholders = ','.join(['?' for _ in columns])
            query = f"INSERT INTO assembly_articles ({','.join(columns)}) VALUES ({placeholders})"
            
            cursor.execute(query, values)
            article_db_id = cursor.lastrowid
            
            self.conn.commit()
            return article_db_id
            
        except Exception as e:
            logger.error(f"Error inserting article {article_data.get('article_number', 'unknown')}: {e}")
            raise
    
    def update_law_article_count(self, law_id: str, article_count: int):
        """법률의 조문 수 업데이트"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE assembly_laws SET article_count = ? WHERE law_id = ?",
                (article_count, law_id)
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error updating article count for law {law_id}: {e}")
    
    def get_statistics(self) -> Dict[str, int]:
        """데이터베이스 통계 조회"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM assembly_laws")
            law_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM assembly_articles")
            article_count = cursor.fetchone()[0]
            
            return {
                'laws': law_count,
                'articles': article_count
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'laws': 0, 'articles': 0}

def process_raw_data_files(raw_dir: Path, db_manager: DatabaseManager) -> Dict[str, Any]:
    """Raw 데이터 파일들을 처리하여 데이터베이스에 저장 (조문 포함)"""
    
    parser = RawDataParser()
    results = {
        'processed_files': 0,
        'total_laws': 0,
        'total_articles': 0,
        'successful_law_inserts': 0,
        'successful_article_inserts': 0,
        'errors': []
    }
    
    try:
        # Raw 데이터 파일들 찾기
        json_files = list(raw_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        for file_path in json_files:
            try:
                logger.info(f"Processing file: {file_path.name}")
                
                # 파일 파싱
                parsed_data = parser.parse_raw_file(file_path)
                laws = parsed_data.get('laws', [])
                articles = parsed_data.get('articles', [])
                
                if not laws:
                    logger.warning(f"No laws found in {file_path.name}")
                    continue
                
                # 법률 데이터베이스에 저장
                for law in laws:
                    try:
                        law_db_id = db_manager.insert_law(law)
                        results['successful_law_inserts'] += 1
                        logger.debug(f"Inserted law: {law['law_name']} (ID: {law_db_id})")
                    except Exception as e:
                        error_msg = f"Error inserting law {law.get('law_name', 'unknown')}: {e}"
                        logger.error(error_msg)
                        results['errors'].append(error_msg)
                
                # 조문 데이터베이스에 저장
                for article in articles:
                    try:
                        article_db_id = db_manager.insert_article(article)
                        results['successful_article_inserts'] += 1
                        logger.debug(f"Inserted article: {article['law_id']} 제{article['article_number']}조 (ID: {article_db_id})")
                    except Exception as e:
                        error_msg = f"Error inserting article {article.get('article_number', 'unknown')}: {e}"
                        logger.error(error_msg)
                        results['errors'].append(error_msg)
                
                # 법률의 조문 수 업데이트
                if articles:
                    law_id = laws[0]['law_id'] if laws else ''
                    if law_id:
                        db_manager.update_law_article_count(law_id, len(articles))
                
                results['processed_files'] += 1
                results['total_laws'] += len(laws)
                results['total_articles'] += len(articles)
                
                logger.info(f"Successfully processed {file_path.name}: {len(laws)} laws, {len(articles)} articles")
                
            except Exception as e:
                error_msg = f"Error processing file {file_path.name}: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
    
    except Exception as e:
        error_msg = f"Critical error in process_raw_data_files: {e}"
        logger.error(error_msg)
        results['errors'].append(error_msg)
    
    return results

def main():
    """메인 함수"""
    logger.info("Starting Full Raw Data Database Rebuild (Laws + Articles)")
    logger.info("="*70)
    
    start_time = datetime.now()
    
    # Raw 데이터 디렉토리 설정
    raw_dir = Path("data/raw/assembly/law/20251016")
    
    if not raw_dir.exists():
        logger.error(f"Raw data directory not found: {raw_dir}")
        return 1
    
    # 데이터베이스 관리자 초기화
    db_manager = DatabaseManager()
    
    try:
        # 데이터베이스 연결
        db_manager.connect()
        
        # 기존 데이터 정리
        logger.info("Clearing existing data...")
        db_manager.clear_existing_data()
        
        # Raw 데이터 처리
        logger.info("Processing raw data files...")
        results = process_raw_data_files(raw_dir, db_manager)
        
        # 통계 출력
        logger.info("="*70)
        logger.info("Processing Results:")
        logger.info(f"  - Processed Files: {results['processed_files']}")
        logger.info(f"  - Total Laws Found: {results['total_laws']}")
        logger.info(f"  - Total Articles Found: {results['total_articles']}")
        logger.info(f"  - Successfully Inserted Laws: {results['successful_law_inserts']}")
        logger.info(f"  - Successfully Inserted Articles: {results['successful_article_inserts']}")
        logger.info(f"  - Errors: {len(results['errors'])}")
        
        # 데이터베이스 통계
        stats = db_manager.get_statistics()
        logger.info(f"  - Laws in Database: {stats['laws']}")
        logger.info(f"  - Articles in Database: {stats['articles']}")
        
        # 오류 출력
        if results['errors']:
            logger.info("Errors encountered:")
            for i, error in enumerate(results['errors'][:10], 1):  # 처음 10개만
                logger.info(f"  {i}. {error}")
            if len(results['errors']) > 10:
                logger.info(f"  ... and {len(results['errors']) - 10} more errors")
        
        # 처리 시간
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        
        # 결과 저장
        report = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_time_seconds': total_time,
            'results': results,
            'database_stats': stats
        }
        
        report_path = Path("data/full_raw_data_rebuild_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Report saved to: {report_path}")
        
        logger.info("="*70)
        logger.info("Full Raw Data Database Rebuild Completed!")
        
        return 0 if results['successful_law_inserts'] > 0 else 1
        
    except Exception as e:
        logger.error(f"Critical error: {e}")
        return 1
    
    finally:
        db_manager.disconnect()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)



