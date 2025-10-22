#!/usr/bin/env python3
"""
실제 Raw 데이터를 사용한 데이터베이스 재구축 스크립트
"""

import sys
import os
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
import re
from typing import Dict, List, Any, Optional
import hashlib
import uuid

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/real_data_rebuild.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RealDataProcessor:
    """실제 Raw 데이터 처리기"""
    
    def __init__(self, db_path: str = "data/lawfirm.db"):
        self.db_path = db_path
        self.processed_count = 0
        self.error_count = 0
        
    def process_raw_law_files(self, raw_dir: str) -> Dict[str, Any]:
        """Raw 법률 파일들을 처리"""
        logger.info(f"Processing raw law files from: {raw_dir}")
        
        results = {
            'processed_files': 0,
            'total_laws': 0,
            'total_articles': 0,
            'errors': []
        }
        
        raw_path = Path(raw_dir)
        if not raw_path.exists():
            logger.error(f"Raw directory not found: {raw_dir}")
            results['errors'].append(f"Raw directory not found: {raw_dir}")
            return results
        
        # 모든 JSON 파일 처리
        for file_path in raw_path.glob("**/*.json"):
            try:
                logger.info(f"Processing file: {file_path.name}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                # 법률 데이터 추출 및 처리
                laws_data = self._extract_laws_from_raw_data(raw_data, file_path)
                
                # 데이터베이스에 저장
                for law_data in laws_data:
                    self._save_law_to_database(law_data)
                    results['total_laws'] += 1
                    results['total_articles'] += len(law_data.get('articles', []))
                
                results['processed_files'] += 1
                logger.info(f"Successfully processed {file_path.name}")
                
            except Exception as e:
                error_msg = f"Error processing {file_path.name}: {e}"
                logger.error(error_msg, exc_info=True)
                results['errors'].append(error_msg)
                self.error_count += 1
        
        logger.info(f"Processing completed. Files: {results['processed_files']}, Laws: {results['total_laws']}, Articles: {results['total_articles']}")
        return results
    
    def _extract_laws_from_raw_data(self, raw_data: Dict[str, Any], file_path: Path) -> List[Dict[str, Any]]:
        """Raw 데이터에서 법률 정보 추출"""
        laws = []
        
        # Raw 데이터 구조 분석
        if isinstance(raw_data, list):
            law_items = raw_data
        elif isinstance(raw_data, dict) and 'laws' in raw_data:
            law_items = raw_data['laws']
        else:
            law_items = [raw_data]
        
        for law_item in law_items:
            try:
                law_data = self._parse_single_law(law_item, file_path)
                if law_data:
                    laws.append(law_data)
            except Exception as e:
                logger.error(f"Error parsing law item: {e}")
                continue
        
        return laws
    
    def _parse_single_law(self, law_item: Dict[str, Any], file_path: Path) -> Optional[Dict[str, Any]]:
        """단일 법률 파싱"""
        try:
            # 기본 정보 추출
            law_name = self._extract_law_name(law_item)
            law_content = self._extract_law_content(law_item)
            
            if not law_name or not law_content:
                logger.warning(f"Insufficient data for law in {file_path.name}")
                return None
            
            # 법률 ID 생성
            law_id = self._generate_law_id(law_name, law_content)
            
            # 조문 추출
            articles = self._extract_articles(law_content)
            
            # 법률 데이터 구성
            law_data = {
                'law_id': law_id,
                'source': 'assembly',
                'law_name': law_name,
                'law_type': self._extract_law_type(law_item),
                'category': self._extract_category(law_item),
                'row_number': None,
                'promulgation_number': self._extract_promulgation_number(law_item),
                'promulgation_date': self._extract_promulgation_date(law_item),
                'enforcement_date': self._extract_enforcement_date(law_item),
                'amendment_type': self._extract_amendment_type(law_item),
                'ministry': self._extract_ministry(law_item),
                'parent_law': None,
                'related_laws': json.dumps([]),
                'full_text': law_content,
                'searchable_text': self._clean_text_for_search(law_content),
                'keywords': json.dumps(self._extract_keywords(law_content)),
                'summary': self._generate_summary(law_content),
                'html_clean_text': self._clean_html_text(law_content),
                'content_html': law_content,
                'raw_content': json.dumps(law_item),
                'detail_url': self._extract_detail_url(law_item),
                'cont_id': self._extract_cont_id(law_item),
                'cont_sid': self._extract_cont_sid(law_item),
                'collected_at': self._extract_collected_at(law_item),
                'processed_at': datetime.now().isoformat(),
                'processing_version': "2.0",
                'data_quality': json.dumps({'source': 'raw_data', 'quality_score': 0.8}),
                'ml_enhanced': False,
                'parsing_quality_score': 0.8,
                'article_count': len(articles),
                'supplementary_count': sum(1 for a in articles if a.get('is_supplementary', False)),
                'control_characters_removed': True,
                'law_name_hash': hashlib.sha256(law_name.encode()).hexdigest(),
                'content_hash': hashlib.sha256(law_content.encode()).hexdigest(),
                'quality_score': 0.8,
                'parsing_method': 'raw_data_parser',
                'auto_corrected': False,
                'manual_review_required': False,
                'migration_timestamp': None,
                'is_primary_version': True,
                'version_number': 1,
                'articles': articles
            }
            
            return law_data
            
        except Exception as e:
            logger.error(f"Error parsing single law: {e}")
            return None
    
    def _extract_law_name(self, law_item: Dict[str, Any]) -> str:
        """법률명 추출"""
        # 여러 가능한 필드에서 법률명 추출
        possible_fields = ['law_name', 'title', 'name', '법률명']
        
        for field in possible_fields:
            if field in law_item and law_item[field]:
                return str(law_item[field]).strip()
        
        # HTML에서 제목 추출 시도
        content = law_item.get('content_html', '') or law_item.get('full_text', '')
        if content:
            # HTML에서 제목 태그 찾기
            title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
            if title_match:
                return self._clean_html_text(title_match.group(1)).strip()
        
        return "알 수 없는 법률"
    
    def _extract_law_content(self, law_item: Dict[str, Any]) -> str:
        """법률 내용 추출"""
        content_fields = ['content_html', 'full_text', 'content', 'law_content']
        
        for field in content_fields:
            if field in law_item and law_item[field]:
                return str(law_item[field])
        
        return ""
    
    def _extract_law_type(self, law_item: Dict[str, Any]) -> str:
        """법률 유형 추출"""
        return law_item.get('law_type', '법률')
    
    def _extract_category(self, law_item: Dict[str, Any]) -> str:
        """카테고리 추출"""
        return law_item.get('category', '기타')
    
    def _extract_promulgation_number(self, law_item: Dict[str, Any]) -> str:
        """공포번호 추출"""
        return law_item.get('promulgation_number', '')
    
    def _extract_promulgation_date(self, law_item: Dict[str, Any]) -> str:
        """공포일 추출"""
        return law_item.get('promulgation_date', '')
    
    def _extract_enforcement_date(self, law_item: Dict[str, Any]) -> str:
        """시행일 추출"""
        return law_item.get('enforcement_date', '')
    
    def _extract_amendment_type(self, law_item: Dict[str, Any]) -> str:
        """개정 유형 추출"""
        return law_item.get('amendment_type', '')
    
    def _extract_ministry(self, law_item: Dict[str, Any]) -> str:
        """소관부처 추출"""
        return law_item.get('ministry', '')
    
    def _extract_detail_url(self, law_item: Dict[str, Any]) -> str:
        """상세 URL 추출"""
        return law_item.get('detail_url', '')
    
    def _extract_cont_id(self, law_item: Dict[str, Any]) -> str:
        """컨텐츠 ID 추출"""
        return law_item.get('cont_id', '')
    
    def _extract_cont_sid(self, law_item: Dict[str, Any]) -> str:
        """컨텐츠 SID 추출"""
        return law_item.get('cont_sid', '')
    
    def _extract_collected_at(self, law_item: Dict[str, Any]) -> str:
        """수집일시 추출"""
        return law_item.get('collected_at', datetime.now().isoformat())
    
    def _extract_articles(self, content: str) -> List[Dict[str, Any]]:
        """조문 추출"""
        articles = []
        
        # 조문 패턴 찾기 (간단한 정규식 사용)
        article_patterns = [
            r'제(\d+)조\s*\(([^)]+)\)\s*([^제]+?)(?=제\d+조|$)',
            r'제(\d+)조\s*([^제]+?)(?=제\d+조|$)',
        ]
        
        for pattern in article_patterns:
            matches = re.finditer(pattern, content, re.DOTALL | re.MULTILINE)
            for match in matches:
                article_number = match.group(1)
                article_title = match.group(2) if len(match.groups()) > 1 else ""
                article_content = match.group(3) if len(match.groups()) > 2 else match.group(2)
                
                if article_content:
                    article_data = {
                        'article_id': str(uuid.uuid4()),
                        'article_number': article_number,
                        'article_title': self._clean_html_text(article_title).strip(),
                        'article_content': self._clean_html_text(article_content).strip(),
                        'is_supplementary': False,
                        'ml_confidence_score': 0.8,
                        'parsing_method': 'raw_data_parser',
                        'article_type': '조문',
                        'word_count': len(article_content.split()),
                        'char_count': len(article_content),
                        'sub_articles': json.dumps([]),
                        'law_references': json.dumps([]),
                        'parsing_quality_score': 0.8
                    }
                    articles.append(article_data)
        
        return articles
    
    def _clean_text_for_search(self, text: str) -> str:
        """검색용 텍스트 정리"""
        if not text:
            return ""
        
        # HTML 태그 제거
        clean_text = re.sub(r'<[^>]+>', ' ', text)
        # 특수 문자 정리
        clean_text = re.sub(r'&[^;]+;', ' ', clean_text)
        # 공백 정리
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        return clean_text.strip()
    
    def _clean_html_text(self, text: str) -> str:
        """HTML 텍스트 정리"""
        if not text:
            return ""
        
        # HTML 태그 제거
        clean_text = re.sub(r'<[^>]+>', '', text)
        # HTML 엔티티 디코딩
        clean_text = clean_text.replace('&nbsp;', ' ')
        clean_text = clean_text.replace('&lt;', '<')
        clean_text = clean_text.replace('&gt;', '>')
        clean_text = clean_text.replace('&amp;', '&')
        # 공백 정리
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        return clean_text.strip()
    
    def _extract_keywords(self, content: str) -> List[str]:
        """키워드 추출 (간단한 구현)"""
        keywords = []
        
        # 법률 관련 키워드 패턴
        keyword_patterns = [
            r'제(\d+)조',
            r'제(\d+)장',
            r'제(\d+)절',
            r'법률',
            r'시행령',
            r'시행규칙'
        ]
        
        for pattern in keyword_patterns:
            matches = re.findall(pattern, content)
            keywords.extend(matches)
        
        return list(set(keywords))[:10]  # 최대 10개
    
    def _generate_summary(self, content: str) -> str:
        """요약 생성 (간단한 구현)"""
        if not content:
            return ""
        
        # 첫 번째 문단을 요약으로 사용
        clean_content = self._clean_html_text(content)
        sentences = clean_content.split('.')
        
        if sentences:
            return sentences[0][:200] + "..." if len(sentences[0]) > 200 else sentences[0]
        
        return clean_content[:200] + "..." if len(clean_content) > 200 else clean_content
    
    def _generate_law_id(self, law_name: str, content: str) -> str:
        """법률 ID 생성"""
        # 법률명과 내용의 해시를 사용하여 고유 ID 생성
        combined = f"{law_name}_{content[:100]}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _save_law_to_database(self, law_data: Dict[str, Any]):
        """법률 데이터를 데이터베이스에 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # assembly_laws 테이블에 삽입
                law_record = (
                    law_data['law_id'],
                    law_data['source'],
                    law_data['law_name'],
                    law_data['law_type'],
                    law_data['category'],
                    law_data['row_number'],
                    law_data['promulgation_number'],
                    law_data['promulgation_date'],
                    law_data['enforcement_date'],
                    law_data['amendment_type'],
                    law_data['ministry'],
                    law_data['parent_law'],
                    law_data['related_laws'],
                    law_data['full_text'],
                    law_data['searchable_text'],
                    law_data['keywords'],
                    law_data['summary'],
                    law_data['html_clean_text'],
                    law_data['content_html'],
                    law_data['raw_content'],
                    law_data['detail_url'],
                    law_data['cont_id'],
                    law_data['cont_sid'],
                    law_data['collected_at'],
                    law_data['processed_at'],
                    law_data['processing_version'],
                    law_data['data_quality'],
                    law_data['ml_enhanced'],
                    law_data['parsing_quality_score'],
                    law_data['article_count'],
                    law_data['supplementary_count'],
                    law_data['control_characters_removed'],
                    law_data['law_name_hash'],
                    law_data['content_hash'],
                    law_data['quality_score'],
                    law_data['parsing_method'],
                    law_data['auto_corrected'],
                    law_data['manual_review_required'],
                    law_data['migration_timestamp'],
                    law_data['is_primary_version'],
                    law_data['version_number']
                )
                
                cursor.execute("""
                    INSERT OR REPLACE INTO assembly_laws 
                    (law_id, source, law_name, law_type, category, row_number,
                     promulgation_number, promulgation_date, enforcement_date, amendment_type,
                     ministry, parent_law, related_laws, full_text, searchable_text,
                     keywords, summary, html_clean_text, content_html, raw_content,
                     detail_url, cont_id, cont_sid, collected_at, processed_at,
                     processing_version, data_quality, ml_enhanced, parsing_quality_score,
                     article_count, supplementary_count, control_characters_removed,
                     law_name_hash, content_hash, quality_score, parsing_method,
                     auto_corrected, manual_review_required, migration_timestamp,
                     is_primary_version, version_number)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, law_record)
                
                # assembly_articles 테이블에 삽입
                for article in law_data.get('articles', []):
                    article_record = (
                        article['article_id'],
                        law_data['law_id'],
                        article['article_number'],
                        article['article_title'],
                        article['article_content'],
                        article['is_supplementary'],
                        article['ml_confidence_score'],
                        article['parsing_method'],
                        article['article_type'],
                        article['word_count'],
                        article['char_count'],
                        article['sub_articles'],
                        article['law_references'],
                        article['parsing_quality_score']
                    )
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO assembly_articles
                        (article_id, law_id, article_number, article_title, article_content,
                         is_supplementary, ml_confidence_score, parsing_method, article_type,
                         word_count, char_count, sub_articles, law_references, parsing_quality_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, article_record)
                
                conn.commit()
                self.processed_count += 1
                
        except Exception as e:
            logger.error(f"Error saving law to database: {e}")
            raise


def main():
    """메인 함수"""
    print("Starting Real Data Database Rebuild Process...")
    print("This process will rebuild the database from actual raw law data.")
    print("="*80)
    
    start_time = datetime.now()
    
    try:
        # 데이터베이스 백업
        db_path = Path("data/lawfirm.db")
        if db_path.exists():
            backup_path = Path(f"data/lawfirm_backup_real_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
            import shutil
            shutil.copy2(db_path, backup_path)
            logger.info(f"Database backed up to: {backup_path}")
        
        # 기존 데이터 정리
        logger.info("Clearing existing data...")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM assembly_articles")
            cursor.execute("DELETE FROM assembly_laws")
            conn.commit()
        logger.info("Existing data cleared.")
        
        # Raw 데이터 처리
        processor = RealDataProcessor()
        raw_dir = "data/raw/assembly/law"
        
        logger.info(f"Processing raw data from: {raw_dir}")
        results = processor.process_raw_law_files(raw_dir)
        
        # 결과 출력
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        print("\n" + "="*80)
        print("REAL DATA DATABASE REBUILD COMPLETED!")
        print("="*80)
        print(f"Processing Summary:")
        print(f"  - Total Time: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s")
        print(f"  - Processed Files: {results['processed_files']:,}")
        print(f"  - Total Laws: {results['total_laws']:,}")
        print(f"  - Total Articles: {results['total_articles']:,}")
        print(f"  - Errors: {len(results['errors'])}")
        
        if results['errors']:
            print(f"\nErrors Encountered:")
            for i, error in enumerate(results['errors'][:5], 1):
                print(f"  {i}. {error}")
            if len(results['errors']) > 5:
                print(f"  ... and {len(results['errors']) - 5} more errors")
        
        # 데이터베이스 검증
        logger.info("Validating database...")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM assembly_laws")
            law_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM assembly_articles")
            article_count = cursor.fetchone()[0]
        
        print(f"\nDatabase Validation:")
        print(f"  - Laws in database: {law_count:,}")
        print(f"  - Articles in database: {article_count:,}")
        
        # 리포트 저장
        report_data = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_time_seconds': total_time,
            'results': results,
            'database_validation': {
                'law_count': law_count,
                'article_count': article_count
            }
        }
        
        report_path = Path("data/real_data_rebuild_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Report saved to: {report_path}")
        print(f"\nReport saved to: {report_path}")
        
        print("\nReal data database rebuild process completed!")
        print("You can now use the LawFirmAI system with the rebuilt database.")
        
        return 0
        
    except Exception as e:
        logger.error(f"Critical error during real data rebuild: {e}", exc_info=True)
        print(f"\nProcess failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
