#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
배치 파일에서 조문을 추출하여 개별 조문 테이블에 저장하는 스크립트
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.database import DatabaseManager
from source.utils.logger import setup_logger

class BatchArticleParser:
    """배치 파일에서 조문을 파싱하는 클래스"""
    
    def __init__(self, batch_dir: str):
        self.batch_dir = Path(batch_dir)
        self.logger = setup_logger("batch_article_parser")
        self.db_manager = DatabaseManager()
        
        # 통계 정보
        self.stats = {
            'total_batches': 0,
            'total_laws': 0,
            'total_articles': 0,
            'total_paragraphs': 0,
            'parsing_errors': []
        }
    
    def parse_all_batches(self) -> Dict[str, Any]:
        """모든 배치 파일을 파싱하여 조문 추출"""
        self.logger.info("배치 파일 파싱 시작")
        
        # 1. 조문 테이블 생성
        self._create_articles_table()
        
        # 2. 배치 파일 목록 가져오기
        batch_files = self._get_batch_files()
        self.stats['total_batches'] = len(batch_files)
        
        self.logger.info(f"총 {len(batch_files)}개 배치 파일 발견")
        
        # 3. 각 배치 파일 파싱
        all_articles = []
        for batch_file in batch_files:
            try:
                articles = self._parse_batch_file(batch_file)
                all_articles.extend(articles)
                self.logger.info(f"배치 파일 {batch_file.name} 파싱 완료: {len(articles)}개 조문")
            except Exception as e:
                error_msg = f"배치 파일 {batch_file.name} 파싱 실패: {e}"
                self.logger.error(error_msg)
                self.stats['parsing_errors'].append(error_msg)
        
        # 4. 데이터베이스에 저장
        if all_articles:
            self._save_articles_to_database(all_articles)
        
        # 5. 통계 출력
        self._print_statistics()
        
        return self.stats
    
    def _create_articles_table(self):
        """조문 테이블 생성"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # current_laws_articles 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS current_laws_articles (
                    article_id TEXT PRIMARY KEY,
                    law_id TEXT NOT NULL,
                    law_name_korean TEXT NOT NULL,
                    article_number INTEGER NOT NULL,
                    article_title TEXT,
                    article_content TEXT NOT NULL,
                    paragraph_number INTEGER,
                    paragraph_content TEXT,
                    sub_paragraph_number TEXT,
                    sub_paragraph_content TEXT,
                    is_supplementary BOOLEAN DEFAULT FALSE,
                    amendment_type TEXT,
                    effective_date TEXT,
                    parsing_method TEXT DEFAULT 'batch_parser',
                    quality_score REAL DEFAULT 0.9,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (law_id) REFERENCES current_laws(law_id)
                )
            """)
            
            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_current_laws_articles_law_article 
                ON current_laws_articles(law_id, article_number)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_current_laws_articles_law_name 
                ON current_laws_articles(law_name_korean, article_number)
            """)
            
            # FTS 테이블 생성
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS current_laws_articles_fts USING fts5(
                    article_content,
                    article_title,
                    paragraph_content,
                    content='current_laws_articles',
                    content_rowid='rowid'
                )
            """)
            
            self.logger.info("조문 테이블 생성 완료")
    
    def _get_batch_files(self) -> List[Path]:
        """배치 파일 목록 가져오기"""
        pattern = "current_law_batch_*.json"
        batch_files = list(self.batch_dir.glob(pattern))
        
        # 요약 파일 제외
        batch_files = [f for f in batch_files if "summary" not in f.name]
        
        return sorted(batch_files)
    
    def _parse_batch_file(self, batch_file: Path) -> List[Dict[str, Any]]:
        """배치 파일에서 조문 추출"""
        with open(batch_file, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        
        articles = []
        laws = batch_data.get('laws', [])
        
        for law in laws:
            try:
                law_articles = self._parse_law_articles(law)
                articles.extend(law_articles)
                self.stats['total_laws'] += 1
            except Exception as e:
                self.logger.error(f"법령 {law.get('법령ID', 'Unknown')} 파싱 실패: {e}")
        
        return articles
    
    def _parse_law_articles(self, law: Dict[str, Any]) -> List[Dict[str, Any]]:
        """법령에서 조문 추출"""
        articles = []
        
        law_id = law.get('법령ID', '')
        law_name = law.get('법령명한글', '')
        detailed_info = law.get('detailed_info', {})
        
        if not detailed_info:
            return articles
        
        # 조문 배열 추출 (detailed_info -> 법령 -> 조문 -> 조문단위)
        beopryeong = detailed_info.get('법령', {})
        law_articles_data = beopryeong.get('조문', {})
        law_articles = law_articles_data.get('조문단위', []) if isinstance(law_articles_data, dict) else []
        
        for article_data in law_articles:
            try:
                # 기본 조문 정보 (리스트 타입 처리)
                article_number_raw = article_data.get('조문번호', 0)
                article_number = int(article_number_raw) if not isinstance(article_number_raw, list) else int(article_number_raw[0]) if article_number_raw else 0
                
                article_title_raw = article_data.get('조문제목', '')
                article_title = article_title_raw if not isinstance(article_title_raw, list) else ' '.join(str(x) for x in article_title_raw) if article_title_raw else ''
                
                article_content_raw = article_data.get('조문내용', '')
                article_content = article_content_raw if not isinstance(article_content_raw, list) else ' '.join(str(x) for x in article_content_raw) if article_content_raw else ''
                
                amendment_type_raw = article_data.get('조문제개정유형', '')
                amendment_type = amendment_type_raw if not isinstance(amendment_type_raw, list) else ' '.join(str(x) for x in amendment_type_raw) if amendment_type_raw else ''
                
                effective_date_raw = article_data.get('조문시행일자', '')
                effective_date = effective_date_raw if not isinstance(effective_date_raw, list) else ' '.join(str(x) for x in effective_date_raw) if effective_date_raw else ''
                
                # 항(paragraph) 정보 추출
                paragraphs_data = article_data.get('항', [])
                
                # 항이 배열인지 객체인지 확인
                if isinstance(paragraphs_data, dict):
                    # 항이 객체인 경우 (호가 있는 경우)
                    paragraphs = [paragraphs_data]
                elif isinstance(paragraphs_data, list):
                    # 항이 배열인 경우
                    paragraphs = paragraphs_data
                else:
                    paragraphs = []
                
                if paragraphs:
                    # 각 항별로 저장
                    for para_data in paragraphs:
                        para_number_raw = para_data.get('항번호', '')
                        para_number = self._extract_paragraph_number(para_number_raw)
                        
                        para_content_raw = para_data.get('항내용', '')
                        para_content = para_content_raw if not isinstance(para_content_raw, list) else ' '.join(str(x) for x in para_content_raw) if para_content_raw else ''
                        
                        # 호(sub-paragraph) 정보 추출
                        sub_paragraphs_data = para_data.get('호', [])
                        
                        if isinstance(sub_paragraphs_data, list) and sub_paragraphs_data:
                            # 각 호별로 저장
                            for sub_para_data in sub_paragraphs_data:
                                sub_para_number_raw = sub_para_data.get('호번호', '')
                                sub_para_number = sub_para_number_raw if not isinstance(sub_para_number_raw, list) else ' '.join(str(x) for x in sub_para_number_raw) if sub_para_number_raw else ''
                                
                                sub_para_content_raw = sub_para_data.get('호내용', '')
                                sub_para_content = sub_para_content_raw if not isinstance(sub_para_content_raw, list) else ' '.join(str(x) for x in sub_para_content_raw) if sub_para_content_raw else ''
                                
                                article_id = f"{law_id}_{article_number}_{para_number}_{sub_para_number}"
                                
                                articles.append({
                                    'article_id': article_id,
                                    'law_id': law_id,
                                    'law_name_korean': law_name,
                                    'article_number': article_number,
                                    'article_title': article_title,
                                    'article_content': article_content,
                                    'paragraph_number': para_number,
                                    'paragraph_content': para_content,
                                    'sub_paragraph_number': sub_para_number,
                                    'sub_paragraph_content': sub_para_content,
                                    'is_supplementary': False,
                                    'amendment_type': amendment_type,
                                    'effective_date': effective_date,
                                    'parsing_method': 'batch_parser',
                                    'quality_score': 0.9
                                })
                                
                                self.stats['total_paragraphs'] += 1
                        else:
                            # 항만 있는 경우
                            article_id = f"{law_id}_{article_number}_{para_number}"
                            
                            articles.append({
                                'article_id': article_id,
                                'law_id': law_id,
                                'law_name_korean': law_name,
                                'article_number': article_number,
                                'article_title': article_title,
                                'article_content': article_content,
                                'paragraph_number': para_number,
                                'paragraph_content': para_content,
                                'is_supplementary': False,
                                'amendment_type': amendment_type,
                                'effective_date': effective_date,
                                'parsing_method': 'batch_parser',
                                'quality_score': 0.9
                            })
                            
                            self.stats['total_paragraphs'] += 1
                else:
                    # 조문만 있는 경우
                    article_id = f"{law_id}_{article_number}"
                    
                    articles.append({
                        'article_id': article_id,
                        'law_id': law_id,
                        'law_name_korean': law_name,
                        'article_number': article_number,
                        'article_title': article_title,
                        'article_content': article_content,
                        'is_supplementary': False,
                        'amendment_type': amendment_type,
                        'effective_date': effective_date,
                        'parsing_method': 'batch_parser',
                        'quality_score': 0.9
                    })
                
                self.stats['total_articles'] += 1
                
            except Exception as e:
                self.logger.error(f"조문 {article_data.get('조문번호', 'Unknown')} 파싱 실패: {e}")
        
        return articles
    
    def _extract_paragraph_number(self, para_number_str: str) -> int:
        """항번호 문자열에서 숫자 추출"""
        if not para_number_str:
            return 0
        
        # ①, ②, ③... 형태에서 숫자 추출
        if para_number_str in ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩']:
            return ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩'].index(para_number_str) + 1
        
        # 숫자만 있는 경우
        try:
            return int(para_number_str)
        except ValueError:
            return 0
    
    def _save_articles_to_database(self, articles: List[Dict[str, Any]]):
        """조문을 데이터베이스에 저장"""
        self.logger.info(f"총 {len(articles)}개 조문을 데이터베이스에 저장 중...")
        
        batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            
            try:
                inserted_count = self._insert_articles_batch(batch)
                total_inserted += inserted_count
                self.logger.info(f"배치 {i//batch_size + 1} 저장 완료: {inserted_count}개")
            except Exception as e:
                self.logger.error(f"배치 {i//batch_size + 1} 저장 실패: {e}")
        
        self.logger.info(f"총 {total_inserted}개 조문 저장 완료")
    
    def _insert_articles_batch(self, articles: List[Dict[str, Any]]) -> int:
        """조문 배치 삽입"""
        query = """
            INSERT OR REPLACE INTO current_laws_articles (
                article_id, law_id, law_name_korean, article_number,
                article_title, article_content, paragraph_number, paragraph_content,
                sub_paragraph_number, sub_paragraph_content, is_supplementary,
                amendment_type, effective_date, parsing_method, quality_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            params_list = []
            for article in articles:
                params = (
                    article['article_id'],
                    article['law_id'],
                    article['law_name_korean'],
                    article['article_number'],
                    article['article_title'],
                    article['article_content'],
                    article.get('paragraph_number'),
                    article.get('paragraph_content'),
                    article.get('sub_paragraph_number'),
                    article.get('sub_paragraph_content'),
                    article['is_supplementary'],
                    article.get('amendment_type'),
                    article.get('effective_date'),
                    article['parsing_method'],
                    article['quality_score']
                )
                params_list.append(params)
            
            cursor.executemany(query, params_list)
            conn.commit()
            
            return len(params_list)
    
    def _print_statistics(self):
        """통계 정보 출력"""
        print("\n" + "="*60)
        print("📊 배치 파일 파싱 통계")
        print("="*60)
        print(f"총 배치 파일: {self.stats['total_batches']:,}개")
        print(f"총 법령: {self.stats['total_laws']:,}개")
        print(f"총 조문: {self.stats['total_articles']:,}개")
        print(f"총 항/호: {self.stats['total_paragraphs']:,}개")
        print(f"파싱 오류: {len(self.stats['parsing_errors'])}개")
        
        if self.stats['parsing_errors']:
            print("\n⚠️ 파싱 오류 목록:")
            for error in self.stats['parsing_errors'][:5]:  # 최대 5개만 표시
                print(f"  - {error}")


def main():
    """메인 실행 함수"""
    batch_dir = "data/raw/law_open_api/current_laws/batches"
    
    if not Path(batch_dir).exists():
        print(f"❌ 배치 디렉토리가 존재하지 않습니다: {batch_dir}")
        return
    
    parser = BatchArticleParser(batch_dir)
    stats = parser.parse_all_batches()
    
    print(f"\n🎉 배치 파일 파싱 완료!")
    print(f"총 {stats['total_articles']:,}개 조문이 추출되었습니다.")


if __name__ == "__main__":
    main()
