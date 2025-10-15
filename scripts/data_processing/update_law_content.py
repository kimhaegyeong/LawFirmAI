#!/usr/bin/env python3
"""
기존 파일의 law_content 업데이트 스크립트

이 스크립트는 원본 파일의 laws.law_content만 HTML에서 추출한 내용으로 업데이트합니다.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.test_improved_html_parser import ImprovedLawHTMLParser

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/update_law_content.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LawContentUpdater:
    """법령 내용 업데이트 클래스"""
    
    def __init__(self):
        """초기화"""
        self.html_parser = ImprovedLawHTMLParser()
    
    def update_law_content(self, file_path: Path) -> Dict[str, Any]:
        """
        기존 파일의 law_content 업데이트
        
        Args:
            file_path (Path): 업데이트할 파일 경로
            
        Returns:
            Dict[str, Any]: 업데이트 결과
        """
        try:
            logger.info(f"법령 내용 업데이트 시작: {file_path}")
            
            # 원본 데이터 로드
            with open(file_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            update_stats = {
                'total_laws': len(original_data.get('laws', [])),
                'successful_updates': 0,
                'failed_updates': 0,
                'errors': []
            }
            
            # 각 법령의 law_content 업데이트
            for i, law_data in enumerate(original_data.get('laws', [])):
                try:
                    law_name = law_data.get('law_name', 'Unknown')
                    logger.info(f"법령 {i+1}/{update_stats['total_laws']} 업데이트 중: {law_name}")
                    
                    # HTML에서 추출한 내용으로 law_content 업데이트
                    updated_content = self._extract_content_from_html(law_data.get('content_html', ''))
                    
                    if updated_content:
                        # 원본 law_content 백업
                        original_content = law_data.get('law_content', '')
                        
                        # law_content 업데이트
                        original_data['laws'][i]['law_content'] = updated_content
                        
                        # 업데이트 정보 추가
                        original_data['laws'][i]['content_updated_at'] = datetime.now().isoformat()
                        original_data['laws'][i]['original_content_length'] = len(original_content)
                        original_data['laws'][i]['updated_content_length'] = len(updated_content)
                        original_data['laws'][i]['content_improvement_ratio'] = len(updated_content) / len(original_content) if original_content else 0
                        
                        update_stats['successful_updates'] += 1
                        logger.info(f"법령 '{law_name}' 업데이트 완료: {len(original_content)} -> {len(updated_content)} 문자")
                    else:
                        update_stats['failed_updates'] += 1
                        logger.warning(f"법령 '{law_name}' 업데이트 실패: HTML 내용 없음")
                        
                except Exception as e:
                    error_msg = f"법령 {i+1} 업데이트 중 오류: {str(e)}"
                    logger.error(error_msg)
                    update_stats['errors'].append(error_msg)
                    update_stats['failed_updates'] += 1
            
            # 파일 메타데이터 업데이트
            original_data['content_updated_at'] = datetime.now().isoformat()
            original_data['content_update_version'] = '1.0'
            original_data['update_stats'] = update_stats
            
            # 업데이트된 데이터 저장 (원본 파일 덮어쓰기)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(original_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"파일 업데이트 완료: {file_path}")
            logger.info(f"업데이트 통계: {update_stats}")
            
            return {
                'success': True,
                'file_path': str(file_path),
                'update_stats': update_stats
            }
            
        except Exception as e:
            error_msg = f"파일 업데이트 중 오류: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def _extract_content_from_html(self, html_content: str) -> str:
        """
        HTML에서 법령 내용 추출
        
        Args:
            html_content (str): HTML 내용
            
        Returns:
            str: 추출된 법령 내용
        """
        try:
            if not html_content:
                return ""
            
            # HTML 파싱
            parsed_html = self.html_parser.parse_html(html_content)
            
            # 조문들을 정리된 형태로 결합
            articles = parsed_html.get('articles', [])
            
            if not articles:
                # 조문이 없으면 깨끗한 텍스트 반환
                return parsed_html.get('clean_text', '')
            
            # 조문들을 하나의 텍스트로 결합
            content_parts = []
            
            for article in articles:
                article_text = f"{article['article_number']}"
                if article.get('article_title'):
                    article_text += f"({article['article_title']})"
                article_text += f" {article['article_content']}"
                content_parts.append(article_text)
            
            return '\n\n'.join(content_parts)
            
        except Exception as e:
            logger.error(f"HTML에서 내용 추출 중 오류: {e}")
            return ""


def main():
    """메인 함수"""
    try:
        # 파일 경로 설정
        file_path = Path("data/raw/assembly/law/20251010/law_page_001_181503.json")
        
        # 파일 존재 확인
        if not file_path.exists():
            logger.error(f"파일이 존재하지 않음: {file_path}")
            return
        
        # 업데이트 실행
        updater = LawContentUpdater()
        result = updater.update_law_content(file_path)
        
        if result['success']:
            logger.info("법령 내용 업데이트 완료!")
            logger.info(f"파일: {result['file_path']}")
            logger.info(f"업데이트 통계: {result['update_stats']}")
        else:
            logger.error(f"업데이트 실패: {result['error']}")
            
    except Exception as e:
        logger.error(f"메인 함수 실행 중 오류: {str(e)}")


if __name__ == "__main__":
    main()



