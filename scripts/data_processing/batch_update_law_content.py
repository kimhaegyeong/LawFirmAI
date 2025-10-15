#!/usr/bin/env python3
"""
모든 law JSON 파일들의 law_content 일괄 업데이트 스크립트

이 스크립트는 data/raw/assembly/law 폴더의 모든 law로 시작하는 JSON 파일들의
law_content를 HTML에서 추출한 내용으로 업데이트합니다.
"""

import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import concurrent.futures
from threading import Lock

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.test_improved_html_parser import ImprovedLawHTMLParser

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/batch_update_law_content.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 전역 통계를 위한 락
stats_lock = Lock()
global_stats = {
    'total_files': 0,
    'successful_updates': 0,
    'failed_updates': 0,
    'total_laws': 0,
    'successful_laws': 0,
    'failed_laws': 0,
    'errors': []
}


class BatchLawContentUpdater:
    """법령 내용 일괄 업데이트 클래스"""
    
    def __init__(self, max_workers: int = 4):
        """초기화"""
        self.html_parser = ImprovedLawHTMLParser()
        self.max_workers = max_workers
    
    def update_all_law_files(self, law_dir: Path) -> Dict[str, Any]:
        """
        모든 law 파일 업데이트
        
        Args:
            law_dir (Path): law 폴더 경로
            
        Returns:
            Dict[str, Any]: 업데이트 결과
        """
        try:
            logger.info(f"법령 파일 일괄 업데이트 시작: {law_dir}")
            
            # 모든 law 파일 찾기
            law_files = self._find_all_law_files(law_dir)
            
            logger.info(f"총 {len(law_files)}개 파일 발견")
            
            # 파일별로 업데이트 실행
            results = []
            
            # 병렬 처리로 업데이트
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self._update_single_file, file_path): file_path 
                    for file_path in law_files
                }
                
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result['success']:
                            logger.info(f"✅ {file_path.name}: {result['stats']['successful_laws']}/{result['stats']['total_laws']} 법령 업데이트 완료")
                        else:
                            logger.error(f"❌ {file_path.name}: {result['error']}")
                            
                    except Exception as e:
                        error_msg = f"파일 {file_path.name} 처리 중 예외: {str(e)}"
                        logger.error(error_msg)
                        results.append({
                            'success': False,
                            'file_path': str(file_path),
                            'error': error_msg
                        })
            
            # 전체 통계 계산
            final_stats = self._calculate_final_stats(results)
            
            logger.info("법령 파일 일괄 업데이트 완료!")
            logger.info(f"전체 통계: {final_stats}")
            
            return {
                'success': True,
                'results': results,
                'final_stats': final_stats
            }
            
        except Exception as e:
            error_msg = f"일괄 업데이트 중 오류: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def _find_all_law_files(self, law_dir: Path) -> List[Path]:
        """모든 law 파일 찾기"""
        law_files = []
        
        # 각 날짜 폴더에서 law로 시작하는 JSON 파일 찾기
        for date_folder in law_dir.iterdir():
            if date_folder.is_dir():
                for file_path in date_folder.iterdir():
                    if (file_path.is_file() and 
                        file_path.name.startswith('law') and 
                        file_path.name.endswith('.json')):
                        law_files.append(file_path)
        
        return sorted(law_files)
    
    def _update_single_file(self, file_path: Path) -> Dict[str, Any]:
        """단일 파일 업데이트"""
        try:
            logger.info(f"파일 업데이트 시작: {file_path.name}")
            
            # 원본 데이터 로드
            with open(file_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            file_stats = {
                'total_laws': len(original_data.get('laws', [])),
                'successful_laws': 0,
                'failed_laws': 0,
                'errors': []
            }
            
            # 각 법령의 law_content 업데이트
            for i, law_data in enumerate(original_data.get('laws', [])):
                try:
                    law_name = law_data.get('law_name', 'Unknown')
                    
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
                        
                        file_stats['successful_laws'] += 1
                    else:
                        file_stats['failed_laws'] += 1
                        logger.warning(f"법령 '{law_name}' 업데이트 실패: HTML 내용 없음")
                        
                except Exception as e:
                    error_msg = f"법령 {i+1} 업데이트 중 오류: {str(e)}"
                    logger.error(error_msg)
                    file_stats['errors'].append(error_msg)
                    file_stats['failed_laws'] += 1
            
            # 파일 메타데이터 업데이트
            original_data['content_updated_at'] = datetime.now().isoformat()
            original_data['content_update_version'] = '1.0'
            original_data['update_stats'] = file_stats
            
            # 업데이트된 데이터 저장 (원본 파일 덮어쓰기)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(original_data, f, ensure_ascii=False, indent=2)
            
            # 전역 통계 업데이트
            with stats_lock:
                global_stats['total_files'] += 1
                global_stats['total_laws'] += file_stats['total_laws']
                global_stats['successful_laws'] += file_stats['successful_laws']
                global_stats['failed_laws'] += file_stats['failed_laws']
                global_stats['errors'].extend(file_stats['errors'])
                
                if file_stats['failed_laws'] == 0:
                    global_stats['successful_updates'] += 1
                else:
                    global_stats['failed_updates'] += 1
            
            return {
                'success': True,
                'file_path': str(file_path),
                'stats': file_stats
            }
            
        except Exception as e:
            error_msg = f"파일 업데이트 중 오류: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'file_path': str(file_path),
                'error': error_msg
            }
    
    def _extract_content_from_html(self, html_content: str) -> str:
        """HTML에서 법령 내용 추출"""
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
    
    def _calculate_final_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """최종 통계 계산"""
        successful_files = sum(1 for r in results if r['success'])
        failed_files = len(results) - successful_files
        
        return {
            'total_files': len(results),
            'successful_files': successful_files,
            'failed_files': failed_files,
            'success_rate': (successful_files / len(results) * 100) if results else 0,
            'total_laws': global_stats['total_laws'],
            'successful_laws': global_stats['successful_laws'],
            'failed_laws': global_stats['failed_laws'],
            'law_success_rate': (global_stats['successful_laws'] / global_stats['total_laws'] * 100) if global_stats['total_laws'] > 0 else 0,
            'total_errors': len(global_stats['errors'])
        }


def main():
    """메인 함수"""
    try:
        # law 폴더 경로 설정
        law_dir = Path("data/raw/assembly/law")
        
        # 폴더 존재 확인
        if not law_dir.exists():
            logger.error(f"폴더가 존재하지 않음: {law_dir}")
            return
        
        # 일괄 업데이트 실행
        updater = BatchLawContentUpdater(max_workers=4)
        result = updater.update_all_law_files(law_dir)
        
        if result['success']:
            logger.info("법령 파일 일괄 업데이트 완료!")
            logger.info(f"최종 통계: {result['final_stats']}")
            
            # 결과를 파일로 저장
            result_file = Path("logs/batch_update_results.json")
            result_file.parent.mkdir(parents=True, exist_ok=True)
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"결과 파일 저장: {result_file}")
        else:
            logger.error(f"일괄 업데이트 실패: {result['error']}")
            
    except Exception as e:
        logger.error(f"메인 함수 실행 중 오류: {str(e)}")


if __name__ == "__main__":
    main()



