#!/usr/bin/env python3
"""
ëª¨ë“  law JSON ?Œì¼?¤ì˜ law_content ?¼ê´„ ?…ë°?´íŠ¸ ?¤í¬ë¦½íŠ¸

???¤í¬ë¦½íŠ¸??data/raw/assembly/law ?´ë”??ëª¨ë“  lawë¡??œì‘?˜ëŠ” JSON ?Œì¼?¤ì˜
law_contentë¥?HTML?ì„œ ì¶”ì¶œ???´ìš©?¼ë¡œ ?…ë°?´íŠ¸?©ë‹ˆ??
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

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.test_improved_html_parser import ImprovedLawHTMLParser

# ë¡œê¹… ?¤ì •
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/batch_update_law_content.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ?„ì—­ ?µê³„ë¥??„í•œ ??
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
    """ë²•ë ¹ ?´ìš© ?¼ê´„ ?…ë°?´íŠ¸ ?´ë˜??""
    
    def __init__(self, max_workers: int = 4):
        """ì´ˆê¸°??""
        self.html_parser = ImprovedLawHTMLParser()
        self.max_workers = max_workers
    
    def update_all_law_files(self, law_dir: Path) -> Dict[str, Any]:
        """
        ëª¨ë“  law ?Œì¼ ?…ë°?´íŠ¸
        
        Args:
            law_dir (Path): law ?´ë” ê²½ë¡œ
            
        Returns:
            Dict[str, Any]: ?…ë°?´íŠ¸ ê²°ê³¼
        """
        try:
            logger.info(f"ë²•ë ¹ ?Œì¼ ?¼ê´„ ?…ë°?´íŠ¸ ?œì‘: {law_dir}")
            
            # ëª¨ë“  law ?Œì¼ ì°¾ê¸°
            law_files = self._find_all_law_files(law_dir)
            
            logger.info(f"ì´?{len(law_files)}ê°??Œì¼ ë°œê²¬")
            
            # ?Œì¼ë³„ë¡œ ?…ë°?´íŠ¸ ?¤í–‰
            results = []
            
            # ë³‘ë ¬ ì²˜ë¦¬ë¡??…ë°?´íŠ¸
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
                            logger.info(f"??{file_path.name}: {result['stats']['successful_laws']}/{result['stats']['total_laws']} ë²•ë ¹ ?…ë°?´íŠ¸ ?„ë£Œ")
                        else:
                            logger.error(f"??{file_path.name}: {result['error']}")
                            
                    except Exception as e:
                        error_msg = f"?Œì¼ {file_path.name} ì²˜ë¦¬ ì¤??ˆì™¸: {str(e)}"
                        logger.error(error_msg)
                        results.append({
                            'success': False,
                            'file_path': str(file_path),
                            'error': error_msg
                        })
            
            # ?„ì²´ ?µê³„ ê³„ì‚°
            final_stats = self._calculate_final_stats(results)
            
            logger.info("ë²•ë ¹ ?Œì¼ ?¼ê´„ ?…ë°?´íŠ¸ ?„ë£Œ!")
            logger.info(f"?„ì²´ ?µê³„: {final_stats}")
            
            return {
                'success': True,
                'results': results,
                'final_stats': final_stats
            }
            
        except Exception as e:
            error_msg = f"?¼ê´„ ?…ë°?´íŠ¸ ì¤??¤ë¥˜: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def _find_all_law_files(self, law_dir: Path) -> List[Path]:
        """ëª¨ë“  law ?Œì¼ ì°¾ê¸°"""
        law_files = []
        
        # ê°?? ì§œ ?´ë”?ì„œ lawë¡??œì‘?˜ëŠ” JSON ?Œì¼ ì°¾ê¸°
        for date_folder in law_dir.iterdir():
            if date_folder.is_dir():
                for file_path in date_folder.iterdir():
                    if (file_path.is_file() and 
                        file_path.name.startswith('law') and 
                        file_path.name.endswith('.json')):
                        law_files.append(file_path)
        
        return sorted(law_files)
    
    def _update_single_file(self, file_path: Path) -> Dict[str, Any]:
        """?¨ì¼ ?Œì¼ ?…ë°?´íŠ¸"""
        try:
            logger.info(f"?Œì¼ ?…ë°?´íŠ¸ ?œì‘: {file_path.name}")
            
            # ?ë³¸ ?°ì´??ë¡œë“œ
            with open(file_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            file_stats = {
                'total_laws': len(original_data.get('laws', [])),
                'successful_laws': 0,
                'failed_laws': 0,
                'errors': []
            }
            
            # ê°?ë²•ë ¹??law_content ?…ë°?´íŠ¸
            for i, law_data in enumerate(original_data.get('laws', [])):
                try:
                    law_name = law_data.get('law_name', 'Unknown')
                    
                    # HTML?ì„œ ì¶”ì¶œ???´ìš©?¼ë¡œ law_content ?…ë°?´íŠ¸
                    updated_content = self._extract_content_from_html(law_data.get('content_html', ''))
                    
                    if updated_content:
                        # ?ë³¸ law_content ë°±ì—…
                        original_content = law_data.get('law_content', '')
                        
                        # law_content ?…ë°?´íŠ¸
                        original_data['laws'][i]['law_content'] = updated_content
                        
                        # ?…ë°?´íŠ¸ ?•ë³´ ì¶”ê?
                        original_data['laws'][i]['content_updated_at'] = datetime.now().isoformat()
                        original_data['laws'][i]['original_content_length'] = len(original_content)
                        original_data['laws'][i]['updated_content_length'] = len(updated_content)
                        original_data['laws'][i]['content_improvement_ratio'] = len(updated_content) / len(original_content) if original_content else 0
                        
                        file_stats['successful_laws'] += 1
                    else:
                        file_stats['failed_laws'] += 1
                        logger.warning(f"ë²•ë ¹ '{law_name}' ?…ë°?´íŠ¸ ?¤íŒ¨: HTML ?´ìš© ?†ìŒ")
                        
                except Exception as e:
                    error_msg = f"ë²•ë ¹ {i+1} ?…ë°?´íŠ¸ ì¤??¤ë¥˜: {str(e)}"
                    logger.error(error_msg)
                    file_stats['errors'].append(error_msg)
                    file_stats['failed_laws'] += 1
            
            # ?Œì¼ ë©”í??°ì´???…ë°?´íŠ¸
            original_data['content_updated_at'] = datetime.now().isoformat()
            original_data['content_update_version'] = '1.0'
            original_data['update_stats'] = file_stats
            
            # ?…ë°?´íŠ¸???°ì´???€??(?ë³¸ ?Œì¼ ??–´?°ê¸°)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(original_data, f, ensure_ascii=False, indent=2)
            
            # ?„ì—­ ?µê³„ ?…ë°?´íŠ¸
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
            error_msg = f"?Œì¼ ?…ë°?´íŠ¸ ì¤??¤ë¥˜: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'file_path': str(file_path),
                'error': error_msg
            }
    
    def _extract_content_from_html(self, html_content: str) -> str:
        """HTML?ì„œ ë²•ë ¹ ?´ìš© ì¶”ì¶œ"""
        try:
            if not html_content:
                return ""
            
            # HTML ?Œì‹±
            parsed_html = self.html_parser.parse_html(html_content)
            
            # ì¡°ë¬¸?¤ì„ ?•ë¦¬???•íƒœë¡?ê²°í•©
            articles = parsed_html.get('articles', [])
            
            if not articles:
                # ì¡°ë¬¸???†ìœ¼ë©?ê¹¨ë—???ìŠ¤??ë°˜í™˜
                return parsed_html.get('clean_text', '')
            
            # ì¡°ë¬¸?¤ì„ ?˜ë‚˜???ìŠ¤?¸ë¡œ ê²°í•©
            content_parts = []
            
            for article in articles:
                article_text = f"{article['article_number']}"
                if article.get('article_title'):
                    article_text += f"({article['article_title']})"
                article_text += f" {article['article_content']}"
                content_parts.append(article_text)
            
            return '\n\n'.join(content_parts)
            
        except Exception as e:
            logger.error(f"HTML?ì„œ ?´ìš© ì¶”ì¶œ ì¤??¤ë¥˜: {e}")
            return ""
    
    def _calculate_final_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ìµœì¢… ?µê³„ ê³„ì‚°"""
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
    """ë©”ì¸ ?¨ìˆ˜"""
    try:
        # law ?´ë” ê²½ë¡œ ?¤ì •
        law_dir = Path("data/raw/assembly/law")
        
        # ?´ë” ì¡´ì¬ ?•ì¸
        if not law_dir.exists():
            logger.error(f"?´ë”ê°€ ì¡´ì¬?˜ì? ?ŠìŒ: {law_dir}")
            return
        
        # ?¼ê´„ ?…ë°?´íŠ¸ ?¤í–‰
        updater = BatchLawContentUpdater(max_workers=4)
        result = updater.update_all_law_files(law_dir)
        
        if result['success']:
            logger.info("ë²•ë ¹ ?Œì¼ ?¼ê´„ ?…ë°?´íŠ¸ ?„ë£Œ!")
            logger.info(f"ìµœì¢… ?µê³„: {result['final_stats']}")
            
            # ê²°ê³¼ë¥??Œì¼ë¡??€??
            result_file = Path("logs/batch_update_results.json")
            result_file.parent.mkdir(parents=True, exist_ok=True)
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"ê²°ê³¼ ?Œì¼ ?€?? {result_file}")
        else:
            logger.error(f"?¼ê´„ ?…ë°?´íŠ¸ ?¤íŒ¨: {result['error']}")
            
    except Exception as e:
        logger.error(f"ë©”ì¸ ?¨ìˆ˜ ?¤í–‰ ì¤??¤ë¥˜: {str(e)}")


if __name__ == "__main__":
    main()



