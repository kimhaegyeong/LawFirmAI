#!/usr/bin/env python3
"""
ê¸°ì¡´ ?Œì¼??law_content ?…ë°?´íŠ¸ ?¤í¬ë¦½íŠ¸

???¤í¬ë¦½íŠ¸???ë³¸ ?Œì¼??laws.law_contentë§?HTML?ì„œ ì¶”ì¶œ???´ìš©?¼ë¡œ ?…ë°?´íŠ¸?©ë‹ˆ??
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.test_improved_html_parser import ImprovedLawHTMLParser

# ë¡œê¹… ?¤ì •
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
    """ë²•ë ¹ ?´ìš© ?…ë°?´íŠ¸ ?´ë˜??""
    
    def __init__(self):
        """ì´ˆê¸°??""
        self.html_parser = ImprovedLawHTMLParser()
    
    def update_law_content(self, file_path: Path) -> Dict[str, Any]:
        """
        ê¸°ì¡´ ?Œì¼??law_content ?…ë°?´íŠ¸
        
        Args:
            file_path (Path): ?…ë°?´íŠ¸???Œì¼ ê²½ë¡œ
            
        Returns:
            Dict[str, Any]: ?…ë°?´íŠ¸ ê²°ê³¼
        """
        try:
            logger.info(f"ë²•ë ¹ ?´ìš© ?…ë°?´íŠ¸ ?œì‘: {file_path}")
            
            # ?ë³¸ ?°ì´??ë¡œë“œ
            with open(file_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            update_stats = {
                'total_laws': len(original_data.get('laws', [])),
                'successful_updates': 0,
                'failed_updates': 0,
                'errors': []
            }
            
            # ê°?ë²•ë ¹??law_content ?…ë°?´íŠ¸
            for i, law_data in enumerate(original_data.get('laws', [])):
                try:
                    law_name = law_data.get('law_name', 'Unknown')
                    logger.info(f"ë²•ë ¹ {i+1}/{update_stats['total_laws']} ?…ë°?´íŠ¸ ì¤? {law_name}")
                    
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
                        
                        update_stats['successful_updates'] += 1
                        logger.info(f"ë²•ë ¹ '{law_name}' ?…ë°?´íŠ¸ ?„ë£Œ: {len(original_content)} -> {len(updated_content)} ë¬¸ì")
                    else:
                        update_stats['failed_updates'] += 1
                        logger.warning(f"ë²•ë ¹ '{law_name}' ?…ë°?´íŠ¸ ?¤íŒ¨: HTML ?´ìš© ?†ìŒ")
                        
                except Exception as e:
                    error_msg = f"ë²•ë ¹ {i+1} ?…ë°?´íŠ¸ ì¤??¤ë¥˜: {str(e)}"
                    logger.error(error_msg)
                    update_stats['errors'].append(error_msg)
                    update_stats['failed_updates'] += 1
            
            # ?Œì¼ ë©”í??°ì´???…ë°?´íŠ¸
            original_data['content_updated_at'] = datetime.now().isoformat()
            original_data['content_update_version'] = '1.0'
            original_data['update_stats'] = update_stats
            
            # ?…ë°?´íŠ¸???°ì´???€??(?ë³¸ ?Œì¼ ??–´?°ê¸°)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(original_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"?Œì¼ ?…ë°?´íŠ¸ ?„ë£Œ: {file_path}")
            logger.info(f"?…ë°?´íŠ¸ ?µê³„: {update_stats}")
            
            return {
                'success': True,
                'file_path': str(file_path),
                'update_stats': update_stats
            }
            
        except Exception as e:
            error_msg = f"?Œì¼ ?…ë°?´íŠ¸ ì¤??¤ë¥˜: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def _extract_content_from_html(self, html_content: str) -> str:
        """
        HTML?ì„œ ë²•ë ¹ ?´ìš© ì¶”ì¶œ
        
        Args:
            html_content (str): HTML ?´ìš©
            
        Returns:
            str: ì¶”ì¶œ??ë²•ë ¹ ?´ìš©
        """
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


def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    try:
        # ?Œì¼ ê²½ë¡œ ?¤ì •
        file_path = Path("data/raw/assembly/law/20251010/law_page_001_181503.json")
        
        # ?Œì¼ ì¡´ì¬ ?•ì¸
        if not file_path.exists():
            logger.error(f"?Œì¼??ì¡´ì¬?˜ì? ?ŠìŒ: {file_path}")
            return
        
        # ?…ë°?´íŠ¸ ?¤í–‰
        updater = LawContentUpdater()
        result = updater.update_law_content(file_path)
        
        if result['success']:
            logger.info("ë²•ë ¹ ?´ìš© ?…ë°?´íŠ¸ ?„ë£Œ!")
            logger.info(f"?Œì¼: {result['file_path']}")
            logger.info(f"?…ë°?´íŠ¸ ?µê³„: {result['update_stats']}")
        else:
            logger.error(f"?…ë°?´íŠ¸ ?¤íŒ¨: {result['error']}")
            
    except Exception as e:
        logger.error(f"ë©”ì¸ ?¨ìˆ˜ ?¤í–‰ ì¤??¤ë¥˜: {str(e)}")


if __name__ == "__main__":
    main()



