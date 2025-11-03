#!/usr/bin/env python3
"""
ë²•ë ¹ ?°ì´??HTML ?•ì œ ?¤í¬ë¦½íŠ¸

???¤í¬ë¦½íŠ¸??law_page_001_181503.json ?Œì¼??content_html???¬ìš©?˜ì—¬
ëª¨ë“  ì¡°ë¬¸??ì¶”ì¶œ?˜ê³  ?•ì œ???°ì´?°ë? ?ì„±?©ë‹ˆ??
"""

import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.data.assembly_playwright_client import AssemblyPlaywrightClient
from scripts.test_improved_html_parser import ImprovedLawHTMLParser

# ë¡œê¹… ?¤ì •
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
    """ë²•ë ¹ ?°ì´??HTML ?•ì œ ?´ë˜??""
    
    def __init__(self):
        """ì´ˆê¸°??""
        self.html_parser = ImprovedLawHTMLParser()
        self.assembly_client = AssemblyPlaywrightClient()
    
    def refine_law_file(self, input_file: Path, output_file: Path) -> Dict[str, Any]:
        """
        ë²•ë ¹ ?Œì¼??HTML???¬ìš©?˜ì—¬ ?•ì œ
        
        Args:
            input_file (Path): ?…ë ¥ ?Œì¼ ê²½ë¡œ
            output_file (Path): ì¶œë ¥ ?Œì¼ ê²½ë¡œ
            
        Returns:
            Dict[str, Any]: ì²˜ë¦¬ ê²°ê³¼
        """
        try:
            logger.info(f"ë²•ë ¹ ?Œì¼ ?•ì œ ?œì‘: {input_file}")
            
            # ?ë³¸ ?°ì´??ë¡œë“œ
            with open(input_file, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            refined_laws = []
            processing_stats = {
                'total_laws': len(original_data.get('laws', [])),
                'successful_refinements': 0,
                'failed_refinements': 0,
                'errors': []
            }
            
            # ê°?ë²•ë ¹ ì²˜ë¦¬
            for i, law_data in enumerate(original_data.get('laws', [])):
                try:
                    logger.info(f"ë²•ë ¹ {i+1}/{processing_stats['total_laws']} ì²˜ë¦¬ ì¤? {law_data.get('law_name', 'Unknown')}")
                    
                    refined_law = self._refine_single_law(law_data)
                    if refined_law:
                        refined_laws.append(refined_law)
                        processing_stats['successful_refinements'] += 1
                    else:
                        processing_stats['failed_refinements'] += 1
                        
                except Exception as e:
                    error_msg = f"ë²•ë ¹ {i+1} ì²˜ë¦¬ ì¤??¤ë¥˜: {str(e)}"
                    logger.error(error_msg)
                    processing_stats['errors'].append(error_msg)
                    processing_stats['failed_refinements'] += 1
            
            # ?•ì œ???°ì´??êµ¬ì„±
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
            
            # ?•ì œ???°ì´???€??
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(refined_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"?•ì œ ?„ë£Œ: {output_file}")
            logger.info(f"ì²˜ë¦¬ ?µê³„: {processing_stats}")
            
            return {
                'success': True,
                'output_file': str(output_file),
                'processing_stats': processing_stats
            }
            
        except Exception as e:
            error_msg = f"?Œì¼ ?•ì œ ì¤??¤ë¥˜: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def _refine_single_law(self, law_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        ?¨ì¼ ë²•ë ¹ ?°ì´???•ì œ
        
        Args:
            law_data (Dict[str, Any]): ?ë³¸ ë²•ë ¹ ?°ì´??
            
        Returns:
            Optional[Dict[str, Any]]: ?•ì œ??ë²•ë ¹ ?°ì´??
        """
        try:
            law_name = law_data.get('law_name', '')
            content_html = law_data.get('content_html', '')
            original_content = law_data.get('law_content', '')
            
            if not content_html:
                logger.warning(f"HTML ?´ìš©???†ìŒ: {law_name}")
                return None
            
            # HTML ?Œì‹±
            parsed_html = self.html_parser.parse_html(content_html)
            
            # ?•ì œ??ë²•ë ¹ ?°ì´??êµ¬ì„±
            refined_law = {
                # ê¸°ë³¸ ?•ë³´
                'law_id': f"assembly_law_{law_data.get('row_number', 'unknown')}",
                'law_name': law_name,
                'law_type': law_data.get('law_type', ''),
                'category': law_data.get('category', ''),
                'row_number': law_data.get('row_number', ''),
                
                # ê³µí¬ ?•ë³´
                'promulgation_info': {
                    'number': law_data.get('promulgation_number', ''),
                    'date': law_data.get('promulgation_date', ''),
                    'enforcement_date': law_data.get('enforcement_date', ''),
                    'amendment_type': law_data.get('amendment_type', '')
                },
                
                # ?˜ì§‘ ?•ë³´
                'collection_info': {
                    'cont_id': law_data.get('cont_id', ''),
                    'cont_sid': law_data.get('cont_sid', ''),
                    'detail_url': law_data.get('detail_url', ''),
                    'collected_at': law_data.get('collected_at', '')
                },
                
                # ?•ì œ???´ìš©
                'refined_content': {
                    'full_text': parsed_html.get('clean_text', ''),
                    'articles': parsed_html.get('articles', []),
                    'html_metadata': parsed_html.get('metadata', {})
                },
                
                # ?ë³¸ ?°ì´??(ì°¸ì¡°??
                'original_content': original_content,
                'content_html': content_html,
                
                # ì²˜ë¦¬ ë©”í??°ì´??
                'refined_at': datetime.now().isoformat(),
                'refinement_version': '1.0',
                'data_quality': self._calculate_data_quality(parsed_html, original_content)
            }
            
            return refined_law
            
        except Exception as e:
            logger.error(f"?¨ì¼ ë²•ë ¹ ?•ì œ ì¤??¤ë¥˜: {str(e)}")
            return None
    
    def _calculate_data_quality(self, parsed_html: Dict[str, Any], original_content: str) -> Dict[str, Any]:
        """
        ?°ì´???ˆì§ˆ ê³„ì‚°
        
        Args:
            parsed_html (Dict[str, Any]): ?Œì‹±??HTML ?°ì´??
            original_content (str): ?ë³¸ ?´ìš©
            
        Returns:
            Dict[str, Any]: ?°ì´???ˆì§ˆ ?•ë³´
        """
        try:
            clean_text = parsed_html.get('clean_text', '')
            articles = parsed_html.get('articles', [])
            
            # ê¸°ë³¸ ?µê³„
            stats = {
                'original_content_length': len(original_content),
                'clean_text_length': len(clean_text),
                'articles_count': len(articles),
                'improvement_ratio': len(clean_text) / len(original_content) if original_content else 0,
                'has_articles': len(articles) > 0,
                'quality_score': 0.0
            }
            
            # ?ˆì§ˆ ?ìˆ˜ ê³„ì‚°
            quality_score = 0.0
            
            # ?ìŠ¤??ê¸¸ì´ ê°œì„  (ìµœë? 30??
            if stats['improvement_ratio'] > 1.5:
                quality_score += 30
            elif stats['improvement_ratio'] > 1.0:
                quality_score += 20
            elif stats['improvement_ratio'] > 0.5:
                quality_score += 10
            
            # ì¡°ë¬¸ ì¶”ì¶œ ?±ê³µ (ìµœë? 40??
            if stats['articles_count'] > 0:
                quality_score += min(40, stats['articles_count'] * 5)
            
            # ì¡°ë¬¸ ?´ìš© ?ˆì§ˆ (ìµœë? 30??
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
            logger.error(f"?°ì´???ˆì§ˆ ê³„ì‚° ì¤??¤ë¥˜: {str(e)}")
            return {'quality_score': 0.0, 'error': str(e)}


def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    try:
        # ?Œì¼ ê²½ë¡œ ?¤ì •
        input_file = Path("data/raw/assembly/law/20251010/law_page_001_181503.json")
        output_file = Path("data/processed/assembly/law/20251011/refined_law_page_001_181503.json")
        
        # ?Œì¼ ì¡´ì¬ ?•ì¸
        if not input_file.exists():
            logger.error(f"?…ë ¥ ?Œì¼??ì¡´ì¬?˜ì? ?ŠìŒ: {input_file}")
            return
        
        # ?•ì œ ?¤í–‰
        refiner = LawDataRefiner()
        result = refiner.refine_law_file(input_file, output_file)
        
        if result['success']:
            logger.info("ë²•ë ¹ ?°ì´???•ì œ ?„ë£Œ!")
            logger.info(f"ì¶œë ¥ ?Œì¼: {result['output_file']}")
            logger.info(f"ì²˜ë¦¬ ?µê³„: {result['processing_stats']}")
        else:
            logger.error(f"?•ì œ ?¤íŒ¨: {result['error']}")
            
    except Exception as e:
        logger.error(f"ë©”ì¸ ?¨ìˆ˜ ?¤í–‰ ì¤??¤ë¥˜: {str(e)}")


if __name__ == "__main__":
    main()
