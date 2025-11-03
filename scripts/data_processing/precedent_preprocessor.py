#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?ë? ?°ì´???„ì²˜ë¦¬ê¸°

?ë? ?°ì´?°ì˜ êµ¬ì¡°?”ëœ ?´ìš©???Œì‹±?˜ê³  ?•ë¦¬?˜ëŠ” ?„ìš© ?„ì²˜ë¦¬ê¸°?…ë‹ˆ??
ë²•ë¥  ?°ì´?°ì????¤ë¥¸ êµ¬ì¡°ë¥?ê°€ì§€???ë? ?°ì´?°ë? ì²˜ë¦¬?©ë‹ˆ??
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)


class PrecedentPreprocessor:
    """?ë? ?°ì´???„ì²˜ë¦¬ê¸° ?´ë˜??""
    
    def __init__(self, enable_term_normalization: bool = True):
        """
        ?ë? ?„ì²˜ë¦¬ê¸° ì´ˆê¸°??
        
        Args:
            enable_term_normalization: ë²•ë¥  ?©ì–´ ?•ê·œ???œì„±??
        """
        self.enable_term_normalization = enable_term_normalization
        
        # ?ë? ?¹ì…˜ ?€???•ì˜
        self.section_types = {
            '?ì‹œ?¬í•­': 'points_at_issue',
            '?ê²°?”ì?': 'decision_summary', 
            'ì°¸ì¡°ì¡°ë¬¸': 'referenced_statutes',
            'ì°¸ì¡°?ë?': 'referenced_cases',
            'ì£¼ë¬¸': 'disposition',
            '?´ìœ ': 'reasoning'
        }
        
        logger.info("PrecedentPreprocessor initialized")
    
    def process_precedent_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ?ë? ?°ì´???„ì²˜ë¦?
        
        Args:
            raw_data: ?ë³¸ ?ë? ?°ì´??
            
        Returns:
            Dict[str, Any]: ?„ì²˜ë¦¬ëœ ?ë? ?°ì´??
        """
        try:
            metadata = raw_data.get('metadata', {})
            items = raw_data.get('items', [])
            
            processed_cases = []
            
            for item in items:
                processed_case = self._process_single_case(item, metadata)
                if processed_case:
                    processed_cases.append(processed_case)
            
            return {
                'metadata': {
                    'data_type': 'precedent',
                    'category': metadata.get('category', 'unknown'),
                    'processed_at': datetime.now().isoformat(),
                    'total_cases': len(processed_cases),
                    'processing_version': '1.0'
                },
                'cases': processed_cases
            }
            
        except Exception as e:
            logger.error(f"Error processing precedent data: {e}")
            return {'cases': []}
    
    def _process_single_case(self, case_item: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        ?¨ì¼ ?ë? ì¼€?´ìŠ¤ ì²˜ë¦¬
        
        Args:
            case_item: ?ë? ì¼€?´ìŠ¤ ?°ì´??
            metadata: ë©”í??°ì´??
            
        Returns:
            Optional[Dict[str, Any]]: ì²˜ë¦¬??ì¼€?´ìŠ¤ ?°ì´??
        """
        try:
            # ê¸°ë³¸ ì¼€?´ìŠ¤ ?•ë³´ ì¶”ì¶œ
            case_id = self._generate_case_id(case_item)
            case_name = case_item.get('case_name', '')
            case_number = case_item.get('case_number', '')
            decision_date = case_item.get('decision_date', '')
            field = case_item.get('field', '')
            court = case_item.get('court', '')
            detail_url = case_item.get('detail_url', '')
            
            # êµ¬ì¡°?”ëœ ?´ìš© ?Œì‹±
            structured_content = case_item.get('structured_content', {})
            case_info = structured_content.get('case_info', {})
            legal_sections = structured_content.get('legal_sections', {})
            parties = structured_content.get('parties', {})
            
            # ë²•ë¥  ?¹ì…˜ ì¶”ì¶œ
            sections = self._extract_legal_sections(legal_sections)
            
            # ?¹ì‚¬???•ë³´ ì¶”ì¶œ
            party_info = self._extract_party_info(parties)
            
            # ?„ì²´ ?ìŠ¤???ì„±
            full_text = self._generate_full_text(case_item, legal_sections, parties)
            
            # ê²€?‰ìš© ?ìŠ¤???ì„±
            searchable_text = self._generate_searchable_text(case_name, case_number, sections)
            
            processed_case = {
                'case_id': case_id,
                'category': metadata.get('category', field),
                'case_name': case_name,
                'case_number': case_number,
                'decision_date': decision_date,
                'field': field,
                'court': court,
                'detail_url': detail_url,
                'full_text': full_text,
                'searchable_text': searchable_text,
                'sections': sections,
                'parties': party_info,
                'case_info': case_info,
                'data_quality': {
                    'parsing_quality_score': self._calculate_quality_score(sections),
                    'content_length': len(full_text),
                    'section_count': len(sections)
                },
                'processed_at': datetime.now().isoformat(),
                'status': 'success'
            }
            
            return processed_case
            
        except Exception as e:
            logger.error(f"Error processing single case: {e}")
            return None
    
    def _generate_case_id(self, case_item: Dict[str, Any]) -> str:
        """ì¼€?´ìŠ¤ ID ?ì„±"""
        case_number = case_item.get('case_number', '')
        case_name = case_item.get('case_name', '')
        
        if case_number:
            return f"case_{case_number}"
        elif case_name:
            clean_name = re.sub(r'[^\wê°€-??', '_', case_name)
            return f"case_{clean_name}"
        else:
            return f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _extract_legal_sections(self, legal_sections: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        ë²•ë¥  ?¹ì…˜ ì¶”ì¶œ
        
        Args:
            legal_sections: êµ¬ì¡°?”ëœ ë²•ë¥  ?¹ì…˜ ?°ì´??
            
        Returns:
            List[Dict[str, Any]]: ì¶”ì¶œ???¹ì…˜ ëª©ë¡
        """
        sections = []
        
        for korean_name, english_key in self.section_types.items():
            content = legal_sections.get(korean_name, '')
            if content and content.strip():
                section = {
                    'section_type': english_key,
                    'section_type_korean': korean_name,
                    'section_content': self._clean_content(content),
                    'section_length': len(content),
                    'has_content': True
                }
                sections.append(section)
            else:
                # ë¹??¹ì…˜???¬í•¨ (êµ¬ì¡°???¼ê???
                section = {
                    'section_type': english_key,
                    'section_type_korean': korean_name,
                    'section_content': '',
                    'section_length': 0,
                    'has_content': False
                }
                sections.append(section)
        
        return sections
    
    def _extract_party_info(self, parties: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        ?¹ì‚¬???•ë³´ ì¶”ì¶œ
        
        Args:
            parties: ?¹ì‚¬???°ì´??
            
        Returns:
            List[Dict[str, Any]]: ?¹ì‚¬???•ë³´ ëª©ë¡
        """
        party_list = []
        
        # ?ê³  ?•ë³´
        plaintiff = parties.get('plaintiff', '')
        if plaintiff and plaintiff.strip():
            party_list.append({
                'party_type': 'plaintiff',
                'party_type_korean': '?ê³ ',
                'party_content': self._clean_content(plaintiff),
                'party_length': len(plaintiff)
            })
        
        # ?¼ê³  ?•ë³´
        defendant = parties.get('defendant', '')
        if defendant and defendant.strip():
            party_list.append({
                'party_type': 'defendant',
                'party_type_korean': '?¼ê³ ',
                'party_content': self._clean_content(defendant),
                'party_length': len(defendant)
            })
        
        return party_list
    
    def _generate_full_text(self, case_item: Dict[str, Any], legal_sections: Dict[str, Any], parties: Dict[str, Any]) -> str:
        """?„ì²´ ?ìŠ¤???ì„±"""
        text_parts = []
        
        # ì¼€?´ìŠ¤ ê¸°ë³¸ ?•ë³´
        case_name = case_item.get('case_name', '')
        case_number = case_item.get('case_number', '')
        decision_date = case_item.get('decision_date', '')
        court = case_item.get('court', '')
        
        if case_name:
            text_parts.append(f"?¬ê±´ëª? {case_name}")
        if case_number:
            text_parts.append(f"?¬ê±´ë²ˆí˜¸: {case_number}")
        if decision_date:
            text_parts.append(f"? ê³ ?? {decision_date}")
        if court:
            text_parts.append(f"ë²•ì›: {court}")
        
        # ë²•ë¥  ?¹ì…˜ ?´ìš©
        for korean_name, english_key in self.section_types.items():
            content = legal_sections.get(korean_name, '')
            if content and content.strip():
                text_parts.append(f"{korean_name}: {content}")
        
        # ?¹ì‚¬???•ë³´
        plaintiff = parties.get('plaintiff', '')
        defendant = parties.get('defendant', '')
        
        if plaintiff:
            text_parts.append(f"?ê³ : {plaintiff}")
        if defendant:
            text_parts.append(f"?¼ê³ : {defendant}")
        
        return '\n'.join(text_parts)
    
    def _generate_searchable_text(self, case_name: str, case_number: str, sections: List[Dict[str, Any]]) -> str:
        """ê²€?‰ìš© ?ìŠ¤???ì„±"""
        search_parts = [case_name, case_number]
        
        for section in sections:
            if section['has_content']:
                search_parts.append(section['section_content'])
        
        return ' '.join(search_parts)
    
    def _clean_content(self, content: str) -> str:
        """?´ìš© ?•ë¦¬"""
        if not content:
            return ""
        
        # HTML ?œê·¸ ?œê±°
        cleaned = re.sub(r'<[^>]+>', '', content)
        
        # ?°ì†??ê³µë°±???˜ë‚˜ë¡?
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # ?œì–´ ë¬¸ì ?œê±°
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)
        
        return cleaned.strip()
    
    def _calculate_quality_score(self, sections: List[Dict[str, Any]]) -> float:
        """?Œì‹± ?ˆì§ˆ ?ìˆ˜ ê³„ì‚°"""
        if not sections:
            return 0.0
        
        total_sections = len(sections)
        filled_sections = sum(1 for section in sections if section['has_content'])
        
        # ê¸°ë³¸ ?ìˆ˜: ì±„ì›Œì§??¹ì…˜ ë¹„ìœ¨
        base_score = filled_sections / total_sections
        
        # ?´ìš© ê¸¸ì´ ë³´ë„ˆ??
        total_length = sum(section['section_length'] for section in sections)
        length_bonus = min(total_length / 10000, 0.2)  # ìµœë? 0.2 ë³´ë„ˆ??
        
        return min(base_score + length_bonus, 1.0)


def main():
    """?ŒìŠ¤?¸ìš© ë©”ì¸ ?¨ìˆ˜"""
    import argparse
    
    parser = argparse.ArgumentParser(description="?ë? ?°ì´???„ì²˜ë¦¬ê¸°")
    parser.add_argument('--input-file', required=True, help='?…ë ¥ ?Œì¼ ê²½ë¡œ')
    parser.add_argument('--output-file', help='ì¶œë ¥ ?Œì¼ ê²½ë¡œ')
    parser.add_argument('--verbose', '-v', action='store_true', help='?ì„¸ ë¡œê·¸ ì¶œë ¥')
    
    args = parser.parse_args()
    
    # ë¡œê¹… ?¤ì •
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # ?„ì²˜ë¦¬ê¸° ì´ˆê¸°??
        preprocessor = PrecedentPreprocessor()
        
        # ?Œì¼ ?½ê¸°
        with open(args.input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # ?„ì²˜ë¦??¤í–‰
        processed_data = preprocessor.process_precedent_data(raw_data)
        
        # ê²°ê³¼ ?€??
        output_file = args.output_file or f"processed_{Path(args.input_file).name}"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Preprocessing completed. Output saved to: {output_file}")
        logger.info(f"Processed {processed_data['metadata']['total_cases']} cases")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
