#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?ë™ ?°ì´??ê°ì? ?œìŠ¤??

?ˆë¡œ???°ì´???ŒìŠ¤ë¥??ë™?¼ë¡œ ê°ì??˜ê³  ë¶„ë¥˜?˜ëŠ” ?œìŠ¤?œìž…?ˆë‹¤.
? ì§œë³??´ë”?€ ?Œì¼ ?¨í„´??ë¶„ì„?˜ì—¬ ì²˜ë¦¬???°ì´?°ë? ?ë³„?©ë‹ˆ??
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import argparse
from collections import defaultdict

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.database import DatabaseManager

logger = logging.getLogger(__name__)


class AutoDataDetector:
    """?ë™ ?°ì´??ê°ì? ?´ëž˜??""
    
    def __init__(self, raw_data_base_path: str = "data/raw/assembly", db_manager: DatabaseManager = None):
        """
        ?ë™ ?°ì´??ê°ì?ê¸?ì´ˆê¸°??
        
        Args:
            raw_data_base_path: ?ë³¸ ?°ì´??ê¸°ë³¸ ê²½ë¡œ
            db_manager: ?°ì´?°ë² ?´ìŠ¤ ê´€ë¦¬ìž (? íƒ?¬í•­)
        """
        self.raw_data_base_path = Path(raw_data_base_path)
        self.db_manager = db_manager or DatabaseManager()
        
        # ?°ì´???¨í„´ ?•ì˜
        self.data_patterns = {
            'law_only': {
                'file_pattern': r'law_only_page_\d+_\d+_\d+\.json',
                'directory_pattern': r'\d{8}',  # YYYYMMDD ?•ì‹
                'metadata_key': 'data_type',
                'expected_value': 'law_only'
            },
            'precedent_civil': {
                'file_pattern': r'precedent_civil_page_\d+_\d+_\d+_\d+\.json',
                'directory_pattern': r'\d{8}/civil',
                'metadata_key': 'category',
                'expected_value': 'civil'
            },
            'precedent_criminal': {
                'file_pattern': r'precedent_criminal_page_\d+_\d+_\d+_\d+\.json',
                'directory_pattern': r'\d{8}/criminal',
                'metadata_key': 'category',
                'expected_value': 'criminal'
            },
            'precedent_family': {
                'file_pattern': r'precedent_family_page_\d+_\d+_\d+_\d+\.json',
                'directory_pattern': r'\d{8}/family',
                'metadata_key': 'category',
                'expected_value': 'family'
            },
            'precedent_tax': {
                'file_pattern': r'precedent_tax_page_\d+_\d+_\d+_\d+\.json',
                'directory_pattern': r'\d{8}/tax',
                'metadata_key': 'category',
                'expected_value': 'tax'
            },
            'precedent_administrative': {
                'file_pattern': r'precedent_administrative_page_\d+_\d+_\d+_\d+\.json',
                'directory_pattern': r'\d{8}/administrative',
                'metadata_key': 'category',
                'expected_value': 'administrative'
            },
            'precedent_patent': {
                'file_pattern': r'precedent_patent_page_\d+_\d+_\d+_\d+\.json',
                'directory_pattern': r'\d{8}/patent',
                'metadata_key': 'category',
                'expected_value': 'patent'
            },
            'precedents': {
                'file_pattern': r'precedent_page_\d+_\d+\.json',
                'directory_pattern': r'\d{8}',
                'metadata_key': 'data_type',
                'expected_value': 'precedents'
            },
            'constitutional': {
                'file_pattern': r'constitutional_page_\d+\.json',
                'directory_pattern': r'\d{8}',
                'metadata_key': 'data_type',
                'expected_value': 'constitutional'
            }
        }
        
        # ê¸°ë³¸ ê²½ë¡œ ?¤ì •
        self.base_paths = {
            'law_only': 'data/raw/assembly/law_only',
            'precedent_civil': 'data/raw/assembly/precedent',
            'precedent_criminal': 'data/raw/assembly/precedent',
            'precedent_family': 'data/raw/assembly/precedent',
            'precedent_tax': 'data/raw/assembly/precedent',
            'precedent_administrative': 'data/raw/assembly/precedent',
            'precedent_patent': 'data/raw/assembly/precedent',
            'precedents': 'data/raw/assembly/precedents',
            'constitutional': 'data/raw/constitutional'
        }
        
        logger.info("AutoDataDetector initialized")
    
    def detect_new_data_sources(self, base_path: str, data_type: str = None) -> Dict[str, List[Path]]:
        """
        ?ˆë¡œ???°ì´???ŒìŠ¤ ê°ì?
        
        Args:
            base_path: ê²€?‰í•  ê¸°ë³¸ ê²½ë¡œ
            data_type: ?¹ì • ?°ì´??? í˜• (None?´ë©´ ëª¨ë“  ? í˜•)
        
        Returns:
            Dict[str, List[Path]]: ?°ì´??? í˜•ë³??Œì¼ ëª©ë¡
        """
        logger.info(f"Detecting new data sources in: {base_path}")
        
        detected_files = defaultdict(list)
        base_path_obj = Path(base_path)
        
        if not base_path_obj.exists():
            logger.warning(f"Base path does not exist: {base_path}")
            return dict(detected_files)
        
        # ? ì§œë³??´ë” ?¤ìº”
        for date_folder in base_path_obj.iterdir():
            if not date_folder.is_dir():
                continue
            
            # ? ì§œ ?´ë” ?¨í„´ ?•ì¸ (YYYYMMDD)
            if not self._is_date_folder(date_folder.name):
                continue
            
            logger.info(f"Scanning date folder: {date_folder.name}")
            
            # ?´ë” ???Œì¼ ?¤ìº” (ì§ì ‘ ?Œì¼ ?ëŠ” ì¹´í…Œê³ ë¦¬ ?˜ìœ„ ?´ë”)
            self._scan_folder_for_files(date_folder, detected_files, data_type)
        
        # ê²°ê³¼ ?”ì•½
        total_files = sum(len(files) for files in detected_files.values())
        logger.info(f"Detection completed: {total_files} new files found")
        for data_type, files in detected_files.items():
            logger.info(f"  {data_type}: {len(files)} files")
        
        return dict(detected_files)
    
    def _scan_folder_for_files(self, folder: Path, detected_files: Dict[str, List[Path]], data_type: str = None):
        """
        ?´ë” ???Œì¼?¤ì„ ?¤ìº”?˜ì—¬ ê°ì????Œì¼ ëª©ë¡??ì¶”ê?
        
        Args:
            folder: ?¤ìº”???´ë”
            detected_files: ê°ì????Œì¼?¤ì„ ?€?¥í•  ?•ì…”?ˆë¦¬
            data_type: ?¹ì • ?°ì´??? í˜• ?„í„°
        """
        # ì§ì ‘ ?Œì¼ ?¤ìº”
        for file_path in folder.glob("*.json"):
            if not file_path.is_file():
                continue
            
            # ?Œì¼ ? í˜• ë¶„ë¥˜
            file_data_type = self.classify_data_type(file_path)
            
            if file_data_type and (data_type is None or file_data_type == data_type):
                # ?´ë? ì²˜ë¦¬???Œì¼?¸ì? ?•ì¸
                if not self.db_manager.is_file_processed(str(file_path)):
                    detected_files[file_data_type].append(file_path)
                    logger.debug(f"New file detected: {file_path} (type: {file_data_type})")
                else:
                    logger.debug(f"File already processed: {file_path}")
        
        # ?˜ìœ„ ì¹´í…Œê³ ë¦¬ ?´ë” ?¤ìº” (precedent??ê²½ìš° civil, criminal, family, tax ??
        for subfolder in folder.iterdir():
            if subfolder.is_dir() and subfolder.name in ['civil', 'criminal', 'family', 'tax']:
                logger.debug(f"Scanning category subfolder: {subfolder.name}")
                self._scan_folder_for_files(subfolder, detected_files, data_type)
    
    def get_file_hash(self, file_path: Path) -> str:
        """?Œì¼ ?´ìš©??SHA256 ?´ì‹œë¥?ê³„ì‚°?˜ì—¬ ë°˜í™˜"""
        import hashlib
        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()
    
    def classify_data_type(self, file_path: Path) -> Optional[str]:
        """
        ?Œì¼ ?´ìš© ê¸°ë°˜ ?°ì´??? í˜• ë¶„ë¥˜
        
        Args:
            file_path: ë¶„ë¥˜???Œì¼ ê²½ë¡œ
        
        Returns:
            Optional[str]: ?°ì´??? í˜• ?ëŠ” None
        """
        try:
            # ?Œì¼ ?¬ê¸° ?•ì¸ (?ˆë¬´ ???Œì¼?€ ?¤í‚µ)
            if file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB
                logger.warning(f"File too large to analyze: {file_path}")
                return None
            
            # JSON ?Œì¼ ?½ê¸°
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # ë©”í??°ì´?°ì—???°ì´??? í˜• ?•ì¸
            if isinstance(data, dict) and 'metadata' in data:
                metadata = data['metadata']
                data_type = metadata.get('data_type')
                category = metadata.get('category')
                
                # precedent ?°ì´?°ì˜ ê²½ìš° ì¹´í…Œê³ ë¦¬ ê¸°ë°˜?¼ë¡œ ë¶„ë¥˜
                if data_type == 'precedent' and category:
                    precedent_type = f'precedent_{category}'
                    if precedent_type in self.data_patterns:
                        return precedent_type
                
                if data_type in self.data_patterns:
                    return data_type
            
            # ?Œì¼ëª??¨í„´?¼ë¡œ ë¶„ë¥˜
            filename = file_path.name
            for data_type, pattern_info in self.data_patterns.items():
                import re
                if re.match(pattern_info['file_pattern'], filename):
                    return data_type
            
            # items êµ¬ì¡°ë¡?ë¶„ë¥˜ (law_only ë°?precedent ?¹í™”)
            if isinstance(data, dict) and 'items' in data:
                items = data['items']
                if items and isinstance(items, list):
                    first_item = items[0]
                    if isinstance(first_item, dict):
                        # law_name???ˆìœ¼ë©?ë²•ë¥  ?°ì´?°ë¡œ ë¶„ë¥˜
                        if 'law_name' in first_item and 'law_content' in first_item:
                            return 'law_only'
                        # case_numberê°€ ?ˆìœ¼ë©??ë? ?°ì´?°ë¡œ ë¶„ë¥˜
                        elif 'case_number' in first_item:
                            # ì¹´í…Œê³ ë¦¬ ?•ë³´ê°€ ?ˆìœ¼ë©?êµ¬ì²´?ì¸ precedent ?€??ë°˜í™˜
                            if 'field' in first_item:
                                field = first_item['field']
                                if field == 'ë¯¼ì‚¬':
                                    return 'precedent_civil'
                                elif field == '?•ì‚¬':
                                    return 'precedent_criminal'
                                elif field == 'ê°€??:
                                    return 'precedent_family'
                                elif field == 'ì¡°ì„¸':
                                    return 'precedent_tax'
                            return 'precedents'
            
            logger.warning(f"Could not classify file: {file_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error classifying file {file_path}: {e}")
            return None
    
    def get_data_statistics(self, files: List[Path]) -> Dict[str, Any]:
        """
        ?Œì¼ ëª©ë¡???µê³„ ?•ë³´ ?ì„±
        
        Args:
            files: ë¶„ì„???Œì¼ ëª©ë¡
        
        Returns:
            Dict[str, Any]: ?µê³„ ?•ë³´
        """
        if not files:
            return {
                'total_files': 0,
                'total_size': 0,
                'date_range': None,
                'file_types': {},
                'estimated_records': 0
            }
        
        total_size = 0
        file_types = defaultdict(int)
        dates = []
        estimated_records = 0
        
        for file_path in files:
            # ?Œì¼ ?¬ê¸°
            total_size += file_path.stat().st_size
            
            # ?Œì¼ ? í˜•
            file_type = self.classify_data_type(file_path)
            if file_type:
                file_types[file_type] += 1
            
            # ? ì§œ ì¶”ì¶œ
            date_folder = file_path.parent.name
            if self._is_date_folder(date_folder):
                dates.append(date_folder)
            
            # ?ˆìƒ ?ˆì½”????ì¶”ì •
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'items' in data:
                        estimated_records += len(data['items'])
            except Exception:
                pass
        
        # ? ì§œ ë²”ìœ„ ê³„ì‚°
        date_range = None
        if dates:
            dates.sort()
            date_range = {
                'start': dates[0],
                'end': dates[-1],
                'total_days': len(set(dates))
            }
        
        return {
            'total_files': len(files),
            'total_size': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'date_range': date_range,
            'file_types': dict(file_types),
            'estimated_records': estimated_records,
            'avg_file_size_mb': round(total_size / len(files) / (1024 * 1024), 2) if files else 0
        }
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """
        ?Œì¼ ?´ì‹œ ê³„ì‚°
        
        Args:
            file_path: ?´ì‹œë¥?ê³„ì‚°???Œì¼ ê²½ë¡œ
        
        Returns:
            str: ?Œì¼??SHA-256 ?´ì‹œê°?
        """
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def _is_date_folder(self, folder_name: str) -> bool:
        """
        ?´ë”ëª…ì´ ? ì§œ ?•ì‹?¸ì? ?•ì¸
        
        Args:
            folder_name: ?•ì¸???´ë”ëª?
        
        Returns:
            bool: ? ì§œ ?•ì‹ ?¬ë?
        """
        import re
        return bool(re.match(r'^\d{8}$', folder_name))
    
    def get_processing_priority(self, data_type: str) -> int:
        """
        ?°ì´??? í˜•ë³?ì²˜ë¦¬ ?°ì„ ?œìœ„ ë°˜í™˜
        
        Args:
            data_type: ?°ì´??? í˜•
        
        Returns:
            int: ?°ì„ ?œìœ„ (??„?˜ë¡ ?’ì? ?°ì„ ?œìœ„)
        """
        priority_map = {
            'law_only': 1,
            'precedents': 2,
            'constitutional': 3,
            'legal_interpretations': 4,
            'administrative_rules': 5
        }
        return priority_map.get(data_type, 99)
    
    def generate_detection_report(self, detected_files: Dict[str, List[Path]]) -> Dict[str, Any]:
        """
        ê°ì? ê²°ê³¼ ë¦¬í¬???ì„±
        
        Args:
            detected_files: ê°ì????Œì¼ ëª©ë¡
        
        Returns:
            Dict[str, Any]: ê°ì? ë¦¬í¬??
        """
        report = {
            'detection_time': datetime.now().isoformat(),
            'total_data_types': len(detected_files),
            'total_files': sum(len(files) for files in detected_files.values()),
            'data_types': {}
        }
        
        for data_type, files in detected_files.items():
            stats = self.get_data_statistics(files)
            stats['priority'] = self.get_processing_priority(data_type)
            report['data_types'][data_type] = stats
        
        return report


def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    parser = argparse.ArgumentParser(description='?ë™ ?°ì´??ê°ì? ?œìŠ¤??)
    parser.add_argument('--base-path', default='data/raw/assembly/law_only',
                       help='ê²€?‰í•  ê¸°ë³¸ ê²½ë¡œ')
    parser.add_argument('--data-type', choices=['law_only', 'precedents', 'constitutional'],
                       help='?¹ì • ?°ì´??? í˜•ë§?ê²€??)
    parser.add_argument('--output-report', help='ê°ì? ë¦¬í¬?¸ë? ?€?¥í•  ?Œì¼ ê²½ë¡œ')
    parser.add_argument('--verbose', '-v', action='store_true', help='?ì„¸ ë¡œê·¸ ì¶œë ¥')
    
    args = parser.parse_args()
    
    # ë¡œê¹… ?¤ì •
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # ?°ì´??ê°ì?ê¸?ì´ˆê¸°??
        detector = AutoDataDetector()
        
        # ?°ì´??ê°ì? ?¤í–‰
        logger.info("Starting data detection...")
        detected_files = detector.detect_new_data_sources(args.base_path, args.data_type)
        
        # ê²°ê³¼ ì¶œë ¥
        if detected_files:
            logger.info("Detection Results:")
            for data_type, files in detected_files.items():
                logger.info(f"  {data_type}: {len(files)} files")
                
                # ì²˜ìŒ ëª?ê°??Œì¼ ê²½ë¡œ ì¶œë ¥
                for i, file_path in enumerate(files[:3]):
                    logger.info(f"    - {file_path}")
                if len(files) > 3:
                    logger.info(f"    ... and {len(files) - 3} more files")
        else:
            logger.info("No new files detected")
        
        # ë¦¬í¬???ì„± ë°??€??
        if args.output_report:
            report = detector.generate_detection_report(detected_files)
            report_path = Path(args.output_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Detection report saved to: {report_path}")
        
        return len(detected_files) > 0
        
    except Exception as e:
        logger.error(f"Error in data detection: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
