#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?°ì´?°ë² ?´ìŠ¤ ?´ìš© ë¶„ì„ ?¤í¬ë¦½íŠ¸

SQLite ?°ì´?°ë² ?´ìŠ¤???´ìš©??ë¶„ì„?˜ì—¬ ?´ë–¤ ?°ì´?°ê? ?ˆëŠ”ì§€ ?•ì¸?©ë‹ˆ??
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
sys.path.append(str(Path(__file__).parent.parent))

# ì§ì ‘ DatabaseManager ?´ë˜???•ì˜ (import ?¤ë¥˜ ë°©ì?)
import sqlite3
from contextlib import contextmanager
from typing import List, Dict, Any

class DatabaseManager:
    """?°ì´?°ë² ?´ìŠ¤ ê´€ë¦??´ë˜??""
    
    def __init__(self, db_path: str = "data/lawfirm.db"):
        """?°ì´?°ë² ?´ìŠ¤ ê´€ë¦¬ì ì´ˆê¸°??""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"DatabaseManager initialized with path: {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """?°ì´?°ë² ?´ìŠ¤ ?°ê²° ì»¨í…?¤íŠ¸ ë§¤ë‹ˆ?€"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """ì¿¼ë¦¬ ?¤í–‰"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

# ë¡œê¹… ?¤ì •
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/database_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DatabaseAnalyzer:
    """?°ì´?°ë² ?´ìŠ¤ ë¶„ì„ ?´ë˜??""
    
    def __init__(self, db_path: str = "data/lawfirm.db"):
        """?°ì´?°ë² ?´ìŠ¤ ë¶„ì„ê¸?ì´ˆê¸°??""
        self.db_path = Path(db_path)
        self.db_manager = DatabaseManager(str(self.db_path))
        self.analysis_result = {}
        
        logger.info(f"DatabaseAnalyzer initialized with path: {self.db_path}")
    
    def analyze_database_content(self) -> Dict[str, Any]:
        """?°ì´?°ë² ?´ìŠ¤ ?´ìš© ë¶„ì„"""
        logger.info("?°ì´?°ë² ?´ìŠ¤ ?´ìš© ë¶„ì„ ?œì‘...")
        
        analysis = {
            "database_path": str(self.db_path),
            "analysis_timestamp": datetime.now().isoformat(),
            "table_counts": {},
            "document_types": {},
            "content_samples": {},
            "metadata_analysis": {},
            "total_documents": 0,
            "analysis_summary": {}
        }
        
        try:
            # 1. ?Œì´ë¸”ë³„ ?°ì´?????•ì¸
            analysis["table_counts"] = self._analyze_table_counts()
            
            # 2. ë¬¸ì„œ ?€?…ë³„ ë¶„ì„
            analysis["document_types"] = self._analyze_document_types()
            
            # 3. ë©”í??°ì´??ë¶„ì„
            analysis["metadata_analysis"] = self._analyze_metadata()
            
            # 4. ì½˜í…ì¸??˜í”Œ ì¶”ì¶œ
            analysis["content_samples"] = self._extract_content_samples()
            
            # 5. ?„ì²´ ë¬¸ì„œ ??ê³„ì‚°
            analysis["total_documents"] = sum(analysis["table_counts"].values())
            
            # 6. ë¶„ì„ ?”ì•½ ?ì„±
            analysis["analysis_summary"] = self._generate_analysis_summary(analysis)
            
            logger.info(f"?°ì´?°ë² ?´ìŠ¤ ë¶„ì„ ?„ë£Œ: ì´?{analysis['total_documents']}ê°?ë¬¸ì„œ")
            
        except Exception as e:
            logger.error(f"?°ì´?°ë² ?´ìŠ¤ ë¶„ì„ ì¤??¤ë¥˜: {e}")
            analysis["error"] = str(e)
        
        self.analysis_result = analysis
        return analysis
    
    def _analyze_table_counts(self) -> Dict[str, int]:
        """?Œì´ë¸”ë³„ ?°ì´????ë¶„ì„"""
        table_counts = {}
        
        tables = [
            "documents", "law_metadata", "precedent_metadata", 
            "constitutional_metadata", "interpretation_metadata",
            "administrative_rule_metadata", "local_ordinance_metadata",
            "chat_history"
        ]
        
        for table in tables:
            try:
                count_query = f"SELECT COUNT(*) FROM {table}"
                result = self.db_manager.execute_query(count_query)
                table_counts[table] = result[0][0] if result else 0
                logger.info(f"{table} ?Œì´ë¸? {table_counts[table]}ê°??ˆì½”??)
            except Exception as e:
                logger.warning(f"{table} ?Œì´ë¸?ë¶„ì„ ?¤íŒ¨: {e}")
                table_counts[table] = 0
        
        return table_counts
    
    def _analyze_document_types(self) -> Dict[str, Any]:
        """ë¬¸ì„œ ?€?…ë³„ ë¶„ì„"""
        document_types = {}
        
        try:
            # ë¬¸ì„œ ?€?…ë³„ ê°œìˆ˜ ì¡°íšŒ
            type_query = """
                SELECT document_type, COUNT(*) as count
                FROM documents 
                GROUP BY document_type
            """
            results = self.db_manager.execute_query(type_query)
            
            for row in results:
                doc_type = row["document_type"]
                count = row["count"]
                document_types[doc_type] = {
                    "count": count,
                    "percentage": 0  # ?˜ì¤‘??ê³„ì‚°
                }
                logger.info(f"{doc_type} ë¬¸ì„œ: {count}ê°?)
            
            # ë¹„ìœ¨ ê³„ì‚°
            total = sum(doc_type["count"] for doc_type in document_types.values())
            if total > 0:
                for doc_type in document_types.values():
                    doc_type["percentage"] = round((doc_type["count"] / total) * 100, 2)
            
        except Exception as e:
            logger.warning(f"ë¬¸ì„œ ?€??ë¶„ì„ ?¤íŒ¨: {e}")
        
        return document_types
    
    def _analyze_metadata(self) -> Dict[str, Any]:
        """ë©”í??°ì´??ë¶„ì„"""
        metadata_analysis = {}
        
        try:
            # ë²•ë ¹ ë©”í??°ì´??ë¶„ì„
            law_query = """
                SELECT law_name, COUNT(*) as count
                FROM law_metadata 
                WHERE law_name IS NOT NULL
                GROUP BY law_name
                ORDER BY count DESC
                LIMIT 10
            """
            law_results = self.db_manager.execute_query(law_query)
            metadata_analysis["top_laws"] = [dict(row) for row in law_results]
            
            # ?ë? ë©”í??°ì´??ë¶„ì„
            precedent_query = """
                SELECT court_name, COUNT(*) as count
                FROM precedent_metadata 
                WHERE court_name IS NOT NULL
                GROUP BY court_name
                ORDER BY count DESC
                LIMIT 10
            """
            precedent_results = self.db_manager.execute_query(precedent_query)
            metadata_analysis["top_courts"] = [dict(row) for row in precedent_results]
            
        except Exception as e:
            logger.warning(f"ë©”í??°ì´??ë¶„ì„ ?¤íŒ¨: {e}")
        
        return metadata_analysis
    
    def _extract_content_samples(self) -> Dict[str, List[str]]:
        """ì½˜í…ì¸??˜í”Œ ì¶”ì¶œ"""
        content_samples = {}
        
        try:
            # ê°?ë¬¸ì„œ ?€?…ë³„ ?˜í”Œ ì¶”ì¶œ
            sample_query = """
                SELECT document_type, title, content
                FROM documents 
                WHERE content IS NOT NULL AND LENGTH(content) > 50
                ORDER BY RANDOM()
                LIMIT 3
            """
            results = self.db_manager.execute_query(sample_query)
            
            for row in results:
                doc_type = row["document_type"]
                if doc_type not in content_samples:
                    content_samples[doc_type] = []
                
                content_samples[doc_type].append({
                    "title": row["title"],
                    "content_preview": row["content"][:200] + "..." if len(row["content"]) > 200 else row["content"]
                })
            
        except Exception as e:
            logger.warning(f"ì½˜í…ì¸??˜í”Œ ì¶”ì¶œ ?¤íŒ¨: {e}")
        
        return content_samples
    
    def _generate_analysis_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """ë¶„ì„ ?”ì•½ ?ì„±"""
        summary = {
            "has_data": analysis["total_documents"] > 0,
            "primary_data_types": [],
            "recommended_approach": "",
            "estimated_qa_potential": 0
        }
        
        if analysis["total_documents"] > 0:
            # ì£¼ìš” ?°ì´???€???ë³„
            doc_types = analysis["document_types"]
            if doc_types:
                sorted_types = sorted(doc_types.items(), key=lambda x: x[1]["count"], reverse=True)
                summary["primary_data_types"] = [doc_type for doc_type, _ in sorted_types[:3]]
            
            # ì¶”ì²œ ?‘ê·¼ë²?ê²°ì •
            if "law" in doc_types and doc_types["law"]["count"] > 0:
                summary["recommended_approach"] = "law_and_precedent_focused"
            elif "precedent" in doc_types and doc_types["precedent"]["count"] > 0:
                summary["recommended_approach"] = "precedent_focused"
            else:
                summary["recommended_approach"] = "mixed_approach"
            
            # ?ˆìƒ Q&A ?ì„± ê°€????
            summary["estimated_qa_potential"] = analysis["total_documents"] * 2  # ë¬¸ì„œ???‰ê·  2ê°?Q&A
        
        else:
            summary["recommended_approach"] = "template_based_generation"
            summary["estimated_qa_potential"] = 200  # ?œí”Œë¦?ê¸°ë°˜ ?ì„± ê°€????
        
        return summary
    
    def save_analysis_report(self, output_path: str = "logs/database_analysis_report.json"):
        """ë¶„ì„ ë³´ê³ ???€??""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"ë¶„ì„ ë³´ê³ ???€???„ë£Œ: {output_file}")
        return str(output_file)
    
    def print_summary(self):
        """ë¶„ì„ ê²°ê³¼ ?”ì•½ ì¶œë ¥"""
        if not self.analysis_result:
            logger.warning("ë¶„ì„ ê²°ê³¼ê°€ ?†ìŠµ?ˆë‹¤.")
            return
        
        summary = self.analysis_result.get("analysis_summary", {})
        table_counts = self.analysis_result.get("table_counts", {})
        doc_types = self.analysis_result.get("document_types", {})
        
        print("\n" + "="*60)
        print("?“Š ?°ì´?°ë² ?´ìŠ¤ ë¶„ì„ ê²°ê³¼ ?”ì•½")
        print("="*60)
        
        print(f"?“ ?°ì´?°ë² ?´ìŠ¤ ê²½ë¡œ: {self.analysis_result.get('database_path', 'N/A')}")
        print(f"?“… ë¶„ì„ ?œê°„: {self.analysis_result.get('analysis_timestamp', 'N/A')}")
        print(f"?“„ ì´?ë¬¸ì„œ ?? {self.analysis_result.get('total_documents', 0)}ê°?)
        
        print("\n?“‹ ?Œì´ë¸”ë³„ ?°ì´????")
        for table, count in table_counts.items():
            print(f"  - {table}: {count}ê°?)
        
        print("\n?“š ë¬¸ì„œ ?€?…ë³„ ë¶„í¬:")
        for doc_type, info in doc_types.items():
            print(f"  - {doc_type}: {info['count']}ê°?({info['percentage']}%)")
        
        print(f"\n?¯ ì¶”ì²œ ?‘ê·¼ë²? {summary.get('recommended_approach', 'N/A')}")
        print(f"?”¢ ?ˆìƒ Q&A ?ì„± ê°€???? {summary.get('estimated_qa_potential', 0)}ê°?)
        
        if summary.get("primary_data_types"):
            print(f"â­?ì£¼ìš” ?°ì´???€?? {', '.join(summary['primary_data_types'])}")
        
        print("="*60)


def main():
    """ë©”ì¸ ?¤í–‰ ?¨ìˆ˜"""
    logger.info("?°ì´?°ë² ?´ìŠ¤ ë¶„ì„ ?œì‘...")
    
    # ?°ì´?°ë² ?´ìŠ¤ ë¶„ì„ê¸?ì´ˆê¸°??
    analyzer = DatabaseAnalyzer()
    
    # ?°ì´?°ë² ?´ìŠ¤ ?´ìš© ë¶„ì„
    analysis_result = analyzer.analyze_database_content()
    
    # ë¶„ì„ ë³´ê³ ???€??
    report_path = analyzer.save_analysis_report()
    
    # ë¶„ì„ ê²°ê³¼ ?”ì•½ ì¶œë ¥
    analyzer.print_summary()
    
    logger.info("?°ì´?°ë² ?´ìŠ¤ ë¶„ì„ ?„ë£Œ!")
    logger.info(f"?ì„¸ ë³´ê³ ?? {report_path}")
    
    return analysis_result


if __name__ == "__main__":
    main()
