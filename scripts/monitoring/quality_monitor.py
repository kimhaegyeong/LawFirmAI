#!/usr/bin/env python3
"""
?ˆì§ˆ ëª¨ë‹ˆ?°ë§ ?œìŠ¤??
ML ê°•í™” ?Œì‹± ?ˆì§ˆ???¤ì‹œê°„ìœ¼ë¡?ëª¨ë‹ˆ?°ë§?˜ê³  ?€?œë³´??ë©”íŠ¸ë¦?„ ?œê³µ?©ë‹ˆ??
"""

import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import sqlite3
from contextlib import contextmanager

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lawfirm_langgraph.core.search.connectors.legal_data_connector_v2 import LegalDataConnectorV2 as DatabaseManager
from source.utils.config import Config

# Windows ì½˜ì†”?ì„œ UTF-8 ?¸ì½”???¤ì •
if os.name == 'nt':  # Windows
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except AttributeError:
        # ?´ë? UTF-8ë¡??¤ì •??ê²½ìš° ë¬´ì‹œ
        pass

logger = logging.getLogger(__name__)


class QualityMonitor:
    """?ˆì§ˆ ëª¨ë‹ˆ?°ë§ ?´ë˜??""
    
    def __init__(self, config: Config):
        """?ˆì§ˆ ëª¨ë‹ˆ?°ë§ ì´ˆê¸°??""
        self.config = config
        
        # SQLite URL???Œì¼ ê²½ë¡œë¡?ë³€??
        db_url = config.database_url
        if db_url.startswith("sqlite:///"):
            db_path = db_url.replace("sqlite:///", "")
            if db_path.startswith("./"):
                db_path = db_path[2:]  # "./" ?œê±°
        else:
            db_path = config.database_path
        
        # ?ˆë? ê²½ë¡œë¡?ë³€??
        db_path = os.path.abspath(db_path)
        
        self.database = DatabaseManager(db_path)
        self.logger = logging.getLogger(__name__)
        
        # ëª¨ë‹ˆ?°ë§ ?¤ì •
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'fair': 0.5,
            'poor': 0.0
        }
        
        self.logger.info("QualityMonitor initialized")
    
    def get_overall_quality_stats(self) -> Dict[str, Any]:
        """?„ì²´ ?ˆì§ˆ ?µê³„ ì¡°íšŒ"""
        try:
            stats_query = """
                SELECT 
                    COUNT(*) as total_articles,
                    SUM(CASE WHEN ml_enhanced = 1 THEN 1 ELSE 0 END) as ml_enhanced_articles,
                    AVG(al.parsing_quality_score) as avg_quality_score,
                    MIN(al.parsing_quality_score) as min_quality_score,
                    MAX(al.parsing_quality_score) as max_quality_score,
                    SUM(CASE WHEN aa.article_type = 'main' THEN 1 ELSE 0 END) as main_articles,
                    SUM(CASE WHEN aa.article_type = 'supplementary' THEN 1 ELSE 0 END) as supplementary_articles,
                    SUM(CASE WHEN aa.parsing_method = 'ml_enhanced' THEN 1 ELSE 0 END) as ml_parsing_method,
                    SUM(CASE WHEN aa.parsing_method = 'rule_based' THEN 1 ELSE 0 END) as rule_parsing_method
                FROM assembly_articles aa
                LEFT JOIN assembly_laws al ON aa.law_id = al.law_id
            """
            
            result = self.database.execute_query(stats_query)
            stats = result[0] if result else {}
            
            # ?ˆì§ˆ ?±ê¸‰ë³?ë¶„í¬ ê³„ì‚°
            quality_distribution = self._calculate_quality_distribution()
            
            return {
                'total_articles': stats.get('total_articles', 0),
                'ml_enhanced_articles': stats.get('ml_enhanced_articles', 0),
                'avg_quality_score': round(stats.get('avg_quality_score', 0.0), 3),
                'min_quality_score': round(stats.get('min_quality_score', 0.0), 3),
                'max_quality_score': round(stats.get('max_quality_score', 0.0), 3),
                'main_articles': stats.get('main_articles', 0),
                'supplementary_articles': stats.get('supplementary_articles', 0),
                'ml_parsing_method': stats.get('ml_parsing_method', 0),
                'rule_parsing_method': stats.get('rule_parsing_method', 0),
                'quality_distribution': quality_distribution,
                'ml_enhancement_rate': round(
                    stats.get('ml_enhanced_articles', 0) / max(stats.get('total_articles', 1), 1) * 100, 2
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error getting overall quality stats: {e}")
            return {}
    
    def _calculate_quality_distribution(self) -> Dict[str, int]:
        """?ˆì§ˆ ?±ê¸‰ë³?ë¶„í¬ ê³„ì‚°"""
        try:
            distribution = {}
            
            for grade, threshold in self.quality_thresholds.items():
                if grade == 'poor':
                    # poor??ìµœí•˜???±ê¸‰
                    query = f"""
                        SELECT COUNT(*) as count 
                        FROM assembly_articles aa
                        LEFT JOIN assembly_laws al ON aa.law_id = al.law_id
                        WHERE al.parsing_quality_score < {self.quality_thresholds['fair']}
                    """
                else:
                    # ?¤ë¥¸ ?±ê¸‰?¤ì? ?„ê³„ê°??´ìƒ
                    query = f"""
                        SELECT COUNT(*) as count 
                        FROM assembly_articles aa
                        LEFT JOIN assembly_laws al ON aa.law_id = al.law_id
                        WHERE al.parsing_quality_score >= {threshold}
                    """
                
                result = self.database.execute_query(query)
                count = result[0]['count'] if result else 0
                distribution[grade] = count
            
            return distribution
            
        except Exception as e:
            self.logger.error(f"Error calculating quality distribution: {e}")
            return {}
    
    def get_law_quality_stats(self, limit: int = 20) -> List[Dict[str, Any]]:
        """ë²•ë¥ ë³??ˆì§ˆ ?µê³„ ì¡°íšŒ"""
        try:
            law_stats_query = """
                SELECT 
                    al.law_name,
                    al.law_id,
                    COUNT(aa.id) as article_count,
                    AVG(al.parsing_quality_score) as avg_quality_score,
                    SUM(CASE WHEN al.ml_enhanced = 1 THEN 1 ELSE 0 END) as ml_enhanced_count,
                    SUM(CASE WHEN aa.article_type = 'main' THEN 1 ELSE 0 END) as main_count,
                    SUM(CASE WHEN aa.article_type = 'supplementary' THEN 1 ELSE 0 END) as supplementary_count,
                    al.parsing_quality_score as law_quality_score
                FROM assembly_laws al
                LEFT JOIN assembly_articles aa ON al.law_id = aa.law_id
                GROUP BY al.law_id, al.law_name, al.parsing_quality_score
                ORDER BY AVG(al.parsing_quality_score) DESC
                LIMIT ?
            """
            
            results = self.database.execute_query(law_stats_query, (limit,))
            
            law_stats = []
            for row in results:
                law_stat = {
                    'law_name': row['law_name'],
                    'law_id': row['law_id'],
                    'article_count': row['article_count'],
                    'avg_quality_score': round(row['avg_quality_score'] or 0.0, 3),
                    'ml_enhanced_count': row['ml_enhanced_count'],
                    'main_count': row['main_count'],
                    'supplementary_count': row['supplementary_count'],
                    'law_quality_score': round(row['law_quality_score'] or 0.0, 3),
                    'ml_enhancement_rate': round(
                        row['ml_enhanced_count'] / max(row['article_count'], 1) * 100, 2
                    )
                }
                law_stats.append(law_stat)
            
            return law_stats
            
        except Exception as e:
            self.logger.error(f"Error getting law quality stats: {e}")
            return []
    
    def get_parsing_method_comparison(self) -> Dict[str, Any]:
        """?Œì‹± ë°©ë²•ë³?ë¹„êµ ?µê³„"""
        try:
            comparison_query = """
                SELECT 
                    aa.parsing_method,
                    COUNT(*) as count,
                    AVG(al.parsing_quality_score) as avg_quality_score,
                    AVG(aa.ml_confidence_score) as avg_ml_confidence,
                    SUM(CASE WHEN aa.article_type = 'main' THEN 1 ELSE 0 END) as main_count,
                    SUM(CASE WHEN aa.article_type = 'supplementary' THEN 1 ELSE 0 END) as supplementary_count
                FROM assembly_articles aa
                LEFT JOIN assembly_laws al ON aa.law_id = al.law_id
                GROUP BY aa.parsing_method
            """
            
            results = self.database.execute_query(comparison_query)
            
            comparison = {}
            for row in results:
                method = row['parsing_method'] or 'unknown'
                comparison[method] = {
                    'count': row['count'],
                    'avg_quality_score': round(row['avg_quality_score'] or 0.0, 3),
                    'avg_ml_confidence': round(row['avg_ml_confidence'] or 0.0, 3),
                    'main_count': row['main_count'],
                    'supplementary_count': row['supplementary_count']
                }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error getting parsing method comparison: {e}")
            return {}
    
    def get_quality_trends(self, days: int = 30) -> Dict[str, List[Dict[str, Any]]]:
        """?ˆì§ˆ ?¸ë Œ??ë¶„ì„ (ê°„ì†Œ?”ëœ ë²„ì „)"""
        try:
            # ìµœê·¼ ?°ì´?°ê? ?ˆëŠ”ì§€ ?•ì¸ (created_at ê¸°ì?)
            trend_query = """
                SELECT 
                    DATE(aa.created_at) as date,
                    COUNT(*) as articles_processed,
                    AVG(al.parsing_quality_score) as avg_quality_score,
                    SUM(CASE WHEN al.ml_enhanced = 1 THEN 1 ELSE 0 END) as ml_enhanced_count
                FROM assembly_articles aa
                LEFT JOIN assembly_laws al ON aa.law_id = al.law_id
                WHERE aa.created_at >= date('now', '-{} days')
                GROUP BY DATE(aa.created_at)
                ORDER BY date
            """.format(days)
            
            results = self.database.execute_query(trend_query)
            
            trends = {
                'daily_stats': [],
                'quality_trend': [],
                'ml_adoption_trend': []
            }
            
            for row in results:
                date_str = row['date']
                daily_stat = {
                    'date': date_str,
                    'articles_processed': row['articles_processed'],
                    'avg_quality_score': round(row['avg_quality_score'] or 0.0, 3),
                    'ml_enhanced_count': row['ml_enhanced_count'],
                    'ml_adoption_rate': round(
                        row['ml_enhanced_count'] / max(row['articles_processed'], 1) * 100, 2
                    )
                }
                trends['daily_stats'].append(daily_stat)
                trends['quality_trend'].append({
                    'date': date_str,
                    'quality_score': daily_stat['avg_quality_score']
                })
                trends['ml_adoption_trend'].append({
                    'date': date_str,
                    'adoption_rate': daily_stat['ml_adoption_rate']
                })
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error getting quality trends: {e}")
            return {'daily_stats': [], 'quality_trend': [], 'ml_adoption_trend': []}
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """?ëŸ¬ ë¶„ì„"""
        try:
            # ?ˆì§ˆ ?ìˆ˜ê°€ ??? ì¡°ë¬¸??ë¶„ì„
            low_quality_query = """
                SELECT 
                    al.law_name,
                    aa.article_number,
                    aa.article_title,
                    al.parsing_quality_score,
                    aa.parsing_method,
                    aa.ml_confidence_score,
                    aa.word_count,
                    aa.char_count
                FROM assembly_articles aa
                JOIN assembly_laws al ON aa.law_id = al.law_id
                WHERE al.parsing_quality_score < 0.5
                ORDER BY al.parsing_quality_score ASC
                LIMIT 50
            """
            
            results = self.database.execute_query(low_quality_query)
            
            error_analysis = {
                'low_quality_articles': [],
                'error_patterns': {},
                'recommendations': []
            }
            
            for row in results:
                article_info = {
                    'law_name': row['law_name'],
                    'article_number': row['article_number'],
                    'article_title': row['article_title'],
                    'parsing_quality_score': round(row['parsing_quality_score'], 3),
                    'parsing_method': row['parsing_method'],
                    'ml_confidence_score': round(row['ml_confidence_score'] or 0.0, 3),
                    'word_count': row['word_count'],
                    'char_count': row['char_count']
                }
                error_analysis['low_quality_articles'].append(article_info)
            
            # ?ëŸ¬ ?¨í„´ ë¶„ì„
            error_patterns = self._analyze_error_patterns(results)
            error_analysis['error_patterns'] = error_patterns
            
            # ê°œì„  ê¶Œì¥?¬í•­ ?ì„±
            recommendations = self._generate_recommendations(error_patterns)
            error_analysis['recommendations'] = recommendations
            
            return error_analysis
            
        except Exception as e:
            self.logger.error(f"Error getting error analysis: {e}")
            return {'low_quality_articles': [], 'error_patterns': {}, 'recommendations': []}
    
    def _analyze_error_patterns(self, low_quality_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """?ëŸ¬ ?¨í„´ ë¶„ì„"""
        try:
            patterns = {
                'common_issues': {},
                'method_performance': {},
                'content_characteristics': {}
            }
            
            # ?Œì‹± ë°©ë²•ë³??±ëŠ¥ ë¶„ì„
            method_counts = {}
            for row in low_quality_results:
                method = row['parsing_method'] or 'unknown'
                method_counts[method] = method_counts.get(method, 0) + 1
            
            patterns['method_performance'] = method_counts
            
            # ?´ìš© ?¹ì„± ë¶„ì„
            short_articles = sum(1 for row in low_quality_results if row['word_count'] < 10)
            long_articles = sum(1 for row in low_quality_results if row['word_count'] > 1000)
            
            patterns['content_characteristics'] = {
                'short_articles': short_articles,
                'long_articles': long_articles,
                'avg_word_count': sum(row['word_count'] for row in low_quality_results) / len(low_quality_results) if low_quality_results else 0
            }
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing error patterns: {e}")
            return {}
    
    def _generate_recommendations(self, error_patterns: Dict[str, Any]) -> List[str]:
        """ê°œì„  ê¶Œì¥?¬í•­ ?ì„±"""
        try:
            recommendations = []
            
            method_performance = error_patterns.get('method_performance', {})
            content_characteristics = error_patterns.get('content_characteristics', {})
            
            # ?Œì‹± ë°©ë²•ë³?ê¶Œì¥?¬í•­
            if method_performance.get('rule_based', 0) > method_performance.get('ml_enhanced', 0):
                recommendations.append("ê·œì¹™ ê¸°ë°˜ ?Œì„œ???±ëŠ¥??ML ê°•í™” ?Œì„œë³´ë‹¤ ??Šµ?ˆë‹¤. ML ëª¨ë¸ ?¬í›ˆ?¨ì„ ê³ ë ¤?˜ì„¸??")
            
            # ?´ìš© ?¹ì„±ë³?ê¶Œì¥?¬í•­
            if content_characteristics.get('short_articles', 0) > 10:
                recommendations.append("ì§§ì? ì¡°ë¬¸?¤ì˜ ?Œì‹± ?ˆì§ˆ????Šµ?ˆë‹¤. ìµœì†Œ ê¸¸ì´ ?„ê³„ê°’ì„ ì¡°ì •?˜ì„¸??")
            
            if content_characteristics.get('long_articles', 0) > 5:
                recommendations.append("ê¸?ì¡°ë¬¸?¤ì˜ ?Œì‹± ?ˆì§ˆ????Šµ?ˆë‹¤. ì²?¬ ë¶„í•  ë¡œì§??ê°œì„ ?˜ì„¸??")
            
            # ?¼ë°˜?ì¸ ê¶Œì¥?¬í•­
            recommendations.extend([
                "?•ê¸°?ì¸ ?ˆì§ˆ ëª¨ë‹ˆ?°ë§???µí•´ ?Œì‹± ?±ëŠ¥??ì¶”ì ?˜ì„¸??",
                "ML ëª¨ë¸??? ë¢°???ìˆ˜ë¥??œìš©?˜ì—¬ ?ˆì§ˆ ?„ê³„ê°’ì„ ì¡°ì •?˜ì„¸??",
                "ë¶€ì¹??Œì‹± ë¡œì§??ë³„ë„ë¡?ìµœì ?”í•˜?¸ìš”."
            ])
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return []
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """ì¢…í•© ?ˆì§ˆ ë³´ê³ ???ì„±"""
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'overall_stats': self.get_overall_quality_stats(),
                'law_stats': self.get_law_quality_stats(limit=10),
                'parsing_method_comparison': self.get_parsing_method_comparison(),
                'quality_trends': self.get_quality_trends(days=30),
                'error_analysis': self.get_error_analysis()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating quality report: {e}")
            return {'error': str(e)}
    
    def save_quality_report(self, report: Dict[str, Any], output_path: str):
        """?ˆì§ˆ ë³´ê³ ???€??""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Quality report saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving quality report: {e}")


def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    import argparse
    
    parser = argparse.ArgumentParser(description="?ˆì§ˆ ëª¨ë‹ˆ?°ë§ ?œìŠ¤??)
    parser.add_argument("--config", default="config.json", help="?¤ì • ?Œì¼ ê²½ë¡œ")
    parser.add_argument("--output", default="reports/quality_report.json", help="ë³´ê³ ??ì¶œë ¥ ê²½ë¡œ")
    parser.add_argument("--log-level", default="INFO", help="ë¡œê·¸ ?ˆë²¨")
    
    args = parser.parse_args()
    
    # ë¡œê¹… ?¤ì •
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # ?¤ì • ë¡œë“œ
    config = Config()
    
    # ?ˆì§ˆ ëª¨ë‹ˆ??ì´ˆê¸°??
    monitor = QualityMonitor(config)
    
    # ?ˆì§ˆ ë³´ê³ ???ì„±
    report = monitor.generate_quality_report()
    
    # ë³´ê³ ???€??
    monitor.save_quality_report(report, args.output)
    
    # ?”ì•½ ì¶œë ¥
    overall_stats = report.get('overall_stats', {})
    print(f"\n=== ?ˆì§ˆ ëª¨ë‹ˆ?°ë§ ë³´ê³ ??===")
    print(f"ì´?ì¡°ë¬¸ ?? {overall_stats.get('total_articles', 0):,}")
    print(f"ML ê°•í™” ì¡°ë¬¸ ?? {overall_stats.get('ml_enhanced_articles', 0):,}")
    print(f"?‰ê·  ?ˆì§ˆ ?ìˆ˜: {overall_stats.get('avg_quality_score', 0.0):.3f}")
    print(f"ML ê°•í™” ë¹„ìœ¨: {overall_stats.get('ml_enhancement_rate', 0.0):.1f}%")
    print(f"ë³¸ì¹™ ì¡°ë¬¸: {overall_stats.get('main_articles', 0):,}")
    print(f"ë¶€ì¹?ì¡°ë¬¸: {overall_stats.get('supplementary_articles', 0):,}")
    
    quality_dist = overall_stats.get('quality_distribution', {})
    print(f"\n?ˆì§ˆ ?±ê¸‰ë³?ë¶„í¬:")
    for grade, count in quality_dist.items():
        print(f"  {grade}: {count:,}")


if __name__ == "__main__":
    main()
