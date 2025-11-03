#!/usr/bin/env python3
"""
ë²•ë¥  ?©ì–´ ?ˆì§ˆ ê²€ì¦??œìŠ¤??
ì¶”ì¶œ???©ì–´?¤ì˜ ?ˆì§ˆ??ê²€ì¦í•˜ê³?ê°œì„  ?œì•ˆ???œê³µ?©ë‹ˆ??
"""

import json
import os
import re
import logging
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime

# ë¡œê¹… ?¤ì •
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/legal_term_quality_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """?ˆì§ˆ ë©”íŠ¸ë¦??°ì´???´ë˜??""
    term: str
    frequency_score: float
    diversity_score: float
    legal_relevance_score: float
    domain_coherence_score: float
    overall_score: float
    issues: List[str]
    recommendations: List[str]

class LegalTermQualityValidator:
    """ë²•ë¥  ?©ì–´ ?ˆì§ˆ ê²€ì¦ê¸°"""
    
    def __init__(self):
        self.extracted_terms_file = "data/extracted_terms/extracted_legal_terms.json"
        self.semantic_relations_file = "data/extracted_terms/semantic_relations.json"
        self.output_dir = "data/extracted_terms/quality_validation"
        
        # ?ˆì§ˆ ê²€ì¦?ê¸°ì?
        self.quality_thresholds = {
            "excellent": 0.8,
            "good": 0.6,
            "fair": 0.4,
            "poor": 0.2
        }
        
        # ë²•ë¥  ?©ì–´ ?¨í„´
        self.legal_patterns = self._initialize_legal_patterns()
        
        # ë¶ˆìš©??ë°??€?ˆì§ˆ ?©ì–´
        self.stop_words = self._initialize_stop_words()
        self.low_quality_patterns = self._initialize_low_quality_patterns()
    
    def _initialize_legal_patterns(self) -> Dict[str, List[str]]:
        """ë²•ë¥  ?©ì–´ ?¨í„´ ì´ˆê¸°??""
        return {
            "high_quality": [
                r"[ê°€-??+ë²?",  # ë²•ë¥ ëª?
                r"[ê°€-??+ê¶?",  # ê¶Œë¦¬
                r"[ê°€-??+?˜ë¬´$",  # ?˜ë¬´
                r"[ê°€-??+?ˆì°¨$",  # ?ˆì°¨
                r"[ê°€-??+? ì²­$",  # ? ì²­
                r"[ê°€-??+? ê³ $",  # ? ê³ 
                r"[ê°€-??+?ˆê?$",  # ?ˆê?
                r"[ê°€-??+?¸ê?$",  # ?¸ê?
                r"[ê°€-??+??",  # ê¸°ê?
                r"[ê°€-??+ì²?",  # ê¸°ê?
                r"[ê°€-??+ë¶€$",  # ê¸°ê?
                r"[ê°€-??+?„ì›??",  # ê¸°ê?
                r"[ê°€-??+ë²•ì›$",  # ë²•ì›
                r"[ê°€-??+?Œì†¡$",  # ?Œì†¡
                r"[ê°€-??+?¬íŒ$",  # ?¬íŒ
                r"[ê°€-??+?ê²°$",  # ?ê²°
                r"[ê°€-??+ì²˜ë¶„$",  # ì²˜ë¶„
                r"[ê°€-??+ê²°ì •$",  # ê²°ì •
                r"[ê°€-??+ëª…ë ¹$",  # ëª…ë ¹
                r"[ê°€-??+ì§€??"  # ì§€??
            ],
            "medium_quality": [
                r"[ê°€-??{2,4}$",  # ?¼ë°˜?ì¸ 2-4ê¸€???©ì–´
                r"??d+ì¡?",  # ì¡°ë¬¸
                r"??d+??",  # ??
                r"??d+??",  # ??
                r"??d+??",  # ??
                r"??d+??",  # ??
                r"??d+??"  # ??
            ],
            "low_quality": [
                r"^\d+$",  # ?«ìë§?
                r"^[ê°€-??{1}$",  # ??ê¸€??
                r"^[a-zA-Z]+$",  # ?ë¬¸ë§?
                r"^[ê°€-??*[0-9]+[ê°€-??*$",  # ?«ì ?¬í•¨
                r"^[ê°€-??*[a-zA-Z]+[ê°€-??*$"  # ?ë¬¸ ?¬í•¨
            ]
        }
    
    def _initialize_stop_words(self) -> Set[str]:
        """ë¶ˆìš©??ì´ˆê¸°??""
        return {
            "ê²?, "??, "??, "ë°?, "?ëŠ”", "ê·?, "??, "?€", "??, "ê°€", "??, "ë¥?,
            "??, "?ì„œ", "ë¡?, "?¼ë¡œ", "?€", "ê³?, "??, "?€", "??, "ë§?, "ë¶€??,
            "ê¹Œì?", "ê¹Œì???, "?ì˜", "?ë???, "?ê???, "?ë”°ë¥?, "?ì˜??,
            "??, "??, "??, "??, "??, "??, "??, "??, "??, "??, "??,
            "??, "??, "??, "??, "??, "??, "??, "??, "??, "??, "??
        }
    
    def _initialize_low_quality_patterns(self) -> List[str]:
        """?€?ˆì§ˆ ?¨í„´ ì´ˆê¸°??""
        return [
            r"^[0-9]+$",  # ?«ìë§?
            r"^[a-zA-Z]+$",  # ?ë¬¸ë§?
            r"^[ê°€-??{1}$",  # ??ê¸€??
            r"^[ê°€-??*[0-9]+[ê°€-??*$",  # ?«ì ?¬í•¨
            r"^[ê°€-??*[a-zA-Z]+[ê°€-??*$",  # ?ë¬¸ ?¬í•¨
            r"^[ê°€-??*[!@#$%^&*()]+[ê°€-??*$"  # ?¹ìˆ˜ë¬¸ì ?¬í•¨
        ]
    
    def load_extracted_terms(self) -> Dict[str, Any]:
        """ì¶”ì¶œ???©ì–´ ë¡œë“œ"""
        logger.info("ì¶”ì¶œ???©ì–´ ë¡œë“œ ì¤?..")
        
        with open(self.extracted_terms_file, 'r', encoding='utf-8') as f:
            extracted_terms = json.load(f)
        
        logger.info(f"ë¡œë“œ???©ì–´ ?? {len(extracted_terms)}")
        return extracted_terms
    
    def validate_term_quality(self, term: str, term_data: Dict[str, Any]) -> QualityMetrics:
        """ê°œë³„ ?©ì–´ ?ˆì§ˆ ê²€ì¦?""
        issues = []
        recommendations = []
        
        # 1. ë¹ˆë„???ìˆ˜ (0-0.3)
        frequency_score = self._calculate_frequency_score(term_data.get('frequency', 0))
        
        # 2. ?¤ì–‘???ìˆ˜ (0-0.2)
        diversity_score = self._calculate_diversity_score(term_data)
        
        # 3. ë²•ë¥  ê´€?¨ì„± ?ìˆ˜ (0-0.3)
        legal_relevance_score = self._calculate_legal_relevance_score(term)
        
        # 4. ?„ë©”???¼ê????ìˆ˜ (0-0.2)
        domain_coherence_score = self._calculate_domain_coherence_score(term, term_data)
        
        # ?„ì²´ ?ìˆ˜ ê³„ì‚°
        overall_score = frequency_score + diversity_score + legal_relevance_score + domain_coherence_score
        
        # ë¬¸ì œ???ë³„
        if frequency_score < 0.1:
            issues.append("ë¹ˆë„?˜ê? ?ˆë¬´ ??Œ")
            recommendations.append("??ë§ì? ?°ì´?°ì—???¬ìš©?˜ëŠ” ?©ì–´?¸ì? ?•ì¸")
        
        if diversity_score < 0.1:
            issues.append("?ŒìŠ¤ ?¤ì–‘?±ì´ ë¶€ì¡±í•¨")
            recommendations.append("?¤ì–‘??ë²•ë ¹/?ë??ì„œ ?¬ìš©?˜ëŠ” ?©ì–´?¸ì? ?•ì¸")
        
        if legal_relevance_score < 0.1:
            issues.append("ë²•ë¥  ê´€?¨ì„±????Œ")
            recommendations.append("ë²•ë¥  ?©ì–´ë¡œì„œ???ì ˆ??ê²€??)
        
        if domain_coherence_score < 0.1:
            issues.append("?„ë©”???¼ê??±ì´ ë¶€ì¡±í•¨")
            recommendations.append("?„ë©”??ë¶„ë¥˜???•í™•??ê²€??)
        
        # ë¶ˆìš©??ê²€??
        if term in self.stop_words:
            issues.append("ë¶ˆìš©?´ë¡œ ë¶„ë¥˜??)
            recommendations.append("ë¶ˆìš©??ëª©ë¡?ì„œ ?œê±° ê³ ë ¤")
        
        # ?€?ˆì§ˆ ?¨í„´ ê²€??
        for pattern in self.low_quality_patterns:
            if re.match(pattern, term):
                issues.append("?€?ˆì§ˆ ?¨í„´???´ë‹¹")
                recommendations.append("?©ì–´ ?•ì‹ ê°œì„  ?„ìš”")
                break
        
        return QualityMetrics(
            term=term,
            frequency_score=frequency_score,
            diversity_score=diversity_score,
            legal_relevance_score=legal_relevance_score,
            domain_coherence_score=domain_coherence_score,
            overall_score=overall_score,
            issues=issues,
            recommendations=recommendations
        )
    
    def _calculate_frequency_score(self, frequency: int) -> float:
        """ë¹ˆë„???ìˆ˜ ê³„ì‚°"""
        if frequency >= 50:
            return 0.3
        elif frequency >= 20:
            return 0.25
        elif frequency >= 10:
            return 0.2
        elif frequency >= 5:
            return 0.15
        elif frequency >= 2:
            return 0.1
        else:
            return 0.05
    
    def _calculate_diversity_score(self, term_data: Dict[str, Any]) -> float:
        """?¤ì–‘???ìˆ˜ ê³„ì‚°"""
        sources = term_data.get('sources', [])
        contexts = term_data.get('context', [])
        
        # ?ŒìŠ¤ ?¤ì–‘??(0-0.1)
        unique_sources = len(set(sources))
        source_score = min(unique_sources / 10.0, 0.1)
        
        # ì»¨í…?¤íŠ¸ ?¤ì–‘??(0-0.1)
        context_score = min(len(contexts) / 5.0, 0.1)
        
        return source_score + context_score
    
    def _calculate_legal_relevance_score(self, term: str) -> float:
        """ë²•ë¥  ê´€?¨ì„± ?ìˆ˜ ê³„ì‚°"""
        # ê³ í’ˆì§??¨í„´ ê²€??
        for pattern in self.legal_patterns["high_quality"]:
            if re.match(pattern, term):
                return 0.3
        
        # ì¤‘í’ˆì§??¨í„´ ê²€??
        for pattern in self.legal_patterns["medium_quality"]:
            if re.match(pattern, term):
                return 0.2
        
        # ?€?ˆì§ˆ ?¨í„´ ê²€??
        for pattern in self.legal_patterns["low_quality"]:
            if re.match(pattern, term):
                return 0.05
        
        # ?¼ë°˜?ì¸ ë²•ë¥  ?©ì–´ ì§€??
        legal_indicators = [
            'ë²?, 'ê·œì¹™', '??, 'ê¶?, '?˜ë¬´', 'ì±…ì„', '?ˆì°¨', '? ì²­', '? ê³ ',
            '?ˆê?', '?¸ê?', '?¹ì¸', '??, 'ì²?, 'ë¶€', '?„ì›??, 'ë²•ì›',
            '?‰ìœ„', 'ì²˜ë¶„', 'ê²°ì •', 'ëª…ë ¹', 'ì§€??, '?Œì†¡', '?¬íŒ', '?ê²°'
        ]
        
        if any(indicator in term for indicator in legal_indicators):
            return 0.15
        
        return 0.1
    
    def _calculate_domain_coherence_score(self, term: str, term_data: Dict[str, Any]) -> float:
        """?„ë©”???¼ê????ìˆ˜ ê³„ì‚°"""
        domain = term_data.get('domain', 'ê¸°í?')
        category = term_data.get('category', '?¼ë°˜')
        
        # ?„ë©”?¸ë³„ ?¼ê???ê²€??
        domain_keywords = {
            "?•ì‚¬ë²?: ["ë²”ì£„", "ì²˜ë²Œ", "?•ë²Œ", "êµ¬ì†", "ê¸°ì†Œ", "ê³µì†Œ", "?¼ê³ ", "ê²€??],
            "ë¯¼ì‚¬ë²?: ["ê³„ì•½", "?í•´ë°°ìƒ", "?Œìœ ê¶?, "ì±„ê¶Œ", "ì±„ë¬´", "?´í–‰", "?„ë°˜"],
            "ê°€ì¡±ë²•": ["?¼ì¸", "?´í˜¼", "?ì†", "?‘ìœ¡", "?„ìë£?, "?¬ì‚°ë¶„í• ", "?‘ìœ¡ê¶?],
            "?ì‚¬ë²?: ["?Œì‚¬", "ì£¼ì‹", "?´ìŒ", "?˜í‘œ", "?í–‰??, "?Œì‚¬ë²?, "?ë²•"],
            "?¸ë™ë²?: ["ê·¼ë¡œ", "ê·¼ë¡œ??, "ê·¼ë¡œê³„ì•½", "?„ê¸ˆ", "ê·¼ë¡œ?œê°„", "?´ê³ "],
            "ë¶€?™ì‚°ë²?: ["ë¶€?™ì‚°", "? ì?", "ê±´ë¬¼", "?±ê¸°", "?Œìœ ê¶Œì´??, "ë§¤ë§¤"],
            "?¹í—ˆë²?: ["?¹í—ˆ", "?¹í—ˆê¶?, "?¹í—ˆì¶œì›", "?¹í—ˆ?±ë¡", "?¹í—ˆì¹¨í•´"],
            "?‰ì •ë²?: ["?‰ì •ì²˜ë¶„", "?‰ì •?Œì†¡", "?‰ì •ë²?, "?ˆê?", "?¸ê?", "?¹ì¸"]
        }
        
        if domain in domain_keywords:
            for keyword in domain_keywords[domain]:
                if keyword in term:
                    return 0.2
        
        # ì¹´í…Œê³ ë¦¬ë³??¼ê???ê²€??
        category_keywords = {
            "ë²•ë¥ ëª?: ["ë²?, "ê·œì¹™", "??],
            "ê¶Œë¦¬": ["ê¶?],
            "?˜ë¬´": ["?˜ë¬´", "ì±…ì„"],
            "?ˆì°¨": ["?ˆì°¨", "? ì²­", "? ê³ "],
            "ê¸°ê?": ["??, "ì²?, "ë¶€", "?„ì›??, "ë²•ì›"],
            "?Œì†¡": ["?Œì†¡", "?¬íŒ", "?ê²°"]
        }
        
        if category in category_keywords:
            for keyword in category_keywords[category]:
                if keyword in term:
                    return 0.15
        
        return 0.1
    
    def validate_all_terms(self, extracted_terms: Dict[str, Any]) -> Dict[str, QualityMetrics]:
        """ëª¨ë“  ?©ì–´ ?ˆì§ˆ ê²€ì¦?""
        logger.info("ëª¨ë“  ?©ì–´ ?ˆì§ˆ ê²€ì¦??œì‘")
        
        quality_metrics = {}
        total_terms = len(extracted_terms)
        
        for i, (term, term_data) in enumerate(extracted_terms.items()):
            if i % 1000 == 0:
                logger.info(f"ì§„í–‰ë¥? {i}/{total_terms} ({i/total_terms*100:.1f}%)")
            
            quality_metrics[term] = self.validate_term_quality(term, term_data)
        
        logger.info("ëª¨ë“  ?©ì–´ ?ˆì§ˆ ê²€ì¦??„ë£Œ")
        return quality_metrics
    
    def generate_quality_report(self, quality_metrics: Dict[str, QualityMetrics]) -> Dict[str, Any]:
        """?ˆì§ˆ ë³´ê³ ???ì„±"""
        logger.info("?ˆì§ˆ ë³´ê³ ???ì„± ì¤?)
        
        total_terms = len(quality_metrics)
        
        # ?ˆì§ˆ ?±ê¸‰ë³?ë¶„í¬
        quality_distribution = {
            "excellent": 0,
            "good": 0,
            "fair": 0,
            "poor": 0
        }
        
        # ë¬¸ì œ?ë³„ ?µê³„
        issue_stats = defaultdict(int)
        
        # ?„ë©”?¸ë³„ ?ˆì§ˆ ?µê³„
        domain_quality = defaultdict(list)
        
        for term, metrics in quality_metrics.items():
            # ?ˆì§ˆ ?±ê¸‰ ë¶„ë¥˜
            if metrics.overall_score >= self.quality_thresholds["excellent"]:
                quality_distribution["excellent"] += 1
            elif metrics.overall_score >= self.quality_thresholds["good"]:
                quality_distribution["good"] += 1
            elif metrics.overall_score >= self.quality_thresholds["fair"]:
                quality_distribution["fair"] += 1
            else:
                quality_distribution["poor"] += 1
            
            # ë¬¸ì œ???µê³„
            for issue in metrics.issues:
                issue_stats[issue] += 1
        
        # ?ìœ„/?˜ìœ„ ?ˆì§ˆ ?©ì–´
        sorted_terms = sorted(quality_metrics.items(), key=lambda x: x[1].overall_score, reverse=True)
        top_quality_terms = [term for term, metrics in sorted_terms[:20]]
        bottom_quality_terms = [term for term, metrics in sorted_terms[-20:]]
        
        # ?ˆì§ˆ ê°œì„  ê¶Œì¥?¬í•­
        improvement_recommendations = self._generate_improvement_recommendations(quality_metrics)
        
        report = {
            "summary": {
                "total_terms": total_terms,
                "validation_date": datetime.now().isoformat(),
                "average_quality_score": sum(m.overall_score for m in quality_metrics.values()) / total_terms
            },
            "quality_distribution": quality_distribution,
            "issue_statistics": dict(issue_stats),
            "top_quality_terms": top_quality_terms,
            "bottom_quality_terms": bottom_quality_terms,
            "improvement_recommendations": improvement_recommendations
        }
        
        logger.info("?ˆì§ˆ ë³´ê³ ???ì„± ?„ë£Œ")
        return report
    
    def _generate_improvement_recommendations(self, quality_metrics: Dict[str, QualityMetrics]) -> List[str]:
        """?ˆì§ˆ ê°œì„  ê¶Œì¥?¬í•­ ?ì„±"""
        recommendations = []
        
        # ë¹ˆë„??ë¬¸ì œ
        low_frequency_count = sum(1 for m in quality_metrics.values() if m.frequency_score < 0.1)
        if low_frequency_count > 0:
            recommendations.append(f"ë¹ˆë„?˜ê? ??? ?©ì–´ {low_frequency_count}ê°??œê±° ê³ ë ¤")
        
        # ?¤ì–‘??ë¬¸ì œ
        low_diversity_count = sum(1 for m in quality_metrics.values() if m.diversity_score < 0.1)
        if low_diversity_count > 0:
            recommendations.append(f"?ŒìŠ¤ ?¤ì–‘?±ì´ ë¶€ì¡±í•œ ?©ì–´ {low_diversity_count}ê°?ê²€???„ìš”")
        
        # ë²•ë¥  ê´€?¨ì„± ë¬¸ì œ
        low_relevance_count = sum(1 for m in quality_metrics.values() if m.legal_relevance_score < 0.1)
        if low_relevance_count > 0:
            recommendations.append(f"ë²•ë¥  ê´€?¨ì„±????? ?©ì–´ {low_relevance_count}ê°??¬ê????„ìš”")
        
        # ?„ë©”???¼ê???ë¬¸ì œ
        low_coherence_count = sum(1 for m in quality_metrics.values() if m.domain_coherence_score < 0.1)
        if low_coherence_count > 0:
            recommendations.append(f"?„ë©”???¼ê??±ì´ ë¶€ì¡±í•œ ?©ì–´ {low_coherence_count}ê°?ë¶„ë¥˜ ?¬ê????„ìš”")
        
        return recommendations
    
    def filter_high_quality_terms(self, quality_metrics: Dict[str, QualityMetrics], threshold: float = 0.6) -> Dict[str, QualityMetrics]:
        """ê³ í’ˆì§??©ì–´ ?„í„°ë§?""
        logger.info(f"ê³ í’ˆì§??©ì–´ ?„í„°ë§?(?„ê³„ê°? {threshold})")
        
        filtered_metrics = {
            term: metrics for term, metrics in quality_metrics.items()
            if metrics.overall_score >= threshold
        }
        
        logger.info(f"?„í„°ë§?ê²°ê³¼: {len(filtered_metrics)}/{len(quality_metrics)} ?©ì–´ ? ì?")
        return filtered_metrics
    
    def save_quality_validation_results(self, quality_metrics: Dict[str, QualityMetrics], quality_report: Dict[str, Any]):
        """?ˆì§ˆ ê²€ì¦?ê²°ê³¼ ?€??""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # ?ˆì§ˆ ë©”íŠ¸ë¦??€??
        metrics_data = {}
        for term, metrics in quality_metrics.items():
            metrics_data[term] = {
                "term": metrics.term,
                "frequency_score": metrics.frequency_score,
                "diversity_score": metrics.diversity_score,
                "legal_relevance_score": metrics.legal_relevance_score,
                "domain_coherence_score": metrics.domain_coherence_score,
                "overall_score": metrics.overall_score,
                "issues": metrics.issues,
                "recommendations": metrics.recommendations
            }
        
        metrics_file = os.path.join(self.output_dir, "quality_metrics.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=2)
        
        # ?ˆì§ˆ ë³´ê³ ???€??
        report_file = os.path.join(self.output_dir, "quality_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"?ˆì§ˆ ê²€ì¦?ê²°ê³¼ ?€???„ë£Œ: {self.output_dir}")
    
    def run_quality_validation(self):
        """?ˆì§ˆ ê²€ì¦??¤í–‰"""
        logger.info("ë²•ë¥  ?©ì–´ ?ˆì§ˆ ê²€ì¦??œì‘")
        
        try:
            # ì¶”ì¶œ???©ì–´ ë¡œë“œ
            extracted_terms = self.load_extracted_terms()
            
            # ?ˆì§ˆ ê²€ì¦?
            quality_metrics = self.validate_all_terms(extracted_terms)
            
            # ?ˆì§ˆ ë³´ê³ ???ì„±
            quality_report = self.generate_quality_report(quality_metrics)
            
            # ê³ í’ˆì§??©ì–´ ?„í„°ë§?
            high_quality_metrics = self.filter_high_quality_terms(quality_metrics, 0.6)
            
            # ê²°ê³¼ ?€??
            self.save_quality_validation_results(quality_metrics, quality_report)
            
            # ê³ í’ˆì§??©ì–´ë§?ë³„ë„ ?€??
            high_quality_file = os.path.join(self.output_dir, "high_quality_terms.json")
            with open(high_quality_file, 'w', encoding='utf-8') as f:
                json.dump(list(high_quality_metrics.keys()), f, ensure_ascii=False, indent=2)
            
            logger.info("ë²•ë¥  ?©ì–´ ?ˆì§ˆ ê²€ì¦??„ë£Œ")
            
            # ê²°ê³¼ ?”ì•½ ì¶œë ¥
            print(f"\n=== ?ˆì§ˆ ê²€ì¦?ê²°ê³¼ ?”ì•½ ===")
            print(f"?„ì²´ ?©ì–´ ?? {len(quality_metrics)}")
            print(f"ê³ í’ˆì§??©ì–´ ?? {len(high_quality_metrics)}")
            print(f"?‰ê·  ?ˆì§ˆ ?ìˆ˜: {quality_report['summary']['average_quality_score']:.3f}")
            print(f"?ˆì§ˆ ?±ê¸‰ ë¶„í¬: {quality_report['quality_distribution']}")
            
        except Exception as e:
            logger.error(f"?ˆì§ˆ ê²€ì¦?ì¤??¤ë¥˜ ë°œìƒ: {e}")
            raise

def main():
    """ë©”ì¸ ?¤í–‰ ?¨ìˆ˜"""
    validator = LegalTermQualityValidator()
    validator.run_quality_validation()

if __name__ == "__main__":
    main()
