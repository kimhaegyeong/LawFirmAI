#!/usr/bin/env python3
"""
?¥ìƒ??ë²•ë¥  ì§ˆì˜ ?ŒìŠ¤???œìŠ¤??
?¤ì œ ?¬ìš©??ì§ˆë¬¸???€???¤ì›Œ??ë§¤í•‘ ?±ëŠ¥?????•í™•?˜ê²Œ ì¸¡ì •?©ë‹ˆ??
"""

import sys
import os
import json
import time
import logging
from typing import Dict, List, Any
from datetime import datetime

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from source.agents.keyword_mapper import EnhancedKeywordMapper, LegalKeywordMapper

# ë¡œê¹… ?¤ì •
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_query_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedQueryTester:
    """?¥ìƒ??ì§ˆì˜ ?ŒìŠ¤?¸ê¸°"""

    def __init__(self):
        self.test_queries = self._initialize_realistic_queries()
        self.output_dir = "data/extracted_terms/enhanced_query_test"

    def _initialize_realistic_queries(self) -> List[Dict[str, Any]]:
        """?„ì‹¤?ì¸ ?ŒìŠ¤??ì§ˆì˜ ì´ˆê¸°??""
        return [
            {
                "question": "ê³„ì•½?œì—???„ì•½ê¸?ì¡°í•­???ˆë¬´ ?’ê²Œ ?¤ì •?˜ì–´ ?ˆëŠ”?? ë²•ì ?¼ë¡œ ë¬¸ì œê°€ ? ê¹Œ??",
                "query_type": "contract_review",
                "domain": "ë¯¼ì‚¬ë²?,
                "legal_keywords": ["ê³„ì•½??, "?„ì•½ê¸?, "ì¡°í•­", "ë¯¼ë²•", "ê³„ì•½ë²?, "?í•´ë°°ìƒ", "?„ì•½ê¸ˆì œ??],
                "context_keywords": ["ë²•ì ", "ë¬¸ì œ", "?¤ì •", "?’ê²Œ", "? íš¨??],
                "expected_terms": ["ë¯¼ë²•", "398ì¡?, "?„ì•½ê¸?, "?í•´ë°°ìƒ", "ê³„ì•½??, "ì¡°í•­"]
            },
            {
                "question": "êµí†µ?¬ê³ ë¡??¸í•œ ?í•´ë°°ìƒ ì²?µ¬ ???„ìš”??ì¦ê±°?ë£Œ??ë¬´ì—‡?¸ê???",
                "query_type": "damage_compensation",
                "domain": "ë¯¼ì‚¬ë²?,
                "legal_keywords": ["êµí†µ?¬ê³ ", "?í•´ë°°ìƒ", "ì²?µ¬", "ì¦ê±°?ë£Œ", "ë¶ˆë²•?‰ìœ„", "ê³¼ì‹¤", "?¸ê³¼ê´€ê³?],
                "context_keywords": ["?„ìš”??, "??, "?¸í•œ", "ë¡?, "ë¬´ì—‡?¸ê???],
                "expected_terms": ["ë¯¼ë²•", "750ì¡?, "ë¶ˆë²•?‰ìœ„", "?í•´ë°°ìƒ", "êµí†µ?¬ê³ ", "ì¦ê±°"]
            },
            {
                "question": "?´í˜¼ ?Œì†¡?ì„œ ?ë? ?‘ìœ¡ê¶Œì„ ê²°ì •?˜ëŠ” ê¸°ì??€ ë¬´ì—‡?¸ê???",
                "query_type": "divorce_proceedings",
                "domain": "ê°€ì¡±ë²•",
                "legal_keywords": ["?´í˜¼", "?Œì†¡", "?ë?", "?‘ìœ¡ê¶?, "ê²°ì •", "ê¸°ì?", "ê°€?•ë²•??, "ê°€ì¡±ë²•"],
                "context_keywords": ["?ì„œ", "??, "?˜ëŠ”", "??, "ë¬´ì—‡?¸ê???],
                "expected_terms": ["ê°€ì¡±ë²•", "?‘ìœ¡ê¶?, "?´í˜¼", "?ë?", "ê°€?•ë²•??, "ê¸°ì?"]
            },
            {
                "question": "ë¶€?™ì‚° ë§¤ë§¤ ê³„ì•½ ???±ê¸° ?´ì „ ?ˆì°¨?€ ?„ìš”???œë¥˜??ë¬´ì—‡?¸ê???",
                "query_type": "real_estate_transaction",
                "domain": "ë¶€?™ì‚°ë²?,
                "legal_keywords": ["ë¶€?™ì‚°", "ë§¤ë§¤", "ê³„ì•½", "?±ê¸°", "?´ì „", "?ˆì°¨", "?œë¥˜", "?±ê¸°ë¶€?±ë³¸"],
                "context_keywords": ["??, "?€", "?„ìš”??, "ë¬´ì—‡?¸ê???],
                "expected_terms": ["ë¶€?™ì‚°?±ê¸°ë²?, "?±ê¸°", "?Œìœ ê¶Œì´??, "ë§¤ë§¤ê³„ì•½", "?±ê¸°ë¶€?±ë³¸"]
            },
            {
                "question": "?¹í—ˆ ì¶œì› ??ë°œëª…??? ê·œ?±ê³¼ ì§„ë³´?±ì„ ?´ë–»ê²??…ì¦?´ì•¼ ?˜ë‚˜??",
                "query_type": "patent_application",
                "domain": "?¹í—ˆë²?,
                "legal_keywords": ["?¹í—ˆ", "ì¶œì›", "ë°œëª…", "? ê·œ??, "ì§„ë³´??, "?…ì¦", "?¹í—ˆì²?, "?¹í—ˆë²?],
                "context_keywords": ["??, "??, "ê³?, "ë¥?, "?´ë–»ê²?, "?´ì•¼", "?˜ë‚˜??],
                "expected_terms": ["?¹í—ˆë²?, "? ê·œ??, "ì§„ë³´??, "?¹í—ˆì¶œì›", "ë°œëª…", "?¹í—ˆì²?]
            }
        ]

    def test_keyword_extraction_accuracy(self) -> Dict[str, Any]:
        """?¤ì›Œ??ì¶”ì¶œ ?•í™•???ŒìŠ¤??""
        logger.info("?¤ì›Œ??ì¶”ì¶œ ?•í™•???ŒìŠ¤???œì‘")

        enhanced_mapper = EnhancedKeywordMapper()
        results = {}

        for i, query in enumerate(self.test_queries):
            question = query["question"]
            query_type = query["query_type"]
            legal_keywords = query["legal_keywords"]
            context_keywords = query["context_keywords"]
            expected_terms = query["expected_terms"]

            logger.info(f"?ŒìŠ¤??{i+1}/{len(self.test_queries)}: {query_type}")

            start_time = time.time()

            # ì¢…í•©?ì¸ ?¤ì›Œ??ë§¤í•‘
            comprehensive_result = enhanced_mapper.get_comprehensive_keyword_mapping(question, query_type)

            end_time = time.time()

            # ì¶”ì¶œ???¤ì›Œ?œë“¤
            all_keywords = comprehensive_result.get("all_keywords", [])
            base_keywords = comprehensive_result.get("base_keywords", [])
            contextual_keywords = comprehensive_result.get("contextual_data", {}).get("all_keywords", [])
            semantic_keywords = comprehensive_result.get("semantic_data", {}).get("recommended_keywords", [])

            # ?•í™•??ê³„ì‚°
            legal_match = [kw for kw in legal_keywords if kw in all_keywords]
            context_match = [kw for kw in context_keywords if kw in all_keywords]
            expected_match = [kw for kw in expected_terms if kw in all_keywords]

            legal_accuracy = len(legal_match) / len(legal_keywords) if legal_keywords else 0
            context_accuracy = len(context_match) / len(context_keywords) if context_keywords else 0
            expected_accuracy = len(expected_match) / len(expected_terms) if expected_terms else 0

            results[query_type] = {
                "question": question,
                "legal_keywords": legal_keywords,
                "context_keywords": context_keywords,
                "expected_terms": expected_terms,
                "extracted_keywords": {
                    "all_keywords": all_keywords,
                    "base_keywords": base_keywords,
                    "contextual_keywords": contextual_keywords,
                    "semantic_keywords": semantic_keywords
                },
                "accuracy_metrics": {
                    "legal_accuracy": legal_accuracy,
                    "context_accuracy": context_accuracy,
                    "expected_accuracy": expected_accuracy,
                    "overall_accuracy": (legal_accuracy + context_accuracy + expected_accuracy) / 3
                },
                "matched_keywords": {
                    "legal_matched": legal_match,
                    "context_matched": context_match,
                    "expected_matched": expected_match
                },
                "processing_time": end_time - start_time,
                "comprehensive_result": comprehensive_result
            }

        logger.info("?¤ì›Œ??ì¶”ì¶œ ?•í™•???ŒìŠ¤???„ë£Œ")
        return results

    def test_keyword_coverage_analysis(self, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
        """?¤ì›Œ??ì»¤ë²„ë¦¬ì? ë¶„ì„"""
        logger.info("?¤ì›Œ??ì»¤ë²„ë¦¬ì? ë¶„ì„ ?œì‘")

        coverage_analysis = {
            "domain_coverage": {},
            "keyword_type_coverage": {},
            "missing_keywords": {},
            "recommendations": []
        }

        # ?„ë©”?¸ë³„ ì»¤ë²„ë¦¬ì?
        domain_stats = {}
        for query_type, result in extraction_results.items():
            domain = next((q["domain"] for q in self.test_queries if q["query_type"] == query_type), "ê¸°í?")

            if domain not in domain_stats:
                domain_stats[domain] = {
                    "total_legal_keywords": 0,
                    "matched_legal_keywords": 0,
                    "total_expected_terms": 0,
                    "matched_expected_terms": 0,
                    "queries": 0
                }

            domain_stats[domain]["total_legal_keywords"] += len(result["legal_keywords"])
            domain_stats[domain]["matched_legal_keywords"] += len(result["matched_keywords"]["legal_matched"])
            domain_stats[domain]["total_expected_terms"] += len(result["expected_terms"])
            domain_stats[domain]["matched_expected_terms"] += len(result["matched_keywords"]["expected_matched"])
            domain_stats[domain]["queries"] += 1

        # ?„ë©”?¸ë³„ ì»¤ë²„ë¦¬ì? ê³„ì‚°
        for domain, stats in domain_stats.items():
            legal_coverage = stats["matched_legal_keywords"] / stats["total_legal_keywords"] if stats["total_legal_keywords"] > 0 else 0
            expected_coverage = stats["matched_expected_terms"] / stats["total_expected_terms"] if stats["total_expected_terms"] > 0 else 0

            coverage_analysis["domain_coverage"][domain] = {
                "legal_coverage": legal_coverage,
                "expected_coverage": expected_coverage,
                "overall_coverage": (legal_coverage + expected_coverage) / 2,
                "query_count": stats["queries"]
            }

        # ?¤ì›Œ???€?…ë³„ ì»¤ë²„ë¦¬ì?
        total_legal = sum(len(r["legal_keywords"]) for r in extraction_results.values())
        total_context = sum(len(r["context_keywords"]) for r in extraction_results.values())
        total_expected = sum(len(r["expected_terms"]) for r in extraction_results.values())

        matched_legal = sum(len(r["matched_keywords"]["legal_matched"]) for r in extraction_results.values())
        matched_context = sum(len(r["matched_keywords"]["context_matched"]) for r in extraction_results.values())
        matched_expected = sum(len(r["matched_keywords"]["expected_matched"]) for r in extraction_results.values())

        coverage_analysis["keyword_type_coverage"] = {
            "legal_keywords": {
                "total": total_legal,
                "matched": matched_legal,
                "coverage_rate": matched_legal / total_legal if total_legal > 0 else 0
            },
            "context_keywords": {
                "total": total_context,
                "matched": matched_context,
                "coverage_rate": matched_context / total_context if total_context > 0 else 0
            },
            "expected_terms": {
                "total": total_expected,
                "matched": matched_expected,
                "coverage_rate": matched_expected / total_expected if total_expected > 0 else 0
            }
        }

        # ?„ë½???¤ì›Œ??ë¶„ì„
        for query_type, result in extraction_results.items():
            missing_legal = [kw for kw in result["legal_keywords"] if kw not in result["extracted_keywords"]["all_keywords"]]
            missing_expected = [kw for kw in result["expected_terms"] if kw not in result["extracted_keywords"]["all_keywords"]]

            coverage_analysis["missing_keywords"][query_type] = {
                "missing_legal_keywords": missing_legal,
                "missing_expected_terms": missing_expected,
                "missing_count": len(missing_legal) + len(missing_expected)
            }

        # ê°œì„  ê¶Œì¥?¬í•­ ?ì„±
        overall_coverage = sum(c["overall_coverage"] for c in coverage_analysis["domain_coverage"].values()) / len(coverage_analysis["domain_coverage"])

        if overall_coverage < 0.3:
            coverage_analysis["recommendations"].append("?„ì²´ ?¤ì›Œ??ì»¤ë²„ë¦¬ì?ê°€ ë§¤ìš° ??Šµ?ˆë‹¤. ?¤ì›Œ??ë§¤í•‘ ?œìŠ¤?œì„ ?„ë©´ ?¬ê?? í•˜?¸ìš”.")
        elif overall_coverage < 0.5:
            coverage_analysis["recommendations"].append("?¤ì›Œ??ì»¤ë²„ë¦¬ì?ê°€ ??Šµ?ˆë‹¤. ?˜ë???ê´€ê³„ë? ?•ì¥?˜ê³  ?„ë©”?¸ë³„ ?©ì–´ë¥?ë³´ê°•?˜ì„¸??")
        elif overall_coverage < 0.7:
            coverage_analysis["recommendations"].append("?¤ì›Œ??ì»¤ë²„ë¦¬ì?ê°€ ë³´í†µ?…ë‹ˆ?? ì¶”ê? ê°œì„ ???µí•´ ???¥ìƒ?œí‚¬ ???ˆìŠµ?ˆë‹¤.")
        else:
            coverage_analysis["recommendations"].append("?¤ì›Œ??ì»¤ë²„ë¦¬ì?ê°€ ?‘í˜¸?©ë‹ˆ?? ?„ì¬ ?¤ì •??? ì??˜ì„¸??")

        # ?„ë©”?¸ë³„ ê¶Œì¥?¬í•­
        for domain, coverage in coverage_analysis["domain_coverage"].items():
            if coverage["overall_coverage"] < 0.3:
                coverage_analysis["recommendations"].append(f"{domain} ?„ë©”?¸ì˜ ì»¤ë²„ë¦¬ì?ê°€ ë§¤ìš° ??Šµ?ˆë‹¤. ?´ë‹¹ ?„ë©”???©ì–´ë¥??€???•ì¥?˜ì„¸??")
            elif coverage["overall_coverage"] < 0.5:
                coverage_analysis["recommendations"].append(f"{domain} ?„ë©”?¸ì˜ ì»¤ë²„ë¦¬ì?ë¥?ê°œì„ ?˜ì„¸??")

        logger.info("?¤ì›Œ??ì»¤ë²„ë¦¬ì? ë¶„ì„ ?„ë£Œ")
        return coverage_analysis

    def test_keyword_expansion_effectiveness(self, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
        """?¤ì›Œ???•ì¥ ?¨ê³¼???ŒìŠ¤??""
        logger.info("?¤ì›Œ???•ì¥ ?¨ê³¼???ŒìŠ¤???œì‘")

        expansion_analysis = {
            "expansion_metrics": {},
            "semantic_effectiveness": {},
            "contextual_effectiveness": {},
            "overall_effectiveness": {}
        }

        for query_type, result in extraction_results.items():
            base_keywords = result["extracted_keywords"]["base_keywords"]
            contextual_keywords = result["extracted_keywords"]["contextual_keywords"]
            semantic_keywords = result["extracted_keywords"]["semantic_keywords"]
            all_keywords = result["extracted_keywords"]["all_keywords"]

            # ?•ì¥ ë©”íŠ¸ë¦?
            base_count = len(base_keywords)
            contextual_count = len(contextual_keywords)
            semantic_count = len(semantic_keywords)
            total_count = len(all_keywords)

            expansion_ratio = total_count / base_count if base_count > 0 else 0
            contextual_expansion = contextual_count / base_count if base_count > 0 else 0
            semantic_expansion = semantic_count / base_count if base_count > 0 else 0

            expansion_analysis["expansion_metrics"][query_type] = {
                "base_keywords": base_count,
                "contextual_keywords": contextual_count,
                "semantic_keywords": semantic_count,
                "total_keywords": total_count,
                "expansion_ratio": expansion_ratio,
                "contextual_expansion": contextual_expansion,
                "semantic_expansion": semantic_expansion
            }

            # ?˜ë????¨ê³¼??(?ˆìƒ ?©ì–´?€??ë§¤ì¹­)
            semantic_matches = [kw for kw in semantic_keywords if kw in result["expected_terms"]]
            semantic_effectiveness = len(semantic_matches) / len(semantic_keywords) if semantic_keywords else 0

            expansion_analysis["semantic_effectiveness"][query_type] = {
                "semantic_matches": semantic_matches,
                "effectiveness_rate": semantic_effectiveness
            }

            # ì»¨í…?¤íŠ¸ ?¨ê³¼??(ë²•ë¥  ?¤ì›Œ?œì???ë§¤ì¹­)
            contextual_matches = [kw for kw in contextual_keywords if kw in result["legal_keywords"]]
            contextual_effectiveness = len(contextual_matches) / len(contextual_keywords) if contextual_keywords else 0

            expansion_analysis["contextual_effectiveness"][query_type] = {
                "contextual_matches": contextual_matches,
                "effectiveness_rate": contextual_effectiveness
            }

        # ?„ì²´ ?¨ê³¼??ê³„ì‚°
        avg_expansion_ratio = sum(m["expansion_ratio"] for m in expansion_analysis["expansion_metrics"].values()) / len(expansion_analysis["expansion_metrics"])
        avg_semantic_effectiveness = sum(e["effectiveness_rate"] for e in expansion_analysis["semantic_effectiveness"].values()) / len(expansion_analysis["semantic_effectiveness"])
        avg_contextual_effectiveness = sum(e["effectiveness_rate"] for e in expansion_analysis["contextual_effectiveness"].values()) / len(expansion_analysis["contextual_effectiveness"])

        expansion_analysis["overall_effectiveness"] = {
            "average_expansion_ratio": avg_expansion_ratio,
            "average_semantic_effectiveness": avg_semantic_effectiveness,
            "average_contextual_effectiveness": avg_contextual_effectiveness,
            "overall_score": (avg_expansion_ratio + avg_semantic_effectiveness + avg_contextual_effectiveness) / 3
        }

        logger.info("?¤ì›Œ???•ì¥ ?¨ê³¼???ŒìŠ¤???„ë£Œ")
        return expansion_analysis

    def generate_comprehensive_report(self, extraction_results: Dict[str, Any], coverage_analysis: Dict[str, Any], expansion_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """ì¢…í•© ë³´ê³ ???ì„±"""
        logger.info("ì¢…í•© ë³´ê³ ???ì„± ì¤?)

        report = {
            "test_summary": {
                "test_date": datetime.now().isoformat(),
                "total_queries": len(self.test_queries),
                "test_type": "enhanced_keyword_extraction_accuracy"
            },
            "extraction_results": extraction_results,
            "coverage_analysis": coverage_analysis,
            "expansion_analysis": expansion_analysis,
            "performance_summary": {
                "average_legal_accuracy": sum(r["accuracy_metrics"]["legal_accuracy"] for r in extraction_results.values()) / len(extraction_results),
                "average_context_accuracy": sum(r["accuracy_metrics"]["context_accuracy"] for r in extraction_results.values()) / len(extraction_results),
                "average_expected_accuracy": sum(r["accuracy_metrics"]["expected_accuracy"] for r in extraction_results.values()) / len(extraction_results),
                "average_overall_accuracy": sum(r["accuracy_metrics"]["overall_accuracy"] for r in extraction_results.values()) / len(extraction_results),
                "average_processing_time": sum(r["processing_time"] for r in extraction_results.values()) / len(extraction_results)
            },
            "recommendations": self._generate_final_recommendations(extraction_results, coverage_analysis, expansion_analysis)
        }

        logger.info("ì¢…í•© ë³´ê³ ???ì„± ?„ë£Œ")
        return report

    def _generate_final_recommendations(self, extraction_results: Dict[str, Any], coverage_analysis: Dict[str, Any], expansion_analysis: Dict[str, Any]) -> List[str]:
        """ìµœì¢… ê¶Œì¥?¬í•­ ?ì„±"""
        recommendations = []

        # ?•í™•??ê¸°ë°˜ ê¶Œì¥?¬í•­
        avg_accuracy = sum(r["accuracy_metrics"]["overall_accuracy"] for r in extraction_results.values()) / len(extraction_results)

        if avg_accuracy < 0.3:
            recommendations.append("?¤ì›Œ??ì¶”ì¶œ ?•í™•?„ê? ë§¤ìš° ??Šµ?ˆë‹¤. ?¤ì›Œ??ë§¤í•‘ ?Œê³ ë¦¬ì¦˜???„ë©´ ?¬ê?? í•˜?¸ìš”.")
        elif avg_accuracy < 0.5:
            recommendations.append("?¤ì›Œ??ì¶”ì¶œ ?•í™•?„ê? ??Šµ?ˆë‹¤. ?˜ë???ê´€ê³„ì? ì»¨í…?¤íŠ¸ ë§¤í•‘??ê°•í™”?˜ì„¸??")
        elif avg_accuracy < 0.7:
            recommendations.append("?¤ì›Œ??ì¶”ì¶œ ?•í™•?„ê? ë³´í†µ?…ë‹ˆ?? ì¶”ê? ê°œì„ ???µí•´ ???¥ìƒ?œí‚¬ ???ˆìŠµ?ˆë‹¤.")
        else:
            recommendations.append("?¤ì›Œ??ì¶”ì¶œ ?•í™•?„ê? ?‘í˜¸?©ë‹ˆ?? ?„ì¬ ?¤ì •??? ì??˜ì„¸??")

        # ì»¤ë²„ë¦¬ì? ê¸°ë°˜ ê¶Œì¥?¬í•­
        overall_coverage = sum(c["overall_coverage"] for c in coverage_analysis["domain_coverage"].values()) / len(coverage_analysis["domain_coverage"])

        if overall_coverage < 0.3:
            recommendations.append("?¤ì›Œ??ì»¤ë²„ë¦¬ì?ê°€ ë§¤ìš° ??Šµ?ˆë‹¤. ?„ë©”?¸ë³„ ?©ì–´ ?¬ì „???€???•ì¥?˜ì„¸??")
        elif overall_coverage < 0.5:
            recommendations.append("?¤ì›Œ??ì»¤ë²„ë¦¬ì?ë¥?ê°œì„ ?˜ê¸° ?„í•´ ?˜ë???ê´€ê³„ë? ?•ì¥?˜ì„¸??")

        # ?•ì¥ ?¨ê³¼??ê¸°ë°˜ ê¶Œì¥?¬í•­
        expansion_score = expansion_analysis["overall_effectiveness"]["overall_score"]

        if expansion_score < 0.3:
            recommendations.append("?¤ì›Œ???•ì¥ ?¨ê³¼?±ì´ ??Šµ?ˆë‹¤. ?˜ë???ë§¤í•‘ê³?ì»¨í…?¤íŠ¸ ?¸ì‹??ê°œì„ ?˜ì„¸??")
        elif expansion_score < 0.5:
            recommendations.append("?¤ì›Œ???•ì¥ ?¨ê³¼?±ì„ ê°œì„ ?˜ê¸° ?„í•´ ?™ìŠµ ?°ì´?°ë? ?•ì¥?˜ì„¸??")

        # ì²˜ë¦¬ ?œê°„ ê¸°ë°˜ ê¶Œì¥?¬í•­
        avg_processing_time = sum(r["processing_time"] for r in extraction_results.values()) / len(extraction_results)

        if avg_processing_time > 0.1:
            recommendations.append("ì²˜ë¦¬ ?œê°„??ê¸¸ì–´ ?±ëŠ¥ ìµœì ?”ê? ?„ìš”?©ë‹ˆ??")
        elif avg_processing_time > 0.05:
            recommendations.append("ì²˜ë¦¬ ?œê°„???ì ˆ?˜ì?ë§?ì¶”ê? ìµœì ???¬ì?ê°€ ?ˆìŠµ?ˆë‹¤.")
        else:
            recommendations.append("ì²˜ë¦¬ ?œê°„???°ìˆ˜?©ë‹ˆ?? ?„ì¬ ?±ëŠ¥??? ì??˜ì„¸??")

        return recommendations

    def save_test_results(self, test_report: Dict[str, Any]):
        """?ŒìŠ¤??ê²°ê³¼ ?€??""
        os.makedirs(self.output_dir, exist_ok=True)

        # ?ŒìŠ¤??ë³´ê³ ???€??
        report_file = os.path.join(self.output_dir, "enhanced_query_test_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(test_report, f, ensure_ascii=False, indent=2)

        logger.info(f"?ŒìŠ¤??ê²°ê³¼ ?€???„ë£Œ: {self.output_dir}")

    def run_enhanced_test(self):
        """?¥ìƒ???ŒìŠ¤???¤í–‰"""
        logger.info("?¥ìƒ??ë²•ë¥  ì§ˆì˜ ?ŒìŠ¤???œì‘")

        try:
            # ?¤ì›Œ??ì¶”ì¶œ ?•í™•???ŒìŠ¤??
            extraction_results = self.test_keyword_extraction_accuracy()

            # ?¤ì›Œ??ì»¤ë²„ë¦¬ì? ë¶„ì„
            coverage_analysis = self.test_keyword_coverage_analysis(extraction_results)

            # ?¤ì›Œ???•ì¥ ?¨ê³¼???ŒìŠ¤??
            expansion_analysis = self.test_keyword_expansion_effectiveness(extraction_results)

            # ì¢…í•© ë³´ê³ ???ì„±
            test_report = self.generate_comprehensive_report(extraction_results, coverage_analysis, expansion_analysis)

            # ê²°ê³¼ ?€??
            self.save_test_results(test_report)

            logger.info("?¥ìƒ??ë²•ë¥  ì§ˆì˜ ?ŒìŠ¤???„ë£Œ")

            # ê²°ê³¼ ?”ì•½ ì¶œë ¥
            print(f"\n=== ?¥ìƒ??ë²•ë¥  ì§ˆì˜ ?ŒìŠ¤??ê²°ê³¼ ?”ì•½ ===")
            print(f"ì´??ŒìŠ¤??ì§ˆì˜ ?? {test_report['test_summary']['total_queries']}")
            print(f"?‰ê·  ë²•ë¥  ?¤ì›Œ???•í™•?? {test_report['performance_summary']['average_legal_accuracy']:.3f}")
            print(f"?‰ê·  ì»¨í…?¤íŠ¸ ?¤ì›Œ???•í™•?? {test_report['performance_summary']['average_context_accuracy']:.3f}")
            print(f"?‰ê·  ?ˆìƒ ?©ì–´ ?•í™•?? {test_report['performance_summary']['average_expected_accuracy']:.3f}")
            print(f"?‰ê·  ?„ì²´ ?•í™•?? {test_report['performance_summary']['average_overall_accuracy']:.3f}")
            print(f"?‰ê·  ì²˜ë¦¬ ?œê°„: {test_report['performance_summary']['average_processing_time']:.4f}ì´?)

            print(f"\n=== ?„ë©”?¸ë³„ ì»¤ë²„ë¦¬ì? ===")
            for domain, coverage in test_report['coverage_analysis']['domain_coverage'].items():
                print(f"{domain}: {coverage['overall_coverage']:.3f}")

            print(f"\n=== ?•ì¥ ?¨ê³¼??===")
            expansion_metrics = test_report['expansion_analysis']['overall_effectiveness']
            print(f"?‰ê·  ?•ì¥ë¥? {expansion_metrics['average_expansion_ratio']:.2f}")
            print(f"?‰ê·  ?˜ë????¨ê³¼?? {expansion_metrics['average_semantic_effectiveness']:.3f}")
            print(f"?‰ê·  ì»¨í…?¤íŠ¸ ?¨ê³¼?? {expansion_metrics['average_contextual_effectiveness']:.3f}")
            print(f"?„ì²´ ?¨ê³¼???ìˆ˜: {expansion_metrics['overall_score']:.3f}")

            print(f"\n=== ê°œì„  ê¶Œì¥?¬í•­ ===")
            for i, recommendation in enumerate(test_report['recommendations'], 1):
                print(f"{i}. {recommendation}")

            # ?ì„¸ ê²°ê³¼ ?ˆì‹œ ì¶œë ¥
            print(f"\n=== ?ì„¸ ê²°ê³¼ ?ˆì‹œ (ì²?ë²ˆì§¸ ì§ˆì˜) ===")
            first_query = list(extraction_results.values())[0]
            print(f"ì§ˆë¬¸: {first_query['question']}")
            print(f"ë²•ë¥  ?¤ì›Œ???•í™•?? {first_query['accuracy_metrics']['legal_accuracy']:.3f}")
            print(f"?ˆìƒ ?©ì–´ ?•í™•?? {first_query['accuracy_metrics']['expected_accuracy']:.3f}")
            print(f"?„ì²´ ?•í™•?? {first_query['accuracy_metrics']['overall_accuracy']:.3f}")
            print(f"ë§¤ì¹­??ë²•ë¥  ?¤ì›Œ?? {first_query['matched_keywords']['legal_matched']}")
            print(f"ë§¤ì¹­???ˆìƒ ?©ì–´: {first_query['matched_keywords']['expected_matched']}")

        except Exception as e:
            logger.error(f"?¥ìƒ???ŒìŠ¤??ì¤??¤ë¥˜ ë°œìƒ: {e}")
            raise

def main():
    """ë©”ì¸ ?¤í–‰ ?¨ìˆ˜"""
    tester = EnhancedQueryTester()
    tester.run_enhanced_test()

if __name__ == "__main__":
    main()
