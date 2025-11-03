#!/usr/bin/env python3
"""
?¤ì œ ?¬ìš©??ì§ˆë¬¸ ê¸°ë°˜ ?¤ì›Œ??ë§¤í•‘ ?ŒìŠ¤??
?¤ì œ ë²•ë¥  ?ë‹´?ì„œ ?˜ì˜¬ ???ˆëŠ” ì§ˆë¬¸?¤ë¡œ ?¤ì›Œ??ë§¤í•‘ ?±ëŠ¥???‰ê??©ë‹ˆ??
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
        logging.FileHandler('logs/realistic_query_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealisticQueryTester:
    """?„ì‹¤?ì¸ ì§ˆì˜ ?ŒìŠ¤?¸ê¸°"""

    def __init__(self):
        self.test_queries = self._initialize_realistic_queries()
        self.output_dir = "data/extracted_terms/realistic_query_test"

    def _initialize_realistic_queries(self) -> List[Dict[str, Any]]:
        """?„ì‹¤?ì¸ ?ŒìŠ¤??ì§ˆì˜ ì´ˆê¸°??""
        return [
            {
                "question": "?„íŒŒ??ê³„ì•½?œì— ?„ì•½ê¸ˆì´ ê³„ì•½ê¸ˆì˜ 10ë°°ë¡œ ?˜ì–´ ?ˆëŠ”?? ?´ê²Œ ?©ë²•?ì¸ê°€??",
                "query_type": "contract_review",
                "domain": "ë¯¼ì‚¬ë²?,
                "realistic_keywords": ["?„íŒŒ??, "ê³„ì•½??, "?„ì•½ê¸?, "ê³„ì•½ê¸?, "10ë°?, "?©ë²•", "ë¯¼ë²•", "398ì¡?],
                "legal_concepts": ["?„ì•½ê¸ˆì œ??, "?í•´ë°°ìƒ", "ê³„ì•½??, "ë¯¼ë²•"]
            },
            {
                "question": "êµí†µ?¬ê³  ?¬ëŠ”???ë?ë°©ì´ ë³´í—˜ì²˜ë¦¬ ???´ì£¼?¤ê³  ?´ìš”. ?´ë–»ê²??´ì•¼ ?˜ë‚˜??",
                "query_type": "damage_compensation",
                "domain": "ë¯¼ì‚¬ë²?,
                "realistic_keywords": ["êµí†µ?¬ê³ ", "?ë?ë°?, "ë³´í—˜ì²˜ë¦¬", "?í•´ë°°ìƒ", "ë³´í—˜?Œì‚¬", "ë¯¼ì‚¬?Œì†¡"],
                "legal_concepts": ["êµí†µ?¬ê³ ", "?í•´ë°°ìƒ", "ë³´í—˜", "ë¯¼ì‚¬?Œì†¡", "ë¶ˆë²•?‰ìœ„"]
            },
            {
                "question": "?´í˜¼?˜ë ¤?”ë° ?„ì´ ?‘ìœ¡ê¶Œì„ ?´ë–»ê²?ê²°ì •?˜ë‚˜??",
                "query_type": "divorce_proceedings",
                "domain": "ê°€ì¡±ë²•",
                "realistic_keywords": ["?´í˜¼", "?„ì´", "?‘ìœ¡ê¶?, "ê²°ì •", "ê°€?•ë²•??, "?‘ìœ¡ë¹?],
                "legal_concepts": ["?´í˜¼", "?‘ìœ¡ê¶?, "ê°€?•ë²•??, "ê°€ì¡±ë²•", "?‘ìœ¡ë¹?]
            },
            {
                "question": "ì§??¬ë ¤?”ë° ì¤‘ê°œ?…ìê°€ ?±ê¸°ë¶€?±ë³¸????ë³´ì—¬ì£¼ë ¤ê³??´ìš”. ?´ë–»ê²??´ì•¼ ?˜ë‚˜??",
                "query_type": "real_estate_transaction",
                "domain": "ë¶€?™ì‚°ë²?,
                "realistic_keywords": ["ì§?, "ì¤‘ê°œ?…ì", "?±ê¸°ë¶€?±ë³¸", "ë¶€?™ì‚°", "ë§¤ë§¤", "?±ê¸°"],
                "legal_concepts": ["ë¶€?™ì‚°ë§¤ë§¤", "?±ê¸°ë¶€?±ë³¸", "ì¤‘ê°œ??, "ë¶€?™ì‚°?±ê¸°ë²?]
            },
            {
                "question": "?Œì‚¬?ì„œ ê°‘ìê¸??´ê³  ?µë³´ë¥?ë°›ì•˜?”ë°, ?´ê²Œ ë¶€?¹í•´ê³ ì¸ê°€??",
                "query_type": "employment_termination",
                "domain": "?¸ë™ë²?,
                "realistic_keywords": ["?Œì‚¬", "?´ê³ ", "?µë³´", "ë¶€?¹í•´ê³?, "ê·¼ë¡œê¸°ì?ë²?, "?¸ë™?„ì›??],
                "legal_concepts": ["?´ê³ ", "ë¶€?¹í•´ê³?, "ê·¼ë¡œê¸°ì?ë²?, "?¸ë™?„ì›??, "êµ¬ì œ? ì²­"]
            },
            {
                "question": "?¹í—ˆ ì¶œì›?˜ë ¤?”ë° ë¹„ìŠ·??ë°œëª…???´ë? ?ˆëŠ”ì§€ ?´ë–»ê²??•ì¸?˜ë‚˜??",
                "query_type": "patent_application",
                "domain": "?¹í—ˆë²?,
                "realistic_keywords": ["?¹í—ˆ", "ì¶œì›", "ë°œëª…", "? ê·œ??, "?¹í—ˆì²?, "? í–‰ê¸°ìˆ "],
                "legal_concepts": ["?¹í—ˆì¶œì›", "? ê·œ??, "?¹í—ˆì²?, "? í–‰ê¸°ìˆ ì¡°ì‚¬", "?¹í—ˆë²?]
            },
            {
                "question": "?Œì‚¬ ?¤ë¦½?˜ë ¤?”ë° ?ë³¸ê¸ˆì? ?¼ë§ˆ???„ìš”?œê???",
                "query_type": "company_establishment",
                "domain": "?ì‚¬ë²?,
                "realistic_keywords": ["?Œì‚¬", "?¤ë¦½", "?ë³¸ê¸?, "ì£¼ì‹?Œì‚¬", "?ë²•", "?±ê¸°"],
                "legal_concepts": ["?Œì‚¬?¤ë¦½", "?ë³¸ê¸?, "ì£¼ì‹?Œì‚¬", "?ë²•", "?±ê¸°"]
            },
            {
                "question": "?•ì‚¬?¬ê±´?¼ë¡œ ê¸°ì†Œ?˜ì—ˆ?”ë° ë³€?¸ì‚¬ ? ì„???„ìˆ˜?¸ê???",
                "query_type": "criminal_defense",
                "domain": "?•ì‚¬ë²?,
                "realistic_keywords": ["?•ì‚¬?¬ê±´", "ê¸°ì†Œ", "ë³€?¸ì‚¬", "? ì„", "êµ?„ ë³€??, "?•ì‚¬?Œì†¡ë²?],
                "legal_concepts": ["?•ì‚¬?¬ê±´", "ë³€?¸ì‚¬? ì„", "êµ?„ ë³€??, "?•ì‚¬?Œì†¡ë²?, "?¼ê³ "]
            }
        ]

    def test_keyword_relevance(self) -> Dict[str, Any]:
        """?¤ì›Œ??ê´€?¨ì„± ?ŒìŠ¤??""
        logger.info("?¤ì›Œ??ê´€?¨ì„± ?ŒìŠ¤???œì‘")

        enhanced_mapper = EnhancedKeywordMapper()
        results = {}

        for i, query in enumerate(self.test_queries):
            question = query["question"]
            query_type = query["query_type"]
            realistic_keywords = query["realistic_keywords"]
            legal_concepts = query["legal_concepts"]

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

            # ê´€?¨ì„± ë¶„ì„
            realistic_matches = [kw for kw in realistic_keywords if kw in all_keywords]
            legal_concept_matches = [kw for kw in legal_concepts if kw in all_keywords]

            realistic_relevance = len(realistic_matches) / len(realistic_keywords) if realistic_keywords else 0
            legal_concept_relevance = len(legal_concept_matches) / len(legal_concepts) if legal_concepts else 0

            # ?¤ì›Œ???ˆì§ˆ ?‰ê?
            quality_score = self._evaluate_keyword_quality(all_keywords, realistic_keywords, legal_concepts)

            results[query_type] = {
                "question": question,
                "realistic_keywords": realistic_keywords,
                "legal_concepts": legal_concepts,
                "extracted_keywords": {
                    "all_keywords": all_keywords,
                    "base_keywords": base_keywords,
                    "contextual_keywords": contextual_keywords,
                    "semantic_keywords": semantic_keywords
                },
                "relevance_metrics": {
                    "realistic_relevance": realistic_relevance,
                    "legal_concept_relevance": legal_concept_relevance,
                    "overall_relevance": (realistic_relevance + legal_concept_relevance) / 2,
                    "quality_score": quality_score
                },
                "matched_keywords": {
                    "realistic_matched": realistic_matches,
                    "legal_concept_matched": legal_concept_matches
                },
                "processing_time": end_time - start_time,
                "comprehensive_result": comprehensive_result
            }

        logger.info("?¤ì›Œ??ê´€?¨ì„± ?ŒìŠ¤???„ë£Œ")
        return results

    def _evaluate_keyword_quality(self, extracted_keywords: List[str], realistic_keywords: List[str], legal_concepts: List[str]) -> float:
        """?¤ì›Œ???ˆì§ˆ ?‰ê?"""
        quality_score = 0.0

        # ë²•ë¥  ê´€?¨ì„± ?ìˆ˜ (0-0.4)
        legal_indicators = [
            'ë²?, 'ê·œì¹™', '??, 'ê¶?, '?˜ë¬´', 'ì±…ì„', '?ˆì°¨', '? ì²­', '? ê³ ',
            '?ˆê?', '?¸ê?', '?¹ì¸', '??, 'ì²?, 'ë¶€', '?„ì›??, 'ë²•ì›',
            '?‰ìœ„', 'ì²˜ë¶„', 'ê²°ì •', 'ëª…ë ¹', 'ì§€??, '?Œì†¡', '?¬íŒ', '?ê²°',
            'ê³„ì•½', '?í•´ë°°ìƒ', '?´í˜¼', '?ì†', 'ë¶€?™ì‚°', '?¹í—ˆ', 'ê·¼ë¡œ', '?Œì‚¬'
        ]

        legal_count = sum(1 for kw in extracted_keywords if any(indicator in kw for indicator in legal_indicators))
        legal_score = min(legal_count / len(extracted_keywords), 0.4) if extracted_keywords else 0
        quality_score += legal_score

        # ?„ì‹¤???ìˆ˜ (0-0.3)
        realistic_count = sum(1 for kw in extracted_keywords if kw in realistic_keywords)
        realistic_score = min(realistic_count / len(realistic_keywords), 0.3) if realistic_keywords else 0
        quality_score += realistic_score

        # ë²•ë¥  ê°œë… ?¼ì¹˜ ?ìˆ˜ (0-0.3)
        concept_count = sum(1 for kw in extracted_keywords if kw in legal_concepts)
        concept_score = min(concept_count / len(legal_concepts), 0.3) if legal_concepts else 0
        quality_score += concept_score

        return min(quality_score, 1.0)

    def test_keyword_diversity_and_coverage(self, relevance_results: Dict[str, Any]) -> Dict[str, Any]:
        """?¤ì›Œ???¤ì–‘??ë°?ì»¤ë²„ë¦¬ì? ?ŒìŠ¤??""
        logger.info("?¤ì›Œ???¤ì–‘??ë°?ì»¤ë²„ë¦¬ì? ?ŒìŠ¤???œì‘")

        diversity_analysis = {
            "keyword_diversity": {},
            "domain_coverage": {},
            "concept_coverage": {},
            "overall_metrics": {}
        }

        # ?¤ì›Œ???¤ì–‘??ë¶„ì„
        for query_type, result in relevance_results.items():
            all_keywords = result["extracted_keywords"]["all_keywords"]
            base_keywords = result["extracted_keywords"]["base_keywords"]
            contextual_keywords = result["extracted_keywords"]["contextual_keywords"]
            semantic_keywords = result["extracted_keywords"]["semantic_keywords"]

            # ?¤ì–‘??ë©”íŠ¸ë¦?
            total_keywords = len(all_keywords)
            unique_keywords = len(set(all_keywords))
            diversity_ratio = unique_keywords / total_keywords if total_keywords > 0 else 0

            # ?•ì¥ ë©”íŠ¸ë¦?
            expansion_ratio = total_keywords / len(base_keywords) if base_keywords else 0
            contextual_ratio = len(contextual_keywords) / len(base_keywords) if base_keywords else 0
            semantic_ratio = len(semantic_keywords) / len(base_keywords) if base_keywords else 0

            diversity_analysis["keyword_diversity"][query_type] = {
                "total_keywords": total_keywords,
                "unique_keywords": unique_keywords,
                "diversity_ratio": diversity_ratio,
                "expansion_ratio": expansion_ratio,
                "contextual_ratio": contextual_ratio,
                "semantic_ratio": semantic_ratio
            }

        # ?„ë©”?¸ë³„ ì»¤ë²„ë¦¬ì?
        domain_stats = {}
        for query_type, result in relevance_results.items():
            domain = next((q["domain"] for q in self.test_queries if q["query_type"] == query_type), "ê¸°í?")

            if domain not in domain_stats:
                domain_stats[domain] = {
                    "total_realistic_keywords": 0,
                    "matched_realistic_keywords": 0,
                    "total_legal_concepts": 0,
                    "matched_legal_concepts": 0,
                    "queries": 0
                }

            domain_stats[domain]["total_realistic_keywords"] += len(result["realistic_keywords"])
            domain_stats[domain]["matched_realistic_keywords"] += len(result["matched_keywords"]["realistic_matched"])
            domain_stats[domain]["total_legal_concepts"] += len(result["legal_concepts"])
            domain_stats[domain]["matched_legal_concepts"] += len(result["matched_keywords"]["legal_concept_matched"])
            domain_stats[domain]["queries"] += 1

        # ?„ë©”?¸ë³„ ì»¤ë²„ë¦¬ì? ê³„ì‚°
        for domain, stats in domain_stats.items():
            realistic_coverage = stats["matched_realistic_keywords"] / stats["total_realistic_keywords"] if stats["total_realistic_keywords"] > 0 else 0
            concept_coverage = stats["matched_legal_concepts"] / stats["total_legal_concepts"] if stats["total_legal_concepts"] > 0 else 0

            diversity_analysis["domain_coverage"][domain] = {
                "realistic_coverage": realistic_coverage,
                "concept_coverage": concept_coverage,
                "overall_coverage": (realistic_coverage + concept_coverage) / 2,
                "query_count": stats["queries"]
            }

        # ?„ì²´ ë©”íŠ¸ë¦?ê³„ì‚°
        avg_diversity = sum(d["diversity_ratio"] for d in diversity_analysis["keyword_diversity"].values()) / len(diversity_analysis["keyword_diversity"])
        avg_expansion = sum(d["expansion_ratio"] for d in diversity_analysis["keyword_diversity"].values()) / len(diversity_analysis["keyword_diversity"])
        avg_coverage = sum(c["overall_coverage"] for c in diversity_analysis["domain_coverage"].values()) / len(diversity_analysis["domain_coverage"])

        diversity_analysis["overall_metrics"] = {
            "average_diversity": avg_diversity,
            "average_expansion": avg_expansion,
            "average_coverage": avg_coverage,
            "overall_score": (avg_diversity + avg_expansion + avg_coverage) / 3
        }

        logger.info("?¤ì›Œ???¤ì–‘??ë°?ì»¤ë²„ë¦¬ì? ?ŒìŠ¤???„ë£Œ")
        return diversity_analysis

    def generate_realistic_test_report(self, relevance_results: Dict[str, Any], diversity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """?„ì‹¤?ì¸ ?ŒìŠ¤??ë³´ê³ ???ì„±"""
        logger.info("?„ì‹¤?ì¸ ?ŒìŠ¤??ë³´ê³ ???ì„± ì¤?)

        report = {
            "test_summary": {
                "test_date": datetime.now().isoformat(),
                "total_queries": len(self.test_queries),
                "test_type": "realistic_keyword_relevance_test"
            },
            "relevance_results": relevance_results,
            "diversity_analysis": diversity_analysis,
            "performance_summary": {
                "average_realistic_relevance": sum(r["relevance_metrics"]["realistic_relevance"] for r in relevance_results.values()) / len(relevance_results),
                "average_legal_concept_relevance": sum(r["relevance_metrics"]["legal_concept_relevance"] for r in relevance_results.values()) / len(relevance_results),
                "average_overall_relevance": sum(r["relevance_metrics"]["overall_relevance"] for r in relevance_results.values()) / len(relevance_results),
                "average_quality_score": sum(r["relevance_metrics"]["quality_score"] for r in relevance_results.values()) / len(relevance_results),
                "average_processing_time": sum(r["processing_time"] for r in relevance_results.values()) / len(relevance_results)
            },
            "recommendations": self._generate_realistic_recommendations(relevance_results, diversity_analysis)
        }

        logger.info("?„ì‹¤?ì¸ ?ŒìŠ¤??ë³´ê³ ???ì„± ?„ë£Œ")
        return report

    def _generate_realistic_recommendations(self, relevance_results: Dict[str, Any], diversity_analysis: Dict[str, Any]) -> List[str]:
        """?„ì‹¤?ì¸ ê¶Œì¥?¬í•­ ?ì„±"""
        recommendations = []

        # ê´€?¨ì„± ê¸°ë°˜ ê¶Œì¥?¬í•­
        avg_relevance = sum(r["relevance_metrics"]["overall_relevance"] for r in relevance_results.values()) / len(relevance_results)

        if avg_relevance < 0.3:
            recommendations.append("?¤ì›Œ??ê´€?¨ì„±??ë§¤ìš° ??Šµ?ˆë‹¤. ?¤ì œ ?¬ìš©??ì§ˆë¬¸??ë§ëŠ” ?¤ì›Œ??ë§¤í•‘??ê°œì„ ?˜ì„¸??")
        elif avg_relevance < 0.5:
            recommendations.append("?¤ì›Œ??ê´€?¨ì„±????Šµ?ˆë‹¤. ?„ì‹¤?ì¸ ?©ì–´?€ ë²•ë¥  ê°œë…??ë§¤í•‘??ê°•í™”?˜ì„¸??")
        elif avg_relevance < 0.7:
            recommendations.append("?¤ì›Œ??ê´€?¨ì„±??ë³´í†µ?…ë‹ˆ?? ì¶”ê? ê°œì„ ???µí•´ ???¥ìƒ?œí‚¬ ???ˆìŠµ?ˆë‹¤.")
        else:
            recommendations.append("?¤ì›Œ??ê´€?¨ì„±???‘í˜¸?©ë‹ˆ?? ?„ì¬ ?¤ì •??? ì??˜ì„¸??")

        # ?ˆì§ˆ ?ìˆ˜ ê¸°ë°˜ ê¶Œì¥?¬í•­
        avg_quality = sum(r["relevance_metrics"]["quality_score"] for r in relevance_results.values()) / len(relevance_results)

        if avg_quality < 0.3:
            recommendations.append("?¤ì›Œ???ˆì§ˆ??ë§¤ìš° ??Šµ?ˆë‹¤. ë²•ë¥  ê´€?¨ì„±ê³??„ì‹¤?±ì„ ëª¨ë‘ ê³ ë ¤???¤ì›Œ??ì¶”ì¶œ??ê°œì„ ?˜ì„¸??")
        elif avg_quality < 0.5:
            recommendations.append("?¤ì›Œ???ˆì§ˆ??ê°œì„ ?˜ê¸° ?„í•´ ë²•ë¥  ?©ì–´?€ ?„ì‹¤???©ì–´??ê· í˜•??ë§ì¶”?¸ìš”.")

        # ?¤ì–‘??ê¸°ë°˜ ê¶Œì¥?¬í•­
        overall_score = diversity_analysis["overall_metrics"]["overall_score"]

        if overall_score < 0.3:
            recommendations.append("?¤ì›Œ???¤ì–‘?±ê³¼ ì»¤ë²„ë¦¬ì?ê°€ ë§¤ìš° ??Šµ?ˆë‹¤. ?„ë©”?¸ë³„ ?©ì–´ ?•ì¥???„ìš”?©ë‹ˆ??")
        elif overall_score < 0.5:
            recommendations.append("?¤ì›Œ???¤ì–‘?±ì„ ê°œì„ ?˜ê¸° ?„í•´ ?˜ë???ê´€ê³„ì? ì»¨í…?¤íŠ¸ ë§¤í•‘???•ì¥?˜ì„¸??")

        # ?„ë©”?¸ë³„ ê¶Œì¥?¬í•­
        for domain, coverage in diversity_analysis["domain_coverage"].items():
            if coverage["overall_coverage"] < 0.3:
                recommendations.append(f"{domain} ?„ë©”?¸ì˜ ?¤ì›Œ??ì»¤ë²„ë¦¬ì?ê°€ ??Šµ?ˆë‹¤. ?´ë‹¹ ?„ë©”???©ì–´ë¥??•ì¥?˜ì„¸??")

        return recommendations

    def save_test_results(self, test_report: Dict[str, Any]):
        """?ŒìŠ¤??ê²°ê³¼ ?€??""
        os.makedirs(self.output_dir, exist_ok=True)

        # ?ŒìŠ¤??ë³´ê³ ???€??
        report_file = os.path.join(self.output_dir, "realistic_query_test_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(test_report, f, ensure_ascii=False, indent=2)

        logger.info(f"?ŒìŠ¤??ê²°ê³¼ ?€???„ë£Œ: {self.output_dir}")

    def run_realistic_test(self):
        """?„ì‹¤?ì¸ ?ŒìŠ¤???¤í–‰"""
        logger.info("?„ì‹¤?ì¸ ë²•ë¥  ì§ˆì˜ ?ŒìŠ¤???œì‘")

        try:
            # ?¤ì›Œ??ê´€?¨ì„± ?ŒìŠ¤??
            relevance_results = self.test_keyword_relevance()

            # ?¤ì›Œ???¤ì–‘??ë°?ì»¤ë²„ë¦¬ì? ?ŒìŠ¤??
            diversity_analysis = self.test_keyword_diversity_and_coverage(relevance_results)

            # ?„ì‹¤?ì¸ ?ŒìŠ¤??ë³´ê³ ???ì„±
            test_report = self.generate_realistic_test_report(relevance_results, diversity_analysis)

            # ê²°ê³¼ ?€??
            self.save_test_results(test_report)

            logger.info("?„ì‹¤?ì¸ ë²•ë¥  ì§ˆì˜ ?ŒìŠ¤???„ë£Œ")

            # ê²°ê³¼ ?”ì•½ ì¶œë ¥
            print(f"\n=== ?„ì‹¤?ì¸ ë²•ë¥  ì§ˆì˜ ?ŒìŠ¤??ê²°ê³¼ ?”ì•½ ===")
            print(f"ì´??ŒìŠ¤??ì§ˆì˜ ?? {test_report['test_summary']['total_queries']}")
            print(f"?‰ê·  ?„ì‹¤??ê´€?¨ì„±: {test_report['performance_summary']['average_realistic_relevance']:.3f}")
            print(f"?‰ê·  ë²•ë¥  ê°œë… ê´€?¨ì„±: {test_report['performance_summary']['average_legal_concept_relevance']:.3f}")
            print(f"?‰ê·  ?„ì²´ ê´€?¨ì„±: {test_report['performance_summary']['average_overall_relevance']:.3f}")
            print(f"?‰ê·  ?ˆì§ˆ ?ìˆ˜: {test_report['performance_summary']['average_quality_score']:.3f}")
            print(f"?‰ê·  ì²˜ë¦¬ ?œê°„: {test_report['performance_summary']['average_processing_time']:.4f}ì´?)

            print(f"\n=== ?„ë©”?¸ë³„ ì»¤ë²„ë¦¬ì? ===")
            for domain, coverage in test_report['diversity_analysis']['domain_coverage'].items():
                print(f"{domain}: {coverage['overall_coverage']:.3f}")

            print(f"\n=== ?¤ì›Œ???¤ì–‘??===")
            diversity_metrics = test_report['diversity_analysis']['overall_metrics']
            print(f"?‰ê·  ?¤ì–‘?? {diversity_metrics['average_diversity']:.3f}")
            print(f"?‰ê·  ?•ì¥ë¥? {diversity_metrics['average_expansion']:.2f}")
            print(f"?‰ê·  ì»¤ë²„ë¦¬ì?: {diversity_metrics['average_coverage']:.3f}")
            print(f"?„ì²´ ?ìˆ˜: {diversity_metrics['overall_score']:.3f}")

            print(f"\n=== ê°œì„  ê¶Œì¥?¬í•­ ===")
            for i, recommendation in enumerate(test_report['recommendations'], 1):
                print(f"{i}. {recommendation}")

            # ?ì„¸ ê²°ê³¼ ?ˆì‹œ ì¶œë ¥
            print(f"\n=== ?ì„¸ ê²°ê³¼ ?ˆì‹œ (ì²?ë²ˆì§¸ ì§ˆì˜) ===")
            first_query = list(relevance_results.values())[0]
            print(f"ì§ˆë¬¸: {first_query['question']}")
            print(f"?„ì‹¤??ê´€?¨ì„±: {first_query['relevance_metrics']['realistic_relevance']:.3f}")
            print(f"ë²•ë¥  ê°œë… ê´€?¨ì„±: {first_query['relevance_metrics']['legal_concept_relevance']:.3f}")
            print(f"?„ì²´ ê´€?¨ì„±: {first_query['relevance_metrics']['overall_relevance']:.3f}")
            print(f"?ˆì§ˆ ?ìˆ˜: {first_query['relevance_metrics']['quality_score']:.3f}")
            print(f"ë§¤ì¹­???„ì‹¤???¤ì›Œ?? {first_query['matched_keywords']['realistic_matched']}")
            print(f"ë§¤ì¹­??ë²•ë¥  ê°œë…: {first_query['matched_keywords']['legal_concept_matched']}")

        except Exception as e:
            logger.error(f"?„ì‹¤?ì¸ ?ŒìŠ¤??ì¤??¤ë¥˜ ë°œìƒ: {e}")
            raise

def main():
    """ë©”ì¸ ?¤í–‰ ?¨ìˆ˜"""
    tester = RealisticQueryTester()
    tester.run_realistic_test()

if __name__ == "__main__":
    main()
