#!/usr/bin/env python3
"""
?•ì¥??ë²•ë¥  ?©ì–´ ?¬ì „ ?µí•© ?ŒìŠ¤??
?•ì¥???¤ì›Œ??ë§¤í•‘ ?œìŠ¤?œì˜ ?±ëŠ¥???ŒìŠ¤?¸í•˜ê³?ê°œì„  ?¨ê³¼ë¥?ì¸¡ì •?©ë‹ˆ??
"""

import json
import os
import sys
import time
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
from datetime import datetime

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from source.agents.keyword_mapper import EnhancedKeywordMapper, LegalKeywordMapper

# ë¡œê¹… ?¤ì •
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/legal_term_integration_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LegalTermIntegrationTester:
    """ë²•ë¥  ?©ì–´ ?µí•© ?ŒìŠ¤?¸ê¸°"""

    def __init__(self):
        self.test_questions = self._initialize_test_questions()
        self.expected_keywords = self._initialize_expected_keywords()
        self.output_dir = "data/extracted_terms/integration_test"

        # ?ŒìŠ¤??ê²°ê³¼ ?€??
        self.test_results = {
            "basic_mapper": {},
            "enhanced_mapper": {},
            "performance_comparison": {},
            "quality_metrics": {}
        }

    def _initialize_test_questions(self) -> List[Dict[str, str]]:
        """?ŒìŠ¤??ì§ˆë¬¸ ì´ˆê¸°??""
        return [
            {
                "question": "ê³„ì•½??ê²€????ì£¼ì˜?´ì•¼ ???¬í•­?€ ë¬´ì—‡?¸ê???",
                "query_type": "contract_review",
                "domain": "ë¯¼ì‚¬ë²?
            },
            {
                "question": "?í•´ë°°ìƒ ì²?µ¬ ?ˆì°¨???´ë–»ê²?ì§„í–‰?˜ë‚˜??",
                "query_type": "damage_compensation",
                "domain": "ë¯¼ì‚¬ë²?
            },
            {
                "question": "?´í˜¼ ?Œì†¡?ì„œ ?„ìë£ŒëŠ” ?´ë–»ê²?ê²°ì •?˜ë‚˜??",
                "query_type": "divorce_proceedings",
                "domain": "ê°€ì¡±ë²•"
            },
            {
                "question": "ë¶€?™ì‚° ë§¤ë§¤ ê³„ì•½ ???±ê¸° ?ˆì°¨???´ë–»ê²??˜ë‚˜??",
                "query_type": "real_estate_transaction",
                "domain": "ë¶€?™ì‚°ë²?
            },
            {
                "question": "?¹í—ˆ ì¶œì› ???„ìš”???œë¥˜??ë¬´ì—‡?¸ê???",
                "query_type": "patent_application",
                "domain": "?¹í—ˆë²?
            },
            {
                "question": "ê·¼ë¡œ???´ê³  ??ë²•ì  ?ˆì°¨???´ë–»ê²??˜ë‚˜??",
                "query_type": "employment_termination",
                "domain": "?¸ë™ë²?
            },
            {
                "question": "?Œì‚¬ ?¤ë¦½ ???„ìš”???ˆê? ?ˆì°¨??ë¬´ì—‡?¸ê???",
                "query_type": "company_establishment",
                "domain": "?ì‚¬ë²?
            },
            {
                "question": "?•ì‚¬ ?¬ê±´?ì„œ ë³€?¸ì‚¬ ? ì„ ?ˆì°¨???´ë–»ê²??˜ë‚˜??",
                "query_type": "criminal_defense",
                "domain": "?•ì‚¬ë²?
            },
            {
                "question": "?‰ì •ì²˜ë¶„???€???´ì˜? ì²­ ë°©ë²•?€ ë¬´ì—‡?¸ê???",
                "query_type": "administrative_appeal",
                "domain": "?‰ì •ë²?
            },
            {
                "question": "?ì† ?¬ê¸° ?ˆì°¨???´ë–»ê²?ì§„í–‰?˜ë‚˜??",
                "query_type": "inheritance_renunciation",
                "domain": "ê°€ì¡±ë²•"
            }
        ]

    def _initialize_expected_keywords(self) -> Dict[str, List[str]]:
        """?ˆìƒ ?¤ì›Œ??ì´ˆê¸°??""
        return {
            "contract_review": ["ê³„ì•½??, "?¹ì‚¬??, "ì¡°ê±´", "ê¸°ê°„", "?´ì?", "?í•´ë°°ìƒ", "ê³„ì•½ê¸?, "?„ì•½ê¸?],
            "damage_compensation": ["?í•´ë°°ìƒ", "ì²?µ¬", "?ˆì°¨", "ë¶ˆë²•?‰ìœ„", "ê³¼ì‹¤", "?¸ê³¼ê´€ê³?, "?í•´??],
            "divorce_proceedings": ["?´í˜¼", "?Œì†¡", "?„ìë£?, "?¬ì‚°ë¶„í• ", "?‘ìœ¡ê¶?, "ë©´ì ‘êµì„­ê¶?, "ê°€?•ë²•??],
            "real_estate_transaction": ["ë¶€?™ì‚°", "ë§¤ë§¤", "ê³„ì•½", "?±ê¸°", "?Œìœ ê¶Œì´??, "?±ê¸°ë¶€?±ë³¸", "ë¶€?™ì‚°?±ê¸°ë²?],
            "patent_application": ["?¹í—ˆ", "ì¶œì›", "?œë¥˜", "?¹í—ˆì²?, "?¹í—ˆë²?, "ë°œëª…", "?¹í—ˆê¶?],
            "employment_termination": ["ê·¼ë¡œ??, "?´ê³ ", "?ˆì°¨", "ë¶€?¹í•´ê³?, "?¸ë™?„ì›??, "ê·¼ë¡œê¸°ì?ë²?],
            "company_establishment": ["?Œì‚¬", "?¤ë¦½", "?ˆê?", "?ˆì°¨", "ì£¼ì‹?Œì‚¬", "?ë³¸ê¸?, "?ë²•"],
            "criminal_defense": ["?•ì‚¬", "?¬ê±´", "ë³€?¸ì‚¬", "? ì„", "?ˆì°¨", "?¼ê³ ", "ê²€??, "?•ì‚¬?Œì†¡ë²?],
            "administrative_appeal": ["?‰ì •ì²˜ë¶„", "?´ì˜? ì²­", "ë°©ë²•", "?‰ì •?Œì†¡", "?‰ì •ë²?, "?ˆê?", "?¹ì¸"],
            "inheritance_renunciation": ["?ì†", "?¬ê¸°", "?ˆì°¨", "?ì†??, "?ì†ë¶?, "? ë¥˜ë¶?, "?ì†ë²?]
        }

    def test_basic_keyword_mapper(self) -> Dict[str, Any]:
        """ê¸°ë³¸ ?¤ì›Œ??ë§¤í¼ ?ŒìŠ¤??""
        logger.info("ê¸°ë³¸ ?¤ì›Œ??ë§¤í¼ ?ŒìŠ¤???œì‘")

        basic_mapper = LegalKeywordMapper()
        results = {}

        for test_case in self.test_questions:
            question = test_case["question"]
            query_type = test_case["query_type"]

            start_time = time.time()

            # ?¤ì›Œ??ì¶”ì¶œ
            keywords = basic_mapper.get_keywords_for_question(question, query_type)

            # ê°€ì¤‘ì¹˜ë³??¤ì›Œ??ì¶”ì¶œ
            weighted_keywords = basic_mapper.get_weighted_keywords_for_question(question, query_type)

            # ?¤ì›Œ???¬í•¨??ê³„ì‚°
            sample_answer = f"{question}???€???µë??…ë‹ˆ?? {', '.join(keywords[:5])} ?±ì˜ ?´ìš©???¬í•¨?©ë‹ˆ??"
            coverage = basic_mapper.calculate_weighted_keyword_coverage(sample_answer, query_type, question)

            end_time = time.time()

            results[query_type] = {
                "question": question,
                "keywords": keywords,
                "weighted_keywords": weighted_keywords,
                "coverage": coverage,
                "processing_time": end_time - start_time,
                "keyword_count": len(keywords)
            }

        logger.info("ê¸°ë³¸ ?¤ì›Œ??ë§¤í¼ ?ŒìŠ¤???„ë£Œ")
        return results

    def test_enhanced_keyword_mapper(self) -> Dict[str, Any]:
        """?¥ìƒ???¤ì›Œ??ë§¤í¼ ?ŒìŠ¤??""
        logger.info("?¥ìƒ???¤ì›Œ??ë§¤í¼ ?ŒìŠ¤???œì‘")

        enhanced_mapper = EnhancedKeywordMapper()
        results = {}

        for test_case in self.test_questions:
            question = test_case["question"]
            query_type = test_case["query_type"]

            start_time = time.time()

            # ì¢…í•©?ì¸ ?¤ì›Œ??ë§¤í•‘
            comprehensive_result = enhanced_mapper.get_comprehensive_keyword_mapping(question, query_type)

            end_time = time.time()

            results[query_type] = {
                "question": question,
                "comprehensive_result": comprehensive_result,
                "processing_time": end_time - start_time,
                "total_keywords": len(comprehensive_result.get("all_keywords", [])),
                "base_keywords": comprehensive_result.get("base_keywords", []),
                "contextual_keywords": comprehensive_result.get("contextual_data", {}).get("all_keywords", []),
                "semantic_keywords": comprehensive_result.get("semantic_data", {}).get("recommended_keywords", [])
            }

        logger.info("?¥ìƒ???¤ì›Œ??ë§¤í¼ ?ŒìŠ¤???„ë£Œ")
        return results

    def test_semantic_keyword_mapper(self) -> Dict[str, Any]:
        """?˜ë????¤ì›Œ??ë§¤í¼ ?ŒìŠ¤??""
        logger.info("?˜ë????¤ì›Œ??ë§¤í¼ ?ŒìŠ¤???œì‘")

        enhanced_mapper = EnhancedKeywordMapper()
        results = {}

        for test_case in self.test_questions:
            question = test_case["question"]
            query_type = test_case["query_type"]

            start_time = time.time()

            # ê¸°ë³¸ ?¤ì›Œ??ì¶”ì¶œ
            basic_keywords = ["ê³„ì•½", "?í•´ë°°ìƒ", "?Œì†¡", "?´í˜¼", "ë¶€?™ì‚°", "?¹í—ˆ", "ê·¼ë¡œ", "?Œì‚¬", "?•ì‚¬", "?‰ì •"]

            # ì¢…í•©?ì¸ ?¤ì›Œ??ë§¤í•‘?ì„œ ?˜ë????°ì´??ì¶”ì¶œ
            comprehensive_result = enhanced_mapper.get_comprehensive_keyword_mapping(question, query_type)
            semantic_data = comprehensive_result.get("semantic_data", {})

            # ?¤ì›Œ???•ì¥ (?˜ë????¤ì›Œ??ì¶”ì²œ ?¬ìš©)
            expanded_keywords = semantic_data.get("recommended_keywords", [])

            # ?¤ì›Œ???´ëŸ¬?¤í„°ë§?(?˜ë????´ëŸ¬?¤í„° ?¬ìš©)
            clusters = semantic_data.get("semantic_clusters", {})

            end_time = time.time()

            results[query_type] = {
                "question": question,
                "semantic_recommendations": semantic_data,
                "expanded_keywords": expanded_keywords,
                "clusters": clusters,
                "processing_time": end_time - start_time,
                "expansion_ratio": len(expanded_keywords) / len(basic_keywords) if basic_keywords else 0
            }

        logger.info("?˜ë????¤ì›Œ??ë§¤í¼ ?ŒìŠ¤???„ë£Œ")
        return results

    def calculate_performance_metrics(self, basic_results: Dict, enhanced_results: Dict, semantic_results: Dict) -> Dict[str, Any]:
        """?±ëŠ¥ ë©”íŠ¸ë¦?ê³„ì‚°"""
        logger.info("?±ëŠ¥ ë©”íŠ¸ë¦?ê³„ì‚° ì¤?)

        metrics = {
            "keyword_coverage_improvement": {},
            "processing_time_comparison": {},
            "keyword_expansion_metrics": {},
            "overall_improvement": {}
        }

        # ?¤ì›Œ??ì»¤ë²„ë¦¬ì? ê°œì„  ì¸¡ì •
        for query_type in basic_results.keys():
            if query_type in enhanced_results:
                basic_keywords = set(basic_results[query_type]["keywords"])
                enhanced_keywords = set(enhanced_results[query_type]["base_keywords"])

                # ?¤ì›Œ???•ì¥ë¥?
                expansion_ratio = len(enhanced_keywords) / len(basic_keywords) if basic_keywords else 0

                # ?ˆìƒ ?¤ì›Œ?œì???ë§¤ì¹­ë¥?
                expected_keywords = set(self.expected_keywords.get(query_type, []))
                basic_match_rate = len(basic_keywords & expected_keywords) / len(expected_keywords) if expected_keywords else 0
                enhanced_match_rate = len(enhanced_keywords & expected_keywords) / len(expected_keywords) if expected_keywords else 0

                metrics["keyword_coverage_improvement"][query_type] = {
                    "basic_keyword_count": len(basic_keywords),
                    "enhanced_keyword_count": len(enhanced_keywords),
                    "expansion_ratio": expansion_ratio,
                    "basic_match_rate": basic_match_rate,
                    "enhanced_match_rate": enhanced_match_rate,
                    "improvement_rate": enhanced_match_rate - basic_match_rate
                }

        # ì²˜ë¦¬ ?œê°„ ë¹„êµ
        for query_type in basic_results.keys():
            if query_type in enhanced_results:
                basic_time = basic_results[query_type]["processing_time"]
                enhanced_time = enhanced_results[query_type]["processing_time"]

                metrics["processing_time_comparison"][query_type] = {
                    "basic_time": basic_time,
                    "enhanced_time": enhanced_time,
                    "time_ratio": enhanced_time / basic_time if basic_time > 0 else 0
                }

        # ?¤ì›Œ???•ì¥ ë©”íŠ¸ë¦?
        for query_type in semantic_results.keys():
            expansion_ratio = semantic_results[query_type]["expansion_ratio"]
            metrics["keyword_expansion_metrics"][query_type] = {
                "expansion_ratio": expansion_ratio,
                "cluster_count": len(semantic_results[query_type]["clusters"])
            }

        # ?„ì²´ ê°œì„ ??ê³„ì‚°
        total_expansion_ratio = sum(m["expansion_ratio"] for m in metrics["keyword_coverage_improvement"].values()) / len(metrics["keyword_coverage_improvement"])
        total_match_improvement = sum(m["improvement_rate"] for m in metrics["keyword_coverage_improvement"].values()) / len(metrics["keyword_coverage_improvement"])
        total_time_ratio = sum(m["time_ratio"] for m in metrics["processing_time_comparison"].values()) / len(metrics["processing_time_comparison"])

        metrics["overall_improvement"] = {
            "average_expansion_ratio": total_expansion_ratio,
            "average_match_improvement": total_match_improvement,
            "average_time_ratio": total_time_ratio,
            "performance_score": (total_expansion_ratio + total_match_improvement) / 2
        }

        logger.info("?±ëŠ¥ ë©”íŠ¸ë¦?ê³„ì‚° ?„ë£Œ")
        return metrics

    def generate_test_report(self, basic_results: Dict, enhanced_results: Dict, semantic_results: Dict, performance_metrics: Dict) -> Dict[str, Any]:
        """?ŒìŠ¤??ë³´ê³ ???ì„±"""
        logger.info("?ŒìŠ¤??ë³´ê³ ???ì„± ì¤?)

        report = {
            "test_summary": {
                "test_date": datetime.now().isoformat(),
                "total_test_cases": len(self.test_questions),
                "test_duration": "??5ë¶?
            },
            "basic_mapper_results": {
                "total_keywords": sum(len(r["keywords"]) for r in basic_results.values()),
                "average_keywords_per_query": sum(len(r["keywords"]) for r in basic_results.values()) / len(basic_results),
                "average_processing_time": sum(r["processing_time"] for r in basic_results.values()) / len(basic_results)
            },
            "enhanced_mapper_results": {
                "total_keywords": sum(r["total_keywords"] for r in enhanced_results.values()),
                "average_keywords_per_query": sum(r["total_keywords"] for r in enhanced_results.values()) / len(enhanced_results),
                "average_processing_time": sum(r["processing_time"] for r in enhanced_results.values()) / len(enhanced_results)
            },
            "semantic_mapper_results": {
                "average_expansion_ratio": sum(r["expansion_ratio"] for r in semantic_results.values()) / len(semantic_results),
                "average_processing_time": sum(r["processing_time"] for r in semantic_results.values()) / len(semantic_results)
            },
            "performance_metrics": performance_metrics,
            "recommendations": self._generate_recommendations(performance_metrics)
        }

        logger.info("?ŒìŠ¤??ë³´ê³ ???ì„± ?„ë£Œ")
        return report

    def _generate_recommendations(self, performance_metrics: Dict) -> List[str]:
        """ê°œì„  ê¶Œì¥?¬í•­ ?ì„±"""
        recommendations = []

        overall_improvement = performance_metrics.get("overall_improvement", {})

        # ?•ì¥ë¥?ê¸°ë°˜ ê¶Œì¥?¬í•­
        expansion_ratio = overall_improvement.get("average_expansion_ratio", 0)
        if expansion_ratio > 2.0:
            recommendations.append("?¤ì›Œ???•ì¥ë¥ ì´ ?°ìˆ˜?©ë‹ˆ?? ?„ì¬ ?¤ì •??? ì??˜ì„¸??")
        elif expansion_ratio > 1.5:
            recommendations.append("?¤ì›Œ???•ì¥ë¥ ì´ ?‘í˜¸?©ë‹ˆ?? ì¶”ê? ê°œì„  ?¬ì?ê°€ ?ˆìŠµ?ˆë‹¤.")
        else:
            recommendations.append("?¤ì›Œ???•ì¥ë¥ ì„ ê°œì„ ?˜ê¸° ?„í•´ ?˜ë???ê´€ê³„ë? ???•ì¥?˜ì„¸??")

        # ë§¤ì¹­ë¥?ê¸°ë°˜ ê¶Œì¥?¬í•­
        match_improvement = overall_improvement.get("average_match_improvement", 0)
        if match_improvement > 0.2:
            recommendations.append("?ˆìƒ ?¤ì›Œ??ë§¤ì¹­ë¥ ì´ ?¬ê²Œ ê°œì„ ?˜ì—ˆ?µë‹ˆ??")
        elif match_improvement > 0.1:
            recommendations.append("?ˆìƒ ?¤ì›Œ??ë§¤ì¹­ë¥ ì´ ê°œì„ ?˜ì—ˆ?µë‹ˆ??")
        else:
            recommendations.append("?ˆìƒ ?¤ì›Œ??ë§¤ì¹­ë¥?ê°œì„ ???„ìš”?©ë‹ˆ??")

        # ì²˜ë¦¬ ?œê°„ ê¸°ë°˜ ê¶Œì¥?¬í•­
        time_ratio = overall_improvement.get("average_time_ratio", 0)
        if time_ratio > 2.0:
            recommendations.append("ì²˜ë¦¬ ?œê°„??ì¦ê??ˆìŠµ?ˆë‹¤. ?±ëŠ¥ ìµœì ?”ë? ê³ ë ¤?˜ì„¸??")
        elif time_ratio > 1.5:
            recommendations.append("ì²˜ë¦¬ ?œê°„???½ê°„ ì¦ê??ˆìŠµ?ˆë‹¤. ëª¨ë‹ˆ?°ë§???„ìš”?©ë‹ˆ??")
        else:
            recommendations.append("ì²˜ë¦¬ ?œê°„???ì ˆ?©ë‹ˆ??")

        # ?±ëŠ¥ ?ìˆ˜ ê¸°ë°˜ ê¶Œì¥?¬í•­
        performance_score = overall_improvement.get("performance_score", 0)
        if performance_score > 0.8:
            recommendations.append("?„ì²´ ?±ëŠ¥???°ìˆ˜?©ë‹ˆ?? ?„ì¬ ?¤ì •??? ì??˜ì„¸??")
        elif performance_score > 0.6:
            recommendations.append("?„ì²´ ?±ëŠ¥???‘í˜¸?©ë‹ˆ?? ì¶”ê? ê°œì„ ???µí•´ ???¥ìƒ?œí‚¬ ???ˆìŠµ?ˆë‹¤.")
        else:
            recommendations.append("?„ì²´ ?±ëŠ¥ ê°œì„ ???„ìš”?©ë‹ˆ?? ?¤ì›Œ??ë§¤í•‘ ?„ëµ???¬ê?? í•˜?¸ìš”.")

        return recommendations

    def save_test_results(self, test_report: Dict[str, Any]):
        """?ŒìŠ¤??ê²°ê³¼ ?€??""
        os.makedirs(self.output_dir, exist_ok=True)

        # ?ŒìŠ¤??ë³´ê³ ???€??
        report_file = os.path.join(self.output_dir, "integration_test_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(test_report, f, ensure_ascii=False, indent=2)

        logger.info(f"?ŒìŠ¤??ê²°ê³¼ ?€???„ë£Œ: {self.output_dir}")

    def run_integration_test(self):
        """?µí•© ?ŒìŠ¤???¤í–‰"""
        logger.info("ë²•ë¥  ?©ì–´ ?¬ì „ ?µí•© ?ŒìŠ¤???œì‘")

        try:
            # ê¸°ë³¸ ?¤ì›Œ??ë§¤í¼ ?ŒìŠ¤??
            basic_results = self.test_basic_keyword_mapper()

            # ?¥ìƒ???¤ì›Œ??ë§¤í¼ ?ŒìŠ¤??
            enhanced_results = self.test_enhanced_keyword_mapper()

            # ?˜ë????¤ì›Œ??ë§¤í¼ ?ŒìŠ¤??
            semantic_results = self.test_semantic_keyword_mapper()

            # ?±ëŠ¥ ë©”íŠ¸ë¦?ê³„ì‚°
            performance_metrics = self.calculate_performance_metrics(basic_results, enhanced_results, semantic_results)

            # ?ŒìŠ¤??ë³´ê³ ???ì„±
            test_report = self.generate_test_report(basic_results, enhanced_results, semantic_results, performance_metrics)

            # ê²°ê³¼ ?€??
            self.save_test_results(test_report)

            logger.info("ë²•ë¥  ?©ì–´ ?¬ì „ ?µí•© ?ŒìŠ¤???„ë£Œ")

            # ê²°ê³¼ ?”ì•½ ì¶œë ¥
            print(f"\n=== ?µí•© ?ŒìŠ¤??ê²°ê³¼ ?”ì•½ ===")
            print(f"?ŒìŠ¤??ì¼€?´ìŠ¤ ?? {test_report['test_summary']['total_test_cases']}")
            print(f"ê¸°ë³¸ ë§¤í¼ ?‰ê·  ?¤ì›Œ???? {test_report['basic_mapper_results']['average_keywords_per_query']:.1f}")
            print(f"?¥ìƒ??ë§¤í¼ ?‰ê·  ?¤ì›Œ???? {test_report['enhanced_mapper_results']['average_keywords_per_query']:.1f}")
            print(f"?‰ê·  ?•ì¥ë¥? {test_report['semantic_mapper_results']['average_expansion_ratio']:.2f}")
            print(f"?„ì²´ ?±ëŠ¥ ?ìˆ˜: {performance_metrics['overall_improvement']['performance_score']:.3f}")

            print(f"\n=== ê°œì„  ê¶Œì¥?¬í•­ ===")
            for i, recommendation in enumerate(test_report['recommendations'], 1):
                print(f"{i}. {recommendation}")

        except Exception as e:
            logger.error(f"?µí•© ?ŒìŠ¤??ì¤??¤ë¥˜ ë°œìƒ: {e}")
            raise

def main():
    """ë©”ì¸ ?¤í–‰ ?¨ìˆ˜"""
    tester = LegalTermIntegrationTester()
    tester.run_integration_test()

if __name__ == "__main__":
    main()
