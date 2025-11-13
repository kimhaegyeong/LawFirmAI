#!/usr/bin/env python3
"""
ë²•ë¥  ì§ˆì˜ ?ŒìŠ¤???œìŠ¤??
?•ì¥???¤ì›Œ??ë§¤í•‘ ?œìŠ¤?œì„ ?¤ì œ ë²•ë¥  ì§ˆë¬¸?¼ë¡œ ?ŒìŠ¤?¸í•©?ˆë‹¤.
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

# core/agentsë¥??¬ìš©?˜ë„ë¡?ë³€ê²?
from source.agents.keyword_mapper import EnhancedKeywordMapper, LegalKeywordMapper
from source.agents.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
from source.agents.workflow_service import LangGraphWorkflowService

# ë¡œê¹… ?¤ì •
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/legal_query_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LegalQueryTester:
    """ë²•ë¥  ì§ˆì˜ ?ŒìŠ¤?¸ê¸°"""

    def __init__(self):
        self.test_queries = self._initialize_test_queries()
        self.output_dir = "data/extracted_terms/query_test"

        # ?ŒìŠ¤??ê²°ê³¼ ?€??
        self.test_results = {
            "keyword_mapping_tests": {},
            "workflow_tests": {},
            "performance_metrics": {},
            "quality_assessment": {}
        }

    def _initialize_test_queries(self) -> List[Dict[str, str]]:
        """?ŒìŠ¤??ì§ˆì˜ ì´ˆê¸°??""
        return [
            {
                "question": "ê³„ì•½?œì—???„ì•½ê¸?ì¡°í•­???ˆë¬´ ?’ê²Œ ?¤ì •?˜ì–´ ?ˆëŠ”?? ë²•ì ?¼ë¡œ ë¬¸ì œê°€ ? ê¹Œ??",
                "query_type": "contract_review",
                "domain": "ë¯¼ì‚¬ë²?,
                "expected_keywords": ["ê³„ì•½??, "?„ì•½ê¸?, "ì¡°í•­", "ë²•ì ", "ë¬¸ì œ", "ë¯¼ë²•", "ê³„ì•½ë²?, "?í•´ë°°ìƒ"]
            },
            {
                "question": "êµí†µ?¬ê³ ë¡??¸í•œ ?í•´ë°°ìƒ ì²?µ¬ ???„ìš”??ì¦ê±°?ë£Œ??ë¬´ì—‡?¸ê???",
                "query_type": "damage_compensation",
                "domain": "ë¯¼ì‚¬ë²?,
                "expected_keywords": ["êµí†µ?¬ê³ ", "?í•´ë°°ìƒ", "ì²?µ¬", "ì¦ê±°?ë£Œ", "ë¶ˆë²•?‰ìœ„", "ê³¼ì‹¤", "?¸ê³¼ê´€ê³?]
            },
            {
                "question": "?´í˜¼ ?Œì†¡?ì„œ ?ë? ?‘ìœ¡ê¶Œì„ ê²°ì •?˜ëŠ” ê¸°ì??€ ë¬´ì—‡?¸ê???",
                "query_type": "divorce_proceedings",
                "domain": "ê°€ì¡±ë²•",
                "expected_keywords": ["?´í˜¼", "?Œì†¡", "?ë?", "?‘ìœ¡ê¶?, "ê²°ì •", "ê¸°ì?", "ê°€?•ë²•??, "ê°€ì¡±ë²•"]
            },
            {
                "question": "ë¶€?™ì‚° ë§¤ë§¤ ê³„ì•½ ???±ê¸° ?´ì „ ?ˆì°¨?€ ?„ìš”???œë¥˜??ë¬´ì—‡?¸ê???",
                "query_type": "real_estate_transaction",
                "domain": "ë¶€?™ì‚°ë²?,
                "expected_keywords": ["ë¶€?™ì‚°", "ë§¤ë§¤", "ê³„ì•½", "?±ê¸°", "?´ì „", "?ˆì°¨", "?œë¥˜", "?±ê¸°ë¶€?±ë³¸"]
            },
            {
                "question": "?¹í—ˆ ì¶œì› ??ë°œëª…??? ê·œ?±ê³¼ ì§„ë³´?±ì„ ?´ë–»ê²??…ì¦?´ì•¼ ?˜ë‚˜??",
                "query_type": "patent_application",
                "domain": "?¹í—ˆë²?,
                "expected_keywords": ["?¹í—ˆ", "ì¶œì›", "ë°œëª…", "? ê·œ??, "ì§„ë³´??, "?…ì¦", "?¹í—ˆì²?, "?¹í—ˆë²?]
            },
            {
                "question": "ê·¼ë¡œ?ê? ë¶€?¹í•´ê³ ë? ?¹í–ˆ????êµ¬ì œ ?ˆì°¨???´ë–»ê²??˜ë‚˜??",
                "query_type": "employment_termination",
                "domain": "?¸ë™ë²?,
                "expected_keywords": ["ê·¼ë¡œ??, "ë¶€?¹í•´ê³?, "êµ¬ì œ", "?ˆì°¨", "?¸ë™?„ì›??, "ê·¼ë¡œê¸°ì?ë²?, "?´ê³ "]
            },
            {
                "question": "ì£¼ì‹?Œì‚¬ ?¤ë¦½ ???„ìš”???ë³¸ê¸ˆê³¼ ?±ê¸° ?ˆì°¨??ë¬´ì—‡?¸ê???",
                "query_type": "company_establishment",
                "domain": "?ì‚¬ë²?,
                "expected_keywords": ["ì£¼ì‹?Œì‚¬", "?¤ë¦½", "?ë³¸ê¸?, "?±ê¸°", "?ˆì°¨", "?ë²•", "?Œì‚¬ë²?, "ì£¼ì£¼"]
            },
            {
                "question": "?•ì‚¬ ?¬ê±´?ì„œ ë³€?¸ì‚¬ ? ì„ê¶Œê³¼ ë³€?¸ì‚¬ ë¹„ìš©?€ ?´ë–»ê²??˜ë‚˜??",
                "query_type": "criminal_defense",
                "domain": "?•ì‚¬ë²?,
                "expected_keywords": ["?•ì‚¬", "?¬ê±´", "ë³€?¸ì‚¬", "? ì„ê¶?, "ë¹„ìš©", "?¼ê³ ", "?•ì‚¬?Œì†¡ë²?, "êµ?„ ë³€??]
            },
            {
                "question": "?‰ì •ì²˜ë¶„???€???´ì˜? ì²­ê³??‰ì •?Œì†¡??ì°¨ì´?ì? ë¬´ì—‡?¸ê???",
                "query_type": "administrative_appeal",
                "domain": "?‰ì •ë²?,
                "expected_keywords": ["?‰ì •ì²˜ë¶„", "?´ì˜? ì²­", "?‰ì •?Œì†¡", "ì°¨ì´??, "?‰ì •ë²?, "?ˆê?", "?¹ì¸"]
            },
            {
                "question": "?ì† ?¬ê¸°?€ ?œì •?¹ì¸ ì¤??´ë–¤ ê²ƒì„ ? íƒ?´ì•¼ ? ê¹Œ??",
                "query_type": "inheritance_renunciation",
                "domain": "ê°€ì¡±ë²•",
                "expected_keywords": ["?ì†", "?¬ê¸°", "?œì •?¹ì¸", "? íƒ", "?ì†??, "?ì†ë¶?, "?ì†ë²?]
            }
        ]

    def test_keyword_mapping(self) -> Dict[str, Any]:
        """?¤ì›Œ??ë§¤í•‘ ?ŒìŠ¤??""
        logger.info("?¤ì›Œ??ë§¤í•‘ ?ŒìŠ¤???œì‘")

        enhanced_mapper = EnhancedKeywordMapper()
        results = {}

        for i, query in enumerate(self.test_queries):
            question = query["question"]
            query_type = query["query_type"]
            expected_keywords = query["expected_keywords"]

            logger.info(f"?ŒìŠ¤??{i+1}/{len(self.test_queries)}: {query_type}")

            start_time = time.time()

            # ì¢…í•©?ì¸ ?¤ì›Œ??ë§¤í•‘
            comprehensive_result = enhanced_mapper.get_comprehensive_keyword_mapping(question, query_type)

            end_time = time.time()

            # ?ˆìƒ ?¤ì›Œ?œì???ë§¤ì¹­ë¥?ê³„ì‚°
            all_keywords = comprehensive_result.get("all_keywords", [])
            matched_keywords = [kw for kw in expected_keywords if kw in all_keywords]
            match_rate = len(matched_keywords) / len(expected_keywords) if expected_keywords else 0

            results[query_type] = {
                "question": question,
                "expected_keywords": expected_keywords,
                "extracted_keywords": all_keywords,
                "matched_keywords": matched_keywords,
                "match_rate": match_rate,
                "processing_time": end_time - start_time,
                "comprehensive_result": comprehensive_result
            }

        logger.info("?¤ì›Œ??ë§¤í•‘ ?ŒìŠ¤???„ë£Œ")
        return results

    def test_workflow_integration(self) -> Dict[str, Any]:
        """?Œí¬?Œë¡œ???µí•© ?ŒìŠ¤??""
        logger.info("?Œí¬?Œë¡œ???µí•© ?ŒìŠ¤???œì‘")

        try:
            # ?Œí¬?Œë¡œ???œë¹„??ì´ˆê¸°??
            workflow_service = LangGraphWorkflowService()

            results = {}

            for i, query in enumerate(self.test_queries[:3]):  # ì²˜ìŒ 3ê°œë§Œ ?ŒìŠ¤??(?œê°„ ?ˆì•½)
                question = query["question"]
                query_type = query["query_type"]

                logger.info(f"?Œí¬?Œë¡œ???ŒìŠ¤??{i+1}/3: {query_type}")

                start_time = time.time()

                try:
                    # ?Œí¬?Œë¡œ???¤í–‰
                    response = workflow_service.process_question(question, query_type)

                    end_time = time.time()

                    results[query_type] = {
                        "question": question,
                        "response": response,
                        "processing_time": end_time - start_time,
                        "success": True
                    }

                except Exception as e:
                    logger.error(f"?Œí¬?Œë¡œ???¤í–‰ ?¤ë¥˜ ({query_type}): {e}")
                    results[query_type] = {
                        "question": question,
                        "error": str(e),
                        "success": False
                    }

            logger.info("?Œí¬?Œë¡œ???µí•© ?ŒìŠ¤???„ë£Œ")
            return results

        except Exception as e:
            logger.error(f"?Œí¬?Œë¡œ???œë¹„??ì´ˆê¸°???¤ë¥˜: {e}")
            return {"error": str(e)}

    def analyze_keyword_quality(self, mapping_results: Dict[str, Any]) -> Dict[str, Any]:
        """?¤ì›Œ???ˆì§ˆ ë¶„ì„"""
        logger.info("?¤ì›Œ???ˆì§ˆ ë¶„ì„ ?œì‘")

        quality_metrics = {
            "overall_match_rate": 0,
            "domain_coverage": {},
            "keyword_diversity": {},
            "processing_efficiency": {},
            "recommendations": []
        }

        total_match_rate = 0
        total_queries = len(mapping_results)

        for query_type, result in mapping_results.items():
            match_rate = result["match_rate"]
            total_match_rate += match_rate

            # ?„ë©”?¸ë³„ ì»¤ë²„ë¦¬ì?
            domain = next((q["domain"] for q in self.test_queries if q["query_type"] == query_type), "ê¸°í?")
            if domain not in quality_metrics["domain_coverage"]:
                quality_metrics["domain_coverage"][domain] = []
            quality_metrics["domain_coverage"][domain].append(match_rate)

            # ?¤ì›Œ???¤ì–‘??
            extracted_count = len(result["extracted_keywords"])
            expected_count = len(result["expected_keywords"])
            diversity_ratio = extracted_count / expected_count if expected_count > 0 else 0

            quality_metrics["keyword_diversity"][query_type] = {
                "extracted_count": extracted_count,
                "expected_count": expected_count,
                "diversity_ratio": diversity_ratio
            }

            # ì²˜ë¦¬ ?¨ìœ¨??
            quality_metrics["processing_efficiency"][query_type] = {
                "processing_time": result["processing_time"],
                "keywords_per_second": extracted_count / result["processing_time"] if result["processing_time"] > 0 else 0
            }

        # ?„ì²´ ë§¤ì¹­ë¥?
        quality_metrics["overall_match_rate"] = total_match_rate / total_queries

        # ?„ë©”?¸ë³„ ?‰ê·  ë§¤ì¹­ë¥?
        for domain, rates in quality_metrics["domain_coverage"].items():
            quality_metrics["domain_coverage"][domain] = sum(rates) / len(rates)

        # ê°œì„  ê¶Œì¥?¬í•­ ?ì„±
        if quality_metrics["overall_match_rate"] < 0.5:
            quality_metrics["recommendations"].append("?„ì²´ ?¤ì›Œ??ë§¤ì¹­ë¥ ì´ ??Šµ?ˆë‹¤. ?¤ì›Œ??ë§¤í•‘ ?„ëµ???¬ê?? í•˜?¸ìš”.")

        if quality_metrics["overall_match_rate"] > 0.8:
            quality_metrics["recommendations"].append("?¤ì›Œ??ë§¤ì¹­ë¥ ì´ ?°ìˆ˜?©ë‹ˆ?? ?„ì¬ ?¤ì •??? ì??˜ì„¸??")

        # ?„ë©”?¸ë³„ ê¶Œì¥?¬í•­
        for domain, rate in quality_metrics["domain_coverage"].items():
            if rate < 0.4:
                quality_metrics["recommendations"].append(f"{domain} ?„ë©”?¸ì˜ ?¤ì›Œ??ë§¤ì¹­ë¥ ì´ ??Šµ?ˆë‹¤. ?´ë‹¹ ?„ë©”???©ì–´ë¥??•ì¥?˜ì„¸??")

        logger.info("?¤ì›Œ???ˆì§ˆ ë¶„ì„ ?„ë£Œ")
        return quality_metrics

    def generate_test_report(self, mapping_results: Dict[str, Any], workflow_results: Dict[str, Any], quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """?ŒìŠ¤??ë³´ê³ ???ì„±"""
        logger.info("?ŒìŠ¤??ë³´ê³ ???ì„± ì¤?)

        report = {
            "test_summary": {
                "test_date": datetime.now().isoformat(),
                "total_queries": len(self.test_queries),
                "successful_mappings": len([r for r in mapping_results.values() if r.get("match_rate", 0) > 0]),
                "successful_workflows": len([r for r in workflow_results.values() if r.get("success", False)])
            },
            "keyword_mapping_results": {
                "average_match_rate": quality_metrics["overall_match_rate"],
                "domain_performance": quality_metrics["domain_coverage"],
                "processing_efficiency": quality_metrics["processing_efficiency"]
            },
            "workflow_integration_results": workflow_results,
            "quality_assessment": quality_metrics,
            "detailed_results": mapping_results,
            "recommendations": quality_metrics["recommendations"]
        }

        logger.info("?ŒìŠ¤??ë³´ê³ ???ì„± ?„ë£Œ")
        return report

    def save_test_results(self, test_report: Dict[str, Any]):
        """?ŒìŠ¤??ê²°ê³¼ ?€??""
        os.makedirs(self.output_dir, exist_ok=True)

        # ?ŒìŠ¤??ë³´ê³ ???€??
        report_file = os.path.join(self.output_dir, "query_test_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(test_report, f, ensure_ascii=False, indent=2)

        logger.info(f"?ŒìŠ¤??ê²°ê³¼ ?€???„ë£Œ: {self.output_dir}")

    def run_query_test(self):
        """ì§ˆì˜ ?ŒìŠ¤???¤í–‰"""
        logger.info("ë²•ë¥  ì§ˆì˜ ?ŒìŠ¤???œì‘")

        try:
            # ?¤ì›Œ??ë§¤í•‘ ?ŒìŠ¤??
            mapping_results = self.test_keyword_mapping()

            # ?Œí¬?Œë¡œ???µí•© ?ŒìŠ¤??
            workflow_results = self.test_workflow_integration()

            # ?¤ì›Œ???ˆì§ˆ ë¶„ì„
            quality_metrics = self.analyze_keyword_quality(mapping_results)

            # ?ŒìŠ¤??ë³´ê³ ???ì„±
            test_report = self.generate_test_report(mapping_results, workflow_results, quality_metrics)

            # ê²°ê³¼ ?€??
            self.save_test_results(test_report)

            logger.info("ë²•ë¥  ì§ˆì˜ ?ŒìŠ¤???„ë£Œ")

            # ê²°ê³¼ ?”ì•½ ì¶œë ¥
            print(f"\n=== ë²•ë¥  ì§ˆì˜ ?ŒìŠ¤??ê²°ê³¼ ?”ì•½ ===")
            print(f"ì´??ŒìŠ¤??ì§ˆì˜ ?? {test_report['test_summary']['total_queries']}")
            print(f"?±ê³µ?ì¸ ?¤ì›Œ??ë§¤í•‘: {test_report['test_summary']['successful_mappings']}")
            print(f"?±ê³µ?ì¸ ?Œí¬?Œë¡œ?? {test_report['test_summary']['successful_workflows']}")
            print(f"?‰ê·  ?¤ì›Œ??ë§¤ì¹­ë¥? {test_report['keyword_mapping_results']['average_match_rate']:.3f}")

            print(f"\n=== ?„ë©”?¸ë³„ ?±ëŠ¥ ===")
            for domain, rate in test_report['keyword_mapping_results']['domain_performance'].items():
                print(f"{domain}: {rate:.3f}")

            print(f"\n=== ê°œì„  ê¶Œì¥?¬í•­ ===")
            for i, recommendation in enumerate(test_report['recommendations'], 1):
                print(f"{i}. {recommendation}")

            # ?ì„¸ ê²°ê³¼ ?ˆì‹œ ì¶œë ¥
            print(f"\n=== ?ì„¸ ê²°ê³¼ ?ˆì‹œ (ì²?ë²ˆì§¸ ì§ˆì˜) ===")
            first_query = list(mapping_results.values())[0]
            print(f"ì§ˆë¬¸: {first_query['question']}")
            print(f"?ˆìƒ ?¤ì›Œ?? {first_query['expected_keywords']}")
            print(f"ì¶”ì¶œ???¤ì›Œ?? {first_query['extracted_keywords'][:10]}...")  # ì²˜ìŒ 10ê°œë§Œ
            print(f"ë§¤ì¹­???¤ì›Œ?? {first_query['matched_keywords']}")
            print(f"ë§¤ì¹­ë¥? {first_query['match_rate']:.3f}")

        except Exception as e:
            logger.error(f"ì§ˆì˜ ?ŒìŠ¤??ì¤??¤ë¥˜ ë°œìƒ: {e}")
            raise

def main():
    """ë©”ì¸ ?¤í–‰ ?¨ìˆ˜"""
    tester = LegalQueryTester()
    tester.run_query_test()

if __name__ == "__main__":
    main()
