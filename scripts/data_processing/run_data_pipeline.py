#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?µÌï© ?∞Ïù¥???åÏù¥?ÑÎùº???§Ìñâ ?§ÌÅ¨Î¶ΩÌä∏

?∞Ïù¥???òÏßëÍ≥?Î≤°ÌÑ∞DB Íµ¨Ï∂ï???µÌï©?òÏó¨ ?§Ìñâ?????àÎäî ?§ÌÅ¨Î¶ΩÌä∏?ÖÎãà??
--mode ?∏ÏûêÎ•??µÌï¥ collect, build, full Î™®ÎìúÎ•??†ÌÉù?????àÏäµ?àÎã§.

?¨Ïö©Î≤?
    python scripts/run_data_pipeline.py --mode full --oc your_email_id
    python scripts/run_data_pipeline.py --mode collect --oc your_email_id
    python scripts/run_data_pipeline.py --mode build
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import logging

# ?ÑÎ°ú?ùÌä∏ Î£®Ìä∏ ?îÎ†â?†Î¶¨Î•?Python Í≤ΩÎ°ú??Ï∂îÍ?
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.collect_data_only import DataCollector
from scripts.build_vector_db import VectorDBBuilder
from source.utils.logger import get_logger

logger = get_logger(__name__)


class DataPipeline:
    """?µÌï© ?∞Ïù¥???åÏù¥?ÑÎùº???¥Îûò??""
    
    def __init__(self, oc: str = None):
        """Ï¥àÍ∏∞??""
        self.oc = oc
        self.collector = DataCollector(oc) if oc else None
        self.builder = VectorDBBuilder()
        
        # ?åÏù¥?ÑÎùº???µÍ≥Ñ
        self.pipeline_stats = {
            'start_time': datetime.now().isoformat(),
            'mode': None,
            'collect_success': False,
            'build_success': False,
            'total_duration': 0,
            'errors': []
        }
    
    def run_collect_mode(self) -> bool:
        """?∞Ïù¥???òÏßë Î™®Îìú ?§Ìñâ"""
        try:
            logger.info("Starting data collection mode...")
            self.pipeline_stats['mode'] = 'collect'
            
            if not self.collector:
                logger.error("OC parameter is required for data collection mode")
                return False
            
            success = self.collector.collect_all_data()
            self.pipeline_stats['collect_success'] = success
            
            if success:
                logger.info("Data collection completed successfully!")
            else:
                logger.error("Data collection failed!")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in collect mode: {e}")
            self.pipeline_stats['errors'].append(f"Collect mode error: {e}")
            return False
    
    def run_build_mode(self) -> bool:
        """Î≤°ÌÑ∞DB Íµ¨Ï∂ï Î™®Îìú ?§Ìñâ"""
        try:
            logger.info("Starting vector DB build mode...")
            self.pipeline_stats['mode'] = 'build'
            
            success = self.builder.build_vector_db()
            self.pipeline_stats['build_success'] = success
            
            if success:
                logger.info("Vector DB build completed successfully!")
            else:
                logger.error("Vector DB build failed!")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in build mode: {e}")
            self.pipeline_stats['errors'].append(f"Build mode error: {e}")
            return False
    
    def run_full_mode(self) -> bool:
        """?ÑÏ≤¥ ?åÏù¥?ÑÎùº???§Ìñâ (?òÏßë + Íµ¨Ï∂ï)"""
        try:
            logger.info("Starting full data pipeline...")
            self.pipeline_stats['mode'] = 'full'
            
            if not self.collector:
                logger.error("OC parameter is required for full mode")
                return False
            
            # 1. ?∞Ïù¥???òÏßë
            logger.info("Step 1: Data collection...")
            collect_success = self.collector.collect_all_data()
            self.pipeline_stats['collect_success'] = collect_success
            
            if not collect_success:
                logger.error("Data collection failed, stopping pipeline")
                return False
            
            # 2. Î≤°ÌÑ∞DB Íµ¨Ï∂ï
            logger.info("Step 2: Vector DB build...")
            build_success = self.builder.build_vector_db()
            self.pipeline_stats['build_success'] = build_success
            
            if not build_success:
                logger.error("Vector DB build failed")
                return False
            
            # 3. ÏµúÏ¢Ö ?µÍ≥Ñ ?ùÏÑ±
            self._generate_pipeline_report()
            
            logger.info("Full data pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in full mode: {e}")
            self.pipeline_stats['errors'].append(f"Full mode error: {e}")
            return False
    
    def run_specific_collect(self, data_type: str, query: str = None, display: int = 100) -> bool:
        """?πÏ†ï ?∞Ïù¥???Ä???òÏßë"""
        try:
            logger.info(f"Starting {data_type} data collection...")
            self.pipeline_stats['mode'] = f'collect_{data_type}'
            
            if not self.collector:
                logger.error("OC parameter is required for data collection")
                return False
            
            success = False
            if data_type == "laws":
                success = self.collector.collect_laws(query=query or "ÎØºÎ≤ï", display=display)
            elif data_type == "precedents":
                success = self.collector.collect_precedents(query=query or "Í≥ÑÏïΩ ?¥Ï?", display=display)
            elif data_type == "constitutional":
                success = self.collector.collect_constitutional_decisions(query=query or "?åÎ≤ï", display=display)
            elif data_type == "interpretations":
                success = self.collector.collect_legal_interpretations(query=query or "Î≤ïÎ†π?¥ÏÑù", display=display)
            elif data_type == "administrative":
                success = self.collector.collect_administrative_rules(query=query or "?âÏ†ïÍ∑úÏπô", display=display)
            elif data_type == "local":
                success = self.collector.collect_local_ordinances(query=query or "?êÏπòÎ≤ïÍ∑ú", display=display)
            else:
                logger.error(f"Unknown data type: {data_type}")
                return False
            
            self.pipeline_stats['collect_success'] = success
            
            if success:
                logger.info(f"{data_type} data collection completed successfully!")
            else:
                logger.error(f"{data_type} data collection failed!")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in {data_type} collection: {e}")
            self.pipeline_stats['errors'].append(f"{data_type} collection error: {e}")
            return False
    
    def run_specific_build(self, data_type: str) -> bool:
        """?πÏ†ï ?∞Ïù¥???Ä??Î≤°ÌÑ∞DB Íµ¨Ï∂ï"""
        try:
            logger.info(f"Starting {data_type} vector DB build...")
            self.pipeline_stats['mode'] = f'build_{data_type}'
            
            # ?∞Ïù¥???Ä??Îß§Ìïë
            type_mapping = {
                "laws": "laws",
                "precedents": "precedents",
                "constitutional": "constitutional_decisions",
                "interpretations": "legal_interpretations",
                "administrative": "administrative_rules",
                "local": "local_ordinances"
            }
            
            mapped_type = type_mapping.get(data_type)
            if not mapped_type:
                logger.error(f"Unknown data type: {data_type}")
                return False
            
            success = self.builder.build_specific_type(mapped_type)
            self.pipeline_stats['build_success'] = success
            
            if success:
                logger.info(f"{data_type} vector DB build completed successfully!")
            else:
                logger.error(f"{data_type} vector DB build failed!")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in {data_type} build: {e}")
            self.pipeline_stats['errors'].append(f"{data_type} build error: {e}")
            return False
    
    def run_multiple_types_collect(self, data_types: list, query: str = None, display: int = 100) -> bool:
        """?¨Îü¨ ?∞Ïù¥???Ä???òÏßë"""
        try:
            logger.info(f"Starting collection for multiple data types: {', '.join(data_types)}")
            self.pipeline_stats['mode'] = f'collect_multiple_{"_".join(data_types)}'
            
            success_count = 0
            total_types = len(data_types)
            
            for data_type in data_types:
                logger.info(f"Collecting {data_type} data...")
                
                # Í∏∞Î≥∏ ÏøºÎ¶¨ ?§Ï†ï
                default_queries = {
                    "laws": "ÎØºÎ≤ï",
                    "precedents": "Í≥ÑÏïΩ ?¥Ï?",
                    "constitutional": "?åÎ≤ï",
                    "interpretations": "Î≤ïÎ†π?¥ÏÑù",
                    "administrative": "?âÏ†ïÍ∑úÏπô",
                    "local": "?êÏπòÎ≤ïÍ∑ú"
                }
                
                search_query = query or default_queries.get(data_type, data_type)
                
                if self.run_specific_collect(data_type, search_query, display):
                    success_count += 1
                    logger.info(f"{data_type} collection completed successfully!")
                else:
                    logger.error(f"{data_type} collection failed!")
            
            # ÏµúÏ¢Ö ?µÍ≥Ñ ?ùÏÑ±
            self._generate_pipeline_report()
            
            logger.info(f"Multiple types collection completed: {success_count}/{total_types} types successful")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error in multiple types collection: {e}")
            self.pipeline_stats['errors'].append(f"Multiple types collection error: {e}")
            return False
    
    def _generate_pipeline_report(self):
        """?åÏù¥?ÑÎùº??Î≥¥Í≥†???ùÏÑ±"""
        try:
            self.pipeline_stats['end_time'] = datetime.now().isoformat()
            self.pipeline_stats['total_duration'] = (
                datetime.fromisoformat(self.pipeline_stats['end_time']) - 
                datetime.fromisoformat(self.pipeline_stats['start_time'])
            ).total_seconds()
            
            # ?òÏßë ?µÍ≥Ñ Ï∂îÍ?
            if self.collector:
                self.pipeline_stats['collect_stats'] = self.collector.collection_stats
            
            # Íµ¨Ï∂ï ?µÍ≥Ñ Ï∂îÍ?
            self.pipeline_stats['build_stats'] = self.builder.build_stats
            
            # Î≥¥Í≥†???Ä??
            report_file = Path("data/pipeline_report.json")
            report_file.parent.mkdir(exist_ok=True)
            
            import json
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.pipeline_stats, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Pipeline report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating pipeline report: {e}")


def main():
    """Î©îÏù∏ ?®Ïàò"""
    parser = argparse.ArgumentParser(description="LawFirmAI Data Pipeline")
    parser.add_argument("--mode", type=str, 
                        choices=["collect", "build", "full", "laws", "precedents", "constitutional", "interpretations", "administrative", "local", "all_types"], 
                        default="full", help="Pipeline mode")
    parser.add_argument("--oc", type=str, help="OC parameter for Law OpenAPI (user email ID)")
    parser.add_argument("--query", type=str, help="Search query for specific data types")
    parser.add_argument("--display", type=int, default=100, help="Number of items to display per API call")
    parser.add_argument("--types", type=str, nargs="+", 
                        choices=["laws", "precedents", "constitutional", "interpretations", "administrative", "local"],
                        help="Specific data types to collect (use with --mode all_types)")
    
    args = parser.parse_args()
    
    # Î°úÍ∑∏ ?îÎ†â?†Î¶¨ ?ùÏÑ±
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # OC ?åÎùºÎØ∏ÌÑ∞ Í≤ÄÏ¶?
    if args.mode in ["collect", "full", "laws", "precedents", "constitutional", "interpretations", "administrative", "local", "all_types"]:
        if not args.oc:
            logger.error("OC parameter is required for data collection modes")
            logger.info("Usage: python scripts/run_data_pipeline.py --mode collect --oc your_email_id")
            return
    
    # ?åÏù¥?ÑÎùº???§Ìñâ
    pipeline = DataPipeline(args.oc)
    
    success = False
    if args.mode == "collect":
        success = pipeline.run_collect_mode()
    elif args.mode == "build":
        success = pipeline.run_build_mode()
    elif args.mode == "full":
        success = pipeline.run_full_mode()
    elif args.mode == "all_types":
        # ÏßÄ?ïÎêú ?Ä?ÖÎì§Îß??òÏßë
        if not args.types:
            logger.error("--types parameter is required when using --mode all_types")
            logger.info("Example: python scripts/run_data_pipeline.py --mode all_types --oc your_email_id --types laws precedents")
            return
        success = pipeline.run_multiple_types_collect(args.types, args.query, args.display)
    elif args.mode in ["laws", "precedents", "constitutional", "interpretations", "administrative", "local"]:
        # ?∞Ïù¥???Ä?ÖÎ≥Ñ ?òÏßë
        success = pipeline.run_specific_collect(args.mode, args.query, args.display)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return
    
    if success:
        logger.info("Data pipeline completed successfully!")
    else:
        logger.error("Data pipeline failed!")


if __name__ == "__main__":
    main()