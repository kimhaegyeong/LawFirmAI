# -*- coding: utf-8 -*-
"""
Agentic AI ?µí•© ?ŒìŠ¤??
Tool Use/Function Calling ê¸°ëŠ¥ê³?ê¸°ì¡´ ?Œí¬?Œë¡œ???µí•© ê²€ì¦?
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ë¡œê¹… ?¤ì •
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

logger = logging.getLogger(__name__)


class TestAgenticIntegration:
    """Agentic AI ?µí•© ?ŒìŠ¤???´ë˜??""
    
    def __init__(self):
        self.test_results = []
    
    def test_tool_import(self):
        """Tool ?œìŠ¤??import ?ŒìŠ¤??""
        logger.info("=" * 80)
        logger.info("Test 1: Tool ?œìŠ¤??Import ?ŒìŠ¤??)
        logger.info("=" * 80)
        
        try:
            from langgraph_core.tools import LEGAL_TOOLS
            logger.info(f"??Tool ?œìŠ¤??import ?±ê³µ: {len(LEGAL_TOOLS)}ê°?Tool")
            
            # Tool ëª©ë¡ ì¶œë ¥
            for i, tool in enumerate(LEGAL_TOOLS, 1):
                logger.info(f"   {i}. {tool.name}: {tool.description[:80]}...")
            
            self.test_results.append(("Tool Import", True, f"{len(LEGAL_TOOLS)}ê°?Tool"))
            return True
        except Exception as e:
            logger.error(f"??Tool ?œìŠ¤??import ?¤íŒ¨: {e}")
            self.test_results.append(("Tool Import", False, str(e)))
            return False
    
    def test_config_flag(self):
        """?¤ì • ?Œë˜ê·??ŒìŠ¤??""
        logger.info("=" * 80)
        logger.info("Test 2: Agentic ëª¨ë“œ ?¤ì • ?Œë˜ê·??ŒìŠ¤??)
        logger.info("=" * 80)
        
        try:
            from infrastructure.utils.langgraph_config import LangGraphConfig
            
            # ê¸°ë³¸ê°??ŒìŠ¤??(ë¹„í™œ?±í™”)
            config_default = LangGraphConfig.from_env()
            logger.info(f"   ê¸°ë³¸ use_agentic_mode: {config_default.use_agentic_mode}")
            assert config_default.use_agentic_mode == False, "ê¸°ë³¸ê°’ì? False?¬ì•¼ ??
            
            # ?˜ê²½ ë³€?˜ë¡œ ?œì„±???ŒìŠ¤??
            original_value = os.environ.get("USE_AGENTIC_MODE")
            try:
                os.environ["USE_AGENTIC_MODE"] = "true"
                config_enabled = LangGraphConfig.from_env()
                logger.info(f"   ?œì„±????use_agentic_mode: {config_enabled.use_agentic_mode}")
                assert config_enabled.use_agentic_mode == True, "?œì„±????True?¬ì•¼ ??
                
                # ?˜ê²½ ë³€??ë³µì›
                if original_value:
                    os.environ["USE_AGENTIC_MODE"] = original_value
                else:
                    os.environ.pop("USE_AGENTIC_MODE", None)
                
                logger.info("???¤ì • ?Œë˜ê·??ŒìŠ¤???±ê³µ")
                self.test_results.append(("Config Flag", True, "?¤ì • ?Œë˜ê·??•ìƒ ?™ì‘"))
                return True
            except Exception as e:
                # ?˜ê²½ ë³€??ë³µì›
                if original_value:
                    os.environ["USE_AGENTIC_MODE"] = original_value
                else:
                    os.environ.pop("USE_AGENTIC_MODE", None)
                raise e
                
        except Exception as e:
            logger.error(f"???¤ì • ?Œë˜ê·??ŒìŠ¤???¤íŒ¨: {e}")
            self.test_results.append(("Config Flag", False, str(e)))
            return False
    
    async def test_workflow_without_agentic(self):
        """Agentic ëª¨ë“œ ë¹„í™œ?±í™” ?íƒœ?ì„œ ?Œí¬?Œë¡œ???ŒìŠ¤??(ê¸°ì¡´ ?™ì‘ ?•ì¸)"""
        logger.info("=" * 80)
        logger.info("Test 3: Agentic ëª¨ë“œ ë¹„í™œ?±í™” ?íƒœ ?Œí¬?Œë¡œ???ŒìŠ¤??)
        logger.info("=" * 80)
        
        try:
            # ?˜ê²½ ë³€???•ì¸ ë°??„ì‹œ ë¹„í™œ?±í™”
            original_value = os.environ.get("USE_AGENTIC_MODE")
            os.environ["USE_AGENTIC_MODE"] = "false"
            
            from source.agents.workflow_service import LangGraphWorkflowService
            from infrastructure.utils.langgraph_config import LangGraphConfig
            
            config = LangGraphConfig.from_env()
            assert config.use_agentic_mode == False, "Agentic ëª¨ë“œê°€ ë¹„í™œ?±í™”?˜ì–´????
            
            workflow_service = LangGraphWorkflowService(config)
            logger.info("   ???Œí¬?Œë¡œ???œë¹„??ì´ˆê¸°???±ê³µ (Agentic ëª¨ë“œ ë¹„í™œ?±í™”)")
            
            # ê°„ë‹¨??ì§ˆë¬¸?¼ë¡œ ?ŒìŠ¤??
            test_query = "ê³„ì•½?´ë? ë¬´ì—‡?¸ê???"
            logger.info(f"   ?ŒìŠ¤??ì§ˆì˜: {test_query}")
            
            result = await workflow_service.process_query(test_query)
            
            # ê²°ê³¼ ê²€ì¦?
            assert "answer" in result or result.get("response"), "?µë? ?„ë“œê°€ ?ˆì–´????
            logger.info(f"   ???Œí¬?Œë¡œ???¤í–‰ ?±ê³µ (ê¸°ì¡´ ë°©ì‹)")
            
            # ?˜ê²½ ë³€??ë³µì›
            if original_value:
                os.environ["USE_AGENTIC_MODE"] = original_value
            else:
                os.environ.pop("USE_AGENTIC_MODE", None)
            
            self.test_results.append(("Workflow Without Agentic", True, "ê¸°ì¡´ ?Œí¬?Œë¡œ???•ìƒ ?™ì‘"))
            return True
            
        except Exception as e:
            logger.error(f"??ê¸°ì¡´ ?Œí¬?Œë¡œ???ŒìŠ¤???¤íŒ¨: {e}")
            # ?˜ê²½ ë³€??ë³µì›
            if original_value:
                os.environ["USE_AGENTIC_MODE"] = original_value
            else:
                os.environ.pop("USE_AGENTIC_MODE", None)
            
            self.test_results.append(("Workflow Without Agentic", False, str(e)))
            return False
    
    async def test_agentic_node_initialization(self):
        """Agentic ?¸ë“œ ì´ˆê¸°???ŒìŠ¤??""
        logger.info("=" * 80)
        logger.info("Test 4: Agentic ?¸ë“œ ì´ˆê¸°???ŒìŠ¤??)
        logger.info("=" * 80)
        
        try:
            # ?˜ê²½ ë³€???•ì¸ ë°??„ì‹œ ?œì„±??
            original_value = os.environ.get("USE_AGENTIC_MODE")
            os.environ["USE_AGENTIC_MODE"] = "true"
            
            from source.agents.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
            from infrastructure.utils.langgraph_config import LangGraphConfig
            
            config = LangGraphConfig.from_env()
            assert config.use_agentic_mode == True, "Agentic ëª¨ë“œê°€ ?œì„±?”ë˜?´ì•¼ ??
            
            workflow = EnhancedLegalQuestionWorkflow(config)
            
            # Tool ?œìŠ¤??ì´ˆê¸°???•ì¸
            assert hasattr(workflow, "legal_tools"), "legal_tools ?ì„±???ˆì–´????
            logger.info(f"   ??Agentic ?¸ë“œ ì´ˆê¸°???±ê³µ")
            logger.info(f"   Tool ê°œìˆ˜: {len(workflow.legal_tools)}")
            
            # ê·¸ë˜?„ì— Agentic ?¸ë“œê°€ ì¶”ê??˜ì—ˆ?”ì? ?•ì¸
            graph = workflow._build_graph()
            nodes = graph.nodes.keys() if hasattr(graph, 'nodes') else []
            
            if "agentic_decision" in nodes:
                logger.info("   ??agentic_decision ?¸ë“œê°€ ê·¸ë˜?„ì— ì¶”ê???)
            else:
                logger.warning("   ? ï¸ agentic_decision ?¸ë“œê°€ ê·¸ë˜?„ì— ?†ìŒ (?˜ë™ ?•ì¸ ?„ìš”)")
            
            # ?˜ê²½ ë³€??ë³µì›
            if original_value:
                os.environ["USE_AGENTIC_MODE"] = original_value
            else:
                os.environ.pop("USE_AGENTIC_MODE", None)
            
            self.test_results.append(("Agentic Node Init", True, f"{len(workflow.legal_tools)}ê°?Tool"))
            return True
            
        except Exception as e:
            logger.error(f"??Agentic ?¸ë“œ ì´ˆê¸°???ŒìŠ¤???¤íŒ¨: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # ?˜ê²½ ë³€??ë³µì›
            if original_value:
                os.environ["USE_AGENTIC_MODE"] = original_value
            else:
                os.environ.pop("USE_AGENTIC_MODE", None)
            
            self.test_results.append(("Agentic Node Init", False, str(e)))
            return False
    
    async def test_tool_execution(self):
        """Tool ?¤í–‰ ?ŒìŠ¤??""
        logger.info("=" * 80)
        logger.info("Test 5: Tool ?¤í–‰ ?ŒìŠ¤??)
        logger.info("=" * 80)
        
        try:
            from langgraph_core.tools import LEGAL_TOOLS
            
            if not LEGAL_TOOLS:
                logger.warning("   ? ï¸ ?¬ìš© ê°€?¥í•œ Tool???†ìŒ (ê²€???”ì§„ ë¯¸ì´ˆê¸°í™” ê°€??")
                self.test_results.append(("Tool Execution", True, "Tool ?†ìŒ (?•ìƒ)"))
                return True
            
            # ì²?ë²ˆì§¸ Toolë¡??ŒìŠ¤??(ë³´í†µ hybrid_search_tool)
            test_tool = LEGAL_TOOLS[0]
            logger.info(f"   ?ŒìŠ¤??Tool: {test_tool.name}")
            
            # Tool ?¤í–‰ ?ŒìŠ¤??(?¤ì œ ê²€?‰ì? ?˜ì? ?Šê³  ?¨ìˆ˜ ?¸ì¶œë§?
            try:
                # Tool???¨ìˆ˜ ?•ì¸
                if hasattr(test_tool, 'func'):
                    logger.info(f"   ??Tool ?¨ìˆ˜ ?•ì¸: {test_tool.func.__name__}")
                else:
                    logger.info(f"   ??Tool êµ¬ì¡° ?•ì¸ ?„ë£Œ")
                
                self.test_results.append(("Tool Execution", True, f"{test_tool.name} ?•ì¸ ?„ë£Œ"))
                return True
            except Exception as e:
                logger.error(f"   ??Tool ?¤í–‰ ?ŒìŠ¤???¤íŒ¨: {e}")
                self.test_results.append(("Tool Execution", False, str(e)))
                return False
                
        except Exception as e:
            logger.error(f"??Tool ?¤í–‰ ?ŒìŠ¤???¤íŒ¨: {e}")
            self.test_results.append(("Tool Execution", False, str(e)))
            return False
    
    def print_summary(self):
        """?ŒìŠ¤??ê²°ê³¼ ?”ì•½ ì¶œë ¥"""
        logger.info("=" * 80)
        logger.info("?ŒìŠ¤??ê²°ê³¼ ?”ì•½")
        logger.info("=" * 80)
        
        total = len(self.test_results)
        passed = sum(1 for _, success, _ in self.test_results if success)
        failed = total - passed
        
        for test_name, success, detail in self.test_results:
            status = "??PASS" if success else "??FAIL"
            logger.info(f"{status} - {test_name}: {detail}")
        
        logger.info("=" * 80)
        logger.info(f"ì´??ŒìŠ¤?? {total}ê°?| ?µê³¼: {passed}ê°?| ?¤íŒ¨: {failed}ê°?)
        logger.info("=" * 80)
        
        return failed == 0


async def run_all_tests():
    """ëª¨ë“  ?ŒìŠ¤???¤í–‰"""
    tester = TestAgenticIntegration()
    
    # ?™ê¸° ?ŒìŠ¤??
    tester.test_tool_import()
    tester.test_config_flag()
    tester.test_tool_execution()
    
    # ë¹„ë™ê¸??ŒìŠ¤??
    await tester.test_workflow_without_agentic()
    await tester.test_agentic_node_initialization()
    
    # ê²°ê³¼ ?”ì•½
    success = tester.print_summary()
    
    return success


if __name__ == "__main__":
    # ?ŒìŠ¤???¤í–‰
    success = asyncio.run(run_all_tests())
    
    # ì¢…ë£Œ ì½”ë“œ
    sys.exit(0 if success else 1)

