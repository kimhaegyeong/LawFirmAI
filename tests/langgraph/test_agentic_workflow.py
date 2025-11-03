# -*- coding: utf-8 -*-
"""
Agentic AI ?Œí¬?Œë¡œ???¤í–‰ ?ŒìŠ¤??
?¤ì œ Agentic ëª¨ë“œ?ì„œ ?Œí¬?Œë¡œ???¤í–‰ ë°?Tool ?¬ìš© ê²€ì¦?
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

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


async def test_agentic_mode_enabled():
    """Agentic ëª¨ë“œ ?œì„±???íƒœ?ì„œ ?Œí¬?Œë¡œ???ŒìŠ¤??""
    logger.info("=" * 80)
    logger.info("Agentic ëª¨ë“œ ?œì„±???íƒœ ?Œí¬?Œë¡œ???ŒìŠ¤??)
    logger.info("=" * 80)
    
    # ?˜ê²½ ë³€???¤ì •
    original_value = os.environ.get("USE_AGENTIC_MODE")
    os.environ["USE_AGENTIC_MODE"] = "true"
    
    try:
        from source.agents.workflow_service import LangGraphWorkflowService
        from infrastructure.utils.langgraph_config import LangGraphConfig
        
        config = LangGraphConfig.from_env()
        assert config.use_agentic_mode == True, "Agentic ëª¨ë“œê°€ ?œì„±?”ë˜?´ì•¼ ??
        
        logger.info("??Agentic ëª¨ë“œ ?œì„±???•ì¸")
        
        workflow_service = LangGraphWorkflowService(config)
        logger.info("???Œí¬?Œë¡œ???œë¹„??ì´ˆê¸°???„ë£Œ")
        
        # ë³µì¡??ì§ˆë¬¸ (Agentic ?¸ë“œë¡??¼ìš°?…ë  ê°€?¥ì„±???’ìŒ)
        test_queries = [
            "ê³„ì•½ ?„ë°˜ ???„ì•½ê¸ˆì? ?´ë–»ê²?ê³„ì‚°?˜ë‚˜??",
            "?´í˜¼ ?Œì†¡ ?ˆì°¨ë¥??Œë ¤ì£¼ì„¸??,
            "ë¯¼ë²• ??23ì¡°ì? ê´€?¨ëœ ìµœê·¼ ?ë?ë¥?ì°¾ì•„ì£¼ì„¸??
        ]
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n--- ?ŒìŠ¤??ì§ˆì˜ {i}: {query} ---")
            
            try:
                start_time = time.time()
                result = await workflow_service.process_query(query)
                execution_time = time.time() - start_time
                
                logger.info(f"???¤í–‰ ?œê°„: {execution_time:.2f}ì´?)
                
                # ê²°ê³¼ ê²€ì¦?
                if "answer" in result or result.get("response"):
                    answer = result.get("answer") or result.get("response", "")
                    logger.info(f"???µë? ?ì„±??({len(answer)}??")
                    
                    # Agentic Tool ?¸ì¶œ ?•ë³´ ?•ì¸
                    if "agentic_tool_calls" in str(result):
                        logger.info("??Agentic Tool???¬ìš©??ê²ƒìœ¼ë¡?ë³´ì„")
                    else:
                        logger.info("?¹ï¸ Agentic Tool ?¬ìš© ?•ë³´ ?†ìŒ (ê¸°ì¡´ ?Œë¡œ?°ì¼ ???ˆìŒ)")
                    
                    # ê²€??ê²°ê³¼ ?•ì¸
                    if "sources" in result or "retrieved_docs" in result:
                        sources = result.get("sources", []) or result.get("retrieved_docs", [])
                        logger.info(f"??ê²€??ê²°ê³¼: {len(sources)}ê°?)
                    
                else:
                    logger.warning("? ï¸ ?µë? ?„ë“œ ?†ìŒ")
                    
            except Exception as e:
                logger.error(f"??ì§ˆì˜ ì²˜ë¦¬ ?¤íŒ¨: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info("\n" + "=" * 80)
        logger.info("?ŒìŠ¤???„ë£Œ")
        logger.info("=" * 80)
        
    finally:
        # ?˜ê²½ ë³€??ë³µì›
        if original_value:
            os.environ["USE_AGENTIC_MODE"] = original_value
        else:
            os.environ.pop("USE_AGENTIC_MODE", None)


    async def test_tool_selection():
        """Tool ? íƒ ?™ì‘ ?ŒìŠ¤??""
        logger.info("=" * 80)
        logger.info("Tool ? íƒ ?™ì‘ ?ŒìŠ¤??)
        logger.info("=" * 80)
        
        try:
            from langgraph_core.tools import LEGAL_TOOLS
            
            logger.info(f"?¬ìš© ê°€?¥í•œ Tool: {len(LEGAL_TOOLS)}ê°?)
            for i, tool in enumerate(LEGAL_TOOLS, 1):
                logger.info(f"  {i}. {tool.name}")
                logger.info(f"     ?¤ëª…: {tool.description[:100]}...")
            
            # ê°?Tool???…ë ¥ ?¤í‚¤ë§??•ì¸
            for tool in LEGAL_TOOLS:
                if hasattr(tool, 'args_schema'):
                    logger.info(f"\n{tool.name} ?…ë ¥ ?¤í‚¤ë§?")
                    schema = tool.args_schema
                    if hasattr(schema, 'schema'):
                        for field_name, field_info in schema.schema().get('properties', {}).items():
                            logger.info(f"  - {field_name}: {field_info.get('description', 'N/A')}")
            
            logger.info("\n??Tool ? íƒ ?ŒìŠ¤???„ë£Œ")
            
        except Exception as e:
            logger.error(f"??Tool ? íƒ ?ŒìŠ¤???¤íŒ¨: {e}")
            import traceback
            logger.error(traceback.format_exc())


if __name__ == "__main__":
    # ?ŒìŠ¤???¤í–‰
    asyncio.run(test_agentic_mode_enabled())
    asyncio.run(test_tool_selection())

