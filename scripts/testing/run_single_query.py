# -*- coding: utf-8 -*-
"""
Single-query runner for evaluating current workflow answer quality.

Usage:
    python scripts/testing/run_single_query.py "ÏßàÏùòÎ¨?
If no argument is provided, a default legal query will be used.
"""

import asyncio
import sys
from pathlib import Path


def project_bootstrap():
    # Ensure project root on sys.path
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


async def run(query: str):
    from infrastructure.utils.langgraph_config import LangGraphConfig
    from source.agents.workflow_service import LangGraphWorkflowService

    config = LangGraphConfig.from_env()
    service = LangGraphWorkflowService(config)

    result = await service.process_query(query, session_id="single_query_eval", enable_checkpoint=False)
    # Normalize fields for readability
    answer = result.get("answer")
    answer_text = answer
    if isinstance(answer_text, dict):
        # try nested extraction
        for key in ("answer", "content", "text"):
            if isinstance(answer_text, dict) and key in answer_text:
                answer_text = answer_text[key]
        if isinstance(answer_text, dict):
            answer_text = str(answer_text)

    print("\n===== Single Query Evaluation =====", flush=True)
    print(f"Query: {query}", flush=True)
    print(f"Answer length: {len(str(answer_text)) if answer_text else 0}", flush=True)
    print("--- Answer Preview (first 600 chars) ---", flush=True)
    print(str(answer_text)[:600] if answer_text else "<no answer>", flush=True)
    print("\n--- Sources ---", flush=True)
    print(result.get("sources", []), flush=True)
    print("\n--- Legal References ---", flush=True)
    print(result.get("legal_references", []), flush=True)
    print("\n--- Metadata ---", flush=True)
    print(result.get("metadata", {}), flush=True)
    print("\n--- Confidence ---", flush=True)
    print(result.get("confidence", 0.0), flush=True)


def main():
    project_bootstrap()
    query = "ÎØºÎ≤ï ??50Ï°??êÌï¥Î∞∞ÏÉÅ ?îÍ±¥?Ä?"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    asyncio.run(run(query))


if __name__ == "__main__":
    main()


