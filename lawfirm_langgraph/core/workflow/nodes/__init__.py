"""
LangGraph Nodes
워크플로우 노드 모듈

참고: 대부분의 기능이 core.workflow.builders, core.shared.wrappers로 이동했습니다.
이 모듈은 호환성을 위해 re-export만 제공합니다.
"""

try:
    from core.workflow.builders.chain_builders import (
        AnswerGenerationChainBuilder,
        ClassificationChainBuilder,
        DirectAnswerChainBuilder,
        DocumentAnalysisChainBuilder,
        QueryEnhancementChainBuilder,
    )
    from core.shared.wrappers.node_wrappers import with_state_optimization
    from core.agents.node_input_output_spec import NodeIOSpec, NodeCategory
    from core.workflow.builders.prompt_builders import PromptBuilder, QueryBuilder
except ImportError:
    # Fallback: 기존 경로 (호환성 유지)
    from .chain_builders import (
        AnswerGenerationChainBuilder,
        ClassificationChainBuilder,
        DirectAnswerChainBuilder,
        DocumentAnalysisChainBuilder,
        QueryEnhancementChainBuilder,
    )
    from .node_wrappers import with_state_optimization
    from .node_input_output_spec import NodeIOSpec, NodeCategory
    from .prompt_builders import PromptBuilder, QueryBuilder

# Phase 1: 노드 모듈화 - 노드 클래스들 추가
from .classification_nodes import ClassificationNodes
from .search_nodes import SearchNodes
from .document_nodes import DocumentNodes
from .answer_nodes import AnswerNodes
from .agentic_nodes import AgenticNodes
from .ethical_rejection_node import EthicalRejectionNode

__all__ = [
    "AnswerGenerationChainBuilder",
    "ClassificationChainBuilder",
    "DirectAnswerChainBuilder",
    "DocumentAnalysisChainBuilder",
    "QueryEnhancementChainBuilder",
    "with_state_optimization",
    "NodeIOSpec",
    "NodeCategory",
    "PromptBuilder",
    "QueryBuilder",
    # Phase 1: 노드 모듈화
    "ClassificationNodes",
    "SearchNodes",
    "DocumentNodes",
    "AnswerNodes",
    "AgenticNodes",
    "EthicalRejectionNode",
]
