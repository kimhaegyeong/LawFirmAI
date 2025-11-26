# -*- coding: utf-8 -*-
"""
Answer Generation Mixin
답변 생성 관련 노드 및 메서드들을 제공하는 Mixin 클래스
"""

import time
from typing import Any, Dict, List, Tuple

try:
    from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
except ImportError:
    from core.workflow.state.state_definitions import LegalWorkflowState
try:
    from lawfirm_langgraph.core.workflow.utils.workflow_constants import WorkflowConstants, QualityThresholds
except ImportError:
    from core.workflow.utils.workflow_constants import WorkflowConstants, QualityThresholds
try:
    from lawfirm_langgraph.core.shared.wrappers.node_wrappers import with_state_optimization
except ImportError:
    from core.shared.wrappers.node_wrappers import with_state_optimization

# Mock observe decorator (Langfuse 제거됨)
def observe(**kwargs):
    def decorator(func):
        return func
    return decorator


class AnswerGenerationMixin:
    """답변 생성 관련 노드 및 메서드들을 제공하는 Mixin 클래스"""
    
    # ============================================================================
    # Answer Generation 헬퍼 메서드들
    # ============================================================================
    
    def _prepare_answer_generation(self, state: LegalWorkflowState) -> Tuple[bool, float]:
        """답변 생성 초기화 및 재시도 확인"""
        metadata = state.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        if "common" in state and isinstance(state.get("common"), dict):
            common_metadata = state["common"].get("metadata", {})
            if isinstance(common_metadata, dict):
                metadata = {**metadata, **common_metadata}
        
        last_executed_node = metadata.get("_last_executed_node", "")
        is_retry = (last_executed_node == "validate_answer_quality")
        if is_retry:
            if self.retry_manager.should_allow_retry(state, "validation"):
                self.retry_manager.increment_retry_count(state, "validation")
        
        return is_retry, time.time()
    
    def _restore_query_type(self, state: LegalWorkflowState) -> str:
        """query_type 검색 및 복원"""
        query_type = self._get_state_value(state, "query_type", "")
        
        if not query_type:
            self.logger.warning("⚠️ [QUESTION TYPE] query_type not found in state, trying additional search...")
            try:
                from core.shared.wrappers.node_wrappers import _global_search_results_cache
                if _global_search_results_cache:
                    cached_query_type = (
                        _global_search_results_cache.get("common", {}).get("classification", {}).get("query_type", "") or
                        _global_search_results_cache.get("metadata", {}).get("query_type", "") or
                        _global_search_results_cache.get("classification", {}).get("query_type", "") or
                        _global_search_results_cache.get("query_type", "") or
                        ""
                    )
                    if cached_query_type:
                        query_type = cached_query_type
                        self.logger.info(f"✅ [QUESTION TYPE] Found query_type in global cache: {query_type}")
                        self._set_state_value(state, "query_type", query_type)
            except (ImportError, AttributeError, TypeError) as e:
                self.logger.debug(f"Could not access global cache: {e}")
        
        if not query_type:
            query_type = "general_question"
            self.logger.warning(f"⚠️ [QUESTION TYPE] query_type not found in state or global cache, using default: {query_type}")
            self._set_state_value(state, "query_type", query_type)
        else:
            self.logger.info(f"✅ [QUESTION TYPE] Using query_type: {query_type}")
        
        return query_type
    
    def _restore_retrieved_docs(self, state: LegalWorkflowState) -> List[Dict[str, Any]]:
        """retrieved_docs 검색 및 복원"""
        retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
        
        if retrieved_docs:
            self._set_state_value(state, "retrieved_docs", retrieved_docs)
            if "search" not in state:
                state["search"] = {}
            state["search"]["retrieved_docs"] = retrieved_docs
            if "common" not in state:
                state["common"] = {}
            if "search" not in state["common"]:
                state["common"]["search"] = {}
            state["common"]["search"]["retrieved_docs"] = retrieved_docs
        
        if not retrieved_docs or len(retrieved_docs) == 0:
            try:
                from core.shared.wrappers.node_wrappers import _global_search_results_cache
                if _global_search_results_cache:
                    cached_docs = _global_search_results_cache.get("retrieved_docs", [])
                    if cached_docs:
                        retrieved_docs = cached_docs
                        self.logger.info(f"🔄 [ANSWER GENERATION] Restored {len(retrieved_docs)} retrieved_docs from global cache")
                        self._set_state_value(state, "retrieved_docs", retrieved_docs)
            except (ImportError, AttributeError, TypeError) as e:
                self.logger.debug(f"Could not restore from global cache: {e}")
        
        semantic_results_count = sum(1 for doc in retrieved_docs if doc.get("search_type") == "semantic") if retrieved_docs else 0
        keyword_results_count = sum(1 for doc in retrieved_docs if doc.get("search_type") == "keyword") if retrieved_docs else 0
        
        metadata = self._get_state_value(state, "metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        if retrieved_docs:
            metadata["search_results"] = {
                "count": len(retrieved_docs),
                "semantic_count": semantic_results_count,
                "keyword_count": keyword_results_count,
                "sources": [doc.get("source", "Unknown") for doc in retrieved_docs[:10]]
            }
            self._set_state_value(state, "metadata", metadata)
        
        if retrieved_docs:
            self.logger.info(
                f"📊 [ANSWER GENERATION] Using {len(retrieved_docs)} documents for answer generation: "
                f"Semantic: {semantic_results_count}, Keyword: {keyword_results_count}"
            )
        
        return retrieved_docs or []
    
    def _build_context_dict(
        self, 
        state: LegalWorkflowState, 
        query_type: str, 
        retrieved_docs: List[Dict[str, Any]], 
        prompt_optimized_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """context_dict 생성 (중복 코드 제거)"""
        has_valid_optimized_context = (
            prompt_optimized_context
            and isinstance(prompt_optimized_context, dict)
            and prompt_optimized_context.get("prompt_optimized_text")
            and len(prompt_optimized_context.get("prompt_optimized_text", "").strip()) > 0
        )
        
        if has_valid_optimized_context:
            prompt_text = prompt_optimized_context["prompt_optimized_text"]
            doc_count = prompt_optimized_context.get("document_count", 0)
            context_length = prompt_optimized_context.get("total_context_length", 0)
            structured_docs = prompt_optimized_context.get("structured_documents", {})
            
            import os
            debug_mode = os.getenv("DEBUG_PROMPT_VALIDATION", "false").lower() == "true"
            
            if debug_mode and retrieved_docs and len(retrieved_docs) > 0 and doc_count == 0:
                self.logger.warning(
                    f"⚠️ [PROMPT VALIDATION] retrieved_docs exists ({len(retrieved_docs)} docs) "
                    f"but prompt_optimized_context has 0 documents."
                )
            
            if context_length < 300:
                if retrieved_docs and len(retrieved_docs) > 0:
                    self.logger.info(f"🔄 [FALLBACK] Switching to _build_intelligent_context due to short prompt_optimized_text")
                    return self._build_intelligent_context(state)
                else:
                    return {
                        "context": prompt_text,
                        "structured_documents": prompt_optimized_context.get("structured_documents", {}),
                        "document_count": doc_count,
                        "legal_references": self._extract_legal_references_from_docs(retrieved_docs),
                        "query_type": query_type,
                        "context_length": context_length,
                        "docs_included": doc_count
                    }
            else:
                if retrieved_docs and len(retrieved_docs) > 0:
                    docs_in_structured = structured_docs.get("documents", []) if isinstance(structured_docs, dict) else []
                    min_required = max(1, min(3, int(len(retrieved_docs) * 0.5))) if len(retrieved_docs) > 5 else 1
                    
                    if not docs_in_structured or len(docs_in_structured) < min_required:
                        normalized_documents = []
                        for idx, doc in enumerate(retrieved_docs[:10], 1):
                            if isinstance(doc, dict):
                                content = doc.get("content") or doc.get("text") or doc.get("content_text", "")
                                source = doc.get("source") or doc.get("title") or f"Document_{idx}"
                                relevance_score = doc.get("relevance_score") or doc.get("final_weighted_score", 0.0)
                                
                                if content and len(content.strip()) > 10:
                                    normalized_documents.append({
                                        "document_id": idx,
                                        "source": source,
                                        "content": content[:2000],
                                        "relevance_score": float(relevance_score),
                                        "metadata": doc.get("metadata", {})
                                    })
                        
                        if normalized_documents:
                            if not isinstance(structured_docs, dict):
                                structured_docs = {}
                            structured_docs["documents"] = normalized_documents
                            structured_docs["total_count"] = len(normalized_documents)
                            doc_count = len(normalized_documents)
                            self.logger.info(
                                f"✅ [SEARCH RESULTS ENFORCED] Added {len(normalized_documents)} documents "
                                f"from retrieved_docs to structured_documents"
                            )
                
                context_dict = {
                    "context": prompt_text,
                    "prompt_optimized_text": prompt_text,
                    "structured_documents": structured_docs,
                    "document_count": doc_count,
                    "legal_references": self._extract_legal_references_from_docs(retrieved_docs),
                    "query_type": query_type,
                    "context_length": context_length,
                    "docs_included": len(structured_docs.get("documents", [])) if isinstance(structured_docs, dict) else 0
                }
                
                content_validation = prompt_optimized_context.get("content_validation")
                if content_validation:
                    context_dict["content_validation"] = content_validation
                    if not content_validation.get("has_document_content", False):
                        self.logger.warning(f"⚠️ [PROMPT VALIDATION] content_validation indicates no document content in prompt")
                
                self.logger.info(f"✅ [PROMPT OPTIMIZED] Using optimized document context ({doc_count} docs, {context_length} chars)")
                return context_dict
        else:
            if retrieved_docs and len(retrieved_docs) > 0:
                self.logger.warning(
                    f"⚠️ [FALLBACK] prompt_optimized_context is missing or invalid, "
                    f"but retrieved_docs exists ({len(retrieved_docs)} docs). "
                    f"Using _build_intelligent_context as fallback."
                )
            else:
                self.logger.info(f"ℹ️ [FALLBACK] No prompt_optimized_context and no retrieved_docs. Using _build_intelligent_context.")
            return self._build_intelligent_context(state)
    
    # ============================================================================
    # Answer Generation 노드들
    # ============================================================================
    
    @observe(name="validate_answer_quality")
    @with_state_optimization("validate_answer_quality", enable_reduction=False)
    def validate_answer_quality(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """답변 품질 및 법령 검증"""
        try:
            self._save_metadata_safely(state, "_last_executed_node", "validate_answer_quality")

            start_time = time.time()
            answer = self._normalize_answer(self._get_state_value(state, "answer", ""))
            errors = self._get_state_value(state, "errors", [])
            sources = self._get_state_value(state, "sources", [])

            answer_content_preview = ""
            if isinstance(answer, str):
                answer_content_preview = answer[:500] if len(answer) > 500 else answer
            else:
                answer_str = str(answer)
                answer_content_preview = answer_str[:500] if len(answer_str) > 500 else answer_str

            answer_length = len(answer) if isinstance(answer, str) else len(str(answer))
            self.logger.info(
                f"🔍 [QUALITY VALIDATION] Answer received for validation:\n"
                f"   Answer length: {answer_length} characters\n"
                f"   Answer content: '{answer_content_preview}'\n"
                f"   Answer type: {type(answer).__name__}\n"
                f"   Error count: {len(errors)}\n"
                f"   Source count: {len(sources)}"
            )

            answer_str_for_check = answer if isinstance(answer, str) else str(answer) if answer else ""
            quality_checks = {
                "has_answer": len(answer_str_for_check) > 0,
                "min_length": len(answer_str_for_check) >= WorkflowConstants.MIN_ANSWER_LENGTH_VALIDATION,
                "no_errors": len(errors) == 0,
                "has_sources": len(sources) > 0 or len(self._get_state_value(state, "retrieved_docs", [])) > 0
            }

            if self.legal_validator and len(answer_str_for_check) > 0:
                try:
                    query = self._get_state_value(state, "query", "")
                    answer_for_validation = answer if isinstance(answer, str) else answer_str_for_check
                    validation_result = self.legal_validator.validate_legal_basis(query, answer_for_validation)
                    self._set_state_value(state, "legal_validity_check", validation_result.is_valid)
                    self._set_state_value(state, "legal_basis_validation", {
                        "confidence": validation_result.confidence,
                        "issues": validation_result.issues,
                        "recommendations": validation_result.recommendations
                    })
                    quality_checks["legal_basis_valid"] = validation_result.is_valid
                    self.logger.info(f"Legal basis validation: {validation_result.is_valid}")
                except Exception as e:
                    self.logger.warning(f"Legal validation failed: {e}")
                    self._set_state_value(state, "legal_validity_check", True)
                    quality_checks["legal_basis_valid"] = True
            else:
                self._set_state_value(state, "legal_validity_check", True)
                quality_checks["legal_basis_valid"] = True

            passed_checks = sum(quality_checks.values())
            total_checks = len(quality_checks)
            quality_score = passed_checks / total_checks

            self.logger.info(
                f"📊 [QUALITY CHECKS] Detailed validation results:\n"
                f"   Quality checks: {quality_checks}\n"
                f"   Passed checks: {passed_checks}/{total_checks}\n"
                f"   Quality score: {quality_score:.2f} (threshold: {QualityThresholds.QUALITY_PASS_THRESHOLD})"
            )

            quality_check_passed = quality_score >= QualityThresholds.QUALITY_PASS_THRESHOLD

            self._save_metadata_safely(state, "quality_score", quality_score, save_to_top_level=True)
            self._save_metadata_safely(state, "quality_check_passed", quality_check_passed, save_to_top_level=True)

            self._update_processing_time(state, start_time)

            quality_status = "통과" if quality_check_passed else "실패"
            legal_validity = self._get_state_value(state, "legal_validity_check", True)
            self._add_step(state, "답변 검증",
                         f"품질: {quality_score:.2f}, 법령: {legal_validity}")

            self.logger.info(
                f"Answer quality validation: {quality_status}, "
                f"score: {quality_score:.2f}, checks: {passed_checks}/{total_checks}"
            )

        except Exception as e:
            self._handle_error(state, str(e), "답변 검증 중 오류")
            self._set_state_value(state, "legal_validity_check", True)

            self._save_metadata_safely(state, "quality_score", 0.0, save_to_top_level=True)
            self._save_metadata_safely(state, "quality_check_passed", False, save_to_top_level=True)

        return state

