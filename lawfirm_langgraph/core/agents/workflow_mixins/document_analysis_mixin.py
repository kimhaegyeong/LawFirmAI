# -*- coding: utf-8 -*-
"""
Document Analysis Mixin
문서 분석 관련 노드 및 메서드들을 제공하는 Mixin 클래스
"""

import time
from typing import Any, Dict, List

from core.workflow.state.state_definitions import LegalWorkflowState
from core.shared.wrappers.node_wrappers import with_state_optimization

# Mock observe decorator (Langfuse 제거됨)
def observe(**kwargs):
    def decorator(func):
        return func
    return decorator


class DocumentAnalysisMixin:
    """문서 분석 관련 노드 및 메서드들을 제공하는 Mixin 클래스"""
    
    # ============================================================================
    # Document Analysis 노드들
    # ============================================================================
    
    @observe(name="prepare_document_context_for_prompt")
    @with_state_optimization("prepare_document_context_for_prompt", enable_reduction=True)
    def prepare_document_context_for_prompt(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """프롬프트에 최대한 반영되도록 문서 컨텍스트 준비"""
        try:
            start_time = time.time()

            retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
            query = self._get_state_value(state, "query", "")
            extracted_keywords = self._get_state_value(state, "extracted_keywords", [])
            query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", ""))
            legal_field = self._get_state_value(state, "legal_field", "")

            if not retrieved_docs:
                self.logger.warning(
                    f"⚠️ [PREPARE CONTEXT] No retrieved_docs to prepare for prompt. "
                    f"Query: '{query[:50]}...', Query type: {query_type_str}"
                )
                self._set_state_value(state, "prompt_optimized_context", {
                    "prompt_optimized_text": "",
                    "structured_documents": {},
                    "document_count": 0,
                    "total_context_length": 0
                })
                return state

            if not isinstance(retrieved_docs, list):
                self.logger.error(
                    f"⚠️ [PREPARE CONTEXT] retrieved_docs is not a list: {type(retrieved_docs).__name__}"
                )
                self._set_state_value(state, "prompt_optimized_context", {
                    "prompt_optimized_text": "",
                    "structured_documents": {},
                    "document_count": 0,
                    "total_context_length": 0
                })
                return state

            valid_docs_count = 0
            docs_without_content = 0
            total_content_length = 0

            for doc in retrieved_docs:
                if not isinstance(doc, dict):
                    docs_without_content += 1
                    continue

                content = doc.get("content") or doc.get("text") or doc.get("content_text", "")
                if content and len(content.strip()) >= 10:
                    valid_docs_count += 1
                    total_content_length += len(content)
                else:
                    docs_without_content += 1
                    source = doc.get("source", "Unknown")
                    self.logger.debug(
                        f"[PREPARE CONTEXT] Document filtered: content missing or too short "
                        f"(source: {source}, content_length: {len(content) if content else 0})"
                    )

            if docs_without_content > 0:
                self.logger.warning(
                    f"⚠️ [PREPARE CONTEXT] Found {docs_without_content} documents without valid content "
                    f"out of {len(retrieved_docs)} total documents. "
                    f"Valid docs: {valid_docs_count}, Total content: {total_content_length} chars"
                )

            if valid_docs_count == 0:
                self.logger.error(
                    f"❌ [PREPARE CONTEXT] No valid documents with content found! "
                    f"Total docs: {len(retrieved_docs)}, "
                    f"Query: '{query[:50]}...'"
                )
                self._set_state_value(state, "prompt_optimized_context", {
                    "prompt_optimized_text": "",
                    "structured_documents": {},
                    "document_count": 0,
                    "total_context_length": 0
                })
                return state

            self.logger.info(
                f"✅ [PREPARE CONTEXT] Preparing context from {valid_docs_count} valid documents "
                f"(total: {len(retrieved_docs)}, content: {total_content_length} chars)"
            )

            prompt_optimized_context = self._build_prompt_optimized_context(
                retrieved_docs=retrieved_docs,
                query=query,
                extracted_keywords=extracted_keywords,
                query_type=query_type_str,
                legal_field=legal_field
            )

            self._set_state_value(state, "prompt_optimized_context", prompt_optimized_context)

            self._save_metadata_safely(state, "_last_executed_node", "prepare_document_context_for_prompt")
            self._update_processing_time(state, start_time)

            doc_count = prompt_optimized_context.get("document_count", 0)
            context_length = prompt_optimized_context.get("total_context_length", 0)
            content_validation = prompt_optimized_context.get("content_validation", {})

            self.logger.info(
                f"✅ [DOCUMENT PREPARATION] Prepared prompt context: "
                f"{doc_count} documents, "
                f"{context_length} chars, "
                f"input docs: {len(retrieved_docs)}"
            )

            if content_validation:
                has_content = content_validation.get("has_document_content", False)
                docs_with_content = content_validation.get("documents_with_content", 0)
                self.logger.info(
                    f"📊 [DOCUMENT PREPARATION] Content validation: "
                    f"has_content={has_content}, "
                    f"docs_with_content={docs_with_content}"
                )

            prompt_text = prompt_optimized_context.get("prompt_optimized_text", "")
            if prompt_text:
                self.logger.debug(
                    f"📝 [DOCUMENT PREPARATION] Prompt text preview (first 200 chars):\n"
                    f"{prompt_text[:200]}..."
                )
            else:
                self.logger.warning(
                    "⚠️ [DOCUMENT PREPARATION] prompt_optimized_text is empty!"
                )

        except Exception as e:
            self._handle_error(state, str(e), "문서 컨텍스트 준비 중 오류 발생")
            self._set_state_value(state, "prompt_optimized_context", {
                "prompt_optimized_text": "",
                "structured_documents": {},
                "document_count": 0,
                "total_context_length": 0
            })

        return state

    @observe(name="prepare_documents_and_terms")
    @with_state_optimization("prepare_documents_and_terms", enable_reduction=True)
    def prepare_documents_and_terms(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """통합된 문서 준비 및 용어 처리 (prepare_document_context_for_prompt + process_legal_terms)"""
        try:
            overall_start_time = time.time()

            context_start_time = time.time()

            retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
            query = self._get_state_value(state, "query", "")
            extracted_keywords = self._get_state_value(state, "extracted_keywords", [])
            query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", ""))
            legal_field = self._get_state_value(state, "legal_field", "")

            has_valid_docs = False
            if not retrieved_docs:
                self.logger.warning(
                    f"⚠️ [PREPARE CONTEXT] No retrieved_docs to prepare for prompt. "
                    f"Query: '{query[:50]}...', Query type: {query_type_str}. "
                    f"Skipping document context preparation and term extraction."
                )
                self._set_state_value(state, "prompt_optimized_context", {
                    "prompt_optimized_text": "",
                    "structured_documents": {},
                    "document_count": 0,
                    "total_context_length": 0
                })
            elif not isinstance(retrieved_docs, list):
                self.logger.error(
                    f"⚠️ [PREPARE CONTEXT] retrieved_docs is not a list: {type(retrieved_docs).__name__}. "
                    f"Skipping document context preparation and term extraction."
                )
                self._set_state_value(state, "prompt_optimized_context", {
                    "prompt_optimized_text": "",
                    "structured_documents": {},
                    "document_count": 0,
                    "total_context_length": 0
                })
            else:
                valid_docs_count = 0
                docs_without_content = 0
                total_content_length = 0

                for doc in retrieved_docs:
                    if not isinstance(doc, dict):
                        docs_without_content += 1
                        continue

                    content = doc.get("content") or doc.get("text") or doc.get("content_text", "")
                    if content and len(content.strip()) >= 10:
                        valid_docs_count += 1
                        total_content_length += len(content)
                    else:
                        docs_without_content += 1

                if docs_without_content > 0:
                    self.logger.warning(
                        f"⚠️ [PREPARE CONTEXT] Found {docs_without_content} documents without valid content "
                        f"out of {len(retrieved_docs)} total documents. "
                        f"Valid docs: {valid_docs_count}, Total content: {total_content_length} chars"
                    )

                if valid_docs_count == 0:
                    self.logger.error(
                        f"❌ [PREPARE CONTEXT] No valid documents with content found! "
                        f"Total docs: {len(retrieved_docs)}, "
                        f"Query: '{query[:50]}...'"
                    )
                    self._set_state_value(state, "prompt_optimized_context", {
                        "prompt_optimized_text": "",
                        "structured_documents": {},
                        "document_count": 0,
                        "total_context_length": 0
                    })
                else:
                    self.logger.info(
                        f"✅ [PREPARE CONTEXT] Preparing context from {valid_docs_count} valid documents "
                        f"(total: {len(retrieved_docs)}, content: {total_content_length} chars)"
                    )

                    prompt_optimized_context = self._build_prompt_optimized_context(
                        retrieved_docs=retrieved_docs,
                        query=query,
                        extracted_keywords=extracted_keywords,
                        query_type=query_type_str,
                        legal_field=legal_field
                    )

                    self._set_state_value(state, "prompt_optimized_context", prompt_optimized_context)

                    doc_count = prompt_optimized_context.get("document_count", 0)
                    context_length = prompt_optimized_context.get("total_context_length", 0)
                    self.logger.info(
                        f"✅ [DOCUMENT PREPARATION] Prepared prompt context: "
                        f"{doc_count} documents, "
                        f"{context_length} chars, "
                        f"input docs: {len(retrieved_docs)}"
                    )
                    has_valid_docs = True

            self._save_metadata_safely(state, "_last_executed_node", "prepare_documents_and_terms")
            self._update_processing_time(state, context_start_time)
            self._add_step(state, "문서 컨텍스트 준비", "프롬프트 최적화 컨텍스트 준비 완료")

            if has_valid_docs:
                terms_start_time = time.time()

                if "metadata" not in state or not isinstance(state.get("metadata"), dict):
                    state["metadata"] = {}
                state["metadata"] = dict(state["metadata"])
                state["metadata"]["_last_executed_node"] = "prepare_documents_and_terms"

                if "common" not in state or not isinstance(state.get("common"), dict):
                    state["common"] = {}
                if "metadata" not in state["common"]:
                    state["common"]["metadata"] = {}
                state["common"]["metadata"]["_last_executed_node"] = "prepare_documents_and_terms"

                existing_metadata_direct = state.get("metadata", {})
                existing_metadata_common = state.get("common", {}).get("metadata", {}) if isinstance(state.get("common"), dict) else {}
                existing_metadata = existing_metadata_direct if isinstance(existing_metadata_direct, dict) else {}
                if isinstance(existing_metadata_common, dict):
                    existing_metadata = {**existing_metadata, **existing_metadata_common}

                existing_top_level = state.get("retry_count", 0)
                saved_gen_retry = max(
                    existing_metadata.get("generation_retry_count", 0),
                    existing_top_level
                )
                saved_val_retry = existing_metadata.get("validation_retry_count", 0)

                retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
                all_terms = self._extract_terms_from_documents(retrieved_docs)
                self.logger.info(f"추출된 용어 수: {len(all_terms)}")

                if all_terms:
                    representative_terms = self._integrate_and_process_terms(all_terms)
                    metadata = dict(existing_metadata)
                    metadata["extracted_terms"] = representative_terms
                    metadata["total_terms_extracted"] = len(all_terms)
                    metadata["unique_terms"] = len(representative_terms)
                    metadata["generation_retry_count"] = saved_gen_retry
                    metadata["validation_retry_count"] = saved_val_retry
                    metadata["_last_executed_node"] = "prepare_documents_and_terms"
                    state["metadata"] = metadata

                    if "common" not in state:
                        state["common"] = {}
                    if "metadata" not in state["common"]:
                        state["common"]["metadata"] = {}
                    state["common"]["metadata"].update(metadata)
                    state["retry_count"] = saved_gen_retry

                    self._set_state_value(state, "metadata", metadata)
                    self._add_step(state, "용어 통합 완료", f"용어 통합 완료: {len(representative_terms)}개")
                    self.logger.info(f"통합된 용어 수: {len(representative_terms)}")
                else:
                    metadata = dict(existing_metadata)
                    metadata["extracted_terms"] = []
                    metadata["generation_retry_count"] = saved_gen_retry
                    metadata["validation_retry_count"] = saved_val_retry
                    metadata["_last_executed_node"] = "prepare_documents_and_terms"
                    state["metadata"] = metadata

                    if "common" not in state:
                        state["common"] = {}
                    if "metadata" not in state["common"]:
                        state["common"]["metadata"] = {}
                    state["common"]["metadata"].update(metadata)
                    state["retry_count"] = saved_gen_retry

                    self._set_state_value(state, "metadata", metadata)
                    self._add_step(state, "용어 추출 없음", "용어 추출 없음 (문서 내용 부족)")

                self._update_processing_time(state, terms_start_time)
            else:
                self.logger.info(
                    f"⏭️ [TERM PROCESSING] Skipping term extraction and processing "
                    f"(no valid retrieved_docs available)"
                )

        except Exception as e:
            self._handle_error(state, str(e), "문서 준비 및 용어 처리 중 오류 발생")

        return state

