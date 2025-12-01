# -*- coding: utf-8 -*-
"""
검색 결과 타입 균형 조정 유틸리티
각 문서 타입에서 최소 1개씩 선택 후 관련도 순으로 채움
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Dict, List, Any, Optional

logger = get_logger(__name__)


class SearchResultBalancer:
    """검색 결과 타입 균형 조정 클래스"""
    
    def __init__(self, min_per_type: int = 1, max_per_type: int = 5):
        """
        초기화
        
        Args:
            min_per_type: 각 타입당 최소 선택 개수
            max_per_type: 각 타입당 최대 선택 개수
        """
        self.logger = get_logger(__name__)
        self.min_per_type = min_per_type
        self.max_per_type = max_per_type
        # 샘플링된 문서는 항상 포함 (타입 다양성 보장)
        self.force_include_samples = True
    
    def balance_search_results(
        self, 
        results: Dict[str, List[Dict[str, Any]]],
        total_limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        검색 결과 타입 균형 조정
        
        Args:
            results: 타입별 검색 결과 딕셔너리
                예: {
                    "statute_article": [...],
                    "case_paragraph": [...],
                    "decision_paragraph": [...],
                    "interpretation_paragraph": [...]
                }
            total_limit: 전체 결과 제한
            
        Returns:
            List[Dict[str, Any]]: 균형 조정된 검색 결과
        """
        balanced = []
        type_order = [
            "statute_article",
            "case_paragraph", 
            "decision_paragraph",
            "interpretation_paragraph"
        ]
        
        # 1단계: 각 타입에서 최소 1개씩 선택 (샘플링된 문서 우선, 강제 포함)
        for doc_type in type_order:
            if doc_type in results and results[doc_type]:
                docs = results[doc_type]
                
                # 샘플링된 문서와 일반 문서 분리
                sample_docs = []
                normal_docs = []
                for doc in docs:
                    if not isinstance(doc, dict):
                        continue
                    is_sample = doc.get("metadata", {}).get("is_sample", False) or doc.get("search_type") == "type_sample"
                    if is_sample:
                        sample_docs.append(doc)
                    else:
                        normal_docs.append(doc)
                
                # 샘플링된 문서를 우선 포함, 나머지는 관련도 순으로 정렬
                sorted_normal_docs = sorted(
                    normal_docs,
                    key=lambda x: x.get("relevance_score", 0.0) if isinstance(x, dict) else 0.0,
                    reverse=True
                )
                
                # 샘플링된 문서를 먼저 추가, 그 다음 일반 문서
                all_sorted_docs = sample_docs + sorted_normal_docs
                
                # 샘플링된 문서는 항상 포함 (타입 다양성 보장)
                if sample_docs and self.force_include_samples:
                    balanced.extend(sample_docs)
                    self.logger.info(
                        f"✅ [BALANCER] {doc_type}: 샘플링된 문서 {len(sample_docs)}개 강제 포함"
                    )
                    # 나머지는 min_per_type에서 샘플링된 문서 수를 뺀 만큼만 추가
                    remaining_needed = max(0, self.min_per_type - len(sample_docs))
                    if remaining_needed > 0 and sorted_normal_docs:
                        balanced.extend(sorted_normal_docs[:remaining_needed])
                else:
                    # 샘플링된 문서가 없으면 일반 로직
                    selected = all_sorted_docs[:self.min_per_type]
                    balanced.extend(selected)
                
                self.logger.debug(
                    f"Selected {len([d for d in balanced if (d.get('type') or d.get('source_type')) == doc_type])} documents from {doc_type} "
                    f"(samples: {len(sample_docs)}, normal: {len(sorted_normal_docs)}, min_per_type={self.min_per_type})"
                )
        
        # 2단계: 나머지는 관련도 순으로 채움
        all_results = []
        for doc_type in type_order:
            if doc_type in results and results[doc_type]:
                docs = results[doc_type]
                # 이미 선택된 문서 제외
                already_selected = {id(doc) for doc in balanced}
                remaining = [
                    doc for doc in docs 
                    if id(doc) not in already_selected
                ]
                all_results.extend(remaining)
        
        # 관련도 순으로 정렬
        all_results.sort(
            key=lambda x: x.get("relevance_score", 0.0) if isinstance(x, dict) else 0.0,
            reverse=True
        )
        
        # 이미 선택된 것 제외하고 추가
        for doc in all_results:
            if len(balanced) >= total_limit:
                break
            if doc not in balanced:
                balanced.append(doc)
        
        self.logger.info(
            f"Balanced search results: {len(balanced)} total "
            f"(min_per_type={self.min_per_type}, total_limit={total_limit})"
        )
        
        return balanced[:total_limit]
    
    def group_results_by_type(
        self, 
        results: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        검색 결과를 타입별로 그룹화
        
        Args:
            results: 검색 결과 리스트
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: 타입별 그룹화된 결과
        """
        grouped = {}
        
        for doc in results:
            if not isinstance(doc, dict):
                continue
            
            doc_type = (
                doc.get("type") or
                doc.get("source_type") or
                doc.get("metadata", {}).get("type") if isinstance(doc.get("metadata"), dict) else None or
                doc.get("metadata", {}).get("source_type") if isinstance(doc.get("metadata"), dict) else None or
                "unknown"
            )
            
            if doc_type not in grouped:
                grouped[doc_type] = []
            
            grouped[doc_type].append(doc)
        
        return grouped

