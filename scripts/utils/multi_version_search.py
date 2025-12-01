"""
멀티 버전 검색 및 A/B 테스트 시스템

여러 FAISS 버전을 동시에 검색하여 결과를 비교하거나 앙상블합니다.
"""
import logging
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class MultiVersionSearch:
    """여러 FAISS 버전을 동시에 검색하는 클래스"""
    
    def __init__(self, faiss_version_manager):
        """
        초기화
        
        Args:
            faiss_version_manager: FAISSVersionManager 인스턴스
        """
        self.version_manager = faiss_version_manager
        self.loaded_indices = {}  # 메모리에 로드된 인덱스 캐시
    
    def load_version(self, version_name: str) -> Optional[Dict[str, Any]]:
        """
        버전을 메모리에 로드
        
        Args:
            version_name: 버전 이름
        
        Returns:
            Optional[Dict]: 인덱스 데이터 (index, id_mapping, metadata, version_info)
        """
        if version_name in self.loaded_indices:
            return self.loaded_indices[version_name]
        
        index_data = self.version_manager.load_index(version_name)
        if index_data:
            self.loaded_indices[version_name] = index_data
            logger.info(f"Loaded version {version_name} into memory")
            return index_data
        else:
            logger.warning(f"Failed to load version {version_name}")
            return None
    
    def search_all_versions(
        self,
        query_vector: np.ndarray,
        versions: List[str],
        k: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        모든 버전에서 동시 검색
        
        Args:
            query_vector: 쿼리 벡터
            versions: 검색할 버전 이름 리스트
            k: 각 버전에서 반환할 최대 결과 수
        
        Returns:
            Dict[str, List[Dict]]: 버전별 검색 결과
        """
        results = {}
        
        for version in versions:
            try:
                version_data = self.load_version(version)
                if version_data is None:
                    results[version] = []
                    continue
                
                index = version_data['index']
                id_mapping = version_data.get('id_mapping', {})
                metadata = version_data.get('metadata', [])
                
                if query_vector.ndim == 1:
                    query_vector = query_vector.reshape(1, -1)
                
                query_vector = query_vector.astype('float32')
                
                distances, indices = index.search(query_vector, k)
                
                version_results = []
                for dist, idx in zip(distances[0], indices[0]):
                    if idx != -1:
                        chunk_id = id_mapping.get(str(idx), id_mapping.get(idx, None))
                        if chunk_id is None and idx < len(metadata):
                            chunk_id = metadata[idx].get('chunk_id', idx)
                        
                        result = {
                            "chunk_id": chunk_id,
                            "distance": float(dist),
                            "similarity": float(1.0 / (1.0 + dist)) if dist > 0 else 1.0,
                            "metadata": metadata[idx] if idx < len(metadata) else {}
                        }
                        version_results.append(result)
                
                results[version] = version_results
                
            except Exception as e:
                logger.error(f"Error searching version {version}: {e}")
                results[version] = []
        
        return results
    
    def ensemble_search(
        self,
        query_vector: np.ndarray,
        versions: List[str],
        weights: Optional[List[float]] = None,
        k: int = 5
    ) -> List[Dict]:
        """
        여러 버전의 결과를 앙상블
        
        Args:
            query_vector: 쿼리 벡터
            versions: 검색할 버전 이름 리스트
            weights: 버전별 가중치 (None이면 균등 가중치)
            k: 최종 반환할 결과 수
        
        Returns:
            List[Dict]: 앙상블된 검색 결과
        """
        all_results = self.search_all_versions(query_vector, versions, k * 2)
        
        if weights is None:
            weights = [1.0 / len(versions)] * len(versions)
        
        if len(weights) != len(versions):
            logger.warning("Weights length mismatch, using equal weights")
            weights = [1.0 / len(versions)] * len(versions)
        
        doc_scores = {}
        
        for version, weight in zip(versions, weights):
            for rank, result in enumerate(all_results[version]):
                chunk_id = result.get("chunk_id")
                if chunk_id is None:
                    continue
                
                similarity = result.get("similarity", 0.0)
                score = weight * similarity * (1.0 / (rank + 1))
                
                if chunk_id not in doc_scores:
                    doc_scores[chunk_id] = {
                        "chunk_id": chunk_id,
                        "ensemble_score": 0.0,
                        "metadata": result.get("metadata", {}),
                        "found_in_versions": [],
                        "individual_scores": {}
                    }
                
                doc_scores[chunk_id]["ensemble_score"] += score
                doc_scores[chunk_id]["found_in_versions"].append(version)
                doc_scores[chunk_id]["individual_scores"][version] = similarity
        
        ranked = sorted(
            doc_scores.items(),
            key=lambda x: x[1]["ensemble_score"],
            reverse=True
        )
        
        return [
            {
                "chunk_id": doc_id,
                "ensemble_score": data["ensemble_score"],
                "metadata": data["metadata"],
                "found_in_versions": data["found_in_versions"],
                "individual_scores": data["individual_scores"]
            }
            for doc_id, data in ranked[:k]
        ]
    
    def compare_results(
        self,
        query_vector: np.ndarray,
        version1: str,
        version2: str,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        두 버전의 검색 결과 비교
        
        Args:
            query_vector: 쿼리 벡터
            version1: 첫 번째 버전 이름
            version2: 두 번째 버전 이름
            k: 검색할 결과 수
        
        Returns:
            Dict: 비교 결과 (공통 결과, 버전별 고유 결과 등)
        """
        results1 = self.search_all_versions(query_vector, [version1], k)[version1]
        results2 = self.search_all_versions(query_vector, [version2], k)[version2]
        
        chunk_ids1 = {r["chunk_id"] for r in results1 if r.get("chunk_id")}
        chunk_ids2 = {r["chunk_id"] for r in results2 if r.get("chunk_id")}
        
        common = chunk_ids1 & chunk_ids2
        unique_to_v1 = chunk_ids1 - chunk_ids2
        unique_to_v2 = chunk_ids2 - chunk_ids1
        
        return {
            "version1": version1,
            "version2": version2,
            "common_results": len(common),
            "unique_to_v1": len(unique_to_v1),
            "unique_to_v2": len(unique_to_v2),
            "jaccard_similarity": len(common) / len(chunk_ids1 | chunk_ids2) if (chunk_ids1 | chunk_ids2) else 0.0,
            "common_chunk_ids": list(common),
            "unique_to_v1_chunk_ids": list(unique_to_v1),
            "unique_to_v2_chunk_ids": list(unique_to_v2),
            "results1": results1,
            "results2": results2
        }
    
    def clear_cache(self, version_name: Optional[str] = None):
        """
        메모리 캐시 정리
        
        Args:
            version_name: 특정 버전만 정리 (None이면 전체 정리)
        """
        if version_name:
            if version_name in self.loaded_indices:
                del self.loaded_indices[version_name]
                logger.info(f"Cleared cache for version {version_name}")
        else:
            self.loaded_indices.clear()
            logger.info("Cleared all version caches")

