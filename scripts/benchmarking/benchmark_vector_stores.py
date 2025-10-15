#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAISS vs ChromaDB 벡터 스토어 성능 비교 벤치마킹 스크립트
LawFirmAI 프로젝트 - TASK 1.2.2
"""

import os
import sys
import time
import psutil
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime
import logging

# 벡터 스토어 라이브러리
import faiss
import chromadb
from chromadb.config import Settings

# 임베딩 모델
from sentence_transformers import SentenceTransformer

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreBenchmark:
    """벡터 스토어 성능 벤치마킹 클래스"""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.results = {}
        self.test_data = self._load_test_data()
        self.embedding_model = None
        
    def _load_test_data(self) -> List[Dict[str, str]]:
        """법률 도메인 테스트 데이터 로드"""
        return [
            {
                "id": f"precedent_{i}",
                "text": f"판례 {i}: 계약 해지와 관련된 손해배상 청구권에 대한 대법원 판례입니다. 계약 당사자 간의 의무 위반 시 손해배상 책임이 발생하며, 이는 민법상 손해배상 제도에 근거합니다.",
                "metadata": {
                    "type": "precedent",
                    "court": "대법원",
                    "year": 2020 + (i % 4),
                    "category": "contract"
                }
            }
            for i in range(1000)  # 1000개 테스트 문서
        ]
    
    def _load_embedding_model(self):
        """임베딩 모델 로드"""
        if self.embedding_model is None:
            logger.info("임베딩 모델 로딩...")
            self.embedding_model = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            )
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """텍스트 임베딩 생성"""
        self._load_embedding_model()
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings.astype('float32')
    
    def benchmark_faiss(self) -> Dict[str, Any]:
        """FAISS 벡터 스토어 벤치마킹"""
        logger.info("FAISS 벡터 스토어 벤치마킹 시작...")
        
        results = {
            "vector_store": "FAISS",
            "benchmark_time": datetime.now().isoformat()
        }
        
        try:
            # 임베딩 생성
            texts = [doc["text"] for doc in self.test_data]
            embeddings = self._generate_embeddings(texts)
            
            # FAISS 인덱스 생성 및 구축
            start_time = time.time()
            index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product (코사인 유사도)
            index.add(embeddings)
            build_time = time.time() - start_time
            
            # 인덱스 정보
            index_info = {
                "num_vectors": index.ntotal,
                "dimension": index.d,
                "build_time": build_time,
                "index_size_mb": self._get_faiss_index_size(index)
            }
            
            # 검색 성능 테스트
            search_results = self._test_faiss_search(index, embeddings)
            
            # 메모리 사용량 측정
            memory_usage = self._get_memory_usage()
            
            results.update({
                "index_info": index_info,
                "search_results": search_results,
                "memory_usage_mb": memory_usage
            })
            
            # 인덱스 저장 테스트
            save_results = self._test_faiss_save_load(index)
            results["save_load_results"] = save_results
            
        except Exception as e:
            logger.error(f"FAISS 벤치마킹 실패: {e}")
            results["error"] = str(e)
            
        return results
    
    def benchmark_chromadb(self) -> Dict[str, Any]:
        """ChromaDB 벡터 스토어 벤치마킹"""
        logger.info("ChromaDB 벡터 스토어 벤치마킹 시작...")
        
        results = {
            "vector_store": "ChromaDB",
            "benchmark_time": datetime.now().isoformat()
        }
        
        try:
            # ChromaDB 클라이언트 생성
            client = chromadb.Client(Settings(
                persist_directory="./chroma_db",
                anonymized_telemetry=False
            ))
            
            # 컬렉션 생성
            collection = client.create_collection(
                name="legal_documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            # 데이터 추가
            start_time = time.time()
            
            # 배치로 데이터 추가 (ChromaDB는 자동으로 임베딩 생성)
            batch_size = 100
            for i in range(0, len(self.test_data), batch_size):
                batch = self.test_data[i:i+batch_size]
                collection.add(
                    documents=[doc["text"] for doc in batch],
                    metadatas=[doc["metadata"] for doc in batch],
                    ids=[doc["id"] for doc in batch]
                )
            
            build_time = time.time() - start_time
            
            # 컬렉션 정보
            collection_info = {
                "num_documents": collection.count(),
                "build_time": build_time,
                "collection_size_mb": self._get_chromadb_size()
            }
            
            # 검색 성능 테스트
            search_results = self._test_chromadb_search(collection)
            
            # 메모리 사용량 측정
            memory_usage = self._get_memory_usage()
            
            results.update({
                "collection_info": collection_info,
                "search_results": search_results,
                "memory_usage_mb": memory_usage
            })
            
            # 저장/로드 테스트
            save_results = self._test_chromadb_save_load()
            results["save_load_results"] = save_results
            
            # 정리
            client.delete_collection("legal_documents")
            
        except Exception as e:
            logger.error(f"ChromaDB 벤치마킹 실패: {e}")
            results["error"] = str(e)
            
        return results
    
    def _test_faiss_search(self, index, embeddings, top_k: int = 5) -> Dict[str, Any]:
        """FAISS 검색 성능 테스트"""
        results = {
            "total_search_time": 0,
            "average_search_time": 0,
            "queries_per_second": 0,
            "search_results": []
        }
        
        # 테스트 쿼리 생성
        test_queries = [
            "계약 해지 손해배상",
            "근로기준법 휴게시간",
            "부동산 매매계약 중도금",
            "이혼 재산분할",
            "손해배상 청구권 소멸시효"
        ]
        
        total_time = 0
        search_results = []
        
        for query in test_queries:
            try:
                # 쿼리 임베딩 생성
                query_embedding = self.embedding_model.encode([query])
                query_embedding = query_embedding.astype('float32')
                
                # 검색 실행
                start_time = time.time()
                scores, indices = index.search(query_embedding, top_k)
                search_time = time.time() - start_time
                
                total_time += search_time
                
                # 결과 저장
                search_results.append({
                    "query": query,
                    "search_time": search_time,
                    "top_k": top_k,
                    "scores": scores[0].tolist(),
                    "indices": indices[0].tolist()
                })
                
            except Exception as e:
                logger.error(f"FAISS 검색 테스트 실패: {e}")
                continue
        
        results.update({
            "total_search_time": total_time,
            "average_search_time": total_time / len(test_queries) if test_queries else 0,
            "queries_per_second": len(test_queries) / total_time if total_time > 0 else 0,
            "search_results": search_results
        })
        
        return results
    
    def _test_chromadb_search(self, collection, top_k: int = 5) -> Dict[str, Any]:
        """ChromaDB 검색 성능 테스트"""
        results = {
            "total_search_time": 0,
            "average_search_time": 0,
            "queries_per_second": 0,
            "search_results": []
        }
        
        # 테스트 쿼리
        test_queries = [
            "계약 해지 손해배상",
            "근로기준법 휴게시간",
            "부동산 매매계약 중도금",
            "이혼 재산분할",
            "손해배상 청구권 소멸시효"
        ]
        
        total_time = 0
        search_results = []
        
        for query in test_queries:
            try:
                # 검색 실행
                start_time = time.time()
                result = collection.query(
                    query_texts=[query],
                    n_results=top_k
                )
                search_time = time.time() - start_time
                
                total_time += search_time
                
                # 결과 저장
                search_results.append({
                    "query": query,
                    "search_time": search_time,
                    "top_k": top_k,
                    "distances": result["distances"][0] if result["distances"] else [],
                    "ids": result["ids"][0] if result["ids"] else []
                })
                
            except Exception as e:
                logger.error(f"ChromaDB 검색 테스트 실패: {e}")
                continue
        
        results.update({
            "total_search_time": total_time,
            "average_search_time": total_time / len(test_queries) if test_queries else 0,
            "queries_per_second": len(test_queries) / total_time if total_time > 0 else 0,
            "search_results": search_results
        })
        
        return results
    
    def _test_faiss_save_load(self, index) -> Dict[str, Any]:
        """FAISS 저장/로드 테스트"""
        results = {
            "save_time": 0,
            "load_time": 0,
            "file_size_mb": 0
        }
        
        try:
            # 저장 테스트
            start_time = time.time()
            faiss.write_index(index, "test_faiss_index.bin")
            save_time = time.time() - start_time
            
            # 파일 크기 측정
            file_size = os.path.getsize("test_faiss_index.bin") / 1024 / 1024
            
            # 로드 테스트
            start_time = time.time()
            loaded_index = faiss.read_index("test_faiss_index.bin")
            load_time = time.time() - start_time
            
            # 정리
            os.remove("test_faiss_index.bin")
            
            results.update({
                "save_time": save_time,
                "load_time": load_time,
                "file_size_mb": file_size
            })
            
        except Exception as e:
            logger.error(f"FAISS 저장/로드 테스트 실패: {e}")
            results["error"] = str(e)
        
        return results
    
    def _test_chromadb_save_load(self) -> Dict[str, Any]:
        """ChromaDB 저장/로드 테스트"""
        results = {
            "persist_time": 0,
            "load_time": 0,
            "db_size_mb": 0
        }
        
        try:
            # ChromaDB는 자동으로 지속화되므로 별도 저장 과정 없음
            results["persist_time"] = 0
            
            # 로드 테스트
            start_time = time.time()
            client = chromadb.Client(Settings(
                persist_directory="./chroma_db",
                anonymized_telemetry=False
            ))
            collection = client.get_collection("legal_documents")
            load_time = time.time() - start_time
            
            # DB 크기 측정
            db_size = self._get_chromadb_size()
            
            results.update({
                "load_time": load_time,
                "db_size_mb": db_size
            })
            
        except Exception as e:
            logger.error(f"ChromaDB 저장/로드 테스트 실패: {e}")
            results["error"] = str(e)
        
        return results
    
    def _get_faiss_index_size(self, index) -> float:
        """FAISS 인덱스 크기 반환 (MB)"""
        # FAISS 인덱스의 메모리 사용량 추정
        return (index.ntotal * index.d * 4) / 1024 / 1024  # float32 = 4 bytes
    
    def _get_chromadb_size(self) -> float:
        """ChromaDB 크기 반환 (MB)"""
        try:
            total_size = 0
            for root, dirs, files in os.walk("./chroma_db"):
                for file in files:
                    filepath = os.path.join(root, file)
                    total_size += os.path.getsize(filepath)
            return total_size / 1024 / 1024
        except:
            return 0
    
    def _get_memory_usage(self) -> float:
        """현재 메모리 사용량 반환 (MB)"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def run_benchmark(self) -> Dict[str, Any]:
        """전체 벤치마킹 실행"""
        logger.info("벡터 스토어 벤치마킹 시작...")
        
        # 시스템 정보 수집
        system_info = {
            "embedding_dimension": self.embedding_dim,
            "test_data_size": len(self.test_data),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "python_version": sys.version
        }
        
        # 각 벡터 스토어 벤치마킹 실행
        faiss_results = self.benchmark_faiss()
        chromadb_results = self.benchmark_chromadb()
        
        # 결과 종합
        benchmark_results = {
            "system_info": system_info,
            "faiss": faiss_results,
            "chromadb": chromadb_results,
            "comparison": self._compare_vector_stores(faiss_results, chromadb_results)
        }
        
        return benchmark_results
    
    def _compare_vector_stores(self, faiss_results: Dict, chromadb_results: Dict) -> Dict[str, Any]:
        """벡터 스토어 성능 비교"""
        comparison = {
            "build_time_comparison": {},
            "search_speed_comparison": {},
            "memory_usage_comparison": {},
            "storage_size_comparison": {},
            "recommendation": ""
        }
        
        try:
            # 구축 시간 비교
            if "index_info" in faiss_results and "collection_info" in chromadb_results:
                faiss_build_time = faiss_results["index_info"]["build_time"]
                chromadb_build_time = chromadb_results["collection_info"]["build_time"]
                
                comparison["build_time_comparison"] = {
                    "faiss_seconds": faiss_build_time,
                    "chromadb_seconds": chromadb_build_time,
                    "speed_ratio": faiss_build_time / chromadb_build_time if chromadb_build_time > 0 else 0
                }
            
            # 검색 속도 비교
            if "search_results" in faiss_results and "search_results" in chromadb_results:
                faiss_qps = faiss_results["search_results"]["queries_per_second"]
                chromadb_qps = chromadb_results["search_results"]["queries_per_second"]
                
                comparison["search_speed_comparison"] = {
                    "faiss_qps": faiss_qps,
                    "chromadb_qps": chromadb_qps,
                    "speed_ratio": faiss_qps / chromadb_qps if chromadb_qps > 0 else 0
                }
            
            # 메모리 사용량 비교
            faiss_memory = faiss_results.get("memory_usage_mb", 0)
            chromadb_memory = chromadb_results.get("memory_usage_mb", 0)
            
            comparison["memory_usage_comparison"] = {
                "faiss_mb": faiss_memory,
                "chromadb_mb": chromadb_memory,
                "memory_ratio": faiss_memory / chromadb_memory if chromadb_memory > 0 else 0
            }
            
            # 저장소 크기 비교
            faiss_size = faiss_results.get("index_info", {}).get("index_size_mb", 0)
            chromadb_size = chromadb_results.get("collection_info", {}).get("collection_size_mb", 0)
            
            comparison["storage_size_comparison"] = {
                "faiss_mb": faiss_size,
                "chromadb_mb": chromadb_size,
                "size_ratio": faiss_size / chromadb_size if chromadb_size > 0 else 0
            }
            
            # 권장사항 생성
            comparison["recommendation"] = self._generate_recommendation(comparison)
            
        except Exception as e:
            logger.error(f"벡터 스토어 비교 중 오류: {e}")
            comparison["error"] = str(e)
        
        return comparison
    
    def _generate_recommendation(self, comparison: Dict) -> str:
        """벡터 스토어 선택 권장사항 생성"""
        try:
            search_ratio = comparison.get("search_speed_comparison", {}).get("speed_ratio", 1)
            memory_ratio = comparison.get("memory_usage_comparison", {}).get("memory_ratio", 1)
            size_ratio = comparison.get("storage_size_comparison", {}).get("size_ratio", 1)
            
            # HuggingFace Spaces 환경 고려
            if search_ratio > 1.5:  # FAISS가 검색이 더 빠름
                return "FAISS 권장: 검색 성능이 우수하여 실시간 검색에 적합"
            elif memory_ratio > 1.3:  # FAISS가 메모리를 더 많이 사용
                return "ChromaDB 권장: 메모리 효율성이 우수하여 제한된 환경에 적합"
            elif size_ratio > 1.2:  # FAISS가 저장공간을 더 많이 사용
                return "ChromaDB 권장: 저장공간 효율성이 우수"
            else:
                return "FAISS 권장: 높은 성능과 안정성으로 프로덕션 환경에 적합"
                
        except Exception as e:
            return f"권장사항 생성 실패: {e}"
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """벤치마킹 결과 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vector_store_benchmark_results_{timestamp}.json"
        
        filepath = os.path.join("benchmark_results", filename)
        os.makedirs("benchmark_results", exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"벤치마킹 결과 저장: {filepath}")
        return filepath

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="벡터 스토어 성능 벤치마킹")
    parser.add_argument("--embedding-dim", type=int, default=768, help="임베딩 차원")
    parser.add_argument("--output", help="결과 저장 파일명")
    
    args = parser.parse_args()
    
    # 벤치마킹 실행
    benchmark = VectorStoreBenchmark(embedding_dim=args.embedding_dim)
    results = benchmark.run_benchmark()
    
    # 결과 저장
    output_file = benchmark.save_results(results, args.output)
    
    # 결과 요약 출력
    print("\n" + "="*50)
    print("벡터 스토어 벤치마킹 결과 요약")
    print("="*50)
    
    if "faiss" in results and "search_results" in results["faiss"]:
        faiss_info = results["faiss"]["search_results"]
        print(f"FAISS - QPS: {faiss_info.get('queries_per_second', 0):.2f}, "
              f"평균 검색시간: {faiss_info.get('average_search_time', 0):.4f}초")
    
    if "chromadb" in results and "search_results" in results["chromadb"]:
        chromadb_info = results["chromadb"]["search_results"]
        print(f"ChromaDB - QPS: {chromadb_info.get('queries_per_second', 0):.2f}, "
              f"평균 검색시간: {chromadb_info.get('average_search_time', 0):.4f}초")
    
    if "comparison" in results and "recommendation" in results["comparison"]:
        print(f"\n권장사항: {results['comparison']['recommendation']}")
    
    print(f"\n상세 결과: {output_file}")

if __name__ == "__main__":
    main()
