#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAISS vs ChromaDB ë²¡í„° ?¤í† ???±ëŠ¥ ë¹„êµ ë²¤ì¹˜ë§ˆí‚¹ ?¤í¬ë¦½íŠ¸
LawFirmAI ?„ë¡œ?íŠ¸ - TASK 1.2.2
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

# ë²¡í„° ?¤í† ???¼ì´ë¸ŒëŸ¬ë¦?
import faiss
import chromadb
from chromadb.config import Settings

# ?„ë² ??ëª¨ë¸
from sentence_transformers import SentenceTransformer

# ë¡œê¹… ?¤ì •
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreBenchmark:
    """ë²¡í„° ?¤í† ???±ëŠ¥ ë²¤ì¹˜ë§ˆí‚¹ ?´ë˜??""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.results = {}
        self.test_data = self._load_test_data()
        self.embedding_model = None
        
    def _load_test_data(self) -> List[Dict[str, str]]:
        """ë²•ë¥  ?„ë©”???ŒìŠ¤???°ì´??ë¡œë“œ"""
        return [
            {
                "id": f"precedent_{i}",
                "text": f"?ë? {i}: ê³„ì•½ ?´ì??€ ê´€?¨ëœ ?í•´ë°°ìƒ ì²?µ¬ê¶Œì— ?€???€ë²•ì› ?ë??…ë‹ˆ?? ê³„ì•½ ?¹ì‚¬??ê°„ì˜ ?˜ë¬´ ?„ë°˜ ???í•´ë°°ìƒ ì±…ì„??ë°œìƒ?˜ë©°, ?´ëŠ” ë¯¼ë²•???í•´ë°°ìƒ ?œë„??ê·¼ê±°?©ë‹ˆ??",
                "metadata": {
                    "type": "precedent",
                    "court": "?€ë²•ì›",
                    "year": 2020 + (i % 4),
                    "category": "contract"
                }
            }
            for i in range(1000)  # 1000ê°??ŒìŠ¤??ë¬¸ì„œ
        ]
    
    def _load_embedding_model(self):
        """?„ë² ??ëª¨ë¸ ë¡œë“œ"""
        if self.embedding_model is None:
            logger.info("?„ë² ??ëª¨ë¸ ë¡œë”©...")
            self.embedding_model = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            )
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """?ìŠ¤???„ë² ???ì„±"""
        self._load_embedding_model()
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings.astype('float32')
    
    def benchmark_faiss(self) -> Dict[str, Any]:
        """FAISS ë²¡í„° ?¤í† ??ë²¤ì¹˜ë§ˆí‚¹"""
        logger.info("FAISS ë²¡í„° ?¤í† ??ë²¤ì¹˜ë§ˆí‚¹ ?œì‘...")
        
        results = {
            "vector_store": "FAISS",
            "benchmark_time": datetime.now().isoformat()
        }
        
        try:
            # ?„ë² ???ì„±
            texts = [doc["text"] for doc in self.test_data]
            embeddings = self._generate_embeddings(texts)
            
            # FAISS ?¸ë±???ì„± ë°?êµ¬ì¶•
            start_time = time.time()
            index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product (ì½”ì‚¬??? ì‚¬??
            index.add(embeddings)
            build_time = time.time() - start_time
            
            # ?¸ë±???•ë³´
            index_info = {
                "num_vectors": index.ntotal,
                "dimension": index.d,
                "build_time": build_time,
                "index_size_mb": self._get_faiss_index_size(index)
            }
            
            # ê²€???±ëŠ¥ ?ŒìŠ¤??
            search_results = self._test_faiss_search(index, embeddings)
            
            # ë©”ëª¨ë¦??¬ìš©??ì¸¡ì •
            memory_usage = self._get_memory_usage()
            
            results.update({
                "index_info": index_info,
                "search_results": search_results,
                "memory_usage_mb": memory_usage
            })
            
            # ?¸ë±???€???ŒìŠ¤??
            save_results = self._test_faiss_save_load(index)
            results["save_load_results"] = save_results
            
        except Exception as e:
            logger.error(f"FAISS ë²¤ì¹˜ë§ˆí‚¹ ?¤íŒ¨: {e}")
            results["error"] = str(e)
            
        return results
    
    def benchmark_chromadb(self) -> Dict[str, Any]:
        """ChromaDB ë²¡í„° ?¤í† ??ë²¤ì¹˜ë§ˆí‚¹"""
        logger.info("ChromaDB ë²¡í„° ?¤í† ??ë²¤ì¹˜ë§ˆí‚¹ ?œì‘...")
        
        results = {
            "vector_store": "ChromaDB",
            "benchmark_time": datetime.now().isoformat()
        }
        
        try:
            # ChromaDB ?´ë¼?´ì–¸???ì„±
            client = chromadb.Client(Settings(
                persist_directory="./chroma_db",
                anonymized_telemetry=False
            ))
            
            # ì»¬ë ‰???ì„±
            collection = client.create_collection(
                name="legal_documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            # ?°ì´??ì¶”ê?
            start_time = time.time()
            
            # ë°°ì¹˜ë¡??°ì´??ì¶”ê? (ChromaDB???ë™?¼ë¡œ ?„ë² ???ì„±)
            batch_size = 100
            for i in range(0, len(self.test_data), batch_size):
                batch = self.test_data[i:i+batch_size]
                collection.add(
                    documents=[doc["text"] for doc in batch],
                    metadatas=[doc["metadata"] for doc in batch],
                    ids=[doc["id"] for doc in batch]
                )
            
            build_time = time.time() - start_time
            
            # ì»¬ë ‰???•ë³´
            collection_info = {
                "num_documents": collection.count(),
                "build_time": build_time,
                "collection_size_mb": self._get_chromadb_size()
            }
            
            # ê²€???±ëŠ¥ ?ŒìŠ¤??
            search_results = self._test_chromadb_search(collection)
            
            # ë©”ëª¨ë¦??¬ìš©??ì¸¡ì •
            memory_usage = self._get_memory_usage()
            
            results.update({
                "collection_info": collection_info,
                "search_results": search_results,
                "memory_usage_mb": memory_usage
            })
            
            # ?€??ë¡œë“œ ?ŒìŠ¤??
            save_results = self._test_chromadb_save_load()
            results["save_load_results"] = save_results
            
            # ?•ë¦¬
            client.delete_collection("legal_documents")
            
        except Exception as e:
            logger.error(f"ChromaDB ë²¤ì¹˜ë§ˆí‚¹ ?¤íŒ¨: {e}")
            results["error"] = str(e)
            
        return results
    
    def _test_faiss_search(self, index, embeddings, top_k: int = 5) -> Dict[str, Any]:
        """FAISS ê²€???±ëŠ¥ ?ŒìŠ¤??""
        results = {
            "total_search_time": 0,
            "average_search_time": 0,
            "queries_per_second": 0,
            "search_results": []
        }
        
        # ?ŒìŠ¤??ì¿¼ë¦¬ ?ì„±
        test_queries = [
            "ê³„ì•½ ?´ì? ?í•´ë°°ìƒ",
            "ê·¼ë¡œê¸°ì?ë²??´ê²Œ?œê°„",
            "ë¶€?™ì‚° ë§¤ë§¤ê³„ì•½ ì¤‘ë„ê¸?,
            "?´í˜¼ ?¬ì‚°ë¶„í• ",
            "?í•´ë°°ìƒ ì²?µ¬ê¶??Œë©¸?œíš¨"
        ]
        
        total_time = 0
        search_results = []
        
        for query in test_queries:
            try:
                # ì¿¼ë¦¬ ?„ë² ???ì„±
                query_embedding = self.embedding_model.encode([query])
                query_embedding = query_embedding.astype('float32')
                
                # ê²€???¤í–‰
                start_time = time.time()
                scores, indices = index.search(query_embedding, top_k)
                search_time = time.time() - start_time
                
                total_time += search_time
                
                # ê²°ê³¼ ?€??
                search_results.append({
                    "query": query,
                    "search_time": search_time,
                    "top_k": top_k,
                    "scores": scores[0].tolist(),
                    "indices": indices[0].tolist()
                })
                
            except Exception as e:
                logger.error(f"FAISS ê²€???ŒìŠ¤???¤íŒ¨: {e}")
                continue
        
        results.update({
            "total_search_time": total_time,
            "average_search_time": total_time / len(test_queries) if test_queries else 0,
            "queries_per_second": len(test_queries) / total_time if total_time > 0 else 0,
            "search_results": search_results
        })
        
        return results
    
    def _test_chromadb_search(self, collection, top_k: int = 5) -> Dict[str, Any]:
        """ChromaDB ê²€???±ëŠ¥ ?ŒìŠ¤??""
        results = {
            "total_search_time": 0,
            "average_search_time": 0,
            "queries_per_second": 0,
            "search_results": []
        }
        
        # ?ŒìŠ¤??ì¿¼ë¦¬
        test_queries = [
            "ê³„ì•½ ?´ì? ?í•´ë°°ìƒ",
            "ê·¼ë¡œê¸°ì?ë²??´ê²Œ?œê°„",
            "ë¶€?™ì‚° ë§¤ë§¤ê³„ì•½ ì¤‘ë„ê¸?,
            "?´í˜¼ ?¬ì‚°ë¶„í• ",
            "?í•´ë°°ìƒ ì²?µ¬ê¶??Œë©¸?œíš¨"
        ]
        
        total_time = 0
        search_results = []
        
        for query in test_queries:
            try:
                # ê²€???¤í–‰
                start_time = time.time()
                result = collection.query(
                    query_texts=[query],
                    n_results=top_k
                )
                search_time = time.time() - start_time
                
                total_time += search_time
                
                # ê²°ê³¼ ?€??
                search_results.append({
                    "query": query,
                    "search_time": search_time,
                    "top_k": top_k,
                    "distances": result["distances"][0] if result["distances"] else [],
                    "ids": result["ids"][0] if result["ids"] else []
                })
                
            except Exception as e:
                logger.error(f"ChromaDB ê²€???ŒìŠ¤???¤íŒ¨: {e}")
                continue
        
        results.update({
            "total_search_time": total_time,
            "average_search_time": total_time / len(test_queries) if test_queries else 0,
            "queries_per_second": len(test_queries) / total_time if total_time > 0 else 0,
            "search_results": search_results
        })
        
        return results
    
    def _test_faiss_save_load(self, index) -> Dict[str, Any]:
        """FAISS ?€??ë¡œë“œ ?ŒìŠ¤??""
        results = {
            "save_time": 0,
            "load_time": 0,
            "file_size_mb": 0
        }
        
        try:
            # ?€???ŒìŠ¤??
            start_time = time.time()
            faiss.write_index(index, "test_faiss_index.bin")
            save_time = time.time() - start_time
            
            # ?Œì¼ ?¬ê¸° ì¸¡ì •
            file_size = os.path.getsize("test_faiss_index.bin") / 1024 / 1024
            
            # ë¡œë“œ ?ŒìŠ¤??
            start_time = time.time()
            loaded_index = faiss.read_index("test_faiss_index.bin")
            load_time = time.time() - start_time
            
            # ?•ë¦¬
            os.remove("test_faiss_index.bin")
            
            results.update({
                "save_time": save_time,
                "load_time": load_time,
                "file_size_mb": file_size
            })
            
        except Exception as e:
            logger.error(f"FAISS ?€??ë¡œë“œ ?ŒìŠ¤???¤íŒ¨: {e}")
            results["error"] = str(e)
        
        return results
    
    def _test_chromadb_save_load(self) -> Dict[str, Any]:
        """ChromaDB ?€??ë¡œë“œ ?ŒìŠ¤??""
        results = {
            "persist_time": 0,
            "load_time": 0,
            "db_size_mb": 0
        }
        
        try:
            # ChromaDB???ë™?¼ë¡œ ì§€?í™”?˜ë?ë¡?ë³„ë„ ?€??ê³¼ì • ?†ìŒ
            results["persist_time"] = 0
            
            # ë¡œë“œ ?ŒìŠ¤??
            start_time = time.time()
            client = chromadb.Client(Settings(
                persist_directory="./chroma_db",
                anonymized_telemetry=False
            ))
            collection = client.get_collection("legal_documents")
            load_time = time.time() - start_time
            
            # DB ?¬ê¸° ì¸¡ì •
            db_size = self._get_chromadb_size()
            
            results.update({
                "load_time": load_time,
                "db_size_mb": db_size
            })
            
        except Exception as e:
            logger.error(f"ChromaDB ?€??ë¡œë“œ ?ŒìŠ¤???¤íŒ¨: {e}")
            results["error"] = str(e)
        
        return results
    
    def _get_faiss_index_size(self, index) -> float:
        """FAISS ?¸ë±???¬ê¸° ë°˜í™˜ (MB)"""
        # FAISS ?¸ë±?¤ì˜ ë©”ëª¨ë¦??¬ìš©??ì¶”ì •
        return (index.ntotal * index.d * 4) / 1024 / 1024  # float32 = 4 bytes
    
    def _get_chromadb_size(self) -> float:
        """ChromaDB ?¬ê¸° ë°˜í™˜ (MB)"""
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
        """?„ì¬ ë©”ëª¨ë¦??¬ìš©??ë°˜í™˜ (MB)"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def run_benchmark(self) -> Dict[str, Any]:
        """?„ì²´ ë²¤ì¹˜ë§ˆí‚¹ ?¤í–‰"""
        logger.info("ë²¡í„° ?¤í† ??ë²¤ì¹˜ë§ˆí‚¹ ?œì‘...")
        
        # ?œìŠ¤???•ë³´ ?˜ì§‘
        system_info = {
            "embedding_dimension": self.embedding_dim,
            "test_data_size": len(self.test_data),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "python_version": sys.version
        }
        
        # ê°?ë²¡í„° ?¤í† ??ë²¤ì¹˜ë§ˆí‚¹ ?¤í–‰
        faiss_results = self.benchmark_faiss()
        chromadb_results = self.benchmark_chromadb()
        
        # ê²°ê³¼ ì¢…í•©
        benchmark_results = {
            "system_info": system_info,
            "faiss": faiss_results,
            "chromadb": chromadb_results,
            "comparison": self._compare_vector_stores(faiss_results, chromadb_results)
        }
        
        return benchmark_results
    
    def _compare_vector_stores(self, faiss_results: Dict, chromadb_results: Dict) -> Dict[str, Any]:
        """ë²¡í„° ?¤í† ???±ëŠ¥ ë¹„êµ"""
        comparison = {
            "build_time_comparison": {},
            "search_speed_comparison": {},
            "memory_usage_comparison": {},
            "storage_size_comparison": {},
            "recommendation": ""
        }
        
        try:
            # êµ¬ì¶• ?œê°„ ë¹„êµ
            if "index_info" in faiss_results and "collection_info" in chromadb_results:
                faiss_build_time = faiss_results["index_info"]["build_time"]
                chromadb_build_time = chromadb_results["collection_info"]["build_time"]
                
                comparison["build_time_comparison"] = {
                    "faiss_seconds": faiss_build_time,
                    "chromadb_seconds": chromadb_build_time,
                    "speed_ratio": faiss_build_time / chromadb_build_time if chromadb_build_time > 0 else 0
                }
            
            # ê²€???ë„ ë¹„êµ
            if "search_results" in faiss_results and "search_results" in chromadb_results:
                faiss_qps = faiss_results["search_results"]["queries_per_second"]
                chromadb_qps = chromadb_results["search_results"]["queries_per_second"]
                
                comparison["search_speed_comparison"] = {
                    "faiss_qps": faiss_qps,
                    "chromadb_qps": chromadb_qps,
                    "speed_ratio": faiss_qps / chromadb_qps if chromadb_qps > 0 else 0
                }
            
            # ë©”ëª¨ë¦??¬ìš©??ë¹„êµ
            faiss_memory = faiss_results.get("memory_usage_mb", 0)
            chromadb_memory = chromadb_results.get("memory_usage_mb", 0)
            
            comparison["memory_usage_comparison"] = {
                "faiss_mb": faiss_memory,
                "chromadb_mb": chromadb_memory,
                "memory_ratio": faiss_memory / chromadb_memory if chromadb_memory > 0 else 0
            }
            
            # ?€?¥ì†Œ ?¬ê¸° ë¹„êµ
            faiss_size = faiss_results.get("index_info", {}).get("index_size_mb", 0)
            chromadb_size = chromadb_results.get("collection_info", {}).get("collection_size_mb", 0)
            
            comparison["storage_size_comparison"] = {
                "faiss_mb": faiss_size,
                "chromadb_mb": chromadb_size,
                "size_ratio": faiss_size / chromadb_size if chromadb_size > 0 else 0
            }
            
            # ê¶Œì¥?¬í•­ ?ì„±
            comparison["recommendation"] = self._generate_recommendation(comparison)
            
        except Exception as e:
            logger.error(f"ë²¡í„° ?¤í† ??ë¹„êµ ì¤??¤ë¥˜: {e}")
            comparison["error"] = str(e)
        
        return comparison
    
    def _generate_recommendation(self, comparison: Dict) -> str:
        """ë²¡í„° ?¤í† ??? íƒ ê¶Œì¥?¬í•­ ?ì„±"""
        try:
            search_ratio = comparison.get("search_speed_comparison", {}).get("speed_ratio", 1)
            memory_ratio = comparison.get("memory_usage_comparison", {}).get("memory_ratio", 1)
            size_ratio = comparison.get("storage_size_comparison", {}).get("size_ratio", 1)
            
            # HuggingFace Spaces ?˜ê²½ ê³ ë ¤
            if search_ratio > 1.5:  # FAISSê°€ ê²€?‰ì´ ??ë¹ ë¦„
                return "FAISS ê¶Œì¥: ê²€???±ëŠ¥???°ìˆ˜?˜ì—¬ ?¤ì‹œê°?ê²€?‰ì— ?í•©"
            elif memory_ratio > 1.3:  # FAISSê°€ ë©”ëª¨ë¦¬ë? ??ë§ì´ ?¬ìš©
                return "ChromaDB ê¶Œì¥: ë©”ëª¨ë¦??¨ìœ¨?±ì´ ?°ìˆ˜?˜ì—¬ ?œí•œ???˜ê²½???í•©"
            elif size_ratio > 1.2:  # FAISSê°€ ?€?¥ê³µê°„ì„ ??ë§ì´ ?¬ìš©
                return "ChromaDB ê¶Œì¥: ?€?¥ê³µê°??¨ìœ¨?±ì´ ?°ìˆ˜"
            else:
                return "FAISS ê¶Œì¥: ?’ì? ?±ëŠ¥ê³??ˆì •?±ìœ¼ë¡??„ë¡œ?•ì…˜ ?˜ê²½???í•©"
                
        except Exception as e:
            return f"ê¶Œì¥?¬í•­ ?ì„± ?¤íŒ¨: {e}"
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """ë²¤ì¹˜ë§ˆí‚¹ ê²°ê³¼ ?€??""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vector_store_benchmark_results_{timestamp}.json"
        
        filepath = os.path.join("benchmark_results", filename)
        os.makedirs("benchmark_results", exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"ë²¤ì¹˜ë§ˆí‚¹ ê²°ê³¼ ?€?? {filepath}")
        return filepath

def main():
    """ë©”ì¸ ?¤í–‰ ?¨ìˆ˜"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ë²¡í„° ?¤í† ???±ëŠ¥ ë²¤ì¹˜ë§ˆí‚¹")
    parser.add_argument("--embedding-dim", type=int, default=768, help="?„ë² ??ì°¨ì›")
    parser.add_argument("--output", help="ê²°ê³¼ ?€???Œì¼ëª?)
    
    args = parser.parse_args()
    
    # ë²¤ì¹˜ë§ˆí‚¹ ?¤í–‰
    benchmark = VectorStoreBenchmark(embedding_dim=args.embedding_dim)
    results = benchmark.run_benchmark()
    
    # ê²°ê³¼ ?€??
    output_file = benchmark.save_results(results, args.output)
    
    # ê²°ê³¼ ?”ì•½ ì¶œë ¥
    print("\n" + "="*50)
    print("ë²¡í„° ?¤í† ??ë²¤ì¹˜ë§ˆí‚¹ ê²°ê³¼ ?”ì•½")
    print("="*50)
    
    if "faiss" in results and "search_results" in results["faiss"]:
        faiss_info = results["faiss"]["search_results"]
        print(f"FAISS - QPS: {faiss_info.get('queries_per_second', 0):.2f}, "
              f"?‰ê·  ê²€?‰ì‹œê°? {faiss_info.get('average_search_time', 0):.4f}ì´?)
    
    if "chromadb" in results and "search_results" in results["chromadb"]:
        chromadb_info = results["chromadb"]["search_results"]
        print(f"ChromaDB - QPS: {chromadb_info.get('queries_per_second', 0):.2f}, "
              f"?‰ê·  ê²€?‰ì‹œê°? {chromadb_info.get('average_search_time', 0):.4f}ì´?)
    
    if "comparison" in results and "recommendation" in results["comparison"]:
        print(f"\nê¶Œì¥?¬í•­: {results['comparison']['recommendation']}")
    
    print(f"\n?ì„¸ ê²°ê³¼: {output_file}")

if __name__ == "__main__":
    main()
