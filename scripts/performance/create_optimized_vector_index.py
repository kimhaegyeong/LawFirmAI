#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ìµœì ?”ëœ ë²¡í„° ?¸ë±???ì„±
"""

import sys
sys.path.append('source')
from source.data.vector_store import LegalVectorStore
import time
import json
from pathlib import Path

def create_optimized_index():
    """ìµœì ?”ëœ ë²¡í„° ?¸ë±???ì„±"""
    print("=== ìµœì ?”ëœ ë²¡í„° ?¸ë±???ì„± ===")
    
    # ê¸°ì¡´ ?¸ë±??ë¡œë“œ
    print("ê¸°ì¡´ ?¸ë±??ë¡œë“œ ì¤?..")
    vector_store = LegalVectorStore(
        model_name='jhgan/ko-sroberta-multitask',
        dimension=768,
        index_type='flat'
    )
    
    if not vector_store.load_index('data/embeddings/ml_enhanced_ko_sroberta_precedents'):
        print("ê¸°ì¡´ ?¸ë±??ë¡œë“œ ?¤íŒ¨")
        return False
    
    print(f"ê¸°ì¡´ ?¸ë±???¬ê¸°: {vector_store.index.ntotal:,}")
    
    # IVF ?¸ë±?¤ë¡œ ë³€??
    print("IVF ?¸ë±?¤ë¡œ ë³€??ì¤?..")
    
    # IVF ?¸ë±???ì„±
    import faiss
    
    # nlist ê°?ê³„ì‚° (?°ì´???¬ê¸°???°ë¼)
    n_vectors = vector_store.index.ntotal
    nlist = min(1000, max(100, n_vectors // 100))  # 100-1000 ?¬ì´??ê°?
    
    print(f"IVF ?Œë¼ë¯¸í„° - nlist: {nlist}")
    
    # IVF ?¸ë±???ì„±
    quantizer = faiss.IndexFlatIP(768)
    ivf_index = faiss.IndexIVFFlat(quantizer, 768, nlist)
    
    # ê¸°ì¡´ ë²¡í„° ?°ì´??ì¶”ì¶œ
    print("ë²¡í„° ?°ì´??ì¶”ì¶œ ì¤?..")
    vectors = vector_store.index.reconstruct_n(0, n_vectors)
    
    # IVF ?¸ë±???ˆë ¨
    print("IVF ?¸ë±???ˆë ¨ ì¤?..")
    ivf_index.train(vectors)
    
    # ë²¡í„° ì¶”ê?
    print("ë²¡í„° ì¶”ê? ì¤?..")
    ivf_index.add(vectors)
    
    # ?ˆë¡œ??ë²¡í„° ?¤í† ???ì„±
    print("ìµœì ?”ëœ ë²¡í„° ?¤í† ???ì„± ì¤?..")
    optimized_store = LegalVectorStore(
        model_name='jhgan/ko-sroberta-multitask',
        dimension=768,
        index_type='ivf'
    )
    
    # ?¸ë±???¤ì •
    optimized_store.index = ivf_index
    optimized_store.document_texts = vector_store.document_texts.copy()
    optimized_store.document_metadata = vector_store.document_metadata.copy()
    optimized_store._index_loaded = True
    
    # ìµœì ?”ëœ ?¸ë±???€??
    output_path = 'data/embeddings/optimized_ko_sroberta_precedents'
    print(f"ìµœì ?”ëœ ?¸ë±???€??ì¤? {output_path}")
    
    optimized_store.save_index(output_path)
    
    print("ìµœì ?”ëœ ?¸ë±???ì„± ?„ë£Œ!")
    
    # ?±ëŠ¥ ë¹„êµ ?ŒìŠ¤??
    print("\n=== ?±ëŠ¥ ë¹„êµ ?ŒìŠ¤??===")
    test_queries = [
        "?í•´ë°°ìƒ ì²?µ¬",
        "ê³„ì•½ ?´ì?",
        "?¹í—ˆ ì¹¨í•´",
        "?´í˜¼ ?Œì†¡",
        "?•ì‚¬ ì²˜ë²Œ"
    ]
    
    # ê¸°ì¡´ ?¸ë±???±ëŠ¥
    print("ê¸°ì¡´ Flat ?¸ë±???±ëŠ¥:")
    flat_times = []
    for query in test_queries:
        start_time = time.time()
        results = vector_store.search(query, top_k=10)
        search_time = time.time() - start_time
        flat_times.append(search_time)
        print(f"  '{query}': {search_time:.3f}ì´?)
    
    # ìµœì ?”ëœ ?¸ë±???±ëŠ¥
    print("\nìµœì ?”ëœ IVF ?¸ë±???±ëŠ¥:")
    ivf_times = []
    for query in test_queries:
        start_time = time.time()
        results = optimized_store.search(query, top_k=10)
        search_time = time.time() - start_time
        ivf_times.append(search_time)
        print(f"  '{query}': {search_time:.3f}ì´?)
    
    # ?±ëŠ¥ ê°œì„ ??ê³„ì‚°
    avg_flat_time = sum(flat_times) / len(flat_times)
    avg_ivf_time = sum(ivf_times) / len(ivf_times)
    improvement = ((avg_flat_time - avg_ivf_time) / avg_flat_time) * 100
    
    print(f"\n?±ëŠ¥ ê°œì„  ê²°ê³¼:")
    print(f"  ê¸°ì¡´ ?‰ê·  ê²€???œê°„: {avg_flat_time:.3f}ì´?)
    print(f"  ìµœì ?”ëœ ?‰ê·  ê²€???œê°„: {avg_ivf_time:.3f}ì´?)
    print(f"  ?±ëŠ¥ ê°œì„ ?? {improvement:.1f}%")
    
    return True

def create_quantized_index():
    """?‘ì?”ëœ ?¸ë±???ì„±"""
    print("\n=== ?‘ì?”ëœ ?¸ë±???ì„± ===")
    
    # ê¸°ì¡´ ?¸ë±??ë¡œë“œ
    vector_store = LegalVectorStore(
        model_name='jhgan/ko-sroberta-multitask',
        dimension=768,
        index_type='flat'
    )
    
    if not vector_store.load_index('data/embeddings/ml_enhanced_ko_sroberta_precedents'):
        print("ê¸°ì¡´ ?¸ë±??ë¡œë“œ ?¤íŒ¨")
        return False
    
    # PQ ?‘ì???¸ë±???ì„±
    import faiss
    
    print("PQ ?‘ì???¸ë±???ì„± ì¤?..")
    
    # PQ ?Œë¼ë¯¸í„° ?¤ì •
    m = 64  # ?œë¸Œë²¡í„° ??
    nbits = 8  # ê°??œë¸Œë²¡í„°??ë¹„íŠ¸ ??
    
    pq_index = faiss.IndexPQ(768, m, nbits)
    
    # ë²¡í„° ?°ì´??ì¶”ì¶œ
    n_vectors = vector_store.index.ntotal
    vectors = vector_store.index.reconstruct_n(0, n_vectors)
    
    # PQ ?¸ë±???ˆë ¨ ë°?ë²¡í„° ì¶”ê?
    print("PQ ?¸ë±???ˆë ¨ ì¤?..")
    pq_index.train(vectors)
    
    print("ë²¡í„° ì¶”ê? ì¤?..")
    pq_index.add(vectors)
    
    # ?‘ì?”ëœ ë²¡í„° ?¤í† ???ì„±
    quantized_store = LegalVectorStore(
        model_name='jhgan/ko-sroberta-multitask',
        dimension=768,
        index_type='pq'
    )
    
    quantized_store.index = pq_index
    quantized_store.document_texts = vector_store.document_texts.copy()
    quantized_store.document_metadata = vector_store.document_metadata.copy()
    quantized_store._index_loaded = True
    
    # ?‘ì?”ëœ ?¸ë±???€??
    output_path = 'data/embeddings/quantized_ko_sroberta_precedents'
    print(f"?‘ì?”ëœ ?¸ë±???€??ì¤? {output_path}")
    
    quantized_store.save_index(output_path)
    
    print("?‘ì?”ëœ ?¸ë±???ì„± ?„ë£Œ!")
    
    # ë©”ëª¨ë¦??¬ìš©??ë¹„êµ
    print("\n=== ë©”ëª¨ë¦??¬ìš©??ë¹„êµ ===")
    
    # ê¸°ì¡´ ?¸ë±??ë©”ëª¨ë¦??¬ìš©??
    flat_memory = vector_store.get_memory_usage()
    print(f"ê¸°ì¡´ Flat ?¸ë±??ë©”ëª¨ë¦? {flat_memory.get('total_memory_mb', 0):.1f}MB")
    
    # ?‘ì?”ëœ ?¸ë±??ë©”ëª¨ë¦??¬ìš©??
    pq_memory = quantized_store.get_memory_usage()
    print(f"?‘ì?”ëœ PQ ?¸ë±??ë©”ëª¨ë¦? {pq_memory.get('total_memory_mb', 0):.1f}MB")
    
    memory_reduction = ((flat_memory.get('total_memory_mb', 0) - pq_memory.get('total_memory_mb', 0)) / 
                       flat_memory.get('total_memory_mb', 1)) * 100
    print(f"ë©”ëª¨ë¦??ˆì•½?? {memory_reduction:.1f}%")
    
    return True

def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    print("LawFirmAI ë²¡í„° ?¸ë±??ìµœì ??)
    print("=" * 50)
    
    # 1. IVF ?¸ë±???ì„±
    if create_optimized_index():
        print("\n??IVF ?¸ë±???ì„± ?„ë£Œ")
    else:
        print("\n??IVF ?¸ë±???ì„± ?¤íŒ¨")
    
    # 2. ?‘ì?”ëœ ?¸ë±???ì„±
    if create_quantized_index():
        print("\n???‘ì?”ëœ ?¸ë±???ì„± ?„ë£Œ")
    else:
        print("\n???‘ì?”ëœ ?¸ë±???ì„± ?¤íŒ¨")
    
    print("\n=== ìµœì ???„ë£Œ ===")
    print("?ì„±???¸ë±??")
    print("  - data/embeddings/optimized_ko_sroberta_precedents (IVF)")
    print("  - data/embeddings/quantized_ko_sroberta_precedents (PQ)")

if __name__ == "__main__":
    main()
