#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ê°œì„ ??ë²¡í„° ?°ì´?°ë² ?´ìŠ¤ ?¬êµ¬ì¶??¤í¬ë¦½íŠ¸
"""

import os
import sys
import json
import numpy as np
import time
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
sys.path.append(str(Path(__file__).parent.parent))

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Required library not available: {e}")
    sys.exit(1)

def rebuild_improved_vector_database():
    """ê°œì„ ??ë²¡í„° ?°ì´?°ë² ?´ìŠ¤ ?¬êµ¬ì¶?""
    print("Rebuilding improved vector database...")
    
    # ê°œì„ ??ë©”í??°ì´??ë¡œë“œ
    with open('data/embeddings/metadata_improved.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total documents: {len(data)}")
    
    # Sentence-BERT ëª¨ë¸ ë¡œë“œ
    print("Loading Sentence-BERT model...")
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    
    # ?ìŠ¤??ì¶”ì¶œ
    texts = [doc['text'] for doc in data]
    print(f"Generating embeddings for {len(texts)} documents...")
    
    # ë°°ì¹˜ ?¨ìœ„ë¡??„ë² ???ì„±
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=True)
        all_embeddings.append(batch_embeddings)
        print(f"Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} completed")
    
    # ëª¨ë“  ?„ë² ??ê²°í•©
    embeddings = np.vstack(all_embeddings)
    print(f"Total embeddings generated: {embeddings.shape}")
    
    # FAISS ?¸ë±???ì„±
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings.astype('float32'))
    
    # ?Œì¼ ?€??
    print("Saving files...")
    
    # FAISS ?¸ë±???€??
    faiss.write_index(faiss_index, "data/embeddings/faiss_index_improved.bin")
    print("FAISS index saved: data/embeddings/faiss_index_improved.bin")
    
    # ?„ë² ???€??
    np.save("data/embeddings/embeddings_improved.npy", embeddings)
    print("Embeddings saved: data/embeddings/embeddings_improved.npy")
    
    # ë©”í??°ì´???€??
    with open("data/embeddings/metadata_improved.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Metadata saved: data/embeddings/metadata_improved.json")
    
    # êµ¬ì¶• ë³´ê³ ???ì„±
    build_report = {
        'build_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_documents': len(data),
        'embeddings_shape': embeddings.shape,
        'faiss_index_size': faiss_index.ntotal,
        'improvements': {
            'precedent_titles_improved': True,
            'court_names_added': True,
            'case_types_added': True
        }
    }
    
    with open("data/embeddings/vector_db_improved_report.json", 'w', encoding='utf-8') as f:
        json.dump(build_report, f, ensure_ascii=False, indent=2)
    print("Build report saved: data/embeddings/vector_db_improved_report.json")
    
    print("\nImproved vector database rebuild completed!")
    print(f"Index size: {faiss_index.ntotal}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    return faiss_index, embeddings, data

def test_improved_search():
    """ê°œì„ ??ê²€???ŒìŠ¤??""
    print("\nTesting improved search...")
    
    # ê°œì„ ???°ì´??ë¡œë“œ
    faiss_index = faiss.read_index("data/embeddings/faiss_index_improved.bin")
    with open('data/embeddings/metadata_improved.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    
    # ?ŒìŠ¤??ì¿¼ë¦¬
    test_queries = [
        "Supreme Court Decision",
        "District Court Decision", 
        "Civil Case",
        "Criminal Case",
        "Administrative Case"
    ]
    
    print("Search test results:")
    for query in test_queries:
        start_time = time.time()
        
        # ì¿¼ë¦¬ ?„ë² ???ì„±
        query_embedding = model.encode([query])
        
        # FAISS ê²€??
        distances, indices = faiss_index.search(query_embedding.astype('float32'), 3)
        
        search_time = time.time() - start_time
        
        print(f"\nQuery: '{query}' (Search time: {search_time:.4f}s)")
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(metadata):
                doc = metadata[idx]
                title = doc['metadata']['original_document']
                doc_type = doc['metadata']['data_type']
                similarity = 1.0 / (1.0 + distance)
                print(f"  {i+1}. [{doc_type}] {title} (Similarity: {similarity:.3f})")

def main():
    print("Improved Vector Database Rebuild")
    print("=" * 50)
    
    # 1. ê°œì„ ??ë²¡í„° ?°ì´?°ë² ?´ìŠ¤ ?¬êµ¬ì¶?
    faiss_index, embeddings, data = rebuild_improved_vector_database()
    
    # 2. ê°œì„ ??ê²€???ŒìŠ¤??
    test_improved_search()
    
    print("\n" + "=" * 50)
    print("All improvements completed successfully!")

if __name__ == "__main__":
    main()
