#!/usr/bin/env python3
"""
ë²¡í„° ?„ë² ??ê¸°ë³¸ ?ŒìŠ¤??
"""

import sys
import os
from pathlib import Path
import time

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from source.data.vector_store import LegalVectorStore

def test_vector_embedding():
    """ë²¡í„° ?„ë² ??ê¸°ë³¸ ?ŒìŠ¤??""
    print("?” ë²¡í„° ?„ë² ??ê¸°ë³¸ ?ŒìŠ¤???œì‘...")
    
    try:
        # ë²¡í„° ?¤í† ??ì´ˆê¸°??
        vector_store = LegalVectorStore(
            model_name="jhgan/ko-sroberta-multitask",
            dimension=768,
            index_type="flat"
        )
        
        print("??ë²¡í„° ?¤í† ??ì´ˆê¸°???±ê³µ")
        
        # ?¸ë±???Œì¼ ?•ì¸
        index_path = Path("data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index")
        faiss_file = index_path.with_suffix('.faiss')
        json_file = index_path.with_suffix('.json')
        
        print(f"?“ ?¸ë±???Œì¼ ?•ì¸:")
        print(f"   - FAISS ?Œì¼: {faiss_file.exists()} ({faiss_file.stat().st_size / (1024*1024):.1f} MB)")
        print(f"   - JSON ?Œì¼: {json_file.exists()} ({json_file.stat().st_size / (1024*1024):.1f} MB)")
        
        if faiss_file.exists() and json_file.exists():
            # ?¸ë±??ë¡œë“œ ?œë„
            try:
                vector_store.load_index(str(index_path))
                stats = vector_store.get_stats()
                
                print(f"???¸ë±??ë¡œë“œ ?±ê³µ")
                print(f"   - ì´?ë¬¸ì„œ ?? {stats.get('documents_count', 0):,}")
                print(f"   - ?¸ë±???¬ê¸°: {stats.get('index_size', 0):,}")
                
                # ê°„ë‹¨??ê²€???ŒìŠ¤??
                test_query = "ê³„ì•½???´ì? ì¡°ê±´"
                start_time = time.time()
                results = vector_store.search(test_query, top_k=3)
                search_time = time.time() - start_time
                
                print(f"?” ê²€???ŒìŠ¤??")
                print(f"   - ì¿¼ë¦¬: '{test_query}'")
                print(f"   - ê²°ê³¼ ?? {len(results)}")
                print(f"   - ê²€???œê°„: {search_time:.3f}ì´?)
                
                if results:
                    best_result = results[0]
                    print(f"   - ìµœê³  ?ìˆ˜: {best_result.get('score', 0):.3f}")
                    print(f"   - ë¬¸ì„œ ID: {best_result.get('metadata', {}).get('document_id', 'Unknown')}")
                
                return True
                
            except Exception as e:
                print(f"???¸ë±??ë¡œë“œ ?¤íŒ¨: {e}")
                return False
        else:
            print("???¸ë±???Œì¼??ì¡´ì¬?˜ì? ?ŠìŠµ?ˆë‹¤.")
            return False
            
    except Exception as e:
        print(f"??ë²¡í„° ?¤í† ???ŒìŠ¤???¤íŒ¨: {e}")
        return False

def check_file_sizes():
    """?Œì¼ ?¬ê¸° ?•ì¸"""
    print("\n?“Š ?ì„±???Œì¼ ?¬ê¸° ?•ì¸:")
    
    embedding_dir = Path("data/embeddings/ml_enhanced_ko_sroberta")
    
    if embedding_dir.exists():
        for file_path in embedding_dir.iterdir():
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"   - {file_path.name}: {size_mb:.2f} MB")
    else:
        print("???„ë² ???”ë ‰? ë¦¬ê°€ ì¡´ì¬?˜ì? ?ŠìŠµ?ˆë‹¤.")

def main():
    """ë©”ì¸ ?ŒìŠ¤???¨ìˆ˜"""
    print("?? ë²¡í„° ?„ë² ??ê¸°ë³¸ ?ŒìŠ¤??)
    print("=" * 40)
    
    # ?Œì¼ ?¬ê¸° ?•ì¸
    check_file_sizes()
    
    # ë²¡í„° ?„ë² ???ŒìŠ¤??
    success = test_vector_embedding()
    
    print("\n" + "=" * 40)
    if success:
        print("?‰ ë²¡í„° ?„ë² ???ŒìŠ¤???±ê³µ!")
        print("??ë²¡í„° ?„ë² ?©ì´ ?•ìƒ?ìœ¼ë¡??ì„±?˜ê³  ë¡œë“œ?©ë‹ˆ??")
    else:
        print("? ï¸ ë²¡í„° ?„ë² ???ŒìŠ¤???¤íŒ¨")
        print("??ì¶”ê? ?ê????„ìš”?©ë‹ˆ??")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
