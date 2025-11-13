#!/usr/bin/env python3
"""
ë²¡í„° ?„ë² ??ì§ì ‘ ?ŒìŠ¤??
"""

import sys
import os
from pathlib import Path
import json
import faiss
import numpy as np

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_direct_faiss():
    """FAISS ?¸ë±??ì§ì ‘ ?ŒìŠ¤??""
    print("FAISS ?¸ë±??ì§ì ‘ ?ŒìŠ¤??..")
    
    try:
        # FAISS ?¸ë±??ì§ì ‘ ë¡œë“œ
        faiss_file = Path("data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index.faiss")
        json_file = Path("data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index.json")
        
        if not faiss_file.exists():
            print(f"FAISS ?Œì¼??ì¡´ì¬?˜ì? ?ŠìŠµ?ˆë‹¤: {faiss_file}")
            return False
        
        if not json_file.exists():
            print(f"JSON ?Œì¼??ì¡´ì¬?˜ì? ?ŠìŠµ?ˆë‹¤: {json_file}")
            return False
        
        # FAISS ?¸ë±??ë¡œë“œ
        index = faiss.read_index(str(faiss_file))
        print(f"FAISS ?¸ë±??ë¡œë“œ ?±ê³µ")
        print(f"   - ?¸ë±???¬ê¸°: {index.ntotal:,}")
        print(f"   - ë²¡í„° ì°¨ì›: {index.d}")
        
        # ë©”í??°ì´??ë¡œë“œ
        with open(json_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"ë©”í??°ì´??ë¡œë“œ ?±ê³µ")
        print(f"   - ëª¨ë¸ëª? {metadata.get('model_name', 'Unknown')}")
        print(f"   - ì°¨ì›: {metadata.get('dimension', 0)}")
        print(f"   - ë¬¸ì„œ ?? {metadata.get('document_count', 0):,}")
        print(f"   - ?ì„±?? {metadata.get('created_at', 'Unknown')}")
        
        # ë¬¸ì„œ ë©”í??°ì´???•ì¸
        doc_metadata = metadata.get('document_metadata', [])
        print(f"   - ë©”í??°ì´????ª© ?? {len(doc_metadata):,}")
        
        if len(doc_metadata) > 0:
            sample_metadata = doc_metadata[0]
            print(f"   - ?˜í”Œ ë©”í??°ì´??")
            for key, value in sample_metadata.items():
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                print(f"     {key}: {value}")
        
        # ê°„ë‹¨??ê²€???ŒìŠ¤??(?œë¤ ë²¡í„°)
        if index.ntotal > 0:
            print(f"\nê²€???ŒìŠ¤??..")
            
            # ?œë¤ ì¿¼ë¦¬ ë²¡í„° ?ì„±
            query_vector = np.random.random((1, index.d)).astype('float32')
            faiss.normalize_L2(query_vector)
            
            # ê²€???¤í–‰
            scores, indices = index.search(query_vector, 5)
            
            print(f"   - ê²€??ê²°ê³¼ ?? {len(indices[0])}")
            print(f"   - ìµœê³  ?ìˆ˜: {scores[0][0]:.3f}")
            print(f"   - ê²€?‰ëœ ?¸ë±?? {indices[0][:3]}")
            
            # ë©”í??°ì´?°ì? ë§¤ì¹­
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(doc_metadata):
                    doc_info = doc_metadata[idx]
                    print(f"   {i+1}. ?ìˆ˜: {score:.3f}, ë²•ë¥ ëª? {doc_info.get('law_name', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"FAISS ì§ì ‘ ?ŒìŠ¤???¤íŒ¨: {e}")
        return False

def main():
    """ë©”ì¸ ?ŒìŠ¤???¨ìˆ˜"""
    print("FAISS ?¸ë±??ì§ì ‘ ?ŒìŠ¤??)
    print("=" * 40)
    
    success = test_direct_faiss()
    
    print("\n" + "=" * 40)
    if success:
        print("FAISS ?¸ë±???ŒìŠ¤???±ê³µ!")
        print("ë²¡í„° ?„ë² ?©ì´ ?•ìƒ?ìœ¼ë¡??ì„±?˜ì—ˆ?µë‹ˆ??")
        print("FAISS ?¸ë±?¤ì? ë©”í??°ì´?°ê? ?¬ë°”ë¥´ê²Œ ?€?¥ë˜?ˆìŠµ?ˆë‹¤.")
    else:
        print("FAISS ?¸ë±???ŒìŠ¤???¤íŒ¨")
        print("ì¶”ê? ?ê????„ìš”?©ë‹ˆ??")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)