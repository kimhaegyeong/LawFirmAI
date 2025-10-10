"""
벡터DB 구축 스크립트 (TASK 3.2용)
하이브리드 검색 시스템을 위한 벡터DB 구축
"""

import os
import sys
import json
import logging
import sqlite3
from typing import List, Dict, Any

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from source.services.semantic_search_engine import SemanticSearchEngine

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorDBBuilder:
    """벡터DB 구축 클래스"""
    
    def __init__(self, db_path: str = "data/lawfirm.db"):
        self.db_path = db_path
        self.semantic_search = SemanticSearchEngine()
    
    def load_documents_from_db(self) -> List[Dict[str, Any]]:
        """데이터베이스에서 문서 로드"""
        try:
            logger.info("Loading documents from database...")
            
            documents = []
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # 법령 데이터 로드
                cursor.execute("SELECT * FROM laws LIMIT 100")
                for row in cursor.fetchall():
                    documents.append({
                        "id": f"law_{row['id']}",
                        "type": "law",
                        "title": row['law_name'],
                        "content": row['content'],
                        "law_name": row['law_name'],
                        "article_number": row['article_number'],
                        "law_type": row['law_type'],
                        "effective_date": row['effective_date'],
                        "source": "database"
                    })
                
                # 판례 데이터 로드
                cursor.execute("SELECT * FROM precedents LIMIT 100")
                for row in cursor.fetchall():
                    documents.append({
                        "id": f"precedent_{row['id']}",
                        "type": "precedent",
                        "title": row['case_name'],
                        "content": row['content'],
                        "case_number": row['case_number'],
                        "court_name": row['court_name'],
                        "decision_date": row['decision_date'],
                        "case_type": row['case_type'],
                        "source": "database"
                    })
                
                # 헌재결정례 데이터 로드
                cursor.execute("SELECT * FROM constitutional_decisions LIMIT 100")
                for row in cursor.fetchall():
                    documents.append({
                        "id": f"constitutional_{row['id']}",
                        "type": "constitutional",
                        "title": row['case_name'],
                        "content": row['content'],
                        "case_number": row['case_number'],
                        "decision_date": row['decision_date'],
                        "decision_type": row['decision_type'],
                        "source": "database"
                    })
            
            logger.info(f"Loaded {len(documents)} documents from database")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load documents from database: {e}")
            return []
    
    def create_sample_documents(self) -> List[Dict[str, Any]]:
        """샘플 문서 생성 (데이터베이스가 비어있는 경우)"""
        logger.info("Creating sample documents...")
        
        sample_documents = [
            {
                "id": "law_1",
                "type": "law",
                "title": "민법 제1조",
                "content": "민법은 개인의 사생활과 재산관계를 규율하는 법률이다. 모든 민사에 관하여 법률에 특별한 규정이 없으면 관습법에 의하고, 관습법이 없으면 조리에 의한다.",
                "law_name": "민법",
                "article_number": "제1조",
                "law_type": "민법",
                "effective_date": "1960-01-01",
                "source": "sample"
            },
            {
                "id": "law_2",
                "type": "law",
                "title": "민법 제2조",
                "content": "권리의 행사와 의무의 이행은 신의에 좇아 성실히 하여야 한다.",
                "law_name": "민법",
                "article_number": "제2조",
                "law_type": "민법",
                "effective_date": "1960-01-01",
                "source": "sample"
            },
            {
                "id": "precedent_1",
                "type": "precedent",
                "title": "계약서 작성에 관한 판례",
                "content": "계약서는 당사자 간의 합의사항을 명확히 기록한 문서로서, 계약의 성립과 내용을 증명하는 중요한 자료이다. 계약서 작성 시에는 계약의 목적, 당사자, 계약조건 등을 명확히 기재하여야 한다.",
                "case_number": "2024다12345",
                "court_name": "대법원",
                "decision_date": "2024-01-15",
                "case_type": "민사",
                "source": "sample"
            },
            {
                "id": "precedent_2",
                "type": "precedent",
                "title": "부동산 매매계약에 관한 판례",
                "content": "부동산 매매계약은 매도인이 부동산의 소유권을 매수인에게 이전하고, 매수인이 그 대가를 지급하는 계약이다. 매매계약 체결 시에는 부동산의 표시, 매매대금, 인도시기 등을 명확히 정하여야 한다.",
                "case_number": "2024다67890",
                "court_name": "고등법원",
                "decision_date": "2024-02-20",
                "case_type": "민사",
                "source": "sample"
            },
            {
                "id": "constitutional_1",
                "type": "constitutional",
                "title": "재산권 보장에 관한 헌재결정",
                "content": "재산권은 헌법상 보장되는 기본권으로서, 국가는 법률에 의하지 아니하고는 이를 제한할 수 없다. 다만, 공공복리를 위하여 필요한 경우에는 법률로써 제한할 수 있다.",
                "case_number": "2024헌마123",
                "decision_date": "2024-03-10",
                "decision_type": "위헌",
                "source": "sample"
            },
            {
                "id": "law_3",
                "type": "law",
                "title": "상법 제1조",
                "content": "상법은 상행위와 상인에 관한 사항을 규율하는 법률이다. 상행위에 관하여는 상법에 특별한 규정이 없으면 민법의 규정을 적용한다.",
                "law_name": "상법",
                "article_number": "제1조",
                "law_type": "상법",
                "effective_date": "1962-01-01",
                "source": "sample"
            },
            {
                "id": "precedent_3",
                "type": "precedent",
                "title": "손해배상에 관한 판례",
                "content": "손해배상은 불법행위로 인하여 타인에게 손해를 가한 자가 그 손해를 배상하는 제도이다. 손해배상의 범위는 통상의 손해와 특별한 사정으로 인한 손해를 포함한다.",
                "case_number": "2024다11111",
                "court_name": "대법원",
                "decision_date": "2024-04-05",
                "case_type": "민사",
                "source": "sample"
            },
            {
                "id": "constitutional_2",
                "type": "constitutional",
                "title": "계약의 자유에 관한 헌재결정",
                "content": "계약의 자유는 헌법상 보장되는 기본권으로서, 개인이 자유롭게 계약을 체결할 수 있는 권리를 의미한다. 다만, 공공질서나 선량한 풍속에 위반되는 계약은 무효이다.",
                "case_number": "2024헌마456",
                "decision_date": "2024-05-15",
                "decision_type": "합헌",
                "source": "sample"
            }
        ]
        
        logger.info(f"Created {len(sample_documents)} sample documents")
        return sample_documents
    
    def build_vector_db(self) -> bool:
        """벡터DB 구축"""
        try:
            logger.info("Starting vector database construction...")
            
            # 문서 로드
            documents = self.load_documents_from_db()
            
            # 데이터베이스가 비어있는 경우 샘플 문서 사용
            if not documents:
                logger.info("Database is empty, using sample documents")
                documents = self.create_sample_documents()
            
            if not documents:
                logger.error("No documents available for vector DB construction")
                return False
            
            # 벡터DB 구축
            success = self.semantic_search.build_index(documents)
            
            if success:
                logger.info("Vector database construction completed successfully")
                
                # 통계 정보 출력
                stats = self.semantic_search.get_index_stats()
                logger.info(f"Index statistics: {stats}")
                
                return True
            else:
                logger.error("Vector database construction failed")
                return False
                
        except Exception as e:
            logger.error(f"Vector database construction failed: {e}")
            return False
    
    def test_vector_search(self) -> bool:
        """벡터 검색 테스트"""
        try:
            logger.info("Testing vector search...")
            
            test_queries = [
                "계약서 작성 방법",
                "부동산 매매",
                "손해배상",
                "재산권 보장"
            ]
            
            for query in test_queries:
                results = self.semantic_search.search(query, k=3, threshold=0.3)
                logger.info(f"Query '{query}': {len(results)} results")
                
                for i, result in enumerate(results):
                    logger.info(f"  {i+1}. {result.get('title', 'No title')} (score: {result.get('similarity_score', 0):.3f})")
            
            logger.info("Vector search test completed")
            return True
            
        except Exception as e:
            logger.error(f"Vector search test failed: {e}")
            return False

def main():
    """메인 함수"""
    logger.info("=" * 60)
    logger.info("🚀 Starting Vector Database Construction for TASK 3.2")
    logger.info("=" * 60)
    
    builder = VectorDBBuilder()
    
    # 벡터DB 구축
    success = builder.build_vector_db()
    
    if success:
        logger.info("✅ Vector database construction completed successfully")
        
        # 벡터 검색 테스트
        test_success = builder.test_vector_search()
        
        if test_success:
            logger.info("✅ Vector search test completed successfully")
        else:
            logger.warning("⚠️ Vector search test failed")
    else:
        logger.error("❌ Vector database construction failed")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
