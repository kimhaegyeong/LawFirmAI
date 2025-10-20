# 하이브리드 검색 시스템 구현 가이드

## 개요

이 문서는 LawFirmAI 프로젝트에서 관계형 데이터베이스(SQLite)와 벡터 데이터베이스(FAISS)를 결합한 하이브리드 검색 시스템을 구현하는 방법을 설명합니다.

## ✅ 구현 완료 상태 (2025-10-10)

- **SQLite 데이터**: 24개 문서 (laws 13개, precedents 11개)
- **FAISS 벡터**: 24개 벡터 (768차원, jhgan/ko-sroberta-multitask 모델)
- **하이브리드 검색**: 정상 작동 확인
- **검증 완료**: 벡터 검색 및 정확 매칭 모두 정상 동작

## 구현 단계

### 0단계: 헌재결정례 데이터 수집 (신규 추가)

#### 0.1 날짜 기반 수집 시스템

헌재결정례 데이터를 체계적으로 수집하기 위한 날짜 기반 수집 시스템을 구현했습니다.

```python
# scripts/constitutional_decision/date_based_collector.py
class DateBasedConstitutionalCollector:
    """날짜 기반 헌재결정례 수집 클래스"""
    
    def collect_by_year(self, year: int, target_count: Optional[int] = None, 
                       unlimited: bool = False, use_final_date: bool = False) -> bool:
        """특정 연도 헌재결정례 수집"""
        # 종국일자 또는 선고일자 기준으로 수집
        # 배치 단위 저장 (10건마다)
        # 체크포인트 복구 기능
```

#### 0.2 배치 저장 및 체크포인트 시스템

```python
# 10건마다 자동 저장
if len(batch_decisions) >= 10:
    self._save_batch(batch_decisions, output_dir, page, category)
    batch_decisions = []  # 배치 초기화

# 체크포인트 저장
def _save_checkpoint(self, output_dir: Path, page_num: int, collected_count: int):
    checkpoint_data = {
        "checkpoint_info": {
            "last_page": page_num,
            "collected_count": collected_count,
            "timestamp": datetime.now().isoformat(),
            "status": "in_progress"
        }
    }
```

#### 0.3 헌재결정례 데이터 구조

```python
@dataclass
class ConstitutionalDecisionData:
    """헌재결정례 데이터 클래스 - 목록 데이터 내부에 본문 데이터 포함"""
    # 목록 조회 API 응답 (기본 정보)
    id: str
    사건번호: str
    종국일자: str
    헌재결정례일련번호: str
    사건명: str
    헌재결정례상세링크: str
    
    # 상세 조회 API 응답 (본문 데이터)
    사건종류명: Optional[str] = None
    판시사항: Optional[str] = None
    결정요지: Optional[str] = None
    전문: Optional[str] = None
    참조조문: Optional[str] = None
    참조판례: Optional[str] = None
    심판대상조문: Optional[str] = None
```

### 1단계: 데이터베이스 스키마 설계 및 구현

#### 1.1 SQLite 테이블 생성

```python
# source/data/database.py
class DatabaseManager:
    def _create_hybrid_search_tables(self):
        """하이브리드 검색을 위한 테이블 생성"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 법률 문서 메인 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS legal_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    law_name TEXT,
                    article_number INTEGER,
                    court_name TEXT,
                    case_number TEXT,
                    decision_date DATE,
                    source_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 벡터 임베딩 연동 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    embedding_vector BLOB NOT NULL,
                    faiss_index INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES legal_documents (id)
                )
            """)
            
            # 검색 인덱스 생성
            self._create_search_indexes(cursor)
            
            conn.commit()
            self.logger.info("하이브리드 검색 테이블 생성 완료")
    
    def _create_search_indexes(self, cursor):
        """검색 성능을 위한 인덱스 생성"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_document_type ON legal_documents(document_type)",
            "CREATE INDEX IF NOT EXISTS idx_law_name ON legal_documents(law_name)",
            "CREATE INDEX IF NOT EXISTS idx_article_number ON legal_documents(article_number)",
            "CREATE INDEX IF NOT EXISTS idx_court_name ON legal_documents(court_name)",
            "CREATE INDEX IF NOT EXISTS idx_decision_date ON legal_documents(decision_date)",
            "CREATE INDEX IF NOT EXISTS idx_document_id ON legal_documents(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_title ON legal_documents(title)",
            "CREATE INDEX IF NOT EXISTS idx_faiss_index ON document_embeddings(faiss_index)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
```

#### 1.2 데이터 마이그레이션

```python
# scripts/migrate_to_hybrid_search.py
class HybridSearchMigrator:
    def __init__(self):
        self.database = DatabaseManager()
        self.vector_store = LegalVectorStore()
        self.logger = logging.getLogger(__name__)
    
    def migrate_existing_data(self):
        """기존 데이터를 하이브리드 검색 구조로 마이그레이션"""
        try:
            # 1. 기존 데이터 로드
            existing_data = self._load_existing_data()
            
            # 2. 데이터 변환 및 저장
            for data in existing_data:
                self._migrate_document(data)
            
            # 3. 벡터 임베딩 생성
            self._generate_vector_embeddings()
            
            self.logger.info("하이브리드 검색 마이그레이션 완료")
            
        except Exception as e:
            self.logger.error(f"마이그레이션 실패: {e}")
            raise
    
    def _migrate_document(self, data: Dict[str, Any]):
        """개별 문서 마이그레이션"""
        # SQLite에 문서 저장
        document_id = self.database.insert_document({
            'document_id': data.get('id'),
            'title': data.get('title', ''),
            'content': data.get('content', ''),
            'document_type': data.get('type', 'unknown'),
            'law_name': data.get('law_name'),
            'article_number': data.get('article_number'),
            'court_name': data.get('court_name'),
            'case_number': data.get('case_number'),
            'decision_date': data.get('decision_date'),
            'source_url': data.get('source_url')
        })
        
        return document_id
```

### 2단계: 검색 엔진 구현

#### 2.1 정확한 매칭 검색 엔진

```python
# source/services/exact_search_engine.py
class ExactSearchEngine:
    def __init__(self, database: DatabaseManager):
        self.database = database
        self.logger = logging.getLogger(__name__)
    
    def search(self, query: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """정확한 매칭 검색"""
        try:
            # 쿼리 분석
            query_analysis = self._analyze_query(query)
            
            # SQL 쿼리 생성
            sql_query, params = self._build_sql_query(query_analysis, filters)
            
            # 데이터베이스 쿼리 실행
            results = self.database.execute_query(sql_query, params)
            
            # 결과 포맷팅
            formatted_results = self._format_results(results)
            
            self.logger.info(f"정확한 매칭 검색 완료: {len(formatted_results)}개 결과")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"정확한 매칭 검색 실패: {e}")
            return []
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """쿼리 분석"""
        analysis = {
            'original_query': query,
            'law_name': None,
            'article_number': None,
            'case_number': None,
            'court_name': None,
            'keywords': []
        }
        
        # 법령명 패턴 매칭
        law_patterns = [
            r'([가-힣]+법)\s*제(\d+)조',
            r'([가-힣]+법)\s*(\d+)조',
            r'([가-힣]+법)'
        ]
        
        for pattern in law_patterns:
            match = re.search(pattern, query)
            if match:
                analysis['law_name'] = match.group(1)
                if len(match.groups()) > 1:
                    analysis['article_number'] = int(match.group(2))
                break
        
        # 사건번호 패턴 매칭
        case_pattern = r'(\d{4}[가-힣]\d+)'
        case_match = re.search(case_pattern, query)
        if case_match:
            analysis['case_number'] = case_match.group(1)
        
        # 법원명 패턴 매칭
        court_patterns = ['대법원', '고등법원', '지방법원', '가정법원', '행정법원']
        for court in court_patterns:
            if court in query:
                analysis['court_name'] = court
                break
        
        # 키워드 추출
        analysis['keywords'] = self._extract_keywords(query)
        
        return analysis
    
    def _build_sql_query(self, query_analysis: Dict[str, Any], 
                        filters: Dict[str, Any] = None) -> Tuple[str, List]:
        """SQL 쿼리 생성"""
        base_query = "SELECT * FROM legal_documents WHERE 1=1"
        params = []
        
        # 법령명 검색
        if query_analysis['law_name']:
            base_query += " AND law_name LIKE ?"
            params.append(f"%{query_analysis['law_name']}%")
        
        # 조문번호 검색
        if query_analysis['article_number']:
            base_query += " AND article_number = ?"
            params.append(query_analysis['article_number'])
        
        # 사건번호 검색
        if query_analysis['case_number']:
            base_query += " AND case_number = ?"
            params.append(query_analysis['case_number'])
        
        # 법원명 검색
        if query_analysis['court_name']:
            base_query += " AND court_name = ?"
            params.append(query_analysis['court_name'])
        
        # 키워드 검색
        if query_analysis['keywords']:
            keyword_conditions = []
            for keyword in query_analysis['keywords']:
                keyword_conditions.append("(title LIKE ? OR content LIKE ?)")
                params.extend([f"%{keyword}%", f"%{keyword}%"])
            
            if keyword_conditions:
                base_query += " AND (" + " OR ".join(keyword_conditions) + ")"
        
        # 추가 필터 적용
        if filters:
            base_query, params = self._apply_filters(base_query, params, filters)
        
        # 정렬 및 제한
        base_query += " ORDER BY created_at DESC LIMIT 50"
        
        return base_query, params
    
    def _apply_filters(self, base_query: str, params: List, 
                      filters: Dict[str, Any]) -> Tuple[str, List]:
        """추가 필터 적용"""
        if 'document_type' in filters:
            base_query += " AND document_type = ?"
            params.append(filters['document_type'])
        
        if 'year' in filters:
            base_query += " AND strftime('%Y', decision_date) = ?"
            params.append(str(filters['year']))
        
        if 'date_range' in filters:
            date_range = filters['date_range']
            if 'start' in date_range:
                base_query += " AND decision_date >= ?"
                params.append(date_range['start'])
            if 'end' in date_range:
                base_query += " AND decision_date <= ?"
                params.append(date_range['end'])
        
        return base_query, params
    
    def _extract_keywords(self, query: str) -> List[str]:
        """키워드 추출"""
        # 간단한 키워드 추출 (실제로는 더 정교한 NLP 기법 사용)
        stopwords = ['의', '을', '를', '에', '에서', '로', '으로', '와', '과', '는', '은', '이', '가']
        words = re.findall(r'[가-힣]+', query)
        keywords = [word for word in words if word not in stopwords and len(word) > 1]
        return keywords
    
    def _format_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """결과 포맷팅"""
        formatted = []
        for result in results:
            formatted.append({
                'id': result['id'],
                'document_id': result['document_id'],
                'title': result['title'],
                'content': result['content'][:500] + '...' if len(result['content']) > 500 else result['content'],
                'document_type': result['document_type'],
                'law_name': result['law_name'],
                'article_number': result['article_number'],
                'court_name': result['court_name'],
                'case_number': result['case_number'],
                'decision_date': result['decision_date'],
                'source_url': result['source_url'],
                'exact_match': True,
                'similarity_score': 1.0,
                'search_type': 'exact'
            })
        return formatted
```

#### 2.2 의미적 검색 엔진

```python
# source/services/semantic_search_engine.py
class SemanticSearchEngine:
    def __init__(self, vector_store: LegalVectorStore):
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)
    
    def search(self, query: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """의미적 검색"""
        try:
            # 벡터 검색
            similar_docs = self.vector_store.search(query, top_k=20)
            
            # 필터 적용
            if filters:
                similar_docs = self._apply_filters(similar_docs, filters)
            
            # 결과 포맷팅
            formatted_results = self._format_results(similar_docs)
            
            self.logger.info(f"의미적 검색 완료: {len(formatted_results)}개 결과")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"의미적 검색 실패: {e}")
            return []
    
    def _apply_filters(self, results: List[Dict[str, Any]], 
                      filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """결과 필터링"""
        filtered_results = []
        
        for result in results:
            metadata = result.get('metadata', {})
            
            # 문서 타입 필터
            if 'document_type' in filters:
                if metadata.get('document_type') != filters['document_type']:
                    continue
            
            # 법원명 필터
            if 'court_name' in filters:
                if metadata.get('court_name') != filters['court_name']:
                    continue
            
            # 연도 필터
            if 'year' in filters:
                decision_date = metadata.get('decision_date')
                if decision_date and str(decision_date.year) != str(filters['year']):
                    continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def _format_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """결과 포맷팅"""
        formatted = []
        for result in results:
            metadata = result.get('metadata', {})
            formatted.append({
                'id': metadata.get('document_id'),
                'document_id': metadata.get('document_id'),
                'title': metadata.get('law_name', ''),
                'content': result.get('text', ''),
                'document_type': metadata.get('document_type', ''),
                'law_name': metadata.get('law_name'),
                'article_number': metadata.get('article_number'),
                'court_name': metadata.get('court_name'),
                'case_number': metadata.get('case_number'),
                'decision_date': metadata.get('decision_date'),
                'source_url': metadata.get('source_url'),
                'exact_match': False,
                'similarity_score': result.get('score', 0.0),
                'search_type': 'semantic'
            })
        return formatted
```

#### 2.3 하이브리드 검색 엔진

```python
# source/services/hybrid_search_engine.py
class HybridSearchEngine:
    def __init__(self, database: DatabaseManager, vector_store: LegalVectorStore):
        self.database = database
        self.vector_store = vector_store
        self.exact_search = ExactSearchEngine(database)
        self.semantic_search = SemanticSearchEngine(vector_store)
        self.result_merger = ResultMerger()
        self.result_ranker = ResultRanker()
        self.logger = logging.getLogger(__name__)
    
    def search(self, query: str, search_type: str = "hybrid", 
               filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """하이브리드 검색 실행"""
        try:
            if search_type == "exact":
                return self.exact_search.search(query, filters)
            elif search_type == "semantic":
                return self.semantic_search.search(query, filters)
            elif search_type == "hybrid":
                return self._hybrid_search(query, filters)
            else:
                raise ValueError(f"Unknown search type: {search_type}")
                
        except Exception as e:
            self.logger.error(f"하이브리드 검색 실패: {e}")
            return []
    
    def _hybrid_search(self, query: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """하이브리드 검색 실행"""
        # 1. 정확한 매칭 검색
        exact_results = self.exact_search.search(query, filters)
        
        # 2. 의미적 검색
        semantic_results = self.semantic_search.search(query, filters)
        
        # 3. 결과 통합
        merged_results = self.result_merger.merge(exact_results, semantic_results)
        
        # 4. 결과 랭킹
        ranked_results = self.result_ranker.rank(merged_results, query)
        
        self.logger.info(f"하이브리드 검색 완료: {len(ranked_results)}개 결과")
        return ranked_results
```

### 3단계: 결과 통합 및 랭킹

#### 3.1 결과 통합기

```python
# source/services/result_merger.py
class ResultMerger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def merge(self, exact_results: List[Dict[str, Any]], 
              semantic_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """검색 결과 통합"""
        try:
            # 중복 제거
            merged_results = self._remove_duplicates(exact_results, semantic_results)
            
            # 결과 통합
            all_results = exact_results + semantic_results
            unique_results = self._deduplicate_by_id(all_results)
            
            self.logger.info(f"결과 통합 완료: {len(unique_results)}개 고유 결과")
            return unique_results
            
        except Exception as e:
            self.logger.error(f"결과 통합 실패: {e}")
            return exact_results + semantic_results
    
    def _remove_duplicates(self, exact_results: List[Dict[str, Any]], 
                          semantic_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """중복 제거"""
        exact_ids = {result['id'] for result in exact_results}
        unique_semantic = [result for result in semantic_results 
                          if result['id'] not in exact_ids]
        return exact_results + unique_semantic
    
    def _deduplicate_by_id(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ID 기준 중복 제거"""
        seen_ids = set()
        unique_results = []
        
        for result in results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                unique_results.append(result)
        
        return unique_results
```

#### 3.2 결과 랭킹기

```python
# source/services/result_ranker.py
class ResultRanker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def rank(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """결과 랭킹"""
        try:
            # 랭킹 점수 계산
            for result in results:
                result['ranking_score'] = self._calculate_ranking_score(result, query)
            
            # 점수 기준 정렬
            ranked_results = sorted(results, key=lambda x: x['ranking_score'], reverse=True)
            
            self.logger.info(f"결과 랭킹 완료: {len(ranked_results)}개 결과")
            return ranked_results
            
        except Exception as e:
            self.logger.error(f"결과 랭킹 실패: {e}")
            return results
    
    def _calculate_ranking_score(self, result: Dict[str, Any], query: str) -> float:
        """랭킹 점수 계산"""
        score = 0.0
        
        # 정확한 매칭 보너스
        if result.get('exact_match', False):
            score += 1.0
        
        # 유사도 점수
        similarity_score = result.get('similarity_score', 0.0)
        score += similarity_score * 0.8
        
        # 제목 매칭 보너스
        title = result.get('title', '').lower()
        query_lower = query.lower()
        if query_lower in title:
            score += 0.3
        
        # 키워드 매칭 보너스
        content = result.get('content', '').lower()
        query_words = query_lower.split()
        matched_words = sum(1 for word in query_words if word in content)
        score += matched_words * 0.1
        
        # 최신 문서 보너스
        decision_date = result.get('decision_date')
        if decision_date:
            from datetime import datetime
            current_year = datetime.now().year
            if isinstance(decision_date, str):
                doc_year = int(decision_date[:4])
            else:
                doc_year = decision_date.year
            year_diff = current_year - doc_year
            if year_diff <= 5:  # 최근 5년
                score += 0.2 * (1 - year_diff / 5)
        
        return score
```

### 4단계: API 엔드포인트 구현

#### 4.1 검색 API 엔드포인트

```python
# source/api/endpoints.py
from fastapi import APIRouter, HTTPException, Query, Body
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    search_type: str = "hybrid"  # "exact", "semantic", "hybrid"
    filters: Optional[Dict[str, Any]] = None
    limit: int = 20
    offset: int = 0

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_count: int
    search_time: float
    search_type: str

@router.post("/api/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """문서 검색 API"""
    try:
        start_time = time.time()
        
        # 하이브리드 검색 실행
        results = hybrid_search_engine.search(
            query=request.query,
            search_type=request.search_type,
            filters=request.filters
        )
        
        # 결과 제한 및 오프셋 적용
        total_count = len(results)
        paginated_results = results[request.offset:request.offset + request.limit]
        
        search_time = time.time() - start_time
        
        return SearchResponse(
            results=paginated_results,
            total_count=total_count,
            search_time=search_time,
            search_type=request.search_type
        )
        
    except Exception as e:
        logger.error(f"검색 API 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/search/suggestions")
async def get_search_suggestions(q: str = Query(..., min_length=1)):
    """검색 제안 API"""
    try:
        # 간단한 검색 제안 구현
        suggestions = search_service.get_suggestions(q)
        return {"suggestions": suggestions}
        
    except Exception as e:
        logger.error(f"검색 제안 API 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### 5단계: 테스트 및 성능 최적화

#### 5.1 단위 테스트

```python
# tests/unit/test_hybrid_search.py
import pytest
from source.services.hybrid_search_engine import HybridSearchEngine
from source.data.database import DatabaseManager
from source.data.vector_store import LegalVectorStore

class TestHybridSearchEngine:
    def setup_method(self):
        self.database = DatabaseManager(":memory:")
        self.vector_store = LegalVectorStore()
        self.search_engine = HybridSearchEngine(self.database, self.vector_store)
    
    def test_exact_search(self):
        """정확한 매칭 검색 테스트"""
        results = self.search_engine.search("민법 제1조", search_type="exact")
        assert len(results) > 0
        assert all(result['exact_match'] for result in results)
    
    def test_semantic_search(self):
        """의미적 검색 테스트"""
        results = self.search_engine.search("계약 해지", search_type="semantic")
        assert len(results) > 0
        assert all(not result['exact_match'] for result in results)
    
    def test_hybrid_search(self):
        """하이브리드 검색 테스트"""
        results = self.search_engine.search("계약 해지", search_type="hybrid")
        assert len(results) > 0
        assert any(result['exact_match'] for result in results) or any(not result['exact_match'] for result in results)
```

#### 5.2 성능 테스트

```python
# tests/performance/test_search_performance.py
import time
import pytest
from source.services.hybrid_search_engine import HybridSearchEngine

class TestSearchPerformance:
    def test_search_response_time(self):
        """검색 응답 시간 테스트"""
        search_engine = HybridSearchEngine(database, vector_store)
        
        queries = [
            "민법 제1조",
            "계약 해지 손해배상",
            "대법원 2023다12345"
        ]
        
        for query in queries:
            start_time = time.time()
            results = search_engine.search(query, search_type="hybrid")
            response_time = time.time() - start_time
            
            assert response_time < 1.0  # 1초 이내 응답
            assert len(results) > 0
```

## 배포 및 운영

### 1. Docker 설정

```dockerfile
# Dockerfile.hybrid-search
FROM python:3.9-slim

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

# 애플리케이션 코드 복사
COPY source/ ./source/
COPY scripts/ ./scripts/

# 하이브리드 검색 서비스 실행
CMD ["python", "-m", "source.services.hybrid_search_service"]
```

### 2. 환경 설정

```python
# source/utils/config.py
class HybridSearchConfig:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///./data/lawfirm.db")
        self.vector_store_path = os.getenv("VECTOR_STORE_PATH", "./data/embeddings/")
        self.search_cache_size = int(os.getenv("SEARCH_CACHE_SIZE", "1000"))
        self.max_search_results = int(os.getenv("MAX_SEARCH_RESULTS", "100"))
        self.search_timeout = int(os.getenv("SEARCH_TIMEOUT", "30"))
```

### 3. 모니터링 설정

```python
# source/utils/monitoring.py
class SearchMonitoring:
    def __init__(self):
        self.metrics = {
            'search_count': 0,
            'exact_search_count': 0,
            'semantic_search_count': 0,
            'hybrid_search_count': 0,
            'average_response_time': 0.0,
            'error_count': 0
        }
    
    def record_search(self, search_type: str, response_time: float, success: bool):
        """검색 메트릭 기록"""
        self.metrics['search_count'] += 1
        self.metrics[f'{search_type}_search_count'] += 1
        
        if success:
            # 응답 시간 업데이트
            current_avg = self.metrics['average_response_time']
            total_searches = self.metrics['search_count']
            self.metrics['average_response_time'] = (
                (current_avg * (total_searches - 1) + response_time) / total_searches
            )
        else:
            self.metrics['error_count'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """메트릭 반환"""
        return self.metrics.copy()
```

## 사용 예시

### 1. 기본 검색

```python
# 하이브리드 검색 엔진 초기화
search_engine = HybridSearchEngine(database, vector_store)

# 정확한 매칭 검색
exact_results = search_engine.search("민법 제1조", search_type="exact")

# 의미적 검색
semantic_results = search_engine.search("계약 해지", search_type="semantic")

# 하이브리드 검색
hybrid_results = search_engine.search("계약 해지 손해배상", search_type="hybrid")
```

### 2. 필터링 검색

```python
# 필터와 함께 검색
filters = {
    'document_type': 'precedent',
    'court_name': '대법원',
    'year': 2023
}

results = search_engine.search(
    query="계약 해지",
    search_type="hybrid",
    filters=filters
)
```

### 3. API 사용

```python
# API 요청 예시
import requests

response = requests.post("http://localhost:8000/api/search", json={
    "query": "계약 해지 손해배상",
    "search_type": "hybrid",
    "filters": {
        "document_type": "precedent",
        "court_name": "대법원"
    },
    "limit": 10
})

results = response.json()
print(f"검색 결과: {len(results['results'])}개")
print(f"검색 시간: {results['search_time']:.3f}초")
```

이 구현 가이드를 따라 하이브리드 검색 시스템을 구축하면 LawFirmAI에서 정확한 법률 정보 검색과 의미적 검색을 모두 지원할 수 있습니다.
