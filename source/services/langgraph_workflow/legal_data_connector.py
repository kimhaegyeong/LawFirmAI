# -*- coding: utf-8 -*-
"""
실제 법률 데이터 연동 서비스
모의 데이터 대신 실제 법률 데이터베이스 사용
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LegalDataConnector:
    """실제 법률 데이터베이스 연결 및 검색 서비스"""

    def __init__(self, db_path: str = "data/lawfirm.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._ensure_database_exists()

    def _ensure_database_exists(self):
        """데이터베이스 존재 확인 및 초기화"""
        if not Path(self.db_path).exists():
            self.logger.warning(f"Database {self.db_path} not found. Creating with sample data.")
            self._create_sample_database()

    def _create_sample_database(self):
        """샘플 법률 데이터베이스 생성"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 법률 문서 테이블 생성
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS legal_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 샘플 데이터 삽입
        sample_documents = [
            {
                "title": "계약서 작성 가이드",
                "content": "계약서 작성 시 당사자, 목적, 조건, 기간, 해지조건을 명확히 기재해야 합니다. 민법 537조에 따라 계약의 효력이 발생하며, 계약금과 위약금의 차이점을 구분하여 명시해야 합니다. 손해배상의 범위를 명확히 정하고, 특약사항이 있는 경우 별도로 기재해야 합니다.",
                "category": "contract_review",
                "source": "민법 해설서"
            },
            {
                "title": "이혼 절차 안내",
                "content": "이혼 절차는 협의이혼, 조정이혼, 재판이혼으로 구분됩니다. 협의이혼은 부부가 합의하여 가정법원에 신청하는 방식이며, 조정이혼은 가정법원의 조정을 통해 이루어집니다. 재판이혼은 법원의 판결에 의한 이혼으로, 위자료와 재산분할을 고려해야 합니다. 양육권과 면접교섭권은 별개 권리로 양육비 지급과 함께 고려해야 합니다.",
                "category": "family_law",
                "source": "가족법 조문"
            },
            {
                "title": "절도죄 구성요건",
                "content": "절도죄는 타인의 재물을 절취하는 범죄로 형법 329조에 규정되어 있습니다. 구성요건으로는 타인의 재물, 절취행위, 불법영득의사, 고의가 필요합니다. 타인의 재물이란 타인이 소유하거나 점유하는 재물을 의미하며, 절취는 평온을 해하는 방법으로 재물을 취득하는 행위입니다. 불법영득의사는 타인의 소유권을 침해할 의사가 있어야 합니다.",
                "category": "criminal_law",
                "source": "형법 조문"
            },
            {
                "title": "손해배상 청구 방법",
                "content": "손해배상 청구는 민법 750조 불법행위에 근거합니다. 구성요건으로는 고의 또는 과실, 위법한 행위, 손해의 발생, 인과관계가 필요합니다. 청구 방법은 소송을 통한 방법과 합의를 통한 방법이 있습니다. 소송의 경우 관할 법원에 제기하며, 합의의 경우 당사자 간 협의를 통해 이루어집니다. 손해의 범위는 재산적 손해와 정신적 손해를 포함합니다.",
                "category": "civil_law",
                "source": "민법 해설서"
            },
            {
                "title": "부당해고 구제 절차",
                "content": "부당해고 구제는 근로기준법에 근거하여 노동위원회에 신청합니다. 구제신청은 해고일로부터 3개월 이내에 하여야 하며, 원직복직과 임금상당액 지급을 청구할 수 있습니다. 노동위원회는 조정과 중재를 통해 해결을 시도하며, 조정이 성립되지 않으면 중재로 진행됩니다. 중재판정에 불복하는 경우 법원에 소송을 제기할 수 있습니다.",
                "category": "labor_law",
                "source": "근로기준법"
            },
            {
                "title": "부동산 매매계약서 필수 조항",
                "content": "부동산 매매계약서에는 당사자, 부동산의 표시, 매매대금, 계약금, 중도금, 잔금, 소유권 이전, 인도, 특약사항이 포함되어야 합니다. 당사자는 매도인과 매수인의 성명, 주소, 연락처를 명확히 기재하고, 부동산의 표시는 등기부등본과 일치해야 합니다. 매매대금은 계약금, 중도금, 잔금으로 구분하여 지급 시기를 명시하며, 소유권 이전 등기와 인도 시기를 정해야 합니다.",
                "category": "property_law",
                "source": "부동산 거래법"
            },
            {
                "title": "특허권 침해 구제 방법",
                "content": "특허권 침해 시 구제 방법으로는 침해금지, 손해배상, 신용회복이 있습니다. 침해금지 청구는 특허심판원이나 법원에 신청할 수 있으며, 손해배상은 민사소송을 통해 청구합니다. 신용회복은 침해로 인한 명예훼손에 대한 구제 방법입니다. 특허권자는 침해행위의 중지와 예방을 청구할 수 있으며, 침해로 인한 손해의 배상을 청구할 수 있습니다.",
                "category": "intellectual_property",
                "source": "특허법"
            },
            {
                "title": "소득세 가산세 안내",
                "content": "소득세 신고 누락 시 가산세는 무신고가산세와 과소신고가산세로 구분됩니다. 무신고가산세는 신고하지 않은 경우 납부세액의 20%이며, 과소신고가산세는 신고한 세액이 실제 세액보다 적은 경우 부족세액의 10%입니다. 납부지연가산세는 납부기한까지 납부하지 않은 경우 납부세액의 연 14.6%입니다. 가산세는 세법에 따라 계산되며, 정당한 사유가 있는 경우 감면될 수 있습니다.",
                "category": "tax_law",
                "source": "소득세법"
            },
            {
                "title": "법정대리인 권한",
                "content": "법정대리인은 미성년자나 성년후견인을 대신하여 법률행위를 할 수 있는 권한을 가집니다. 권한의 범위는 재산관리와 신분행위로 구분되며, 재산관리는 일상적인 재산관리와 중요한 재산처분으로 나뉩니다. 신분행위는 입양, 혼인 등에 대한 동의권과 취소권을 포함합니다. 법정대리인의 권한은 민법에 의해 제한되며, 미성년자의 이익을 보호하는 것이 원칙입니다.",
                "category": "civil_law",
                "source": "민법"
            },
            {
                "title": "민사소송 관할 법원",
                "content": "민사소송의 관할은 민사소송법에 의해 결정됩니다. 보통재판적은 피고의 주소지 또는 거소지 법원이며, 특별재판적은 사건의 성질에 따라 결정됩니다. 토지관할은 부동산이 있는 곳의 법원이 관할하며, 사물관할은 소송의 목적가액에 따라 결정됩니다. 관할법원은 소장 제출 시점에 결정되며, 관할이 없는 경우 이송신청을 할 수 있습니다.",
                "category": "civil_procedure",
                "source": "민사소송법"
            }
        ]

        for doc in sample_documents:
            cursor.execute('''
                INSERT INTO legal_documents (title, content, category, source)
                VALUES (?, ?, ?, ?)
            ''', (doc["title"], doc["content"], doc["category"], doc["source"]))

        conn.commit()
        conn.close()
        self.logger.info(f"Sample database created with {len(sample_documents)} documents")

    async def search_legal_documents(self, query: str, domain_hints: List[str] = None) -> List[Dict[str, Any]]:
        """법률 문서 비동기 검색 (LangGraph 워크플로우용)"""
        try:
            # 도메인 힌트가 있으면 해당 카테고리로 검색
            category = None
            if domain_hints:
                # 도메인 힌트를 카테고리로 매핑
                domain_mapping = {
                    "labor": "labor_law",
                    "family": "family_law",
                    "criminal": "criminal_law",
                    "civil": "civil_law",
                    "property": "property_law",
                    "intellectual_property": "intellectual_property",
                    "tax": "tax_law",
                    "contract": "contract_review",
                    "procedure": "civil_procedure"
                }

                for hint in domain_hints:
                    if hint in domain_mapping:
                        category = domain_mapping[hint]
                        break

            # 동기 메서드 호출
            results = self.search_documents(query, category, limit=5)

            # 결과를 LangGraph 워크플로우 형식으로 변환
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result["id"],
                    "title": result["title"],
                    "content": result["content"],
                    "category": result["category"],
                    "source": result["source"],
                    "relevance_score": result["relevance_score"],
                    "created_at": result["created_at"]
                })

            self.logger.info(f"Found {len(formatted_results)} legal documents for query: {query}")
            return formatted_results

        except Exception as e:
            self.logger.error(f"Error searching legal documents: {e}")
            return []

    def _extract_keywords(self, query: str) -> List[str]:
        """질문에서 법률 관련 키워드 추출"""
        keywords = []

        # 법률 키워드 매핑
        legal_keywords = {
            "야간수당": ["야간", "야간수당", "야근", "야간근로"],
            "연장근무": ["연장근무", "연장", "초과근무", "휴일근무"],
            "중복": ["중복", "이중"],
            "상속분": ["상속분", "상속"],
            "유언장": ["유언장", "유언"],
            "법정상속인": ["상속인", "법정상속"],
            "손해배상": ["손해배상", "손해", "배상"],
            "불법행위": ["불법행위", "불법"],
        }

        # 질문에서 키워드 찾기
        for keyword, variations in legal_keywords.items():
            if any(v in query for v in variations):
                keywords.append(keyword)

        # 질문 자체를 키워드로 추가
        keywords.append(query)

        return keywords

    def search_documents(self, query: str, category: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """실제 데이터베이스 테이블에서 법률 문서 검색"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            results = []

            # 🆕 키워드 추출 및 확장
            keywords = self._extract_keywords(query)
            self.logger.info(f"🔍 검색 키워드: {keywords}")

            # 검색어 생성 (OR 조건으로 확장된 검색)
            search_conditions = " OR ".join(["article_content LIKE ?" for _ in keywords])
            search_params = [f"%{kw}%" for kw in keywords]

            # 🆕 추가 로깅
            self.logger.info(f"📝 search_conditions: {search_conditions}")
            self.logger.info(f"📝 search_params 개수: {len(search_params)}")
            self.logger.info(f"📝 search_params: {search_params[:3]}...")  # 처음 3개만

            search_term = f"%{query}%"

            # 1. 현행법 조문 검색 (current_laws_articles) - 🆕 키워드 확장 검색
            search_sql = f'''
                SELECT
                    'current_law' as source_type,
                    law_name_korean as title,
                    article_content as content,
                    'current_law' as category,
                    article_number as article_num,
                    law_id as source
                FROM current_laws_articles
                WHERE ({search_conditions}) OR law_name_korean LIKE ?
                ORDER BY
                    CASE
                        WHEN article_content LIKE ? THEN 1
                        WHEN law_name_korean LIKE ? THEN 2
                        ELSE 3
                    END,
                    LENGTH(article_content) DESC
                LIMIT ?
            '''

            # 파라미터 구성: 키워드 검색 조건들 + OR law_name LIKE + 정렬용 + LIMIT
            params = search_params + [search_term] + [search_term, search_term, limit]

            self.logger.info(f"🔍 검색 SQL 실행: {len(keywords)}개 키워드")
            cursor.execute(search_sql, params)

            current_law_results = cursor.fetchall()
            self.logger.info(f"📊 현재법 조문 검색: {len(current_law_results)}개 발견")
            for row in current_law_results:
                results.append({
                    "id": f"current_{row['source']}_{row['article_num']}",
                    "title": f"{row['title']} 제{row['article_num']}조",
                    "content": row["content"],
                    "category": "current_law",
                    "source": row["source"],
                    "created_at": "2024-01-01",
                    "relevance_score": self._calculate_relevance_score(query, row["content"])
                })

            # 2. 법령 조문 검색 (assembly_articles) - 🆕 키워드 확장 검색
            remaining_limit = limit - len(results)
            if remaining_limit > 0:
                assembly_sql = f'''
                    SELECT
                        'assembly_law' as source_type,
                        article_title as title,
                        article_content as content,
                        'assembly_law' as category,
                        article_number as article_num,
                        law_id as source
                    FROM assembly_articles
                    WHERE ({search_conditions}) OR article_title LIKE ?
                    ORDER BY
                        CASE
                            WHEN article_content LIKE ? THEN 1
                            WHEN article_title LIKE ? THEN 2
                            ELSE 3
                        END,
                        LENGTH(article_content) DESC
                    LIMIT ?
                '''

                assembly_params = search_params + [search_term] + [search_term, search_term, remaining_limit]
                cursor.execute(assembly_sql, assembly_params)

                assembly_results = cursor.fetchall()
                for row in assembly_results:
                    results.append({
                        "id": f"assembly_{row['source']}_{row['article_num']}",
                        "title": f"{row['title']} 제{row['article_num']}조",
                        "content": row["content"],
                        "category": "assembly_law",
                        "source": row["source"],
                        "created_at": "2024-01-01",
                        "relevance_score": self._calculate_relevance_score(query, row["content"])
                    })

            # 3. 판례 검색 (precedent_cases) - 🆕 키워드 확장 검색
            remaining_limit = limit - len(results)
            if remaining_limit > 0:
                precedent_sql = f'''
                    SELECT
                        'precedent' as source_type,
                        case_name as title,
                        full_text as content,
                        category as category,
                        case_number as article_num,
                        court as source
                    FROM precedent_cases
                    WHERE ({search_conditions}) OR case_name LIKE ?
                    ORDER BY
                        CASE
                            WHEN full_text LIKE ? THEN 1
                            WHEN case_name LIKE ? THEN 2
                            ELSE 3
                        END,
                        LENGTH(full_text) DESC
                    LIMIT ?
                '''

                precedent_params = search_params + [search_term] + [search_term, search_term, remaining_limit]
                cursor.execute(precedent_sql, precedent_params)

                precedent_results = cursor.fetchall()
                for row in precedent_results:
                    results.append({
                        "id": f"precedent_{row['article_num']}",
                        "title": row["title"],
                        "content": row["content"],
                        "category": row["category"],
                        "source": row["source"],
                        "created_at": "2024-01-01",
                        "relevance_score": self._calculate_relevance_score(query, row["content"])
                    })

            conn.close()
            self.logger.info(f"Found {len(results)} documents for query: {query}")
            return results

        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return []

    def _calculate_relevance_score(self, query: str, content: str) -> float:
        """개선된 관련성 점수 계산"""
        if not query or not content:
            return 0.0

        query_words = set(query.lower().split())
        content_lower = content.lower()

        # 단어별 매칭 점수
        word_matches = 0
        for word in query_words:
            if word in content_lower:
                word_matches += 1

        # 기본 점수 계산
        base_score = word_matches / len(query_words) if query_words else 0.0

        # 구문 매칭 보너스
        phrase_bonus = 0.0
        if len(query) > 2:
            if query.lower() in content_lower:
                phrase_bonus = 0.3

        # 길이 기반 보너스 (너무 짧거나 긴 내용에 대한 패널티)
        length_penalty = 0.0
        if len(content) < 50:
            length_penalty = -0.2
        elif len(content) > 2000:
            length_penalty = -0.1

        final_score = min(1.0, max(0.0, base_score + phrase_bonus + length_penalty))
        return round(final_score, 2)

    def get_document_by_category(self, category: str, limit: int = 3) -> List[Dict[str, Any]]:
        """카테고리별 문서 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id, title, content, category, source, created_at
                FROM legal_documents
                WHERE category = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (category, limit))

            rows = cursor.fetchall()
            results = []

            for row in rows:
                results.append({
                    "id": row["id"],
                    "title": row["title"],
                    "content": row["content"],
                    "category": row["category"],
                    "source": row["source"],
                    "created_at": row["created_at"],
                    "relevance_score": 0.8  # 카테고리 매칭 시 높은 점수
                })

            conn.close()
            return results

        except Exception as e:
            self.logger.error(f"Error getting documents by category: {e}")
            return []

    def get_all_categories(self) -> List[str]:
        """모든 카테고리 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT DISTINCT category FROM legal_documents ORDER BY category')
            categories = [row[0] for row in cursor.fetchall()]

            conn.close()
            return categories

        except Exception as e:
            self.logger.error(f"Error getting categories: {e}")
            return []

    def add_document(self, title: str, content: str, category: str, source: str = "Manual") -> bool:
        """새 문서 추가"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO legal_documents (title, content, category, source)
                VALUES (?, ?, ?, ?)
            ''', (title, content, category, source))

            conn.commit()
            conn.close()

            self.logger.info(f"Added new document: {title}")
            return True

        except Exception as e:
            self.logger.error(f"Error adding document: {e}")
            return False
