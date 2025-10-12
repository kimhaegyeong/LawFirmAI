# Law Data Preprocessing Pipeline - Version Management Plan

## Overview

Raw Assembly law data의 구조가 수집 일자에 따라 변경될 수 있으므로, 파싱 버전 관리 시스템을 구현하여 다양한 데이터 형식을 처리할 수 있도록 합니다.

## Version Management Architecture

### 1. Version Detection System

#### Data Structure Analyzer
**File:** `scripts/assembly/parsers/version_detector.py`

```python
class DataVersionDetector:
    """Raw 데이터 구조를 분석하여 파싱 버전을 자동 감지"""
    
    def __init__(self):
        self.version_patterns = {
            'v1.0': {
                'required_fields': ['law_name', 'law_content', 'content_html'],
                'optional_fields': ['row_number', 'category', 'law_type'],
                'date_patterns': ['YYYY.M.D.', 'YYYY년 M월 D일'],
                'html_structure': 'basic_html'
            },
            'v1.1': {
                'required_fields': ['law_name', 'law_content', 'content_html', 'promulgation_number'],
                'optional_fields': ['row_number', 'category', 'law_type', 'enforcement_date'],
                'date_patterns': ['YYYY.M.D.', 'YYYY년 M월 D일', 'YYYY-MM-DD'],
                'html_structure': 'enhanced_html'
            },
            'v1.2': {
                'required_fields': ['law_name', 'law_content', 'content_html', 'promulgation_number', 'amendment_type'],
                'optional_fields': ['row_number', 'category', 'law_type', 'enforcement_date', 'cont_id'],
                'date_patterns': ['YYYY.M.D.', 'YYYY년 M월 D일', 'YYYY-MM-DD', 'YYYY.MM.DD'],
                'html_structure': 'structured_html'
            }
        }
    
    def detect_version(self, raw_data: dict) -> str:
        """Raw 데이터 구조를 분석하여 버전 감지"""
        field_score = self._analyze_fields(raw_data)
        date_score = self._analyze_date_formats(raw_data)
        html_score = self._analyze_html_structure(raw_data)
        
        # 버전별 점수 계산
        version_scores = {}
        for version, patterns in self.version_patterns.items():
            score = (
                field_score.get(version, 0) * 0.4 +
                date_score.get(version, 0) * 0.3 +
                html_score.get(version, 0) * 0.3
            )
            version_scores[version] = score
        
        # 가장 높은 점수의 버전 반환
        return max(version_scores, key=version_scores.get)
    
    def _analyze_fields(self, raw_data: dict) -> dict:
        """필드 존재 여부로 버전 분석"""
        scores = {}
        for version, patterns in self.version_patterns.items():
            required_count = sum(1 for field in patterns['required_fields'] 
                              if field in raw_data)
            optional_count = sum(1 for field in patterns['optional_fields'] 
                               if field in raw_data)
            
            total_fields = len(patterns['required_fields']) + len(patterns['optional_fields'])
            scores[version] = (required_count + optional_count) / total_fields
        
        return scores
    
    def _analyze_date_formats(self, raw_data: dict) -> dict:
        """날짜 형식으로 버전 분석"""
        scores = {}
        date_fields = ['promulgation_date', 'enforcement_date']
        
        for version, patterns in self.version_patterns.items():
            score = 0
            for field in date_fields:
                if field in raw_data:
                    date_value = raw_data[field]
                    for pattern in patterns['date_patterns']:
                        if self._matches_date_pattern(date_value, pattern):
                            score += 1
                            break
            scores[version] = score / len(date_fields) if date_fields else 0
        
        return scores
    
    def _analyze_html_structure(self, raw_data: dict) -> dict:
        """HTML 구조로 버전 분석"""
        scores = {}
        html_content = raw_data.get('content_html', '')
        
        for version, patterns in self.version_patterns.items():
            structure_type = patterns['html_structure']
            if structure_type == 'basic_html':
                score = 1.0 if '<html>' in html_content else 0.0
            elif structure_type == 'enhanced_html':
                score = 1.0 if '<div' in html_content and '<span' in html_content else 0.0
            elif structure_type == 'structured_html':
                score = 1.0 if '<article' in html_content or 'data-article' in html_content else 0.0
            else:
                score = 0.0
            
            scores[version] = score
        
        return scores
```

### 2. Version-Specific Parsers

#### Version Parser Registry
**File:** `scripts/assembly/parsers/version_parsers.py`

```python
class VersionParserRegistry:
    """버전별 파서 등록 및 관리"""
    
    def __init__(self):
        self.parsers = {
            'v1.0': V1_0Parser(),
            'v1.1': V1_1Parser(),
            'v1.2': V1_2Parser()
        }
        self.default_version = 'v1.2'
    
    def get_parser(self, version: str):
        """버전에 해당하는 파서 반환"""
        return self.parsers.get(version, self.parsers[self.default_version])
    
    def get_supported_versions(self) -> list:
        """지원되는 버전 목록 반환"""
        return list(self.parsers.keys())

class V1_0Parser:
    """버전 1.0 데이터 파서"""
    
    def parse(self, raw_data: dict) -> dict:
        """v1.0 형식의 데이터 파싱"""
        return {
            'law_id': f"assembly_law_{raw_data.get('row_number', 'unknown')}",
            'law_name': raw_data.get('law_name', ''),
            'law_content': raw_data.get('law_content', ''),
            'content_html': raw_data.get('content_html', ''),
            'basic_metadata': {
                'category': raw_data.get('category', ''),
                'law_type': raw_data.get('law_type', '')
            },
            'parsing_version': 'v1.0'
        }

class V1_1Parser:
    """버전 1.1 데이터 파서"""
    
    def parse(self, raw_data: dict) -> dict:
        """v1.1 형식의 데이터 파싱"""
        return {
            'law_id': f"assembly_law_{raw_data.get('row_number', 'unknown')}",
            'law_name': raw_data.get('law_name', ''),
            'law_content': raw_data.get('law_content', ''),
            'content_html': raw_data.get('content_html', ''),
            'promulgation_info': {
                'number': raw_data.get('promulgation_number', ''),
                'date': raw_data.get('promulgation_date', ''),
                'enforcement_date': raw_data.get('enforcement_date', '')
            },
            'basic_metadata': {
                'category': raw_data.get('category', ''),
                'law_type': raw_data.get('law_type', '')
            },
            'parsing_version': 'v1.1'
        }

class V1_2Parser:
    """버전 1.2 데이터 파서 (현재 구현)"""
    
    def parse(self, raw_data: dict) -> dict:
        """v1.2 형식의 데이터 파싱"""
        return {
            'law_id': f"assembly_law_{raw_data.get('row_number', 'unknown')}",
            'law_name': raw_data.get('law_name', ''),
            'law_content': raw_data.get('law_content', ''),
            'content_html': raw_data.get('content_html', ''),
            'promulgation_info': {
                'number': raw_data.get('promulgation_number', ''),
                'date': raw_data.get('promulgation_date', ''),
                'enforcement_date': raw_data.get('enforcement_date', ''),
                'amendment_type': raw_data.get('amendment_type', '')
            },
            'collection_info': {
                'cont_id': raw_data.get('cont_id', ''),
                'cont_sid': raw_data.get('cont_sid', ''),
                'detail_url': raw_data.get('detail_url', ''),
                'collected_at': raw_data.get('collected_at', '')
            },
            'basic_metadata': {
                'category': raw_data.get('category', ''),
                'law_type': raw_data.get('law_type', '')
            },
            'parsing_version': 'v1.2'
        }
```

### 3. Version-Aware Preprocessing Pipeline

#### Updated Main Preprocessing Script
**File:** `scripts/assembly/preprocess_laws.py`

```python
class VersionAwareLawPreprocessor:
    """버전 인식 법률 데이터 전처리기"""
    
    def __init__(self):
        self.version_detector = DataVersionDetector()
        self.version_registry = VersionParserRegistry()
        
        # 버전별 파서들
        self.html_parsers = {
            'v1.0': LawHTMLParserV1_0(),
            'v1.1': LawHTMLParserV1_1(),
            'v1.2': LawHTMLParserV1_2()
        }
        
        self.article_parsers = {
            'v1.0': ArticleParserV1_0(),
            'v1.1': ArticleParserV1_1(),
            'v1.2': ArticleParserV1_2()
        }
        
        # 공통 파서들 (버전 독립적)
        self.metadata_extractor = MetadataExtractor()
        self.text_normalizer = TextNormalizer()
        self.searchable_text_generator = SearchableTextGenerator()
    
    def preprocess_law_file(self, input_file: Path, output_dir: Path) -> Dict[str, Any]:
        """버전 인식 법률 파일 전처리"""
        try:
            logger.info(f"Processing file: {input_file}")
            
            # Raw 데이터 로드
            with open(input_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # 버전 감지
            version = self._detect_file_version(raw_data)
            logger.info(f"Detected version: {version}")
            
            # 버전별 파서 선택
            version_parser = self.version_registry.get_parser(version)
            html_parser = self.html_parsers.get(version, self.html_parsers['v1.2'])
            article_parser = self.article_parsers.get(version, self.article_parsers['v1.2'])
            
            processed_laws = []
            file_stats = {
                'file_name': input_file.name,
                'detected_version': version,
                'total_laws': 0,
                'processed_laws': 0,
                'version_stats': {}
            }
            
            # 각 법률 처리
            laws = raw_data.get('laws', [])
            if not laws and isinstance(raw_data, list):
                laws = raw_data
            
            file_stats['total_laws'] = len(laws)
            
            for law_data in laws:
                try:
                    processed_law = self._process_single_law_with_version(
                        law_data, version, version_parser, html_parser, article_parser
                    )
                    if processed_law:
                        processed_laws.append(processed_law)
                        file_stats['processed_laws'] += 1
                        
                        # 버전별 통계 업데이트
                        if version not in file_stats['version_stats']:
                            file_stats['version_stats'][version] = 0
                        file_stats['version_stats'][version] += 1
                        
                except Exception as e:
                    error_msg = f"Error processing law {law_data.get('law_name', 'Unknown')}: {str(e)}"
                    logger.error(error_msg)
            
            # 처리된 데이터 저장
            if processed_laws:
                output_file = self._get_output_file_path(input_file, output_dir, version)
                self._save_processed_data(processed_laws, output_file)
                logger.info(f"Saved {len(processed_laws)} processed laws to {output_file}")
            
            return file_stats
            
        except Exception as e:
            logger.error(f"Error processing file {input_file}: {e}")
            return {'file_name': input_file.name, 'error': str(e)}
    
    def _detect_file_version(self, raw_data: dict) -> str:
        """파일 버전 감지"""
        if isinstance(raw_data, list) and len(raw_data) > 0:
            # 리스트인 경우 첫 번째 항목으로 버전 감지
            sample_data = raw_data[0]
        else:
            sample_data = raw_data
        
        return self.version_detector.detect_version(sample_data)
    
    def _process_single_law_with_version(self, law_data: Dict[str, Any], 
                                       version: str, version_parser, 
                                       html_parser, article_parser) -> Optional[Dict[str, Any]]:
        """버전별 단일 법률 처리"""
        try:
            # 버전별 기본 파싱
            version_parsed = version_parser.parse(law_data)
            
            # HTML 파싱
            html_content = version_parsed.get('content_html', '')
            html_parsed = html_parser.parse_html(html_content) if html_content else {
                'clean_text': '',
                'articles': [],
                'metadata': {}
            }
            
            # Article 파싱
            law_content = version_parsed.get('law_content', '')
            articles = article_parser.parse_articles(law_content)
            
            # 공통 메타데이터 추출
            metadata = self.metadata_extractor.extract(law_data)
            
            # 텍스트 정규화
            clean_text = self.text_normalizer.normalize(law_content)
            
            # 검색 가능한 텍스트 생성
            searchable_data = self.searchable_text_generator.generate(clean_text, articles)
            
            # 최종 결과 조합
            processed_law = {
                # 버전 정보
                'parsing_version': version,
                'version_metadata': {
                    'detected_at': datetime.now().isoformat(),
                    'version_confidence': self.version_detector.get_confidence(law_data, version)
                },
                
                # 기본 식별 정보
                'law_id': version_parsed.get('law_id', ''),
                'source': 'assembly',
                
                # 버전별 파싱된 데이터
                **version_parsed,
                
                # 추출된 메타데이터
                **metadata,
                
                # 파싱된 콘텐츠
                'articles': articles,
                'full_text': clean_text,
                'searchable_text': searchable_data.get('full_text', ''),
                'keywords': searchable_data.get('keywords', []),
                'summary': searchable_data.get('summary', ''),
                
                # HTML 파싱 데이터
                'html_clean_text': html_parsed.get('clean_text', ''),
                'html_articles': html_parsed.get('articles', []),
                'html_metadata': html_parsed.get('metadata', {}),
                
                # 원본 데이터 (참조용)
                'raw_content': law_content,
                'content_html': html_content,
                
                # 처리 메타데이터
                'processed_at': datetime.now().isoformat(),
                'processing_version': '2.0',  # 파이프라인 버전
                'data_quality': self._calculate_data_quality(articles, metadata, searchable_data, version)
            }
            
            return processed_law
            
        except Exception as e:
            logger.error(f"Error processing single law with version {version}: {e}")
            return None
```

### 4. Version-Specific HTML Parsers

#### HTML Parser Versions
**File:** `scripts/assembly/parsers/html_parsers_v1.py`

```python
class LawHTMLParserV1_0(LawHTMLParser):
    """버전 1.0 HTML 파서"""
    
    def parse_html(self, html_content: str) -> Dict[str, Any]:
        """v1.0 형식의 HTML 파싱"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        return {
            'clean_text': self._extract_basic_text(soup),
            'articles': self._extract_basic_articles(soup),
            'metadata': self._extract_basic_metadata(soup),
            'parser_version': 'v1.0'
        }
    
    def _extract_basic_text(self, soup: BeautifulSoup) -> str:
        """기본 텍스트 추출"""
        # v1.0은 단순한 텍스트 추출
        return soup.get_text()

class LawHTMLParserV1_1(LawHTMLParser):
    """버전 1.1 HTML 파서"""
    
    def parse_html(self, html_content: str) -> Dict[str, Any]:
        """v1.1 형식의 HTML 파싱"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        return {
            'clean_text': self._extract_enhanced_text(soup),
            'articles': self._extract_enhanced_articles(soup),
            'metadata': self._extract_enhanced_metadata(soup),
            'parser_version': 'v1.1'
        }
    
    def _extract_enhanced_text(self, soup: BeautifulSoup) -> str:
        """향상된 텍스트 추출"""
        # v1.1은 div, span 태그 고려
        for script in soup(["script", "style", "nav"]):
            script.decompose()
        
        # div와 span 태그 처리
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return ' '.join(chunk for chunk in chunks if chunk)

class LawHTMLParserV1_2(LawHTMLParser):
    """버전 1.2 HTML 파서 (현재 구현)"""
    
    def parse_html(self, html_content: str) -> Dict[str, Any]:
        """v1.2 형식의 HTML 파싱"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        return {
            'clean_text': self._extract_structured_text(soup),
            'articles': self._extract_structured_articles(soup),
            'metadata': self._extract_structured_metadata(soup),
            'parser_version': 'v1.2'
        }
    
    def _extract_structured_text(self, soup: BeautifulSoup) -> str:
        """구조화된 텍스트 추출"""
        # v1.2는 article 태그나 data-article 속성 고려
        articles = soup.find_all(['article', '[data-article]'])
        if articles:
            return ' '.join(article.get_text() for article in articles)
        else:
            return super()._extract_clean_text(soup)
```

### 5. Version Configuration Management

#### Version Configuration
**File:** `scripts/assembly/config/version_config.py`

```python
class VersionConfig:
    """버전별 설정 관리"""
    
    VERSION_CONFIGS = {
        'v1.0': {
            'name': 'Basic Law Data Format',
            'description': '초기 법률 데이터 형식 (기본 필드만 포함)',
            'supported_features': ['basic_parsing', 'text_extraction'],
            'deprecated': False,
            'migration_path': 'v1.1'
        },
        'v1.1': {
            'name': 'Enhanced Law Data Format',
            'description': '향상된 법률 데이터 형식 (공포 정보 추가)',
            'supported_features': ['basic_parsing', 'text_extraction', 'promulgation_info'],
            'deprecated': False,
            'migration_path': 'v1.2'
        },
        'v1.2': {
            'name': 'Structured Law Data Format',
            'description': '구조화된 법률 데이터 형식 (수정 정보, 수집 메타데이터 포함)',
            'supported_features': ['basic_parsing', 'text_extraction', 'promulgation_info', 
                                  'amendment_info', 'collection_metadata'],
            'deprecated': False,
            'migration_path': None
        }
    }
    
    @classmethod
    def get_config(cls, version: str) -> dict:
        """버전별 설정 반환"""
        return cls.VERSION_CONFIGS.get(version, cls.VERSION_CONFIGS['v1.2'])
    
    @classmethod
    def get_supported_versions(cls) -> list:
        """지원되는 버전 목록 반환"""
        return [v for v, config in cls.VERSION_CONFIGS.items() 
                if not config.get('deprecated', False)]
    
    @classmethod
    def is_version_supported(cls, version: str) -> bool:
        """버전 지원 여부 확인"""
        return version in cls.VERSION_CONFIGS and not cls.VERSION_CONFIGS[version].get('deprecated', False)
```

### 6. Version Migration System

#### Data Migration
**File:** `scripts/assembly/migration/data_migrator.py`

```python
class DataMigrator:
    """데이터 버전 마이그레이션"""
    
    def __init__(self):
        self.migration_rules = {
            'v1.0_to_v1.1': self._migrate_v1_0_to_v1_1,
            'v1.1_to_v1.2': self._migrate_v1_1_to_v1_2
        }
    
    def migrate_data(self, data: dict, from_version: str, to_version: str) -> dict:
        """데이터 버전 마이그레이션"""
        if from_version == to_version:
            return data
        
        migration_key = f"{from_version}_to_{to_version}"
        if migration_key in self.migration_rules:
            return self.migration_rules[migration_key](data)
        else:
            # 직접 마이그레이션이 없는 경우 단계별 마이그레이션
            return self._step_by_step_migration(data, from_version, to_version)
    
    def _migrate_v1_0_to_v1_1(self, data: dict) -> dict:
        """v1.0에서 v1.1로 마이그레이션"""
        migrated = data.copy()
        
        # 공포 정보 추가 (기본값으로 설정)
        migrated['promulgation_number'] = ''
        migrated['promulgation_date'] = ''
        migrated['enforcement_date'] = ''
        
        # 마이그레이션 메타데이터 추가
        migrated['migration_history'] = [
            {
                'from_version': 'v1.0',
                'to_version': 'v1.1',
                'migrated_at': datetime.now().isoformat(),
                'migration_type': 'automatic'
            }
        ]
        
        return migrated
    
    def _migrate_v1_1_to_v1_2(self, data: dict) -> dict:
        """v1.1에서 v1.2로 마이그레이션"""
        migrated = data.copy()
        
        # 수정 정보 추가
        migrated['amendment_type'] = ''
        
        # 수집 메타데이터 추가
        migrated['cont_id'] = ''
        migrated['cont_sid'] = ''
        migrated['detail_url'] = ''
        migrated['collected_at'] = ''
        
        # 마이그레이션 히스토리 업데이트
        migration_history = migrated.get('migration_history', [])
        migration_history.append({
            'from_version': 'v1.1',
            'to_version': 'v1.2',
            'migrated_at': datetime.now().isoformat(),
            'migration_type': 'automatic'
        })
        migrated['migration_history'] = migration_history
        
        return migrated
```

### 7. Version-Aware Database Schema

#### Database Schema Updates
**File:** `scripts/assembly/import_laws_to_db.py`

```python
def _create_tables(self):
    """버전 인식 데이터베이스 테이블 생성"""
    try:
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # assembly_laws 테이블 (버전 정보 추가)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS assembly_laws (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    law_id TEXT UNIQUE NOT NULL,
                    source TEXT NOT NULL DEFAULT 'assembly',
                    
                    -- 버전 정보
                    parsing_version TEXT NOT NULL,
                    version_confidence REAL,
                    migration_history TEXT,  -- JSON array
                    
                    -- 기본 정보
                    law_name TEXT NOT NULL,
                    law_type TEXT,
                    category TEXT,
                    row_number TEXT,
                    
                    -- 공포 정보
                    promulgation_number TEXT,
                    promulgation_date TEXT,
                    enforcement_date TEXT,
                    amendment_type TEXT,
                    
                    -- 추출된 메타데이터
                    ministry TEXT,
                    parent_law TEXT,
                    related_laws TEXT,  -- JSON array
                    
                    -- 콘텐츠
                    full_text TEXT NOT NULL,
                    searchable_text TEXT NOT NULL,
                    keywords TEXT,  -- JSON array
                    summary TEXT,
                    
                    -- HTML 콘텐츠
                    html_clean_text TEXT,
                    content_html TEXT,
                    
                    -- 원본 데이터
                    raw_content TEXT,
                    detail_url TEXT,
                    cont_id TEXT,
                    cont_sid TEXT,
                    collected_at TEXT,
                    
                    -- 처리 메타데이터
                    processed_at TEXT NOT NULL,
                    processing_version TEXT,
                    data_quality TEXT,  -- JSON object
                    
                    -- 타임스탬프
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 버전별 통계 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS assembly_version_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parsing_version TEXT NOT NULL,
                    total_laws INTEGER DEFAULT 0,
                    processed_laws INTEGER DEFAULT 0,
                    failed_laws INTEGER DEFAULT 0,
                    average_quality_score REAL DEFAULT 0.0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(parsing_version)
                )
            ''')
            
            # 버전별 인덱스
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_assembly_laws_parsing_version ON assembly_laws (parsing_version)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_assembly_laws_version_confidence ON assembly_laws (version_confidence)')
            
            conn.commit()
            logger.info("Version-aware database tables created successfully")
            
    except Exception as e:
        logger.error(f"Error creating version-aware database tables: {e}")
        raise
```

### 8. Version Reporting and Analytics

#### Version Analytics
**File:** `scripts/assembly/analytics/version_analytics.py`

```python
class VersionAnalytics:
    """버전별 분석 및 리포팅"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    def generate_version_report(self) -> dict:
        """버전별 리포트 생성"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # 버전별 통계
            cursor.execute('''
                SELECT 
                    parsing_version,
                    COUNT(*) as total_laws,
                    AVG(version_confidence) as avg_confidence,
                    AVG(CAST(JSON_EXTRACT(data_quality, '$.completeness_score') AS REAL)) as avg_quality
                FROM assembly_laws 
                GROUP BY parsing_version
                ORDER BY parsing_version
            ''')
            
            version_stats = cursor.fetchall()
            
            # 최근 처리된 버전들
            cursor.execute('''
                SELECT DISTINCT parsing_version, MAX(processed_at) as last_processed
                FROM assembly_laws 
                GROUP BY parsing_version
                ORDER BY last_processed DESC
            ''')
            
            recent_versions = cursor.fetchall()
            
            return {
                'version_statistics': [
                    {
                        'version': row[0],
                        'total_laws': row[1],
                        'average_confidence': row[2],
                        'average_quality': row[3]
                    }
                    for row in version_stats
                ],
                'recent_versions': [
                    {
                        'version': row[0],
                        'last_processed': row[1]
                    }
                    for row in recent_versions
                ],
                'report_generated_at': datetime.now().isoformat()
            }
    
    def get_version_compatibility_matrix(self) -> dict:
        """버전 호환성 매트릭스"""
        return {
            'supported_versions': ['v1.0', 'v1.1', 'v1.2'],
            'deprecated_versions': [],
            'migration_paths': {
                'v1.0': ['v1.1', 'v1.2'],
                'v1.1': ['v1.2'],
                'v1.2': []
            },
            'compatibility_matrix': {
                'v1.0': {'v1.1': True, 'v1.2': True},
                'v1.1': {'v1.2': True},
                'v1.2': {}
            }
        }
```

## Implementation Steps

### Step 1: Version Detection System
1. `DataVersionDetector` 클래스 구현
2. 버전 패턴 정의 및 테스트
3. 자동 버전 감지 로직 구현

### Step 2: Version-Specific Parsers
1. `VersionParserRegistry` 구현
2. 각 버전별 파서 클래스 구현
3. 버전별 HTML 파서 구현

### Step 3: Updated Preprocessing Pipeline
1. `VersionAwareLawPreprocessor` 구현
2. 버전별 처리 로직 통합
3. 버전별 통계 수집

### Step 4: Database Schema Updates
1. 버전 정보 컬럼 추가
2. 버전별 통계 테이블 생성
3. 버전별 인덱스 생성

### Step 5: Migration System
1. `DataMigrator` 구현
2. 버전간 마이그레이션 규칙 정의
3. 마이그레이션 테스트

### Step 6: Analytics and Reporting
1. `VersionAnalytics` 구현
2. 버전별 리포트 생성
3. 호환성 매트릭스 구현

## Usage Examples

### Version-Aware Preprocessing
```bash
# 자동 버전 감지로 전처리
python scripts/assembly/preprocess_laws.py --input data/raw/assembly/law/20251010 --output data/processed/assembly/law

# 특정 버전으로 강제 처리
python scripts/assembly/preprocess_laws.py --input data/raw/assembly/law/20251010 --output data/processed/assembly/law --force-version v1.1

# 버전별 리포트 생성
python scripts/assembly/version_analytics.py --generate-report
```

### Version Migration
```bash
# 데이터 버전 마이그레이션
python scripts/assembly/migrate_data.py --input data/processed/assembly/law --from-version v1.0 --to-version v1.2
```

## Benefits

1. **Backward Compatibility**: 이전 버전 데이터 처리 가능
2. **Forward Compatibility**: 새로운 버전 데이터 자동 감지
3. **Data Integrity**: 버전별 검증 및 품질 관리
4. **Migration Support**: 버전간 데이터 마이그레이션
5. **Analytics**: 버전별 통계 및 분석
6. **Maintenance**: 버전별 유지보수 및 업데이트

이 버전 관리 시스템을 통해 다양한 수집 일자의 Raw 데이터를 안정적으로 처리할 수 있습니다.
