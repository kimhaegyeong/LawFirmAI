-- lawfirm_v2.db schema (SQLite + FTS5)
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- Domains and Sources
CREATE TABLE IF NOT EXISTS domains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL, -- statute, case, decision, interpretation
    path TEXT NOT NULL,
    hash TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Statutes
CREATE TABLE IF NOT EXISTS statutes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain_id INTEGER NOT NULL REFERENCES domains(id) ON DELETE RESTRICT,
    name TEXT NOT NULL,
    abbrv TEXT,
    statute_type TEXT,
    proclamation_date TEXT,
    effective_date TEXT,
    category TEXT,
    UNIQUE(domain_id, name)
);

CREATE TABLE IF NOT EXISTS statute_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    statute_id INTEGER NOT NULL REFERENCES statutes(id) ON DELETE CASCADE,
    article_no TEXT NOT NULL,   -- 제n조
    clause_no TEXT,             -- 항
    item_no TEXT,               -- 호
    heading TEXT,               -- 조문 제목
    text TEXT NOT NULL,
    version_effective_date TEXT
);

CREATE INDEX IF NOT EXISTS idx_statute_articles_keys
ON statute_articles (statute_id, article_no, clause_no, item_no);

-- Cases (판결문)
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain_id INTEGER NOT NULL REFERENCES domains(id) ON DELETE RESTRICT,
    doc_id TEXT NOT NULL UNIQUE,
    court TEXT,
    case_type TEXT,
    casenames TEXT,
    announce_date TEXT
);

CREATE TABLE IF NOT EXISTS case_paragraphs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    para_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    UNIQUE(case_id, para_index)
);

CREATE INDEX IF NOT EXISTS idx_case_paragraphs_case
ON case_paragraphs (case_id, para_index);

-- Decisions (심결례 등)
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain_id INTEGER NOT NULL REFERENCES domains(id) ON DELETE RESTRICT,
    org TEXT,                   -- 기관
    doc_id TEXT NOT NULL UNIQUE,
    decision_date TEXT,
    result TEXT
);

CREATE TABLE IF NOT EXISTS decision_paragraphs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id INTEGER NOT NULL REFERENCES decisions(id) ON DELETE CASCADE,
    para_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    UNIQUE(decision_id, para_index)
);

CREATE INDEX IF NOT EXISTS idx_decision_paragraphs_decision
ON decision_paragraphs (decision_id, para_index);

-- Interpretations (유권해석)
CREATE TABLE IF NOT EXISTS interpretations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain_id INTEGER NOT NULL REFERENCES domains(id) ON DELETE RESTRICT,
    org TEXT,
    doc_id TEXT NOT NULL UNIQUE,
    title TEXT,
    response_date TEXT
);

CREATE TABLE IF NOT EXISTS interpretation_paragraphs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interpretation_id INTEGER NOT NULL REFERENCES interpretations(id) ON DELETE CASCADE,
    para_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    UNIQUE(interpretation_id, para_index)
);

CREATE INDEX IF NOT EXISTS idx_interpretation_paragraphs_interp
ON interpretation_paragraphs (interpretation_id, para_index);

-- Vector store meta
CREATE TABLE IF NOT EXISTS text_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,           -- statute_article | case_paragraph | decision_paragraph | interpretation_paragraph
    source_id INTEGER NOT NULL,          -- FK to corresponding table
    level TEXT,                          -- article/clause/item or paragraph
    chunk_index INTEGER NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    overlap_chars INTEGER,
    text TEXT NOT NULL,
    token_count INTEGER,
    meta TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_text_chunks_source
ON text_chunks (source_type, source_id, chunk_index);

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id INTEGER NOT NULL REFERENCES text_chunks(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_embeddings_chunk
ON embeddings (chunk_id);

-- Optional cache
CREATE TABLE IF NOT EXISTS retrieval_cache (
    query_hash TEXT PRIMARY KEY,
    topk_ids TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- FTS5 virtual tables with external content
CREATE VIRTUAL TABLE IF NOT EXISTS statute_articles_fts USING fts5(
    text,
    content='statute_articles',
    content_rowid='id',
    tokenize='unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS case_paragraphs_fts USING fts5(
    text,
    content='case_paragraphs',
    content_rowid='id',
    tokenize='unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS decision_paragraphs_fts USING fts5(
    text,
    content='decision_paragraphs',
    content_rowid='id',
    tokenize='unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS interpretation_paragraphs_fts USING fts5(
    text,
    content='interpretation_paragraphs',
    content_rowid='id',
    tokenize='unicode61'
);

-- Triggers to sync base tables with FTS
-- Statute Articles
CREATE TRIGGER IF NOT EXISTS statute_articles_ai AFTER INSERT ON statute_articles BEGIN
  INSERT INTO statute_articles_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS statute_articles_ad AFTER DELETE ON statute_articles BEGIN
  INSERT INTO statute_articles_fts(statute_articles_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS statute_articles_au AFTER UPDATE ON statute_articles BEGIN
  INSERT INTO statute_articles_fts(statute_articles_fts, rowid, text) VALUES ('delete', old.id, old.text);
  INSERT INTO statute_articles_fts(rowid, text) VALUES (new.id, new.text);
END;

-- Case Paragraphs
CREATE TRIGGER IF NOT EXISTS case_paragraphs_ai AFTER INSERT ON case_paragraphs BEGIN
  INSERT INTO case_paragraphs_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS case_paragraphs_ad AFTER DELETE ON case_paragraphs BEGIN
  INSERT INTO case_paragraphs_fts(case_paragraphs_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS case_paragraphs_au AFTER UPDATE ON case_paragraphs BEGIN
  INSERT INTO case_paragraphs_fts(case_paragraphs_fts, rowid, text) VALUES ('delete', old.id, old.text);
  INSERT INTO case_paragraphs_fts(rowid, text) VALUES (new.id, new.text);
END;

-- Decision Paragraphs
CREATE TRIGGER IF NOT EXISTS decision_paragraphs_ai AFTER INSERT ON decision_paragraphs BEGIN
  INSERT INTO decision_paragraphs_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS decision_paragraphs_ad AFTER DELETE ON decision_paragraphs BEGIN
  INSERT INTO decision_paragraphs_fts(decision_paragraphs_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS decision_paragraphs_au AFTER UPDATE ON decision_paragraphs BEGIN
  INSERT INTO decision_paragraphs_fts(decision_paragraphs_fts, rowid, text) VALUES ('delete', old.id, old.text);
  INSERT INTO decision_paragraphs_fts(rowid, text) VALUES (new.id, new.text);
END;

-- Interpretation Paragraphs
CREATE TRIGGER IF NOT EXISTS interpretation_paragraphs_ai AFTER INSERT ON interpretation_paragraphs BEGIN
  INSERT INTO interpretation_paragraphs_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS interpretation_paragraphs_ad AFTER DELETE ON interpretation_paragraphs BEGIN
  INSERT INTO interpretation_paragraphs_fts(interpretation_paragraphs_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS interpretation_paragraphs_au AFTER UPDATE ON interpretation_paragraphs BEGIN
  INSERT INTO interpretation_paragraphs_fts(interpretation_paragraphs_fts, rowid, text) VALUES ('delete', old.id, old.text);
  INSERT INTO interpretation_paragraphs_fts(rowid, text) VALUES (new.id, new.text);
END;

-- Set user_version for migration tracking
PRAGMA user_version = 1;
