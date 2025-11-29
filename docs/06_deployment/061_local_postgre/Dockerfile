# 공식 PostgreSQL 이미지에 pgvector 및 PGroonga 확장 설치
# PostgreSQL 18은 최신 버전이므로 PGroonga 4.0.4+ 사용 (PostgreSQL 18 호환)
FROM postgres:18

# pgvector 버전 (PostgreSQL 18 호환 최신 버전)
ARG PGVECTOR_VERSION=v0.8.1

# PGroonga 버전 (PostgreSQL 18 호환 버전)
# PostgreSQL 18의 index_beginscan API 변경사항을 지원하는 버전 사용
ARG PGROONGA_VERSION=4.0.4

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    postgresql-server-dev-18 \
    libgroonga-dev \
    groonga-tokenizer-mecab \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# pgvector 설치 (PostgreSQL 18 호환 버전)
RUN cd /tmp && \
    git clone --branch ${PGVECTOR_VERSION} https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make && \
    make install && \
    cd / && \
    rm -rf /tmp/pgvector

# PGroonga 설치 (서브모듈 포함)
RUN cd /tmp && \
    git clone --depth 1 --recursive https://github.com/pgroonga/pgroonga.git && \
    cd pgroonga && \
    make && \
    make install && \
    cd / && \
    rm -rf /tmp/pgroonga

# 확장 기능 자동 로드 설정 (선택사항)
RUN echo "shared_preload_libraries = 'vector'" >> /usr/share/postgresql/postgresql.conf.sample

# 초기화 스크립트 추가
RUN echo "CREATE EXTENSION IF NOT EXISTS vector;" > /docker-entrypoint-initdb.d/01-init-extensions.sql && \
    echo "CREATE EXTENSION IF NOT EXISTS pgroonga;" >> /docker-entrypoint-initdb.d/01-init-extensions.sql

# 빌드 의존성 제거 (이미지 크기 최적화)
# libgroonga와 groonga-tokenizer-mecab는 런타임에도 필요하므로 유지
RUN apt-get update && apt-get purge -y \
    build-essential \
    git \
    postgresql-server-dev-18 \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 5432

CMD ["postgres"]

