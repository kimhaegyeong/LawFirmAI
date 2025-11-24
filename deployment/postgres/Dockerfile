# 공식 PostgreSQL 이미지에 pgvector 확장 설치
FROM postgres:18

# pgvector 버전 (PostgreSQL 18 호환 최신 버전)
ARG PGVECTOR_VERSION=v0.8.1

# 빌드 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    postgresql-server-dev-18 \
    && rm -rf /var/lib/apt/lists/*

# pgvector 소스에서 빌드 및 설치
RUN git clone --branch ${PGVECTOR_VERSION} https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && cd /tmp/pgvector \
    && make \
    && make install \
    && rm -rf /tmp/pgvector

# 빌드 의존성 제거 (이미지 크기 최적화)
RUN apt-get purge -y \
    build-essential \
    git \
    postgresql-server-dev-18 \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

