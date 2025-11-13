# -*- coding: utf-8 -*-
"""
LangChain Configuration
LangChain 설정 관리 모듈
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)


class VectorStoreType(Enum):
    """벡터 저장소 타입"""
    FAISS = "faiss"
    CHROMA = "chroma"
    PINECONE = "pinecone"


class LLMProvider(Enum):
    """LLM 제공자 타입"""
    OPENAI = "openai"
    LOCAL = "local"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class LangChainConfig:
    """LangChain 설정 클래스"""
    
    # 벡터 저장소 설정
    vector_store_type: VectorStoreType = VectorStoreType.FAISS
    vector_store_path: str = "./data/embeddings/faiss_index"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # 문서 처리 설정
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_context_length: int = 4000
    
    # 검색 설정
    search_k: int = 5
    similarity_threshold: float = 0.7
    
    # LLM 설정
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1000
    
    # Google AI 설정
    google_api_key: Optional[str] = None
    
    # Langfuse 설정
    langfuse_enabled: bool = True
    langfuse_secret_key: Optional[str] = None
    langfuse_public_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_debug: bool = False
    
    # 로컬 모델 설정
    local_model_path: Optional[str] = None
    local_model_device: str = "cpu"
    
    # 성능 설정
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1시간
    enable_async: bool = True
    
    @classmethod
    def from_env(cls) -> 'LangChainConfig':
        """환경 변수에서 설정 로드"""
        config = cls()
        
        # 벡터 저장소 설정
        vector_store_type = os.getenv("VECTOR_STORE_TYPE", "faiss")
        if vector_store_type == "faiss":
            config.vector_store_type = VectorStoreType.FAISS
        elif vector_store_type == "chroma":
            config.vector_store_type = VectorStoreType.CHROMA
        elif vector_store_type == "pinecone":
            config.vector_store_type = VectorStoreType.PINECONE
        
        config.vector_store_path = os.getenv("VECTOR_STORE_PATH", config.vector_store_path)
        config.embedding_model = os.getenv("EMBEDDING_MODEL", config.embedding_model)
        
        # 문서 처리 설정
        config.chunk_size = int(os.getenv("CHUNK_SIZE", config.chunk_size))
        config.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", config.chunk_overlap))
        config.max_context_length = int(os.getenv("MAX_CONTEXT_LENGTH", config.max_context_length))
        
        # 검색 설정
        config.search_k = int(os.getenv("SEARCH_K", config.search_k))
        config.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", config.similarity_threshold))
        
        # LLM 설정
        llm_provider = os.getenv("LLM_PROVIDER", "openai")
        if llm_provider == "openai":
            config.llm_provider = LLMProvider.OPENAI
        elif llm_provider == "local":
            config.llm_provider = LLMProvider.LOCAL
        elif llm_provider == "anthropic":
            config.llm_provider = LLMProvider.ANTHROPIC
        elif llm_provider == "google":
            config.llm_provider = LLMProvider.GOOGLE
        
        config.llm_model = os.getenv("LLM_MODEL", config.llm_model)
        config.llm_temperature = float(os.getenv("LLM_TEMPERATURE", config.llm_temperature))
        config.llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", config.llm_max_tokens))
        
        # Google AI 설정
        config.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Langfuse 설정
        config.langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "true").lower() == "true"
        config.langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        config.langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        config.langfuse_host = os.getenv("LANGFUSE_HOST", config.langfuse_host)
        config.langfuse_debug = os.getenv("LANGFUSE_DEBUG", "false").lower() == "true"
        
        # 로컬 모델 설정
        config.local_model_path = os.getenv("LOCAL_MODEL_PATH")
        config.local_model_device = os.getenv("LOCAL_MODEL_DEVICE", config.local_model_device)
        
        # 성능 설정
        config.enable_caching = os.getenv("ENABLE_CACHING", "true").lower() == "true"
        config.cache_ttl = int(os.getenv("CACHE_TTL", config.cache_ttl))
        config.enable_async = os.getenv("ENABLE_ASYNC", "true").lower() == "true"
        
        logger.info(f"LangChain configuration loaded: {config}")
        return config
    
    def validate(self) -> List[str]:
        """설정 유효성 검사"""
        errors = []
        
        # 필수 설정 검사
        if self.langfuse_enabled:
            if not self.langfuse_secret_key:
                errors.append("LANGFUSE_SECRET_KEY is required when Langfuse is enabled")
            if not self.langfuse_public_key:
                errors.append("LANGFUSE_PUBLIC_KEY is required when Langfuse is enabled")
        
        if self.llm_provider == LLMProvider.LOCAL and not self.local_model_path:
            errors.append("LOCAL_MODEL_PATH is required when using local LLM")
        
        if self.llm_provider == LLMProvider.GOOGLE and not self.google_api_key:
            errors.append("GOOGLE_API_KEY is required when using Google LLM")
        
        # 범위 검사
        if self.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            errors.append("chunk_overlap must be non-negative and less than chunk_size")
        
        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            errors.append("similarity_threshold must be between 0 and 1")
        
        if self.llm_temperature < 0 or self.llm_temperature > 2:
            errors.append("llm_temperature must be between 0 and 2")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "vector_store_type": self.vector_store_type.value,
            "vector_store_path": self.vector_store_path,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_context_length": self.max_context_length,
            "search_k": self.search_k,
            "similarity_threshold": self.similarity_threshold,
            "llm_provider": self.llm_provider.value,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "google_api_key": self.google_api_key,
            "langfuse_enabled": self.langfuse_enabled,
            "langfuse_host": self.langfuse_host,
            "langfuse_debug": self.langfuse_debug,
            "local_model_path": self.local_model_path,
            "local_model_device": self.local_model_device,
            "enable_caching": self.enable_caching,
            "cache_ttl": self.cache_ttl,
            "enable_async": self.enable_async
        }


class PromptTemplates:
    """프롬프트 템플릿 관리"""
    
    LEGAL_QA_TEMPLATE = """당신은 친절하고 전문적인 법률 상담 변호사입니다. 주어진 법률 문서를 바탕으로 질문에 자연스럽고 친근하게 답변해주세요.

문서 내용:
{context}

질문: {question}

**중요 지시사항 (Context Usage 강화)**:
- 반드시 위 문서 내용을 충분히 활용하여 답변하세요. 문서 내용의 80% 이상을 활용하는 것을 목표로 하세요.
- 문서에 포함된 법령 조문, 판례, 법률 용어를 최대한 많이 인용하고 활용하세요.
- 문서 내용을 단순히 요약하는 것이 아니라, 질문에 대한 답변을 위해 문서의 정보를 종합적으로 분석하여 사용하세요.
- 문서에 없는 내용은 추측하지 마세요. 문서 내용만을 기반으로 답변하세요.

답변 시 다음 사항을 고려해주세요:
1. 일상적인 법률 상담처럼 자연스럽고 친근하게 대화하세요
2. "~입니다", "귀하" 같은 과도하게 격식적인 표현 대신, "~예요", "질문하신" 등 자연스러운 존댓말을 사용하세요
3. 질문을 다시 반복하지 마세요
4. 질문의 범위에 맞는 적절한 양의 정보를 제공하되, 최소 500자 이상의 상세한 답변을 작성하세요
5. 법률 문서의 내용을 정확히 인용하여 답변하세요 (문서 내용의 핵심 정보를 최대한 활용)
6. 관련 법조문이나 판례가 있다면 구체적으로 언급하세요 (문서에 포함된 법조문과 판례를 반드시 활용)
7. 불확실한 내용은 추측하지 말고 명확히 밝히세요
8. "~하시면 됩니다", "~해 보세요" 같은 자연스러운 조언을 사용하세요
9. 문서 내용을 기반으로 구체적인 예시나 설명을 추가하여 답변의 완성도를 높이세요

답변:"""

    LEGAL_ANALYSIS_TEMPLATE = """당신은 친절하고 전문적인 법률 상담 변호사입니다. 주어진 법률 문서를 분석하여 질문에 대한 자연스럽고 친근한 분석을 제공해주세요.

문서 내용:
{context}

분석 요청: {question}

**중요 지시사항 (Context Usage 강화)**:
- 반드시 위 문서 내용을 충분히 활용하여 분석하세요. 문서 내용의 80% 이상을 활용하는 것을 목표로 하세요.
- 문서에 포함된 법령 조문, 판례, 법률 용어를 최대한 많이 인용하고 활용하세요.
- 문서 내용을 단순히 요약하는 것이 아니라, 질문에 대한 분석을 위해 문서의 정보를 종합적으로 분석하여 사용하세요.
- 문서에 없는 내용은 추측하지 마세요. 문서 내용만을 기반으로 분석하세요.

분석 시 다음 원칙을 따라 답변해주세요:
1. 일상적인 법률 상담처럼 자연스럽고 친근하게 대화하세요
2. "~입니다", "귀하" 같은 과도하게 격식적인 표현 대신, "~예요", "질문하신" 등 자연스러운 존댓말을 사용하세요
3. 질문을 다시 반복하지 마세요
4. 질문의 범위에 맞는 적절한 양의 정보를 제공하되, 최소 800자 이상의 상세한 분석을 작성하세요
5. 불필요한 형식(제목, 번호 매기기)은 최소화하세요
6. 핵심 쟁점을 파악하고 법적 근거를 제시하세요 (문서에 포함된 법적 근거를 최대한 활용)
7. 실무적 관점에서 실행 가능한 조언을 제공하세요
8. 문서 내용을 기반으로 구체적인 예시나 설명을 추가하여 분석의 완성도를 높이세요

분석:"""

    CONTRACT_REVIEW_TEMPLATE = """당신은 친절하고 전문적인 법률 상담 변호사입니다. 주어진 계약서 내용을 검토하여 자연스럽고 친근하게 잠재적 위험요소와 개선사항을 제시해주세요.

계약서 내용:
{context}

검토 요청: {question}

검토 시 다음 원칙을 따라 답변해주세요:
1. 일상적인 법률 상담처럼 자연스럽고 친근하게 대화하세요
2. "~입니다", "귀하" 같은 과도하게 격식적인 표현 대신, "~예요", "질문하신" 등 자연스러운 존댓말을 사용하세요
3. 질문을 다시 반복하지 마세요
4. 질문의 범위에 맞는 적절한 양의 정보만 제공하세요
5. 불필요한 형식(제목, 번호 매기기)은 최소화하세요
6. 계약 조건의 명확성과 당사자 간 권리와 의무의 균형을 확인하세요
7. 잠재적 법적 리스크와 개선 권고사항을 제시하세요
8. "~하시면 됩니다", "~해 보세요" 같은 자연스러운 조언을 사용하세요

검토 결과:"""

    @classmethod
    def get_template(cls, template_type: str) -> str:
        """템플릿 타입에 따른 프롬프트 반환"""
        templates = {
            "legal_qa": cls.LEGAL_QA_TEMPLATE,
            "legal_analysis": cls.LEGAL_ANALYSIS_TEMPLATE,
            "contract_review": cls.CONTRACT_REVIEW_TEMPLATE
        }
        return templates.get(template_type, cls.LEGAL_QA_TEMPLATE)


# 전역 설정 인스턴스
langchain_config = LangChainConfig.from_env()
