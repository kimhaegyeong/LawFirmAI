"""
원본 문서 조회 API 라우터
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from api.services.original_document_service import OriginalDocumentService

router = APIRouter(prefix="/api/documents", tags=["documents"])


def get_document_service() -> OriginalDocumentService:
    """의존성 주입: OriginalDocumentService 인스턴스 반환"""
    return OriginalDocumentService()


@router.get("/original/{source_type}/{source_id}")
async def get_original_document(
    source_type: str,
    source_id: int,
    service: OriginalDocumentService = Depends(get_document_service)
):
    """
    원본 문서 조회
    
    Args:
        source_type: 문서 타입 (statute_article, case_paragraph, etc.)
        source_id: 문서 ID
        service: OriginalDocumentService 인스턴스
    
    Returns:
        원본 문서 정보
    """
    if source_type not in ["statute_article", "case_paragraph", "decision_paragraph", "interpretation_paragraph"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid source_type: {source_type}"
        )
    
    document = service.get_original_document(source_type, source_id)
    
    if document is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {source_type}/{source_id}"
        )
    
    return document


@router.get("/chunks/{chunk_group_id}")
async def get_chunks_by_group(
    chunk_group_id: str,
    service: OriginalDocumentService = Depends(get_document_service)
):
    """
    청크 그룹 ID로 관련 청크 조회
    
    Args:
        chunk_group_id: 청크 그룹 ID
        service: OriginalDocumentService 인스턴스
    
    Returns:
        청크 리스트
    """
    chunks = service.get_chunks_by_group(chunk_group_id)
    
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=f"Chunks not found for group: {chunk_group_id}"
        )
    
    return {
        "chunk_group_id": chunk_group_id,
        "chunks": chunks,
        "count": len(chunks)
    }


@router.get("/chunk/{chunk_id}")
async def get_chunk_info(
    chunk_id: int,
    service: OriginalDocumentService = Depends(get_document_service)
):
    """
    청크 정보 조회
    
    Args:
        chunk_id: 청크 ID
        service: OriginalDocumentService 인스턴스
    
    Returns:
        청크 정보
    """
    chunk_info = service.get_chunk_info(chunk_id)
    
    if chunk_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Chunk not found: {chunk_id}"
        )
    
    return chunk_info

