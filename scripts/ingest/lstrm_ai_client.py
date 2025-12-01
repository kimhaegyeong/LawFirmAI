# -*- coding: utf-8 -*-
"""
lstrmAI API 클라이언트
국가법령정보 공동활용 LAW OPEN DATA - 법령용어 검색 API 클라이언트
"""

from typing import Any, Dict, Optional

from scripts.ingest.base.api_client import BaseAPIClient


class LstrmAIClient(BaseAPIClient):
    """lstrmAI API 클라이언트"""
    
    @property
    def target(self) -> str:
        """API target 이름"""
        return 'lstrmAI'
    
    def build_search_params(
        self,
        query: str = "",
        page: int = 1,
        display: int = 100,
        homonym_yn: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """검색 파라미터 생성"""
        params = {
            'OC': self.oc,
            'target': self.target,
            'type': 'JSON',
            'query': query,
            'page': page,
            'display': display
        }
        if homonym_yn:
            params['homonymYn'] = homonym_yn
        return params
    
    def search_terms(
        self,
        query: str = "",
        page: int = 1,
        display: int = 100,
        homonym_yn: Optional[str] = None
    ) -> Dict[str, Any]:
        """법령용어 검색"""
        params = self.build_search_params(query, page, display, homonym_yn=homonym_yn)
        return self._make_request(params)

