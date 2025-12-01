# -*- coding: utf-8 -*-
"""
dlytrm (일상용어) API 클라이언트
국가법령정보 공동활용 LAW OPEN DATA - 일상용어 검색 API 클라이언트
"""

from typing import Any, Dict

from scripts.ingest.base.api_client import BaseAPIClient


class DlytrmClient(BaseAPIClient):
    """dlytrm (일상용어) API 클라이언트"""
    
    @property
    def target(self) -> str:
        """API target 이름"""
        return 'dlytrm'
    
    def build_search_params(
        self,
        query: str = "",
        page: int = 1,
        display: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """검색 파라미터 생성"""
        return {
            'OC': self.oc,
            'target': self.target,
            'type': 'JSON',
            'query': query,
            'page': page,
            'display': display
        }

