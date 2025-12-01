#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cloudflare R2 클라이언트

Cloudflare R2 (S3 호환) 스토리지와 상호작용하는 클라이언트 모듈입니다.
"""

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, BotoCoreError
from pathlib import Path
import os
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class R2Client:
    """Cloudflare R2 클라이언트"""
    
    def __init__(self):
        """
        R2 클라이언트 초기화
        
        환경 변수에서 설정을 읽어옵니다:
        - CLOUDFLARE_R2_ENDPOINT: R2 엔드포인트 URL
        - CLOUDFLARE_R2_ACCESS_KEY_ID: 액세스 키 ID
        - CLOUDFLARE_R2_SECRET_ACCESS_KEY: 시크릿 액세스 키
        - CLOUDFLARE_R2_BUCKET_NAME: 버킷 이름
        """
        self.endpoint = os.getenv("CLOUDFLARE_R2_ENDPOINT")
        self.access_key = os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID")
        self.secret_key = os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY")
        self.bucket_name = os.getenv("CLOUDFLARE_R2_BUCKET_NAME")
        
        # 환경 변수 검증
        if not all([self.endpoint, self.access_key, self.secret_key, self.bucket_name]):
            raise ValueError(
                "R2 설정이 완료되지 않았습니다. 다음 환경 변수를 설정해주세요:\n"
                "- CLOUDFLARE_R2_ENDPOINT\n"
                "- CLOUDFLARE_R2_ACCESS_KEY_ID\n"
                "- CLOUDFLARE_R2_SECRET_ACCESS_KEY\n"
                "- CLOUDFLARE_R2_BUCKET_NAME"
            )
        
        # S3 호환 클라이언트 생성
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4')
        )
        
        logger.info(f"R2 클라이언트 초기화 완료: {self.bucket_name}")
    
    def upload_file(self, local_path: Path, object_key: str, 
                   metadata: Optional[Dict[str, str]] = None) -> bool:
        """
        파일을 R2에 업로드
        
        Args:
            local_path: 로컬 파일 경로
            object_key: R2 객체 키 (경로)
            metadata: 메타데이터 (선택사항)
        
        Returns:
            bool: 업로드 성공 여부
        """
        try:
            if not local_path.exists():
                logger.error(f"파일이 존재하지 않습니다: {local_path}")
                return False
            
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            self.s3_client.upload_file(
                str(local_path),
                self.bucket_name,
                object_key,
                ExtraArgs=extra_args if extra_args else None
            )
            
            logger.info(f"파일 업로드 완료: {object_key}")
            return True
            
        except FileNotFoundError:
            logger.error(f"파일을 찾을 수 없습니다: {local_path}")
            return False
        except ClientError as e:
            logger.error(f"R2 업로드 실패: {e}")
            return False
        except Exception as e:
            logger.error(f"예상치 못한 오류: {e}")
            return False
    
    def download_file(self, object_key: str, local_path: Path) -> bool:
        """
        R2에서 파일 다운로드
        
        Args:
            object_key: R2 객체 키 (경로)
            local_path: 로컬 저장 경로
        
        Returns:
            bool: 다운로드 성공 여부
        """
        try:
            # 디렉토리 생성
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.s3_client.download_file(
                self.bucket_name,
                object_key,
                str(local_path)
            )
            
            logger.info(f"파일 다운로드 완료: {object_key} -> {local_path}")
            return True
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404':
                logger.error(f"파일을 찾을 수 없습니다: {object_key}")
            else:
                logger.error(f"R2 다운로드 실패: {e}")
            return False
        except Exception as e:
            logger.error(f"예상치 못한 오류: {e}")
            return False
    
    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> List[Dict[str, Any]]:
        """
        R2 버킷의 객체 목록 조회
        
        Args:
            prefix: 객체 키 접두사 (필터링)
            max_keys: 최대 반환 개수
        
        Returns:
            List[Dict]: 객체 정보 리스트
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            objects = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    objects.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'],
                        'etag': obj['ETag']
                    })
            
            logger.info(f"객체 목록 조회 완료: {len(objects)}개")
            return objects
            
        except ClientError as e:
            logger.error(f"객체 목록 조회 실패: {e}")
            return []
        except Exception as e:
            logger.error(f"예상치 못한 오류: {e}")
            return []
    
    def object_exists(self, object_key: str) -> bool:
        """
        객체가 존재하는지 확인
        
        Args:
            object_key: R2 객체 키
        
        Returns:
            bool: 객체 존재 여부
        """
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404':
                return False
            logger.error(f"객체 확인 실패: {e}")
            return False
        except Exception as e:
            logger.error(f"예상치 못한 오류: {e}")
            return False
    
    def delete_object(self, object_key: str) -> bool:
        """
        R2에서 객체 삭제
        
        Args:
            object_key: R2 객체 키
        
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            logger.info(f"객체 삭제 완료: {object_key}")
            return True
        except ClientError as e:
            logger.error(f"객체 삭제 실패: {e}")
            return False
        except Exception as e:
            logger.error(f"예상치 못한 오류: {e}")
            return False
    
    def get_object_metadata(self, object_key: str) -> Optional[Dict[str, Any]]:
        """
        객체 메타데이터 조회
        
        Args:
            object_key: R2 객체 키
        
        Returns:
            Dict: 메타데이터 또는 None
        """
        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            
            return {
                'content_length': response.get('ContentLength'),
                'content_type': response.get('ContentType'),
                'last_modified': response.get('LastModified'),
                'etag': response.get('ETag'),
                'metadata': response.get('Metadata', {})
            }
        except ClientError as e:
            logger.error(f"메타데이터 조회 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"예상치 못한 오류: {e}")
            return None


if __name__ == "__main__":
    """테스트용"""
    import sys
    from scripts.utils.env_check import check_venv
    
    if not check_venv():
        print("❌ scripts/.venv 가상환경을 활성화해주세요.")
        sys.exit(1)
    
    # R2 클라이언트 테스트
    try:
        client = R2Client()
        print("✅ R2 클라이언트 초기화 성공")
        
        # 버킷 연결 테스트
        objects = client.list_objects(max_keys=1)
        print(f"✅ R2 연결 성공 (버킷: {client.bucket_name})")
        
    except Exception as e:
        print(f"❌ R2 클라이언트 초기화 실패: {e}")
        sys.exit(1)

