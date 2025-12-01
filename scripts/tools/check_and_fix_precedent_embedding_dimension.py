"""
precedent_chunks 테이블의 실제 임베딩 차원 확인 및 embedding_versions 테이블 정보 수정

이 스크립트는:
1. precedent_chunks 테이블의 실제 임베딩 벡터 차원을 확인
2. embedding_version별 차원 분포 확인
3. embedding_versions 테이블의 모델 정보를 실제 차원에 맞게 수정
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    root_env = _PROJECT_ROOT / ".env"
    if root_env.exists():
        load_dotenv(dotenv_path=str(root_env), override=True)
    scripts_env = _PROJECT_ROOT / "scripts" / ".env"
    if scripts_env.exists():
        load_dotenv(dotenv_path=str(scripts_env), override=True)
except ImportError:
    pass

from scripts.ingest.open_law.utils import build_database_url
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

def get_db_engine(db_url: str):
    return create_engine(
        db_url,
        poolclass=QueuePool,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        echo=False
    )

def check_embedding_dimensions(engine) -> Dict[int, Dict[str, Any]]:
    """precedent_chunks 테이블의 실제 임베딩 차원 확인"""
    print("="*80)
    print("1. precedent_chunks 테이블 임베딩 차원 확인")
    print("="*80)
    
    with engine.connect() as conn:
        # embedding_version별로 샘플 벡터를 가져와 차원 확인
        # 각 버전별로 하나의 샘플 벡터를 가져와 차원 확인
        result = conn.execute(text("""
            SELECT DISTINCT embedding_version
            FROM precedent_chunks
            WHERE embedding_vector IS NOT NULL
            ORDER BY embedding_version
        """))
        
        versions = [row[0] for row in result]
        
        if not versions:
            print("  ⚠️ 임베딩 벡터가 없습니다.")
            return {}
        
        version_dimensions = {}
        dimensions_count = {}
        
        for version in versions:
            # 각 버전별로 샘플 벡터 하나를 가져와 차원 확인
            sample_result = conn.execute(text("""
                SELECT embedding_vector::text
                FROM precedent_chunks
                WHERE embedding_vector IS NOT NULL AND embedding_version = :version
                LIMIT 1
            """), {'version': version})
            
            sample_row = sample_result.fetchone()
            if sample_row:
                # 벡터 텍스트를 파싱하여 차원 확인
                # pgvector 텍스트 형식: "[0.1,0.2,0.3,...]"
                vector_text = sample_row[0]
                # 대괄호 제거 후 쉼표로 분리
                vector_values = vector_text.strip('[]').split(',')
                dimension = len(vector_values)
                
                # 해당 버전의 전체 개수 확인
                count_result = conn.execute(text("""
                    SELECT COUNT(*)
                    FROM precedent_chunks
                    WHERE embedding_vector IS NOT NULL AND embedding_version = :version
                """), {'version': version})
                count = count_result.fetchone()[0]
                
                version_dimensions[version] = {dimension: count}
                dimensions_count[dimension] = dimensions_count.get(dimension, 0) + count
                
                print(f"  version {version}: 차원 {dimension} ({count:,}개 청크)")
        
        print("\n  전체 차원 분포:")
        for dim, count in sorted(dimensions_count.items()):
            print(f"    차원 {dim}: {count:,}개 청크")
        
        return version_dimensions

def check_embedding_versions(engine) -> List[Dict[str, Any]]:
    """embedding_versions 테이블 정보 확인"""
    print("\n" + "="*80)
    print("2. embedding_versions 테이블 정보 확인")
    print("="*80)
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT 
                id,
                version,
                data_type,
                model_name,
                dim,
                is_active,
                chunking_strategy,
                description
            FROM embedding_versions
            WHERE data_type = 'precedents'
            ORDER BY version
        """))
        
        versions = []
        for row in result:
            version_info = {
                'id': row[0],
                'version': row[1],
                'data_type': row[2],
                'model_name': row[3],
                'dim': row[4],
                'is_active': row[5],
                'chunking_strategy': row[6],
                'description': row[7]
            }
            versions.append(version_info)
            
            status = "✅ 활성" if version_info['is_active'] else "❌ 비활성"
            print(f"  {status} | ID={version_info['id']} | version={version_info['version']} | "
                  f"model={version_info['model_name']} | dim={version_info['dim']}")
        
        return versions

def update_embedding_version(engine, version_id: int, model_name: str, dim: int, description: Optional[str] = None):
    """embedding_versions 테이블 정보 수정"""
    print("\n" + "="*80)
    print(f"3. embedding_versions 테이블 정보 수정 (ID={version_id})")
    print("="*80)
    
    with engine.begin() as conn:
        # 현재 정보 확인
        result = conn.execute(text("""
            SELECT id, version, data_type, model_name, dim
            FROM embedding_versions
            WHERE id = :version_id
        """), {'version_id': version_id})
        
        current = result.fetchone()
        if not current:
            print(f"  ❌ 버전 ID {version_id}를 찾을 수 없습니다.")
            return False
        
        print(f"  현재 정보:")
        print(f"    model_name: {current[3]}")
        print(f"    dim: {current[4]}")
        print(f"  수정할 정보:")
        print(f"    model_name: {model_name}")
        print(f"    dim: {dim}")
        
        # 정보 수정
        update_query = text("""
            UPDATE embedding_versions
            SET model_name = :model_name,
                dim = :dim
        """)
        
        params = {
            'version_id': version_id,
            'model_name': model_name,
            'dim': dim
        }
        
        if description:
            update_query = text("""
                UPDATE embedding_versions
                SET model_name = :model_name,
                    dim = :dim,
                    description = :description
            """)
            params['description'] = description
        
        update_query = text("""
            UPDATE embedding_versions
            SET model_name = :model_name,
                dim = :dim
            WHERE id = :version_id
        """)
        
        conn.execute(update_query, params)
        
        print(f"  ✅ 버전 ID {version_id} 정보 수정 완료")
        return True

def determine_correct_model(dimension: int) -> tuple[str, int]:
    """차원에 따라 올바른 모델명 결정"""
    # 일반적인 모델 차원 매핑
    model_mapping = {
        384: ("jhgan/ko-sroberta-multitask", 384),
        768: ("woong0322/ko-legal-sbert-finetuned", 768),
    }
    
    if dimension in model_mapping:
        return model_mapping[dimension]
    
    # 차원만 알 수 있는 경우
    return (f"unknown_model_dim_{dimension}", dimension)

def check_model_name_mismatch(engine, version_dimensions: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """모델명과 차원의 불일치 확인"""
    print("\n" + "="*80)
    print("3-1. 모델명과 차원 불일치 확인")
    print("="*80)
    
    # 모델명-차원 매핑
    model_dim_mapping = {
        "jhgan/ko-sroberta-multitask": 384,
        "woong0322/ko-legal-sbert-finetuned": 768,
    }
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT 
                id,
                version,
                data_type,
                model_name,
                dim,
                is_active
            FROM embedding_versions
            WHERE data_type = 'precedents'
            ORDER BY version
        """))
        
        mismatches = []
        
        for row in result:
            version_id = row[0]
            version_num = row[1]
            model_name = row[3]
            registered_dim = row[4]
            
            # 해당 버전의 실제 차원 확인
            if version_num in version_dimensions:
                actual_dims = version_dimensions[version_num]
                actual_dim = max(actual_dims.keys(), key=lambda k: actual_dims[k])
                
                # 모델명에 따른 예상 차원 확인
                expected_dim = model_dim_mapping.get(model_name)
                
                print(f"\n  버전 {version_num} (ID={version_id}):")
                print(f"    등록된 모델: {model_name}")
                print(f"    등록된 차원: {registered_dim}")
                print(f"    실제 차원: {actual_dim}")
                if expected_dim:
                    print(f"    모델 예상 차원: {expected_dim}")
                
                # 차원 불일치 확인
                if registered_dim != actual_dim:
                    print(f"    ⚠️ 차원 불일치 발견!")
                    correct_model, correct_dim = determine_correct_model(actual_dim)
                    mismatches.append({
                        'version_id': version_id,
                        'version_num': version_num,
                        'current_model': model_name,
                        'current_dim': registered_dim,
                        'correct_model': correct_model,
                        'correct_dim': actual_dim
                    })
                # 모델명과 차원 불일치 확인
                elif expected_dim and expected_dim != actual_dim:
                    print(f"    ⚠️ 모델명과 차원 불일치 발견!")
                    correct_model, correct_dim = determine_correct_model(actual_dim)
                    mismatches.append({
                        'version_id': version_id,
                        'version_num': version_num,
                        'current_model': model_name,
                        'current_dim': registered_dim,
                        'correct_model': correct_model,
                        'correct_dim': actual_dim
                    })
                else:
                    print(f"    ✅ 모델명과 차원 일치")
        
        return mismatches

def main():
    db_url = build_database_url() or os.getenv('DATABASE_URL')
    if not db_url:
        print("❌ 데이터베이스 URL을 찾을 수 없습니다.")
        return
    
    engine = get_db_engine(db_url)
    
    # 1. 실제 임베딩 차원 확인
    version_dimensions = check_embedding_dimensions(engine)
    
    if not version_dimensions:
        print("\n❌ 임베딩 벡터가 없어 수정할 수 없습니다.")
        return
    
    # 2. embedding_versions 테이블 정보 확인
    versions = check_embedding_versions(engine)
    
    if not versions:
        print("\n❌ precedents 타입의 embedding_versions가 없습니다.")
        return
    
    # 3. 차원 불일치 확인 및 수정
    print("\n" + "="*80)
    print("3. 차원 불일치 확인 및 수정")
    print("="*80)
    
    needs_update = []
    
    for version_info in versions:
        version_id = version_info['id']
        version_num = version_info['version']
        registered_dim = version_info['dim']
        registered_model = version_info['model_name']
        
        # 해당 버전의 실제 차원 확인
        if version_num in version_dimensions:
            actual_dims = version_dimensions[version_num]
            # 가장 많은 차원을 실제 차원으로 사용
            actual_dim = max(actual_dims.keys(), key=lambda k: actual_dims[k])
            actual_count = actual_dims[actual_dim]
            
            print(f"\n  버전 {version_num} (ID={version_id}):")
            print(f"    등록된 정보: model={registered_model}, dim={registered_dim}")
            print(f"    실제 차원: {actual_dim} (청크 수: {actual_count:,})")
            
            # 모델명-차원 매핑 확인
            model_dim_mapping = {
                "jhgan/ko-sroberta-multitask": 384,
                "woong0322/ko-legal-sbert-finetuned": 768,
            }
            expected_dim = model_dim_mapping.get(registered_model)
            
            if registered_dim != actual_dim:
                print(f"    ⚠️ 차원 불일치 발견!")
                correct_model, correct_dim = determine_correct_model(actual_dim)
                needs_update.append({
                    'version_id': version_id,
                    'version_num': version_num,
                    'current_model': registered_model,
                    'current_dim': registered_dim,
                    'correct_model': correct_model,
                    'correct_dim': actual_dim
                })
            elif expected_dim and expected_dim != actual_dim:
                print(f"    ⚠️ 모델명과 차원 불일치 발견! (모델 {registered_model}는 {expected_dim}차원이어야 하는데 실제는 {actual_dim}차원)")
                correct_model, correct_dim = determine_correct_model(actual_dim)
                needs_update.append({
                    'version_id': version_id,
                    'version_num': version_num,
                    'current_model': registered_model,
                    'current_dim': registered_dim,
                    'correct_model': correct_model,
                    'correct_dim': actual_dim
                })
            else:
                print(f"    ✅ 차원 일치")
        else:
            print(f"\n  버전 {version_num} (ID={version_id}):")
            print(f"    ⚠️ 해당 버전의 임베딩 벡터가 없습니다.")
    
    # 4. 수정 실행
    if needs_update:
        print("\n" + "="*80)
        print("4. embedding_versions 테이블 정보 수정")
        print("="*80)
        
        for update_info in needs_update:
            print(f"\n  버전 {update_info['version_num']} (ID={update_info['version_id']}) 수정:")
            print(f"    모델: {update_info['current_model']} → {update_info['correct_model']}")
            print(f"    차원: {update_info['current_dim']} → {update_info['correct_dim']}")
            
            # 사용자 확인 (자동 실행)
            update_embedding_version(
                engine,
                update_info['version_id'],
                update_info['correct_model'],
                update_info['correct_dim'],
                description=f"수정됨: 실제 차원 {update_info['correct_dim']}에 맞게 업데이트"
            )
        
        print("\n✅ 모든 수정이 완료되었습니다.")
    else:
        print("\n✅ 차원 불일치가 없습니다. 수정할 필요가 없습니다.")
    
    # 5. 최종 확인
    print("\n" + "="*80)
    print("5. 최종 확인")
    print("="*80)
    check_embedding_versions(engine)

if __name__ == '__main__':
    main()

