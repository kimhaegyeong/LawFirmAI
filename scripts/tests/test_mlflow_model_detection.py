#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MLflow 모델 감지 테스트 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
from lawfirm_langgraph.core.utils.config import Config

def test_mlflow_model_detection():
    """MLflow 모델 감지 테스트"""
    print("=" * 80)
    print("MLflow 모델 감지 테스트")
    print("=" * 80)
    
    # Config에서 설정 확인
    config = Config()
    db_path = config.database_path
    use_mlflow_index = getattr(config, 'use_mlflow_index', True)
    mlflow_run_id = getattr(config, 'mlflow_run_id', None)
    
    print(f"\n📋 설정 정보:")
    print(f"   - Database Path: {db_path}")
    print(f"   - Use MLflow Index: {use_mlflow_index}")
    print(f"   - MLflow Run ID: {mlflow_run_id}")
    
    # SemanticSearchEngineV2 초기화
    print(f"\n🔧 SemanticSearchEngineV2 초기화 중...")
    try:
        engine = SemanticSearchEngineV2(
            db_path=db_path,
            model_name=None,  # None으로 설정하여 MLflow에서 자동 감지하도록 함
            use_mlflow_index=use_mlflow_index,
            mlflow_run_id=mlflow_run_id
        )
        
        print(f"\n✅ 초기화 완료")
        print(f"   - 사용된 모델: {engine.model_name}")
        print(f"   - MLflow Manager: {engine.mlflow_manager is not None}")
        print(f"   - Use MLflow Index: {engine.use_mlflow_index}")
        
        if engine.mlflow_manager:
            print(f"\n📊 MLflow Manager 정보:")
            try:
                production_run = engine.mlflow_manager.get_production_run()
                print(f"   - Production Run ID: {production_run}")
                
                if production_run:
                    import mlflow
                    try:
                        version_info = engine.mlflow_manager.load_version_info_from_local(production_run)
                        if version_info is None:
                            version_info = mlflow.artifacts.load_dict(f"runs:/{production_run}/version_info.json")
                        
                        if version_info:
                            embedding_config = version_info.get('embedding_config', {})
                            mlflow_model = embedding_config.get('model')
                            mlflow_dimension = embedding_config.get('dimension')
                            
                            print(f"   - MLflow Model: {mlflow_model}")
                            print(f"   - MLflow Dimension: {mlflow_dimension}")
                            
                            if mlflow_model:
                                if mlflow_model == engine.model_name:
                                    print(f"\n✅ 성공: MLflow에서 모델을 감지하고 사용 중입니다!")
                                    print(f"   - MLflow Model: {mlflow_model}")
                                    print(f"   - Engine Model: {engine.model_name}")
                                else:
                                    print(f"\n⚠️  경고: MLflow 모델과 Engine 모델이 일치하지 않습니다!")
                                    print(f"   - MLflow Model: {mlflow_model}")
                                    print(f"   - Engine Model: {engine.model_name}")
                            else:
                                print(f"\n⚠️  경고: MLflow version_info에 모델 정보가 없습니다.")
                        else:
                            print(f"\n⚠️  경고: version_info.json을 로드할 수 없습니다.")
                    except Exception as e:
                        print(f"\n❌ 에러: version_info.json 로드 실패: {e}")
                else:
                    print(f"\n⚠️  경고: Production Run ID를 찾을 수 없습니다.")
            except Exception as e:
                print(f"\n❌ 에러: MLflow Manager 정보 조회 실패: {e}")
        else:
            print(f"\n⚠️  경고: MLflow Manager가 초기화되지 않았습니다.")
        
        return engine.model_name
        
    except Exception as e:
        print(f"\n❌ 에러: 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_mlflow_model_detection()


