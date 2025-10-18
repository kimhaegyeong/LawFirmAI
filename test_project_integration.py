import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ["USE_LANGGRAPH"] = "true"
os.environ["LANGGRAPH_ENABLED"] = "true"

print("="*50)
print("LawFirmAI LangGraph Integration Test")
print("="*50)

# Test 1: Import project modules
print("\n[1/5] Testing project module imports...")
try:
    from source.utils.langgraph_config import LangGraphConfig
    print("  OK: LangGraphConfig imported")
    
    config = LangGraphConfig.from_env()
    print(f"  OK: Config loaded (enabled={config.langgraph_enabled})")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# Test 2: State definitions
print("\n[2/5] Testing state definitions...")
try:
    from source.services.langgraph.state_definitions import create_initial_legal_state
    
    state = create_initial_legal_state("test query", "test-session")
    print(f"  OK: Initial state created")
    print(f"      Query: {state['query']}")
    print(f"      Session: {state['session_id']}")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# Test 3: Checkpoint manager
print("\n[3/5] Testing checkpoint manager...")
try:
    import tempfile
    from source.services.langgraph.checkpoint_manager import CheckpointManager
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    manager = CheckpointManager(db_path)
    print("  OK: CheckpointManager initialized")
    
    info = manager.get_database_info()
    print(f"      DB path: {info['database_path']}")
    print(f"      LangGraph available: {info['langgraph_available']}")
    
    os.unlink(db_path)
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# Test 4: Workflow compilation
print("\n[4/5] Testing workflow compilation...")
try:
    from source.services.langgraph.legal_workflow import LegalQuestionWorkflow
    from unittest.mock import Mock, patch
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        config.checkpoint_db_path = f.name
    
    # Mock dependencies
    with patch('source.services.langgraph.legal_workflow.QuestionClassifier'):
        with patch('source.services.langgraph.legal_workflow.HybridSearchEngine'):
            with patch('source.services.langgraph.legal_workflow.OllamaClient'):
                workflow = LegalQuestionWorkflow(config)
                print("  OK: Workflow initialized")
                
                compiled = workflow.compile()
                if compiled:
                    print("  OK: Workflow compiled successfully")
                else:
                    print("  WARNING: Workflow compilation returned None")
    
    os.unlink(config.checkpoint_db_path)
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Workflow service
print("\n[5/5] Testing workflow service...")
try:
    from source.services.langgraph.workflow_service import LangGraphWorkflowService
    from unittest.mock import Mock, patch
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        config.checkpoint_db_path = f.name
    
    with patch('source.services.langgraph.workflow_service.LegalQuestionWorkflow'):
        with patch('source.services.langgraph.workflow_service.CheckpointManager'):
            service = LangGraphWorkflowService(config)
            print("  OK: WorkflowService initialized")
            
            status = service.get_service_status()
            print(f"      Service: {status['service_name']}")
            print(f"      Status: {status['status']}")
    
    os.unlink(config.checkpoint_db_path)
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*50)
print("SUCCESS: All integration tests passed!")
print("="*50)
print("\nNext steps:")
print("1. Start Ollama server: ollama serve")
print("2. Pull model: ollama pull qwen2.5:7b")
print("3. Run Gradio app: python gradio/app.py")

