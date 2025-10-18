#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple LangGraph Integration Test
"""

import os
import sys
import tempfile
from pathlib import Path

# Set environment variables
os.environ["USE_LANGGRAPH"] = "true"
os.environ["LANGGRAPH_ENABLED"] = "true"

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_langgraph_imports():
    """Test LangGraph package imports"""
    print("=== Testing LangGraph Imports ===")
    
    try:
        import langgraph
        print("OK: langgraph imported successfully")
        
        from langgraph.checkpoint.sqlite import SqliteSaver
        print("OK: SqliteSaver imported successfully")
        
        from langgraph.graph import StateGraph, END
        print("OK: StateGraph and END imported successfully")
        
        return True
    except Exception as e:
        print(f"ERROR: Import failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\n=== Testing Configuration Loading ===")
    
    try:
        # Test environment variables
        use_langgraph = os.getenv("USE_LANGGRAPH", "false").lower() == "true"
        print(f"OK: USE_LANGGRAPH: {use_langgraph}")
        
        langgraph_enabled = os.getenv("LANGGRAPH_ENABLED", "false").lower() == "true"
        print(f"OK: LANGGRAPH_ENABLED: {langgraph_enabled}")
        
        return True
    except Exception as e:
        print(f"ERROR: Configuration test failed: {e}")
        return False

def test_sqlite_checkpoint():
    """Test SQLite checkpoint functionality"""
    print("\n=== Testing SQLite Checkpoint ===")
    
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path = tmp_file.name
        
        # Initialize SqliteSaver
        saver = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
        print("✓ SqliteSaver initialized successfully")
        
        # Test basic operations
        thread_id = "test_thread"
        config = {"configurable": {"thread_id": thread_id}}
        
        # List checkpoints (should be empty initially)
        checkpoints = list(saver.list(config))
        print(f"✓ Listed checkpoints: {len(checkpoints)} found")
        
        # Clean up
        os.unlink(db_path)
        print("✓ Temporary database cleaned up")
        
        return True
    except Exception as e:
        print(f"✗ SQLite checkpoint test failed: {e}")
        return False

def test_state_graph():
    """Test StateGraph creation"""
    print("\n=== Testing StateGraph Creation ===")
    
    try:
        from langgraph.graph import StateGraph, END
        from typing import TypedDict
        
        # Define a simple state
        class TestState(TypedDict):
            message: str
            counter: int
        
        # Create a simple workflow
        workflow = StateGraph(TestState)
        
        def simple_node(state: TestState) -> TestState:
            state["counter"] = state.get("counter", 0) + 1
            return state
        
        # Add nodes and edges
        workflow.add_node("process", simple_node)
        workflow.set_entry_point("process")
        workflow.add_edge("process", END)
        
        # Compile the workflow
        app = workflow.compile()
        print("✓ StateGraph created and compiled successfully")
        
        # Test execution
        initial_state = {"message": "test", "counter": 0}
        result = app.invoke(initial_state)
        
        print(f"✓ Workflow executed successfully: counter = {result['counter']}")
        
        return True
    except Exception as e:
        print(f"✗ StateGraph test failed: {e}")
        return False

def test_ollama_integration():
    """Test Ollama integration"""
    print("\n=== Testing Ollama Integration ===")
    
    try:
        from langchain_community.llms import Ollama
        
        # Test Ollama client creation (without actual connection)
        ollama_llm = Ollama(model="qwen2.5:7b", base_url="http://localhost:11434")
        print("✓ Ollama client created successfully")
        
        # Note: We don't actually invoke it since Ollama server might not be running
        print("✓ Ollama integration test completed (no actual connection)")
        
        return True
    except Exception as e:
        print(f"✗ Ollama integration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("LangGraph Integration Test")
    print("=" * 40)
    
    tests = [
        test_langgraph_imports,
        test_config_loading,
        test_sqlite_checkpoint,
        test_state_graph,
        test_ollama_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test execution error: {e}")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All tests passed! LangGraph integration is working.")
    else:
        print("✗ Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
