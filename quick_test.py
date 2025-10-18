import sys
import os

# Test 1: LangGraph packages
print("Test 1: LangGraph packages")
try:
    import langgraph
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import StateGraph, END
    print("PASS: All LangGraph packages imported")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# Test 2: Create simple workflow
print("\nTest 2: Create simple workflow")
try:
    from typing import TypedDict
    
    class SimpleState(TypedDict):
        count: int
    
    workflow = StateGraph(SimpleState)
    
    def increment(state):
        state["count"] += 1
        return state
    
    workflow.add_node("increment", increment)
    workflow.set_entry_point("increment")
    workflow.add_edge("increment", END)
    
    app = workflow.compile()
    result = app.invoke({"count": 0})
    
    assert result["count"] == 1
    print("PASS: Workflow created and executed")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# Test 3: SQLite checkpoint
print("\nTest 3: SQLite checkpoint")
try:
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    saver = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
    print("PASS: SQLite checkpoint created")
    
    os.unlink(db_path)
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

print("\n" + "="*40)
print("All tests passed!")
print("LangGraph integration is ready to use.")

