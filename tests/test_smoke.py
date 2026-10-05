def test_flow_imports():
    from src.pipeline.flow import main_flow
    assert main_flow.name == "defender-servidor"
