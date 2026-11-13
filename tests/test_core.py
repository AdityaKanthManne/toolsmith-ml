import pytest
from toolsmith.core import validate_call, evaluate, baseline

def test_valid_tool_call():
    x = {"tool":"get_revenue","arguments":{"company":"Apple","fiscal_year":2025}}
    assert validate_call(x)==x

def test_invalid_schema():
    with pytest.raises(ValueError):
        validate_call({"tool":"get_revenue","arguments":{"company":"Apple","fiscal_year":"2025"}})
    with pytest.raises(ValueError):
        validate_call({"tool":"delete_data","arguments":{}})

def test_baseline_and_scoring():
    e = {"prompt":"Apple revenue in 2025","expected":{"tool":"get_revenue","arguments":{"company":"Apple","fiscal_year":2025}}}
    assert evaluate([e],[baseline(e)])["exact_call_accuracy"] == 1.0
    assert evaluate([e],['not json'])["schema_validity"] == 0
