"""Reproducible tool-call evaluation: JSON parsing, schema and tool selection."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

TOOLS = {
    "get_revenue": {"company": str, "fiscal_year": int},
    "calculate_growth": {"company": str, "start_year": int, "end_year": int},
    "search_filing": {"company": str, "query": str},
}

def validate_call(raw: str | dict) -> dict:
    obj = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(obj, dict) or set(obj) != {"tool", "arguments"}:
        raise ValueError("Expected exactly 'tool' and 'arguments'")
    tool = obj["tool"]
    if tool not in TOOLS:
        raise ValueError(f"Unknown tool: {tool}")
    schema = TOOLS[tool]
    args = obj["arguments"]
    if not isinstance(args, dict) or set(args) != set(schema):
        raise ValueError("Incorrect argument keys")
    for key, typ in schema.items():
        if type(args[key]) is not typ:
            raise ValueError(f"{key} must be {typ.__name__}")
    return obj

def evaluate(examples: list[dict], predictions: list[str | dict]) -> dict:
    if len(examples) != len(predictions) or not examples:
        raise ValueError("Provide matching, nonempty examples and predictions")
    valid = correct_tool = correct_all = 0
    errors = []
    for idx, (example, prediction) in enumerate(zip(examples, predictions)):
        expected = validate_call(example["expected"])
        try:
            got = validate_call(prediction)
            valid += 1
            correct_tool += got["tool"] == expected["tool"]
            correct_all += got == expected
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append({"index":idx,"error":str(exc)})
    n = len(examples)
    return {"count":n,"schema_validity":valid/n,"tool_accuracy":correct_tool/n,"exact_call_accuracy":correct_all/n,"errors":errors}

def format_prompt(example: dict) -> str:
    tools = {k: {arg: typ.__name__ for arg,typ in v.items()} for k,v in TOOLS.items()}
    return "Choose one tool. Output only a JSON object with keys tool and arguments.\nTools: "+json.dumps(tools)+"\nRequest: "+example["prompt"]+"\nJSON:"

def baseline(example: dict) -> dict:
    """Non-ML heuristic baseline, not a substitute for trained-model results."""
    q = example["prompt"].lower()
    company = next((name for name in ("Apple","Microsoft","Nvidia") if name.lower() in q), None)
    if company is None:
        raise ValueError("Unrecognized company")
    years = [int(t) for t in q.replace(","," ").split() if t.isdigit() and len(t)==4]
    if "growth" in q and len(years)>=2:
        return {"tool":"calculate_growth","arguments":{"company":company,"start_year":years[0],"end_year":years[1]}}
    if "revenue" in q and years:
        return {"tool":"get_revenue","arguments":{"company":company,"fiscal_year":years[0]}}
    return {"tool":"search_filing","arguments":{"company":company,"query":example["prompt"]}}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["baseline","evaluate"])
    parser.add_argument("--dataset", default="data/examples.json")
    parser.add_argument("--predictions")
    args = parser.parse_args()
    examples = json.loads(Path(args.dataset).read_text())
    predictions = [baseline(e) for e in examples] if args.command == "baseline" else json.loads(Path(args.predictions).read_text())
    print(json.dumps(evaluate(examples,predictions),indent=2))

if __name__ == "__main__":
    main()
