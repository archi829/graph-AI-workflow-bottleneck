import json
from pathlib import Path

input_dir = Path("data/raw/agent_system=crewai")
output_file = Path("data/crew_traces.jsonl")

with output_file.open("w", encoding="utf-8") as out:
    for json_file in input_dir.rglob("*.json"):
        with json_file.open("r", encoding="utf-8") as f:
            trace = json.load(f)
        out.write(json.dumps(trace) + "\n")

print(f"Created {output_file}")
