import json
from pathlib import Path

input_dir = Path(r"data\raw\agent_system=crewai")
output_file = "crew_traces.jsonl"

with open(output_file, "w", encoding="utf-8") as out:
    for json_file in input_dir.rglob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            trace = json.load(f)
        out.write(json.dumps(trace) + "\n")

print(f"Created {output_file}")