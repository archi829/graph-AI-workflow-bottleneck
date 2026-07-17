import json
import random

# Input JSONL files
input_files = [
    "data/crew_traces.jsonl",
    "data/open_deep_research_traces.jsonl"
]

output_file = "all_traces.jsonl"

# Read all traces
traces = []
for file in input_files:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))

# Shuffle (reproducible)
random.seed(42)
random.shuffle(traces)

# Write merged JSONL
with open(output_file, "w", encoding="utf-8") as f:
    for trace in traces:
        f.write(json.dumps(trace) + "\n")

print(f"Created {output_file} with {len(traces)} traces.")