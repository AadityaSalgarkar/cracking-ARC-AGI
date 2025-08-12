#!/usr/bin/env python3
"""Generate a JavaScript file with all puzzle filenames."""

import json
from pathlib import Path

# Get all puzzle files
training_dir = Path("../dataset/ARC-1/data/training")
evaluation_dir = Path("../dataset/ARC-1/data/evaluation")

training_files = sorted([f.stem for f in training_dir.glob("*.json")])
evaluation_files = sorted([f.stem for f in evaluation_dir.glob("*.json")])

# Create JavaScript content
js_content = f"""// Auto-generated list of puzzle files
const PUZZLE_FILES = {{
    training: {json.dumps(training_files, indent=4)},
    evaluation: {json.dumps(evaluation_files, indent=4)}
}};
"""

# Write to file
with open("puzzle_files.js", "w") as f:
    f.write(js_content)

print(f"Generated puzzle_files.js with {len(training_files)} training and {len(evaluation_files)} evaluation puzzles")