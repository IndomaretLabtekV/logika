---
name: notebook
description: >
  Made by @zirachw
  Workflow for building and editing Jupyter notebooks that are generated from Python builder
  scripts (using md() / code() / section() helper functions). Use this skill whenever the user
  asks to add, fix, move, or restructure a cell in a notebook that has a corresponding builder
  script, or when editing any build_*.py / build_vN.py / generate_notebook.py style script.
  Trigger even if the user just says "fix the notebook", "add a cell", "move imports", or
  "the notebook has an error" -- if a builder script exists, this workflow applies.
---

# Notebook-from-Script Workflow

Made by @zirachw

## Language

All prose in markdown cells follows the project's language. Invoke `/writing` for English notebooks. Invoke `/indonesia-writing` for Indonesian notebooks. English is the default when the project language is not stated.

## Core Workflow

1. **Read the builder script** before making any change.
2. **Edit the script** using the `md()`, `code()`, and `section()` helpers.
3. **Regenerate** the notebook by running the script (`python src/build_<name>.py`).
4. **Confirm** the output line confirms the notebook was written successfully.

## Builder Script Skeleton

```python
import json
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).parent / "<notebook>.ipynb"
HR  = "---"

def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": uuid4().hex[:8], "metadata": {}, "source": source}

def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "id": uuid4().hex[:8],
        "metadata": {}, "outputs": [], "source": source,
    }

def section(title: str, anchor: str) -> dict:
    return md(
        f"{HR}\n\n"
        f"# {title} <a name=\"{anchor}\"></a>\n\n"
        f"{HR}"
    )

cells = []

# ---------------------------------------------------------------------------
# Notebook writer
# ---------------------------------------------------------------------------
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"},
    },
    "cells": cells,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(cells)} cells -> {OUT}")
```

Cell source strings use plain Python string concatenation. Every line ends with `\n` except the last line of a cell.

```python
cells.append(code(
    "import numpy as np\n"
    "import pandas as pd\n"
    "print('ready')"
))
```

The `# ---...---` section divider comments belong only in the builder `.py` file. They must not appear inside any string passed to `code()`.

## Notebook Structure

Every notebook opens with three fixed banner cells, followed by Introduction (section 1), Initialization (section 2), then content sections (3, 4, ...).

```python
# Cell 1: project name + tagline
cells.append(md(
    f"{HR}\n\n"
    "# Competition / Project Name\n\n"
    "*Brief one-sentence description of the competition or project.*"
))

# Cell 2: team info
cells.append(md(
    f"{HR}\n\n"
    "## Team Name or Group Number\n\n"
    "- Name 1 (Role)\n"
    "- Name 2\n"
    "- Name 3"
))

# Cell 3: Table of Contents
cells.append(md(
    f"{HR}\n\n"
    "## Table of Contents\n\n"
    "1. [**Introduction**](#1)\n"
    "2. [**Initialization**](#2)\n"
    "3. [**Section Title**](#3)\n"
))
```

## Writing a New Builder Script (Long Scripts)

When the expected output is over 1000 lines, do not write the entire file in one shot because the Write call will not complete. Use this approach instead.

1. Write the skeleton (imports, helpers, `cells = []`, notebook writer block) with the Write tool.
2. Append each section with a separate Edit call, inserting before the notebook writer block.
   ```
   old_string = (the notebook writer block, verbatim)
   new_string = (the new section code) + "\n\n" + (the notebook writer block)
   ```
3. After every 3-4 sections, run the script to verify it parses correctly.
4. Do a final run to produce the `.ipynb`.

## Inspecting an Existing Notebook

Never Grep or Read a `.ipynb` file as raw text. Use a Python heredoc instead.

```bash
python3 << 'EOF'
import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('src/notebook.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'keyword' in src:
        print(f'--- Cell {i} ({cell["cell_type"]}) ---')
        print(src)
        print()
EOF
```

## Common Fixes

| Symptom | Fix |
|---------|-----|
| `NameError` / `ModuleNotFoundError` in a cell | Add the import to the Libraries cell, not the failing cell |
| `!pip install` found in script | Replace with `%pip install` |
| Notebook does not reflect recent edits | Re-run the builder script to regenerate the `.ipynb` |
| Non-ASCII symbol in a cell string | Replace with the ASCII equivalent (see cell-style reference) |
| Inline import inside a function in a code cell | Move it to the Libraries cell |
| Formula comment trailing on the right | Move it to the line above |
| `# --- label ---` divider inside a `code()` string | Remove it and use a blank line to group |

## Template

A complete runnable example is available as [build_template.py](build_template.py) (the builder script) and [template.ipynb](template.ipynb) (the generated output). Read `build_template.py` to see how all conventions look assembled into a real script.

## Reference Files

Always load [references/cell-style.md](references/cell-style.md) for notebook-specific style rules (bridging paragraphs, Insights blocks, formula comments, divider comments, docstrings).

Load the following when the task involves that area:

- Introduction section patterns (Overview, Aim, Metric, Dataset variants, Approach) are in [references/introduction.md](references/introduction.md)
- Initialization section patterns (Environment Setup, Seed, Settings, Load Dataset) are in [references/initialization.md](references/initialization.md)
