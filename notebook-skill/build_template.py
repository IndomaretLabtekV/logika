import json
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).parent / "template.ipynb"
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
# Banner (3 cells) + TOC
# ---------------------------------------------------------------------------
cells.append(md(
    f"{HR}\n\n"
    "# Competition / Project Name\n\n"
    "*Brief one-sentence description of the competition or project.*"
))

cells.append(md(
    f"{HR}\n\n"
    "## Team Name or Group Number\n\n"
    "- Name 1 (Role)\n"
    "- Name 2\n"
    "- Name 3"
))

cells.append(md(
    f"{HR}\n\n"
    "## Table of Contents\n\n"
    "1. [**Introduction**](#1)\n"
    "2. [**Initialization**](#2)\n"
    "3. [**Section Title**](#3)\n"
    "4. [**Section Title**](#4)\n"
))

# ---------------------------------------------------------------------------
# Section 1: Introduction
# ---------------------------------------------------------------------------
cells.append(section("Introduction", "1"))

cells.append(md(
    "## Overview\n\n"
    "*Brief description of the problem domain and why it matters. "
    "One to three sentences in third person.*"
))

cells.append(md(
    "## Aim\n\n"
    "*What the notebook aims to accomplish, framed in terms of the model task. "
    "For example: build a classifier that predicts X from Y, achieving metric Z.*"
))

cells.append(md(
    "## Metric\n\n"
    "***Metric Name** measures ... . "
    "It is appropriate here because ...*\n\n"
    "$$\\text{Metric} = \\frac{\\text{numerator}}{\\text{denominator}}$$"
))

cells.append(md(
    "## Dataset\n\n"
    "*Data sourced from <Platform>, specifically from the competition or project "
    "<Name> at: <URL>*\n\n"
    "*The dataset contains <N> samples with <label/target> indicating <description>. "
    "<Important note about distribution or characteristics.>*\n\n"
    "*The dataset is used to <purpose within this notebook>.*\n\n"
    "**Attributes**\n\n"
    "1.  `feature_1`     : Description of first feature\n"
    "2.  `feature_2`     : Description of second feature\n"
    "3.  `feature_3`     : Description of third feature\n\n"
    "**Target**\n"
    " `target`       : Target description (0: Negative class, 1: Positive class)"
))

cells.append(md(
    "## Approach: <Method Name>\n\n"
    "*<Method Name> is a <type of model or technique> that <brief description of what it "
    "does and why it is appropriate for this task>.*\n\n"
    "*The approach uses <key technique> with <training strategy>. "
    "<Detail about the architecture or configuration>.*\n\n"
    "```\n"
    "Model Architecture\n"
    "    |\n"
    "    +-- Feature layer [dim]\n"
    "         |\n"
    "    Output layer -> prediction\n"
    "```\n\n"
    "*<Detail about data splitting strategy, class weighting, or any other "
    "training consideration.>*"
))

# ---------------------------------------------------------------------------
# Section 2: Initialization
# ---------------------------------------------------------------------------
cells.append(section("Initialization", "2"))

cells.append(md(
    "## Environment Setup\n\n"
    "To reproduce results without re-running training, assign `Settings.RUN_NAME` "
    "to `'INFERENCE'` and then run all cells. "
    "Recorded runtime: training takes approximately <duration> on the following GPU."
))

cells.append(code("!nvidia-smi"))

cells.append(md("The following cell installs all libraries used in this notebook."))

cells.append(code(
    "%pip install -q numpy pandas matplotlib scikit-learn"
))

cells.append(md("## Import Libraries"))

cells.append(code(
    "import os\n"
    "import random\n"
    "from pathlib import Path\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt"
))

cells.append(md("## Seed Everything"))

cells.append(code(
    "def seed_everything(seed: int = 42):\n"
    "    random.seed(seed)\n"
    "    os.environ['PYTHONHASHSEED'] = str(seed)\n"
    "    np.random.seed(seed)\n"
    "\n"
    "seed_everything(42)"
))

cells.append(md(
    "## Settings\n\n"
    "All paths, hyperparameters, and environment flags are centralised here. "
    "Switching between Kaggle and local execution requires no changes outside this class."
))

cells.append(code(
    "class Settings:\n"
    "    SEED       = 42\n"
    "    _ON_KAGGLE = Path('/kaggle/input').exists()\n"
    "\n"
    "    DATA_DIR   = (\n"
    "        Path('/kaggle/input/<dataset-slug>')\n"
    "        if _ON_KAGGLE else Path('../dataset/<local-folder>')\n"
    "    )\n"
    "    OUTPUT_DIR = Path('/kaggle/working') if _ON_KAGGLE else Path('../output')\n"
    "    FIG_DIR    = OUTPUT_DIR / 'template'\n"
    "\n"
    "CFG = Settings()\n"
    "CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
    "CFG.FIG_DIR.mkdir(parents=True, exist_ok=True)"
))

cells.append(md(
    "## Load Dataset\n\n"
    "Collect file paths and split labels. No visualisation here."
))

cells.append(code(
    "train_dir = CFG.DATA_DIR / 'train'\n"
    "val_dir   = CFG.DATA_DIR / 'val'\n"
    "test_dir  = CFG.DATA_DIR / 'test'\n"
    "\n"
    "train_files = sorted(train_dir.glob('*'))\n"
    "val_files   = sorted(val_dir.glob('*'))\n"
    "test_files  = sorted(test_dir.glob('*'))\n"
    "\n"
    "print(f'Train: {len(train_files)}  Val: {len(val_files)}  Test: {len(test_files)}')"
))

# ---------------------------------------------------------------------------
# Section 3: Example Content Section
# ---------------------------------------------------------------------------
cells.append(section("Section Title", "3"))

cells.append(md(
    "A bridging paragraph sets context for this section in third person. "
    "One to three sentences before the first code cell."
))

cells.append(code(
    "# example placeholder"
))

cells.append(md(
    "#### Insights\n\n"
    "> First observation derived from the analysis above.\n\n"
    "> Second observation if the insight covers more than one distinct point."
))

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
