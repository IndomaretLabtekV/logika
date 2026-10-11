# Introduction Section

Every notebook has an Introduction section as section 1, before Initialization. It has its own `section()` call.

**Which subsections to include:**
- Notebooks that load their own dataset (training, EDA, modelling) include Overview, Aim, Metric, Dataset, and Approach.
- Comparison or analysis notebooks that receive results from other notebooks include only Overview.

All prose inside Introduction subsections is wrapped in `*italic*`. Each paragraph is its own `*...*` block separated by `\n\n`.

```python
cells.append(section("Introduction", "1"))

cells.append(md("## Overview\n\n*Context paragraph in third person.*"))

cells.append(md("## Aim\n\n*What the notebook/model aims to accomplish.*"))

cells.append(md(
    "## Metric\n\n"
    "***Metric Name** measures ... . "
    "It is appropriate here because ...*\n\n"
    "$$\\text{Metric} = \\frac{\\text{numerator}}{\\text{denominator}}$$"
))
```

## Dataset Cell Variants

The closing block of the Dataset cell depends on the data modality. Choose the matching variant.

**Tabular** (numbered attribute list followed by a target line):
```python
cells.append(md(
    "## Dataset\n\n"
    "*Data sourced from <Platform> at: <URL>*\n\n"
    "*The dataset contains <N> samples. <Distribution note.>*\n\n"
    "*The dataset is used to <purpose>.*\n\n"
    "**Attributes**\n\n"
    "1.  `feature_1`     : Description of first feature\n"
    "2.  `feature_2`     : Description of second feature\n"
    "3.  `feature_3`     : Description of third feature\n\n"
    "**Target**\n"
    " `target`       : Target description (0: Negative class, 1: Positive class)"
))
```

**Image / annotation** (file structure tree):
```python
cells.append(md(
    "## Dataset\n\n"
    "*Data sourced from <Platform> at: <URL>*\n\n"
    "*The dataset contains <N> images annotated with <mask / bounding box / label>. "
    "<Distribution or split note.>*\n\n"
    "*The dataset is used to <purpose>.*\n\n"
    "```\n"
    "data/\n"
    "├── train/\n"
    "|    ├── images/\n"
    "|    └── masks/\n"
    "└── test/\n"
    "     └── images/\n"
    "```"
))
```

**Text / NLP** (corpus file structure):
```python
cells.append(md(
    "## Dataset\n\n"
    "*Data sourced from <Platform> at: <URL>*\n\n"
    "*The dataset contains <N> text samples with <label / caption / sequence>. "
    "<Distribution or split note.>*\n\n"
    "*The dataset is used to <purpose>.*\n\n"
    "```\n"
    "data/\n"
    "├── <corpus_file>.txt\n"
    "├── train/\n"
    "|    └── <class or split files>\n"
    "└── test/\n"
    "     └── <class or split files>\n"
    "```"
))
```

**Audio** (audio file structure):
```python
cells.append(md(
    "## Dataset\n\n"
    "*Data sourced from <Platform> at: <URL>*\n\n"
    "*The dataset contains <N> audio files in <format> with <label / transcript>. "
    "<Distribution or split note.>*\n\n"
    "*The dataset is used to <purpose>.*\n\n"
    "```\n"
    "data/\n"
    "├── train/\n"
    "|    └── <class>/\n"
    "└── test/\n"
    "     └── <class>/\n"
    "```"
))
```

## Approach Cell

All notebooks that load data include the Approach cell. Comparison notebooks omit it.

```python
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
```
