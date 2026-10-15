# Initialization Section

Initialization is always section 2. Adjust all subsequent section anchors accordingly (3, 4, ...).

The Initialization section always contains these subsections in this exact order:

1. `## Environment Setup` contains a bridging paragraph, then `!nvidia-smi`, then a one-sentence markdown cell ("The following cell installs all libraries used in this notebook."), then `%pip install -q ...`
2. `## Import Libraries` centralises every `import` for the entire notebook
3. `## Seed Everything` defines and calls `seed_everything(42)`
4. `## Settings` contains a bridging paragraph, then the Settings class and `CFG = Settings()`
5. `## Load Dataset` collects file paths and split labels, with no visualisation

## Environment Setup

```python
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
```

Always use `%pip install` (IPython magic), never `!pip install`.

## Seed Everything

```python
cells.append(code(
    "def seed_everything(seed: int = 42):\n"
    "    random.seed(seed)\n"
    "    os.environ['PYTHONHASHSEED'] = str(seed)\n"
    "    np.random.seed(seed)\n"
    "\n"
    "seed_everything(42)"
))
```

## Settings Pattern

All paths, hyperparameters, and environment flags go in `Settings`. Never hardcode constants outside of it.

```python
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
    "    FIG_DIR    = OUTPUT_DIR / '<notebook-name>'\n"
    "\n"
    "CFG = Settings()\n"
    "CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
    "CFG.FIG_DIR.mkdir(parents=True, exist_ok=True)"
))
```

## Load Dataset

Collect file paths and split labels. No visualisation in this subsection.

```python
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
```
