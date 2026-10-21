"""Build migrated notebooks in a disposable mirror; never train or overwrite history."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
from IPython.core.inputtransformer2 import TransformerManager

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"eda/out"
BUILDERS=["build_review_notebooks.py", "build_review_notebook_v5.py", "build_reverse_notebook.py",
          "build_kumo_notebook.py", "build_kumo_improvement_notebook.py", "build_hybrid_kumo_notebook.py",
          "build_finetune_pilot_notebook.py", "build_goal054_notebook.py"]


def main():
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"notebook").glob("*.ipynb")}
    results=[]
    with tempfile.TemporaryDirectory(prefix="logika-build-check-") as tmp:
        project=Path(tmp)
        shutil.copytree(ROOT/"scripts",project/"scripts",ignore=shutil.ignore_patterns("__pycache__"))
        (project/"notebook").mkdir()
        for p in (ROOT/"notebook").glob("*.ipynb"):
            if p.name[:2].isdigit() or p.name=="analisis_logika.ipynb":
                shutil.copy2(p,project/"notebook"/p.name)
        for name in ("artifacts","results","third_party"):
            (project/name).symlink_to(ROOT/name,target_is_directory=True)
        for builder in BUILDERS:
            result=subprocess.run([sys.executable,str(project/"scripts"/builder)],cwd=project,
                                  text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
            print(f"{builder}: exit={result.returncode}",flush=True)
            results.append({"builder":builder,"exit_code":result.returncode,"output":result.stdout})
            if result.returncode:
                print(result.stdout,flush=True)
                break
        cells=0
        for p in (project/"notebook").glob("*.ipynb"):
            nb=nbformat.read(p,as_version=4);nbformat.validate(nb)
            for i,cell in enumerate(nb.cells):
                if cell.cell_type=="code":
                    compile(TransformerManager().transform_cell(cell.source),f"{p.name}:cell{i}","exec")
                    assert not any(s in cell.source for s in ("Logika-main","data/raw","notebooks/")),p
                    cells+=1
        # reporting.create_notebook is a ninth builder, without report/model execution.
        result=subprocess.run([sys.executable,"-c","import sys; sys.path.insert(0, 'scripts'); import reporting; reporting.create_notebook()"],
                              cwd=project,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
        results.append({"builder":"reporting.create_notebook","exit_code":result.returncode,"output":result.stdout})
        assert (project/"notebook/analisis_logika.ipynb").exists()
        assert not (project/"analisis_logika.ipynb").exists()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items()),"Original notebook modified"
    report={"builders":results,"compiled_cells":cells,"original_notebooks_unchanged":True,
            "model_execution":False,"temporary_mirror_removed":True}
    (OUT/"160_integration_builders.json").write_text(json.dumps(report,indent=2))
    fig,ax=plt.subplots(figsize=(9,4));ax.barh([r["builder"] for r in results],[int(r["exit_code"]==0) for r in results])
    ax.set(xlabel="Build-only success (no model execution)",xlim=(0,1.2));fig.tight_layout()
    fig.savefig(OUT/"160_integration_builders.png",dpi=120);plt.close(fig)
    print(f"Built {len(results)} entry points; compiled {cells} cells; original notebooks unchanged",flush=True)
    assert len(results)==9 and all(r["exit_code"]==0 for r in results),report


if __name__=="__main__":
    main()
