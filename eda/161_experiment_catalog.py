"""Index experiment code and reported metrics from both origins without mixing protocols."""
import ast
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/"docs/integration"
OUT=ROOT/"eda/out"


def metric_dicts(obj,path="$",context=""):
    if isinstance(obj,dict):
        context=str(obj.get("name",obj.get("candidate",obj.get("model",context))))
        nums={k:v for k,v in obj.items() if isinstance(v,(float,int)) and not isinstance(v,bool)
              and ("ks" in k.lower() or k in {"fold_wins","public_score","p_gt_zero","p_positive"})}
        if nums:
            yield {"object_path":path,"candidate":context,"reported_metrics":json.dumps(nums,sort_keys=True)}
        for k,v in obj.items():
            yield from metric_dicts(v,f"{path}.{k}",context)
    elif isinstance(obj,list):
        for i,v in enumerate(obj): yield from metric_dicts(v,f"{path}[{i}]",context)


def write_csv(path,rows):
    with path.open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    entries=[]
    for source,folder in [("utama",ROOT/"eda"),("teman",ROOT/"scripts")]:
        for p in sorted(folder.glob("*.py")):
            tree=ast.parse(p.read_text(),filename=str(p))
            doc=(ast.get_docstring(tree) or "").splitlines()
            entries.append({"origin":source,"entry_point":str(p.relative_to(ROOT)),
                            "description":doc[0] if doc else "See source and historical journal",
                            "evidence_index":"handoff/05_katalog_eksperimen/" if source=="utama" else "artifacts/optimization/",
                            "protocol":"validation_v2 for new experiments; older versions differ" if source=="utama" else
                                       "Historical 3/5/10fold; do not compare with validation_v2 directly"})
    write_csv(AUDIT/"indeks_eksperimen.csv",entries)
    reports=[]
    # Keep each reported object anchored to its source and JSON path; never join OOF.
    for source,folder in [("teman",ROOT/"artifacts"),("utama",ROOT/"results")]:
        for p in sorted(folder.rglob("*.json")):
            if any(s in p.parts for s in ("__pycache__","node_modules")):continue
            try: obj=json.loads(p.read_text())
            except (ValueError,UnicodeError):continue
            for row in metric_dicts(obj):
                reports.append({"origin":source,"evidence_file":str(p.relative_to(ROOT)),**row,
                                "status":"reported historical metric; not independently rerun"})
    write_csv(AUDIT/"indeks_metrik_dilaporkan.csv",reports)
    counts={s:sum(r["origin"]==s for r in entries) for s in ("utama","teman")}
    fig,axes=plt.subplots(1,2,figsize=(9,3))
    axes[0].bar(counts.keys(),counts.values());axes[0].set_title("Experiment/helper Python entry points")
    axes[1].bar(counts.keys(),[sum(r["origin"]==s for r in reports) for s in counts]);axes[1].set_title("Reported metric objects (not independent fits)")
    fig.tight_layout();fig.savefig(OUT/"161_experiment_catalog.png",dpi=120);plt.close(fig)
    summary={"entry_points":counts,"reported_metric_objects":len(reports),"oof_combined":False,"model_executed":False}
    (OUT/"161_experiment_catalog.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
