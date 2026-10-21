"""Audit the frozen VIP choice, reconstructed splits and feature differences.

CPU only. Does not read outer scores or alter any recipe/checkpoint. Stable ID
digests supplement152's legacy object-array byte hash, which can encode pointers.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

import valid2 as v
from common import OUT, PQ, ROOT


def sha(path):
    with path.open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def stable_ids(values):
    return hashlib.sha256(json.dumps(list(values),ensure_ascii=True,separators=(",",":")).encode()).hexdigest()


def family(name):
    for prefixes,label in [(('bc_','bp_','bh_','bz_'),'bureau'),(('rp_','rq_','sv_'),'repayment'),
                           (('ac_','cf_'),'account'),(('ec_','ec2_'),'ecommerce'),(('pp_',),'price'),
                           (('pv_','ob_'),'previous')]:
        if name.startswith(prefixes):return label
    return 'application/cross'


def main():
    directory=ROOT/"results/v48/vip_raw_cached_f0"
    manifest=json.loads((directory/"manifest.json").read_text())
    selection=json.loads((directory/"s42_f0_selection.json").read_text())
    assert manifest["source_sha256"]==sha(ROOT/"eda/152_vipcop_kumo.py")
    assert manifest["features_sha256"]==sha(PQ/"features.parquet")
    locked=json.loads((ROOT/"docs/integration/protokol_ssl_kumo.json").read_text())
    assert locked["source_sha256"]["eda/valid2.py"]==sha(ROOT/"eda/valid2.py")
    assert selection["seed"]==42 and selection["fold"]==0
    assert manifest["config"]["confirm_nq"]==manifest["config"]["nq"]==4000
    meta=(pl.read_parquet(PQ/"features.parquet",columns=["application_id","customer_id","target","is_train"])
          .filter(pl.col("is_train")==1).sort("application_id"))
    ids,y=meta["application_id"].to_numpy(),meta["target"].to_numpy().astype(int)
    assert meta.height==meta["application_id"].n_unique()==meta["customer_id"].n_unique()==70000
    tr,va=v.outer(y,42)[0];fi,st=v.inner(y[tr],42,0);fi,st=tr[fi],tr[st]
    qs,qc=train_test_split(st,train_size=4000,stratify=y[st],random_state=42)
    qs,qc=np.sort(qs),np.sort(qc)
    assert selection["search_ids"]==ids[qs].tolist()
    assert selection["confirmation_ids"]==ids[qc].tolist()
    partitions={"inner_fit":fi,"inner_search":qs,"inner_confirmation":qc,"outer_validation":va}
    assert sum(len(rows) for rows in partitions.values())==len(np.unique(np.concatenate(list(partitions.values()))))==70000
    stable={name:{"rows":len(rows),"application_ids_json_sha256":stable_ids(ids[rows].tolist())}
            for name,rows in partitions.items()}
    rows=selection["confirmation"]
    baseline=next(row for row in rows if row["name"]=="fm_legacy")
    winner=next(row for row in rows if row["name"]==selection["winner"])
    assert max(rows,key=lambda row:row["ks"])["name"]==selection["winner"]
    basecols,wincols=set(baseline["cols"]),set(winner["cols"])
    assert len(basecols)==len(wincols)==104
    # Demonstrate the exact storage problem using synthetic IDs, not raw data.
    first=np.array([("application_"+str(i)*32).encode().decode() for i in range(3)],dtype=object)
    second=np.array([value.encode().decode() for value in first],dtype=object)
    assert np.array_equal(first,second) and all(a is not b for a,b in zip(first,second))
    assert hashlib.sha256(first.tobytes()).digest()!=hashlib.sha256(second.tobytes()).digest()
    assert stable_ids(first.tolist())==stable_ids(second.tolist())
    report={"winner":selection["winner"],"confirmation_baseline_ks":baseline["ks"],
            "confirmation_winner_ks":winner["ks"],"confirmation_delta":winner["ks"]-baseline["ks"],
            "feature_overlap":len(basecols&wincols),"baseline_only":sorted(basecols-wincols),
            "winner_only":sorted(wincols-basecols),"stable_reconstructed_splits":stable,
            "selection_sha256":sha(directory/"s42_f0_selection.json"),
            "valid2_sha256":sha(ROOT/"eda/valid2.py"),
            "source_sha256":sha(Path(__file__).resolve()),
            "legacy_fit_id_digest":selection["fit_ids_sha256"],
            "legacy_fit_digest_limitation":"Object-array tobytes hashes pointer storage, not stable string content; not used as split proof.",
            "split_evidence":"Canonical feature hash + unchanged152 source + valid2 splits + exact saved search/confirmation IDs.",
            "outer_scores_read":False,"model_executed":False,
            "limits":["Confirmation is inner selection evidence, not outer/generalization or a breakthrough.",
                      "Reconstructed context IDs supplement metadata; no stored historical context tensor was available.",
                      "Surrogate coefficients describe associations in searched coalitions, not causal feature effects."]}
    (OUT/"169_vip_selection_audit.json").write_text(json.dumps(report,indent=2))
    labels=sorted({family(c) for c in basecols|wincols})
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    x=np.arange(len(labels))
    axes[0].bar(x-.18,[sum(family(c)==label for c in basecols) for label in labels],width=.36,label="baseline")
    axes[0].bar(x+.18,[sum(family(c)==label for c in wincols) for label in labels],width=.36,label="VIP selected")
    axes[0].set_xticks(x,labels,rotation=45,ha="right");axes[0].set(ylabel="Features",title=f"104 features each; overlap {len(basecols&wincols)}")
    axes[0].legend()
    axes[1].bar([row["name"] for row in rows],[row["ks"]-baseline["ks"] for row in rows])
    axes[1].axhline(0,color="k",lw=.7);axes[1].tick_params(axis="x",rotation=45)
    axes[1].set(ylabel="DeltaKS vs baseline",title="Inner confirmation only; outer scores not read")
    fig.tight_layout();fig.savefig(OUT/"169_vip_selection_audit.png",dpi=130);plt.close(fig)
    print(f"winner={selection['winner']} confirmation delta={report['confirmation_delta']:+.6f}; sharedfeatures={report['feature_overlap']}/104")
    print("stable reconstructed partitions:",stable)
    print("Legacy object-byte fit-ID hash encodes pointers rather than string content; source/data and saved split IDs verified")
    print("No outer score read; no recipe/checkpoint changed")


if __name__=="__main__":main()
