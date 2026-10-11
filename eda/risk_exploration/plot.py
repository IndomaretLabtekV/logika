#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["duckdb==1.5.6", "numpy==2.5.4", "matplotlib==3.11.2"]
# ///
# How to run: install uv from https://docs.astral.sh/uv/ then
# uv run eda/risk_exploration/plot.py
"""Render aggregate evidence as portable PNG figures without customer identifiers."""
from __future__ import annotations

import sys

import duckdb
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build import DEST


def save(name: str) -> None:
    """Save the current figure using a consistent readable export resolution."""
    plt.savefig(DEST / name, dpi=170, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Plot aligned histories, conditional rates, recurrence and cohort graphs."""
    if "--help" in sys.argv:
        print(__doc__ + "\nRun build.py and diagnose.py first. No arguments required.")
        return
    if len(sys.argv) != 1:
        sys.exit("Unexpected argument. Use --help.")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    with duckdb.connect() as con:
        for path in DEST.glob("*.csv"):
            con.read_csv(str(path)).create_view(path.stem)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for ax, field, label in zip(axes.flat,
                ("income_to_declared", "net_to_declared", "low_balance_fraction", "commerce_to_income"),
                ("Income-like credits / declared income", "Net flow / declared income",
                 "Fraction of low-balance days", "Recorded commerce / declared income"), strict=True):
            for target, color in ((0,"#227b8d"),(1,"#c65337")):
                rows = con.sql(f"SELECT month_lag,{field} FROM lag_profiles WHERE target={target} ORDER BY month_lag").fetchall()
                ax.plot([r[0] for r in rows],[r[1] for r in rows],marker="o",color=color,label=f"target={target}")
            ax.invert_xaxis()
            ax.set(xlabel="Complete calendar months before decision", ylabel=label)
            ax.legend()
        fig.suptitle("Application-aligned histories: descriptive means; cohort size varies by lag")
        save("aligned_histories.png")

        fig, axes = plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
        for ax, question in zip(axes,("cash beyond score","spending beyond score"),strict=True):
            rows = con.sql(f"SELECT stratum,segment,sum(n) AS n,sum(events)/sum(n)::DOUBLE AS rate FROM interactions WHERE question='{question}' GROUP BY 1,2 ORDER BY 1,2").fetchall()
            strata = ["<550", "550-649", "650+", "unscored"]
            segments = ["deficit", "surplus", "unobserved"] if question == "cash beyond score" else ["<10%", "10-30%", "30%+"]
            matrix = np.full((len(segments),len(strata)),np.nan)
            for stratum,segment,n,rate in rows:
                i,j=segments.index(segment),strata.index(stratum)
                matrix[i,j]=rate*100
                mark = "*" if n<200 or n*rate<20 else ""
                ax.text(j,i,f"{rate:.1%}{mark}\nn={n:,}",ha="center",va="center",fontsize=9)
            im=ax.imshow(matrix,cmap="YlOrRd",vmin=0,vmax=70,aspect="auto")
            ax.set(xticks=range(len(strata)),xticklabels=strata,yticks=range(len(segments)),yticklabels=segments,title=question,xlabel="Bureau score band")
        fig.colorbar(im,ax=axes,label="Observed 90-DPD rate (%)",shrink=0.8)
        fig.suptitle("Descriptive rates; * fewer than 200 customers or 20 events")
        save("conditional_risk.png")

        rows=con.sql("SELECT * FROM category_network ORDER BY category_a,category_b").fetchall()
        names=sorted({r[0] for r in rows}|{r[1] for r in rows})
        angles=np.linspace(0,2*np.pi,len(names),endpoint=False)
        positions={name: (np.cos(angle),np.sin(angle)) for name,angle in zip(names,angles,strict=True)}
        fig,ax=plt.subplots(figsize=(9,8))
        for a,b,n,na,nb,jaccard,lift in rows:
            x1,y1=positions[a]
            x2,y2=positions[b]
            ax.plot([x1,x2],[y1,y2],color="#607d8b",alpha=0.25,linewidth=0.5+4*jaccard)
        for name,(x,y) in positions.items():
            ax.scatter([x],[y],s=900,c="#d8ecec",edgecolors="#227b8d",zorder=3)
            ax.text(x*1.18,y*1.18,name,ha="center",va="center")
        ax.set(xlim=(-1.5,1.5),ylim=(-1.4,1.4),aspect="equal",
               title="Category co-occurrence across labeled customers\nEdge width = Jaccard overlap; these are not social connections")
        ax.axis("off")
        save("category_graph.png")

        rows=con.sql("SELECT region,product_category,customers FROM region_category_network ORDER BY 1,2").fetchall()
        regions=sorted({r[0] for r in rows})
        matrix=np.zeros((len(regions),len(names)))
        totals=dict(con.sql("SELECT region,n FROM region_summary").fetchall())
        for region,category,n in rows:
            matrix[regions.index(region),names.index(category)]=100*n/totals[region]
        fig,ax=plt.subplots(figsize=(11,4),constrained_layout=True)
        im=ax.imshow(matrix,cmap="Blues",vmin=0,vmax=100)
        ax.set(xticks=range(len(names)),xticklabels=names,yticks=range(len(regions)),yticklabels=regions,
               title="Fictional region-to-category incidence (% of labeled customers)")
        plt.setp(ax.get_xticklabels(),rotation=25,ha="right")
        for i in range(len(regions)):
            for j in range(len(names)):
                ax.text(j,i,f"{matrix[i,j]:.1f}",ha="center",va="center",color="white")
        fig.colorbar(im,ax=ax)
        save("region_category.png")

        fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        for year in (2023,2024,2025):
            rows=con.sql(f"SELECT calendar_month,within_customer_income,within_customer_spending FROM seasonality WHERE year={year} ORDER BY calendar_month").fetchall()
            for ax,k in zip(axes,(1,2),strict=True):
                ax.plot([r[0] for r in rows],[r[k] for r in rows],marker="o",label=str(year))
        for ax,label in zip(axes,("Income-like credits","Commerce: complete months after first observed order"),strict=True):
            ax.set(xlabel="Calendar month",ylabel="Value / customer's observed monthly mean",title=label,xticks=range(1,13))
            ax.legend()
        fig.suptitle("Seasonality screen: changing cohorts and partial years prevent a causal reading")
        save("seasonality.png")
    print("Wrote five figures to", DEST)


if __name__ == "__main__":
    main()
