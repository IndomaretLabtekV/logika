"""Record the completed real submission against its pre-upload forecast."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'results/v59/v41_lowvar_today';receipt=json.loads((out/'submission_receipt.json').read_text())
    audit=json.loads((ROOT/'eda/out/201_saved_lowvar_audit.json').read_text());pre=json.loads((out/'submission_preflight.json').read_text())
    assert receipt['ref']==57486639 and receipt['status']=='SubmissionStatus.COMPLETE'
    assert audit['public_forecast_mean']==pre['public_forecast_mean'] and audit['public_forecast_sd']==pre['public_forecast_sd']
    score=float(receipt['publicScore']);baseline=.56405;mean=audit['public_forecast_mean'];sd=audit['public_forecast_sd']
    result={'receipt':receipt,'public_score':score,'v31_best_public_historical':baseline,'delta_vs_v31_public':score-baseline,
            'forecast_mean':mean,'forecast_sd':sd,'forecast_z':(score-mean)/sd,'private_score_available':False,'breakthrough':False,
            'quota_remaining_from_kaggle_cli':1,'submission_goal_today_met':True,'larger_research_goal_complete':False,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'decision':'No promotion as breakthrough, no public-driven weight tuning, no automatic second upload'}
    (out/'post_submission_analysis.json').write_text(json.dumps(result,indent=2));(ROOT/'eda/out/211_submission_v59_review.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots(figsize=(8,3));ax.errorbar(mean,0,xerr=sd,fmt='o',capsize=4,label='Forecast ±1SD before upload');ax.scatter([score],[0],marker='x',s=70,label=f'Actual {score:.5f}');ax.axvline(baseline,ls='--',color='grey',label='Historical v31 best');ax.set(yticks=[],xlabel='Public KS',title='Completed submission57486639; no private score available');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'eda/out/211_submission_v59_review.png',dpi=130);plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
