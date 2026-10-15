"""Recompute saved-prediction trajectory; never execute notebook cells."""
from shared7 import *
import re

def main():
    f,tr,te,p,t=saved7();y=p['target'].to_numpy();scores={};rows=[]
    weights={1:dict(lgb=1,xgb=1,cat=1,tabpfn=1,causilo=1),2:dict(lgb=17,lgb_s=4,xgb=5,cat=2,cat_s=2),3:dict(lgb=8,lgb_s=3,xgb=1,xgb_s=3,cat=2,tabm=13),5:dict(lgb=4,xgb_s=1,cat=9,cat_s=3,tabm=13),6:dict(lgb=3,lgb_s=1,cat=2,cat_s=1,tabpfn=6,tabm=10,tabm_p=7)}
    logs=[]
    for v in range(1,8):
        n=json.loads((ROOT/f'notebook/v{v}.ipynb').read_text())
        for i,c in enumerate(n['cells']):
            for o in c.get('outputs',[]):
                for line in ''.join(o.get('text',[])).splitlines():
                    if any(k in line for k in ['OOF KS','BLEND','EXAONE','held-out halves','best_iteration']):logs.append(dict(version=v,cell=i,text=line))
        if v==4:
            rows.append(dict(version=v,model='selected_blend',KS=.5114068357000261,AUC=.837616223664805,source='notebook logged only; OOF artifact unavailable'));continue
        folder=ROOT/f'results/v{v}';pred={}
        for path in sorted(folder.glob('oof_*.npy')):
            name=path.stem[4:];a=np.load(path);assert a.shape==y.shape and np.isfinite(a).all()
            if v==1:
                ids=pl.read_parquet(ROOT/'data_pq/v1_train_order.parquet').select('application_id').with_columns(s=pl.Series(a))
                a=tr.select('application_id').join(ids,on='application_id',validate='1:1',maintain_order='left')['s'].to_numpy()
            if v==7: assert np.allclose(a,p[name].to_numpy())
            pred[name]=a;rows.append(dict(version=v,model=name,KS=ks(y,a),AUC=auc(y,a),source='saved OOF'))
        s=p['blend'].to_numpy() if v==7 else sum(w*rankdata(pred[name])/len(y) for name,w in weights[v].items())/sum(weights[v].values())
        scores[v]=s;rows.append(dict(version=v,model='selected_blend',KS=ks(y,s),AUC=auc(y,s),source='ID-bound OOF' if v==7 else 'legacy OOF; v1 order manifest, v2-v6 log consistency'))
        expected={1:.5051,2:.5135,3:.5142,5:.5274,6:.5283,7:.5270}[v]
        assert abs(ks(y,s)-expected)<.00006,(v,ks(y,s))
    z=csv(rows,'00_scores');csv(logs,'00_notebook_evidence')
    rng=np.random.default_rng(71015);plans={v:ks_plan(y,scores[v]) for v in [5,6,7]};boot=[]
    pos,neg=np.flatnonzero(y),np.flatnonzero(y==0)
    for i in range(400):
        idx=np.r_[rng.choice(pos,len(pos)),rng.choice(neg,len(neg))];w=np.bincount(idx,minlength=len(y));k={v:planned_ks(plans[v],w) for v in plans}
        boot.append(dict(iteration=i,v7_minus_v5=k[7]-k[5],v7_minus_v6=k[7]-k[6]))
    b=csv(boot,'00_paired_bootstrap');js({c:dict(delta=ks(y,scores[7])-ks(y,scores[v]),CI95=b[c].quantile([.025,.975]).tolist()) for v,c in [(5,'v7_minus_v5'),(6,'v7_minus_v6')]},'00_delta_summary')
    cor=pd.DataFrame({v:rankdata(s) for v,s in scores.items()}).corr();csv(cor.reset_index(),'00_version_rank_correlation')
    seg=[]
    for value in [0,1]:
        m=tr['has_history'].to_numpy()==value
        for v,s in scores.items():seg.append(dict(version=v,history=value,n=int(m.sum()),KS=ks(y[m],s[m])))
    csv(seg,'00_history_segments')
    fig,ax=plt.subplots(1,3,figsize=(15,4));q=z[z.model=='selected_blend'];ax[0].plot(q.version,q.KS,'o-');ax[0].set(xlabel='Version (v4 from log)',ylabel='Selected OOF KS')
    for c in b.columns[1:]:ax[1].hist(b[c],bins=30,histtype='step',label=c)
    ax[1].axvline(0,color='black');ax[1].legend();ax[1].set_title('Fixed-prediction paired resampling')
    ax[2].imshow(cor,vmin=.9,vmax=1);ax[2].set_xticks(range(len(cor)),cor.columns);ax[2].set_yticks(range(len(cor)),cor.columns);ax[2].set_title('Version rank correlation');figsave(fig,'00_trajectory')
    js({str(x.relative_to(ROOT)):digest(x) for x in [*[ROOT/f'notebook/v{v}.ipynb' for v in range(1,8)],ROOT/'results/v7/oof_all.parquet',ROOT/'results/v7/test_all.parquet',ROOT/'results/v7/submission.csv',ROOT/'data_pq/features.parquet']},'00_input_hashes')

if __name__=='__main__':main()
