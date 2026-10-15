"""Recover actual submission provenance and paired augmentation diagnostics."""
from shared8 import *

def main():
    f,tr,te,names,raw,test,w,s,res=saved8();y=tr['target'].to_numpy()
    csv([dict(model=n,weight=float(v),OOF_KS=ks(y,raw[n]),AUC=auc(y,raw[n])) for n,v in zip(names,w)],'00_models')
    base=np.mean([rankdata(raw[n])/len(y) for n in TREES],axis=0);aug=np.mean([rankdata(raw[n+'_aug'])/len(y) for n in TREES],axis=0)
    variants={'actual_submission':s,'base_trees':base,'aug_trees':aug,'safety_net':.8*base+.2*rankdata(raw['tabm'])/len(y)}
    # Fixed removal is explanatory, not re-optimization of the submitted blend.
    noaug=np.array([0 if n.endswith('_aug') else v for n,v in zip(names,w)]);noaug/=noaug.sum()
    variants['fixed_remove_aug']=np.column_stack([rankdata(raw[n])/len(y) for n in names])@noaug
    rows=[]
    for group,m in [('all',np.ones(len(y),bool)),('thin',tr['has_history'].to_numpy()==0),('history',tr['has_history'].to_numpy()==1)]:
        for n,a in variants.items():rows.append(dict(group=group,candidate=n,n=int(m.sum()),KS=ks(y[m],a[m])))
    z=csv(rows,'00_segments')
    pairs=[];rng=np.random.default_rng(81015);pos,neg=np.flatnonzero(y),np.flatnonzero(y==0)
    for n in [*TREES,'tabm']:
        a,b=raw[n],raw[n+'_aug'];pa,pb=ks_plan(y,a),ks_plan(y,b);d=[]
        for i in range(200):
            ix=np.r_[rng.choice(pos,len(pos)),rng.choice(neg,len(neg))];ww=np.bincount(ix,minlength=len(y));d.append(planned_ks(pb,ww)-planned_ks(pa,ww))
        pairs.append(dict(model=n,delta=ks(y,b)-ks(y,a),lo95=np.quantile(d,.025),hi95=np.quantile(d,.975)))
    q=csv(pairs,'00_paired_augmentation')
    audit=[]
    for path in ['notebook/v8.ipynb','results/v8/v8_run.ipynb']:
        nb=json.loads((ROOT/path).read_text())
        for i,c in enumerate(nb['cells']):
            if c['cell_type']!='code':continue
            src=''.join(c['source']);audit.append(dict(file=path,cell=i,execution_count=c.get('execution_count'),has_outputs=bool(c.get('outputs')),first_line=src.splitlines()[0] if src else ''))
    csv(audit,'00_execution_manifest')
    js(dict(submission_KS=ks(y,s),submission_AUC=auc(y,s),max_csv_reconstruction_error=res,augmented_weight=float(sum(v for n,v in zip(names,w) if n.endswith('_aug'))),public_reported=.56145,top_public_reported=.57493,missing_predictions=[n for n in ['exaone','realmlp','tabicl','tabm_p'] if n not in names],warning='CSV decomposition recovers existing coefficients without target labels. It does not establish how weights were selected or which later session wrote the CSV.'),'00_summary')
    tr.select('application_id','target').with_columns(blend=pl.Series(s)).write_parquet(OUT/'00_bound_oof.parquet')
    fig,ax=plt.subplots(1,3,figsize=(16,4));ax[0].barh(names,w);ax[0].set_title('Recovered actual CSV weights');ax[1].errorbar(q.delta,np.arange(len(q)),xerr=[q.delta-q.lo95,q.hi95-q.delta],fmt='o');ax[1].set_yticks(range(len(q)),q.model);ax[1].axvline(0,color='black');ax[1].set_title('Augmented minus base: paired KS')
    for n in ['base_trees','aug_trees','actual_submission']:
        a=z[z.candidate==n];ax[2].plot(a.group,a.KS,'o-',label=n)
    ax[2].legend(fontsize=7);figsave(fig,'00_audit')
    js({str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'notebook/v8.ipynb',ROOT/'results/v8/v8_run.ipynb',ROOT/'results/v8/submission.csv',ROOT/'data_pq/hist_labels.parquet',ROOT/'data_pq/hist_feats.parquet']},'00_hashes')

if __name__=='__main__':main()
