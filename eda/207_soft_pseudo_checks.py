"""Check prior preservation and isolation of unused labels in CPU soft SSL."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('soft_ssl_checks',Path(__file__).with_name('206_cpu_capped_union_soft_ssl.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Checks(unittest.TestCase):
    def test_pseudo_prior_weight_and_unique_rows(self):
        p=np.linspace(.001,.99,30000)
        for prior in (.08,.135,.25):
            ix,soft,w,info=module.pseudo(p,prior)
            self.assertEqual(len(set(ix)),2000);self.assertEqual(len(ix),len(soft));self.assertTrue((w>0).all())
            self.assertAlmostEqual(w.sum(),500,places=10);self.assertAlmostEqual(np.average(soft,weights=w),prior,places=12)

    def test_no_pseudo_from_uninformative_teacher(self):
        with self.assertRaisesRegex(ValueError,'bracket'):module.pseudo(np.full(30000,.13),.135)

    def test_held_out_labels_cannot_enter_student(self):
        F=pd.DataFrame({'x':np.linspace(0,1,30200)});real=np.arange(100);test=np.arange(200,30200)
        labels=np.zeros(30200);labels[real[:14]]=1
        class Teacher:
            def predict(self,*args,**kwargs):return np.linspace(.001,.99,30000)
        observed=[]
        def capture(p,dataset,**kwargs):
            observed.append((dataset.data.copy(),np.asarray(dataset.label).copy(),np.asarray(dataset.weight).copy(),kwargs));return object()
        with patch.object(module.lgb,'train',side_effect=capture):
            module.fit_student(F,labels,real,test,Teacher(),23,42)
            labels[100:]=1-labels[100:]
            module.fit_student(F,labels,real,test,Teacher(),23,42)
        self.assertEqual(len(observed),2)
        for a,b in zip(observed[0][:3],observed[1][:3]):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(observed[0][1][:100],labels[real]);self.assertEqual(observed[0][3]['num_boost_round'],23)

    def test_vocabulary_uses_only_fitting_rows(self):
        F=pd.DataFrame({c:['known','known','unseen'] for c in module.v.CAT_COLS})
        G=module.vocab(F,np.array([0,1]))
        for c in module.v.CAT_COLS:
            self.assertEqual(list(G[c].cat.categories),['known']);self.assertTrue(pd.isna(G[c].iloc[2]))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    out=Path(__file__).resolve().parent/'out';report={'checks':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'gpu_used':False}
    (out/'207_soft_pseudo_checks.json').write_text(json.dumps(report,indent=2));fig,ax=plt.subplots(figsize=(6,3));ax.bar(['pass','fail/error'],[result.testsRun-len(result.failures)-len(result.errors),len(result.failures)+len(result.errors)]);ax.set(title='Soft-SSL prior and label isolation');fig.tight_layout();fig.savefig(out/'207_soft_pseudo_checks.png',dpi=130);plt.close(fig)
    raise SystemExit(not result.wasSuccessful())
