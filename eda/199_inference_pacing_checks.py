"""CPU proof that inference pacing preserves outputs, state and RNG."""
import copy
import json
from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from inference_pacing import pace_model
from sdm.models.kumo.tabular.model import _KumoTabular

torch.set_num_threads(1)


class Estimator(torch.nn.Module):
    def __init__(self):
        super().__init__();torch.manual_seed(199)
        self.net=_KumoTabular(num_classes=10,num_quantiles=0,cell_channels=8,num_embedding_layers=2,num_embedding_heads=2,
            num_inducing_points=4,group_size=1,num_frequencies=2,num_readout_tokens=1,icl_channels=8,num_icl_layers=2,
            num_icl_heads=2,num_icl_key_value_heads_for_query=1,device="cpu",dtype=torch.float32)
    def fit(self,**kwargs):return self.net(**kwargs)
    def predict(self,**kwargs):return self.net(**kwargs)


def inputs():
    return {"x":torch.arange(84,dtype=torch.float32).reshape(1,14,6)/84,"y":torch.tensor([[0,1,0,1,0,0,1,0,1,0]]),"categorical_mask":torch.zeros(1,6,dtype=torch.bool)}


class Checks(unittest.TestCase):
    def test_fit_predict_outputs_parameters_and_rng_unchanged(self):
        a=Estimator().eval();b=copy.deepcopy(a);delays=[];events=[]
        state=pace_model(b,synchronize=lambda:None,sleep=delays.append,trace=events.append)
        with torch.no_grad():
            rng=torch.get_rng_state().clone();fa=a.fit(**inputs());fb=b.fit(**inputs())
            torch.testing.assert_close(fa,fb,rtol=0,atol=0)
            torch.testing.assert_close(a.predict(**inputs()),b.predict(**inputs()),rtol=0,atol=0)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        # With no recording cache, each ICL layer runs context and query GQA
        #branches:5embedding calls+2x2ICL calls, while there are7unique blocks.
        self.assertEqual(state["fit_blocks"],9);self.assertEqual(state["query_chunks"],1)
        self.assertEqual(delays,[.25]*9+[.1]);self.assertEqual(state["phase"],"idle")
        self.assertEqual(sum(e["event"]=="block_start" for e in events),9)
        self.assertEqual(sum(e["event"]=="block_complete" for e in events),9)
        for name,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[name],rtol=0,atol=0)

    def test_duplicate_and_invalid_architecture_rejected(self):
        model=Estimator();pace_model(model,synchronize=lambda:None,sleep=lambda _:None)
        with self.assertRaises(ValueError):pace_model(model)
        with self.assertRaises(ValueError):pace_model(torch.nn.Linear(2,2))


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    import hashlib
    source=Path(__file__).with_name("inference_pacing.py")
    out=Path(__file__).resolve().parent/"out"
    (out/"199_inference_pacing_checks.json").write_text(json.dumps({"checks":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"pacing_source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"gpu_executed":False},indent=2))
    fig,ax=plt.subplots(figsize=(7,3));ax.bar(["pass","fail/error"],[result.testsRun-len(result.failures)-len(result.errors),len(result.failures)+len(result.errors)])
    ax.set(title="Inference pacing: exact CPU output/state/RNG parity",ylabel="CPU checks")
    fig.tight_layout();fig.savefig(out/"199_inference_pacing_checks.png",dpi=130);plt.close(fig)
    raise SystemExit(not result.wasSuccessful())
