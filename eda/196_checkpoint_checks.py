"""CPU parity checks before any GPU activation-checkpointing experiment."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sdm.models.kumo.tabular.model import _KumoTabular

spec=importlib.util.spec_from_file_location("checkpoint_probe",Path(__file__).with_name("195_kumo_checkpoint_preflight.py"))
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
torch.set_num_threads(1)


def tiny():
    torch.manual_seed(196)
    return _KumoTabular(num_classes=10,num_quantiles=0,cell_channels=8,num_embedding_layers=2,
          num_embedding_heads=2,num_inducing_points=4,group_size=1,num_frequencies=2,num_readout_tokens=1,
          icl_channels=8,num_icl_layers=2,num_icl_heads=2,num_icl_key_value_heads_for_query=1,device="cpu",dtype=torch.float32)


def inputs():
    generator=torch.Generator().manual_seed(196)
    return {"x":torch.randn(1,14,6,generator=generator),"y":torch.tensor([[0,1,0,1,0,0,1,0,1,0]]),
            "categorical_mask":torch.zeros(1,6,dtype=torch.bool)}


class Checks(unittest.TestCase):
    def test_real_small_kumo_outputs_gradients_and_sgd_match(self):
        a=tiny().train();b=copy.deepcopy(a);keys=list(a.state_dict());state=probe.enable_checkpointing(b)
        self.assertEqual(keys,list(b.state_dict()));self.assertEqual(len(state["modules"]),7)
        xa=a(**inputs());xb=b(**inputs());torch.testing.assert_close(xa,xb,rtol=0,atol=0)
        xa.square().mean().backward();xb.square().mean().backward()
        for (name,pa),(nameb,pb) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(name,nameb);self.assertEqual(pa.grad is None,pb.grad is None)
            if pa.grad is not None:torch.testing.assert_close(pa.grad,pb.grad,rtol=1e-5,atol=1e-6,msg=name)
        torch.optim.SGD(a.parameters(),lr=.01).step();torch.optim.SGD(b.parameters(),lr=.01).step()
        for name,p in a.state_dict().items():torch.testing.assert_close(p,b.state_dict()[name],rtol=1e-5,atol=1e-6,msg=name)
        self.assertTrue(any(count>=2 for count in state["compute_calls"].values()),"No activation recomputation occurred")

    def test_no_grad_inference_bypasses_checkpointing(self):
        a=tiny().eval();b=copy.deepcopy(a);state=probe.enable_checkpointing(b)
        with torch.no_grad():expected=a(**inputs());observed=b(**inputs())
        torch.testing.assert_close(expected,observed,rtol=0,atol=0)
        self.assertEqual(sum(state["compute_calls"].values()),0);self.assertGreater(state["bypass_calls"],0)

    def test_autocast_state_is_preserved_during_backward(self):
        a=tiny().train();b=copy.deepcopy(a);probe.enable_checkpointing(b)
        with torch.amp.autocast("cpu",dtype=torch.bfloat16):
            ya=a(**inputs());yb=b(**inputs())
        torch.testing.assert_close(ya,yb,rtol=0,atol=0)
        ya.float().square().mean().backward();yb.float().square().mean().backward()
        for (name,pa),(_,pb) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(pa.grad is None,pb.grad is None)
            if pa.grad is not None:torch.testing.assert_close(pa.grad,pb.grad,rtol=1e-4,atol=1e-6,msg=name)

    def test_duplicate_or_wrong_architecture_rejected(self):
        model=tiny();probe.enable_checkpointing(model)
        with self.assertRaises(ValueError):probe.enable_checkpointing(model)
        with self.assertRaises(ValueError):probe.enable_checkpointing(torch.nn.Linear(2,2))

    def test_recomputation_preserves_dropout_rng_and_gradients(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.row_embedding=torch.nn.Module()
                self.row_embedding.cell_embedding=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Dropout(.5),torch.nn.Linear(8,2))
            def forward(self,x):return self.row_embedding.cell_embedding(x)
        torch.manual_seed(197);a=Toy().train();b=copy.deepcopy(a);probe.enable_checkpointing(b);x=torch.ones(5,4)
        torch.manual_seed(123);ya=a(x);ya.square().sum().backward()
        torch.manual_seed(123);yb=b(x);yb.square().sum().backward()
        torch.testing.assert_close(ya,yb,rtol=0,atol=0)
        for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    out=Path(__file__).resolve().parent/"out"
    (out/"196_checkpoint_checks.json").write_text(json.dumps({"checks":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"wrapper_sha256":probe.sha(probe.__file__),"gpu_executed":False},indent=2))
    fig,ax=plt.subplots(figsize=(7,3));ax.bar(["pass","fail/error"],[result.testsRun-len(result.failures)-len(result.errors),len(result.failures)+len(result.errors)])
    ax.set(ylabel="CPU checks",title="Activation recomputation: numerical and RNG parity")
    fig.tight_layout();fig.savefig(out/"196_checkpoint_checks.png",dpi=130);plt.close(fig)
    raise SystemExit(not result.wasSuccessful())
