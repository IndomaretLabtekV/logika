# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory-efficient final-hop readout for SDM 0.2.1's InvariantGNN.

Adapted from sdm/models/kumo/relational/invariant_gnn.py. Keep all source nodes
and messages, but aggregate/project only the readout-table destinations on a
single-hop pass. Other hop counts use the original implementation. This changes
no weights, features, neighbors, labels, or random edge-type embeddings.
"""


def readout_forward(
    self,
    x,
    graph,
    *,
    readout_table,
    readout_index,
    num_hops,
    cache=None,
    generator=None
):
    import torch
    import torch.nn.functional as functional
    from sdm._kernels import segment_multi_reduce

    if num_hops != 1:
        return self._original_readout_forward(
            x,
            graph,
            readout_table=readout_table,
            readout_index=readout_index,
            num_hops=num_hops,
            cache=cache,
            generator=generator,
        )
    if cache is None or cache.is_recording:
        edge_type_emb = torch.randn(
            (graph.num_edge_types, self.edge_type_lin.weight.size(-1)),
            dtype=x.dtype,
            device=x.device,
            generator=generator,
        )
        edge_type_emb = self.edge_type_lin(functional.normalize(edge_type_emb, dim=-1))
        if cache is not None and cache.is_recording:
            cache["edge_type_emb"] = edge_type_emb
    else:
        edge_type_emb = cache["edge_type_emb"]
    start, end = (
        graph.start_node_offsets[readout_table],
        graph.end_node_offsets[readout_table],
    )
    # Offsets still refer to the complete sorted edge array. Slicing destination
    # offsets does not remove any incoming message to a readout node.
    stats = segment_multi_reduce(
        src=self.src_lin(x),
        index=graph.row,
        edge_attr=edge_type_emb,
        offsets=graph.colptr[start : end + 1],
        edge_type=graph.edge_type,
    )
    out = torch.addmm(
        self.skip_lin(x[start:end]), stats.flatten(1), self.aggregation_lin.weight.T
    )
    del stats
    out = functional.gelu(self.norm(out[readout_index]))
    return self.out_norm(self.out_lin(out))


def install_gnn(gnn):
    from types import MethodType

    assert not hasattr(gnn, "_original_readout_forward")
    gnn._original_readout_forward = gnn.forward
    gnn.forward = MethodType(readout_forward, gnn)


def install(model):
    for component in model.models.values():
        install_gnn(component.gnn)


def verify():
    import copy
    import torch
    from sdm.cache import Cache
    from sdm.models.kumo.relational.graph import HomogeneousGraph
    from sdm.models.kumo.relational.invariant_gnn import InvariantGNN

    torch.manual_seed(102026)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.1)
    device = "cuda"
    row = torch.randint(128, (600,), device=device)
    col, order = torch.randint(128, (600,), device=device).sort()
    graph = HomogeneousGraph(
        row=row[order],
        col=col,
        colptr=torch._convert_indices_from_coo_to_csr(col, 128),
        edge_type=torch.randint(6, (600,), device=device)[order],
        num_edge_types=6,
        start_node_offsets={"users": 30},
        end_node_offsets={"users": 70},
    )
    index = torch.tensor([0, 2, 2, 39, 1, 19], device=device)
    records = []
    for dtype in [torch.float32, torch.float16]:
        original = InvariantGNN(64, device=device).eval()
        for parameter in original.parameters():
            parameter.requires_grad_(False)
        optimized = copy.deepcopy(original)
        install_gnn(optimized)
        x = torch.randn(128, 64, device=device)
        for hops in [0, 1, 2]:
            caches = [Cache(), Cache()]
            for mode in ["uncached", "record", "replay"]:
                if mode == "replay":
                    for cache in caches:
                        cache.freeze()
                outputs = []
                with torch.inference_mode(), torch.amp.autocast(
                    "cuda", dtype=dtype, enabled=dtype == torch.float16
                ):
                    for model, cache in zip([original, optimized], caches):
                        outputs.append(
                            model(
                                x,
                                graph,
                                readout_table="users",
                                readout_index=index,
                                num_hops=hops,
                                cache=None if mode == "uncached" else cache,
                                generator=torch.Generator(device=device).manual_seed(
                                    314159
                                ),
                            )
                        )
                tolerance = 0.003 if dtype == torch.float16 else 2e-6
                torch.testing.assert_close(
                    outputs[0], outputs[1], rtol=tolerance, atol=tolerance
                )
                if hops and mode != "uncached":
                    assert torch.equal(
                        caches[0]["edge_type_emb"], caches[1]["edge_type_emb"]
                    )
                records.append(
                    dict(
                        dtype=str(dtype),
                        hops=hops,
                        mode=mode,
                        max_abs_difference=float((outputs[0] - outputs[1]).abs().max()),
                    )
                )
    return dict(
        passed=True,
        comparisons=records,
        nonzero_readout_offset_tested=True,
        duplicate_and_reordered_readout_indices_tested=True,
        cache_record_and_replay_tested=True,
        labels_used=False,
        caveat="Algebraically equivalent one-hop readout; FP16 GEMM rounding may differ with matrix row count.",
    )


if __name__ == "__main__":
    import json
    from pathlib import Path
    import research_goal054 as r

    result = verify()
    result["backend_sha256"] = r.checksum(Path(__file__))
    r.write_json(r.DEST / "relational_readout_parity.json", result)
    print(json.dumps(result), flush=True)
