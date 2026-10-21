"""Check real-data inference, then run the fixed five-fold relational experiment."""

if __name__ == "__main__":
    import gc
    import torch
    from research_relational054 import run

    print("RELATIONAL TECHNICAL PREFLIGHT", flush=True)
    run(preflight=True)
    gc.collect()
    torch.cuda.empty_cache()
    print("RELATIONAL FIVE-FOLD EVALUATION", flush=True)
    run()
