"""Bounded, falsifiable RNG probes; a failed probe does not identify a generator."""

from __future__ import annotations

import hashlib
import itertools
import time
import json
import argparse
import random
from pathlib import Path

import numpy as np


def probe_age_sequence(ages, seed_limit=10000):
    """Test uniform integer streams with a small fixed offset/stride hypothesis family."""
    target = np.asarray(ages, dtype=np.int64)
    assert target.min() == 23 and target.max() == 60
    seeds = list(range(seed_limit)) + [
        171327,
        20250101,
        20260101,
        20261010,
        20261013,
        20262026,
    ]
    prefix = target[:12]
    hits = []
    started = time.monotonic()
    for family in ["PCG64", "MT19937_RandomState"]:
        for seed in seeds:
            rng = (
                np.random.Generator(np.random.PCG64(seed))
                if family == "PCG64"
                else np.random.RandomState(seed)
            )
            drawn = (
                rng.integers(23, 61, size=512)
                if family == "PCG64"
                else rng.randint(23, 61, size=512)
            )
            for stride in range(1, 17):
                offsets = np.flatnonzero(
                    (drawn[:-stride] == prefix[0]) & (drawn[stride:] == prefix[1])
                )
                offsets = offsets[offsets + 11 * stride < len(drawn)]
                for offset in offsets:
                    if np.array_equal(drawn[offset + stride * np.arange(12)], prefix):
                        rng = (
                            np.random.Generator(np.random.PCG64(seed))
                            if family == "PCG64"
                            else np.random.RandomState(seed)
                        )
                        length = int(offset + stride * (len(target) - 1) + 1)
                        extended = (
                            rng.integers(23, 61, size=length)
                            if family == "PCG64"
                            else rng.randint(23, 61, size=length)
                        )
                        matched = (
                            extended[offset + stride * np.arange(len(target))] == target
                        )
                        hits.append(
                            {
                                "family": family,
                                "seed": seed,
                                "offset": int(offset),
                                "stride": stride,
                                "full_rows": len(target),
                                "full_match_fraction": float(matched.mean()),
                                "exact_full_sequence": bool(matched.all()),
                            }
                        )
        print("Completed RNG probe:", family, flush=True)
    return {
        "seed_count_per_family": len(seeds),
        "integer_support": [23, 60],
        "prefix_length": 12,
        "stream_length": 512,
        "strides": [1, 16],
        "families": ["PCG64", "MT19937_RandomState"],
        "hits": hits,
        "elapsed_seconds": time.monotonic() - started,
        "inferred_generator_seed": next(
            (h["seed"] for h in hits if h["exact_full_sequence"]), None
        ),
        "limitations": "Only direct uniform integer streams and fixed offsets/strides. Unknown interleaved calls, shuffled rows, other RNGs and seeds outside the tested set are not ruled out.",
    }


def probe_id_hashes(customer_ids):
    """Check common deterministic ID templates using no targets or model features."""
    observed = [str(s).split("_", 1)[1] for s in customer_ids[:3]]
    hashes = {
        "sha256_first16": lambda b: hashlib.sha256(b).hexdigest()[:16],
        "sha1_first16": lambda b: hashlib.sha1(b).hexdigest()[:16],
        "md5_first16": lambda b: hashlib.md5(b).hexdigest()[:16],
        "blake2b_8": lambda b: hashlib.blake2b(b, digest_size=8).hexdigest(),
    }
    matches, tested = [], 0
    for prefix, separator, start, seed in itertools.product(
        ["C", "c", "customer", "cust", "customer_id", "profile"],
        ["_", ":", "-", "/", ""],
        [0, 1],
        ["", "42", "2025", "2026", "20261010", "LOGIKA2026", "logika2026"],
    ):
        formats = [("prefix_index", lambda i: separator.join([prefix, str(i)]))]
        if seed:
            formats += [
                ("prefix_seed_index", lambda i: separator.join([prefix, seed, str(i)])),
                ("seed_prefix_index", lambda i: separator.join([seed, prefix, str(i)])),
                ("prefix_index_seed", lambda i: separator.join([prefix, str(i), seed])),
            ]
        for name, template in formats:
            for hash_name, hash_fn in hashes.items():
                tested += 1
                generated = [hash_fn(template(i + start).encode()) for i in range(3)]
                if generated == observed:
                    matches.append(
                        {
                            "prefix": prefix,
                            "separator": separator,
                            "index_start": start,
                            "seed_or_salt": seed,
                            "template": name,
                            "hash": hash_name,
                        }
                    )
    return {
        "templates_tested": tested,
        "confirmed_three_row_matches": matches,
        "id_model_features_used": False,
        "limitations": "Truncated hexadecimal IDs alone do not reveal an RNG or seed. Template search is intentionally bounded.",
    }


def probe_per_customer_rng(customer_ids, ages):
    ids = np.array(
        [int(str(s).split("_", 1)[1], 16) for s in customer_ids[:1000]], dtype=np.uint64
    )
    observed = np.asarray(ages[:1000])
    results = []
    for family in ["PCG64", "MT19937_RandomState"]:
        methods = {
            "id_low32": ids % 2**32,
            "id_high32": ids // 2**32,
            "profile_index_zero": np.arange(len(ids)),
            "profile_index_one": np.arange(len(ids)) + 1,
        }
        if family == "PCG64":
            methods["id_uint64"] = ids
        for method, seeds in methods.items():
            draws = []
            for seed in seeds:
                if family == "PCG64":
                    draws.append(
                        np.random.Generator(np.random.PCG64(int(seed))).integers(
                            23, 61, size=128
                        )
                    )
                else:
                    draws.append(
                        np.random.RandomState(int(seed)).randint(23, 61, size=128)
                    )
            rates = (np.vstack(draws) == observed[:, None]).mean(axis=0)
            best = int(np.argmax(rates))
            results.append(
                {
                    "family": family,
                    "seed_mapping": method,
                    "best_offset": best,
                    "age_match_fraction_at_best_offset": float(rates[best]),
                    "all_rows_match": bool(rates[best] == 1),
                }
            )
    return {
        "customers": len(ids),
        "offsets": [0, 127],
        "results": results,
        "null_single_offset_expected_match_fraction": 1 / 38,
        "max_offset_selection_inflates_null_match_rate": True,
        "id_values_used_only_for_diagnostic_seed_hypotheses": True,
    }


def probe_raw_identifier_stream(customer_ids, seed_limit=10000):
    """Test whether hex IDs are unmodified 64-bit outputs or 8 random bytes."""
    hexadecimal = [str(s).split("_", 1)[1] for s in customer_ids[:3]]
    targets = {
        "hex_integer_big_endian": np.array(
            [int(s, 16) for s in hexadecimal], dtype=np.uint64
        ),
        "bytes_little_endian": np.array(
            [int.from_bytes(bytes.fromhex(s), "little") for s in hexadecimal],
            dtype=np.uint64,
        ),
    }

    def draws(family, seed, n):
        if family in ["PCG64", "PCG64DXSM", "Philox", "SFC64"]:
            return getattr(np.random, family)(seed).random_raw(n)
        if family == "NumPy_RandomState_uint64":
            return np.random.RandomState(seed).randint(
                0, np.iinfo(np.uint64).max, size=n, dtype=np.uint64
            )
        if family == "NumPy_RandomState_bytes":
            return np.frombuffer(np.random.RandomState(seed).bytes(n * 8), dtype="<u8")
        rng = random.Random(seed)
        return np.fromiter(
            (rng.getrandbits(64) for _ in range(n)), dtype=np.uint64, count=n
        )

    families = [
        "PCG64",
        "PCG64DXSM",
        "NumPy_RandomState_uint64",
        "NumPy_RandomState_bytes",
        "Python_random_getrandbits64",
    ]
    seeds = list(range(seed_limit)) + [
        171327,
        20250101,
        20260101,
        20261010,
        20261013,
        20262026,
    ]
    candidates = []

    def inspect(family, seed, n):
        values = draws(family, seed, n)
        for encoding, target in targets.items():
            positions = np.flatnonzero(values == target[0])
            for position in positions:
                extended = values if n == 1000000 else draws(family, seed, 1000000)
                match_positions = [int(position)]
                for next_target in target[1:]:
                    matches = np.flatnonzero(extended == next_target)
                    matches = matches[matches > match_positions[-1]]
                    if not len(matches):
                        break
                    match_positions.append(int(matches[0]))
                candidates.append(
                    {
                        "family": family,
                        "seed": seed,
                        "encoding": encoding,
                        "raw_positions": match_positions,
                        "three_ordered_ids_reproduced": len(match_positions) == 3,
                    }
                )

    for family in families:
        for seed in seeds:
            inspect(family, seed, 512)
        print("Completed raw ID output probe:", family, flush=True)
    long_seeds = [
        0,
        1,
        7,
        17,
        42,
        123,
        1234,
        2025,
        2026,
        20261010,
        20260101,
        20261013,
        171327,
        9173,
        8247,
    ]
    for family in families + ["Philox", "SFC64"]:
        for seed in long_seeds:
            inspect(family, seed, 1000000)
    return {
        "families_prefix_probe": families,
        "seed_count_per_family": len(seeds),
        "prefix_raw_outputs": 512,
        "extra_long_stream_families": families + ["Philox", "SFC64"],
        "extra_long_stream_seeds": long_seeds,
        "long_stream_length": 1000000,
        "first_id_exact_matches": candidates,
        "any_three_ordered_id_match": any(
            r["three_ordered_ids_reproduced"] for r in candidates
        ),
        "limitations": "Only IDs formed directly from integer outputs or raw bytes. Hashes, UUIDs, separate cryptographic RNGs, other seeds and later call states are not ruled out.",
    }


def untemper_mt_word(value):
    """Undo the four MT19937 output-tempering operations on one 32-bit word."""

    def undo_right(y, shift):
        x = y
        for _ in range(32 // shift + 1):
            x = y ^ (x >> shift)
        return x & 0xFFFFFFFF

    def undo_left(y, shift, mask):
        x = y
        for _ in range(32 // shift + 1):
            x = y ^ ((x << shift) & mask)
        return x & 0xFFFFFFFF

    x = undo_right(int(value), 18)
    x = undo_left(x, 15, 0xEFC60000)
    x = undo_left(x, 7, 0x9D2C5680)
    return undo_right(x, 11)


def predict_mt_after_observed_words(words):
    assert len(words) > 624
    cloned = random.Random(0)
    cloned.setstate((3, tuple(untemper_mt_word(w) for w in words[:624]) + (624,), None))
    return [cloned.getrandbits(32) for _ in words[624:]]


def probe_mt_state_clone(customer_ids):
    """A seed-free falsification of consecutive raw-MT words encoded as 64-bit IDs."""
    control = random.Random(81721)
    for _ in range(37):
        control.getrandbits(32)
    control_words = [control.getrandbits(32) for _ in range(1200)]
    assert predict_mt_after_observed_words(control_words) == control_words[624:]
    numpy_words = (
        np.random.RandomState(231)
        .randint(0, 2**32, size=1200, dtype=np.uint32)
        .tolist()
    )
    assert predict_mt_after_observed_words(numpy_words) == numpy_words[624:]
    hexadecimal = [str(s).split("_", 1)[1] for s in customer_ids[:1000]]
    results = []
    for encoding in ["big", "little"]:
        ids = [int.from_bytes(bytes.fromhex(s), encoding) for s in hexadecimal]
        for word_order in ["low_high", "high_low"]:
            words = []
            for number in ids:
                pair = [number & 0xFFFFFFFF, number >> 32]
                words.extend(pair if word_order == "low_high" else pair[::-1])
            predicted = predict_mt_after_observed_words(words)
            observed = words[624:]
            matches = np.array(predicted, dtype=np.uint32) == np.array(
                observed, dtype=np.uint32
            )
            results.append(
                {
                    "encoding": encoding,
                    "word_order": word_order,
                    "fitting_words": 624,
                    "independent_test_words": len(observed),
                    "matching_words": int(matches.sum()),
                    "all_test_words_match": bool(matches.all()),
                }
            )
    return {
        "python_positive_control_passed": True,
        "numpy_positive_control_passed": True,
        "results": results,
        "any_consecutive_mt_stream_detected": any(
            r["all_test_words_match"] for r in results
        ),
        "limitations": "Requires consecutive unmodified 32-bit MT outputs. Hashing, interleaved calls, row permutation or another RNG defeats this particular test.",
    }


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-limit", type=int, default=10000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    profiles = pd.read_csv(root / "data/customer_profiles.csv")
    destination = root / "artifacts/reverse_engineering"
    destination.mkdir(parents=True, exist_ok=True)
    result = {
        "numpy_version": np.__version__,
        "age_stream": probe_age_sequence(profiles.age.to_numpy(), args.seed_limit),
        "id_templates": probe_id_hashes(profiles.customer_id.to_numpy()),
        "per_customer_seed_probe": probe_per_customer_rng(
            profiles.customer_id.to_numpy(), profiles.age.to_numpy()
        ),
        "original_generator_seed_recovered": False,
    }
    result["raw_identifier_stream_probe"] = probe_raw_identifier_stream(
        profiles.customer_id.to_numpy(), args.seed_limit
    )
    result["mt_state_clone_probe"] = probe_mt_state_clone(
        profiles.customer_id.to_numpy()
    )
    result["original_generator_seed_recovered"] = any(
        h["exact_full_sequence"] for h in result["age_stream"]["hits"]
    )
    (destination / "rng_probe.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
