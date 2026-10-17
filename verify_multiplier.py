#!/usr/bin/env python3
"""
Verification script for the Nano Transformer Multiplier leaderboard (MultiplierBoard).

Tests a model's ability to multiply two 10-digit numbers (producing a 20-digit product).
A model must achieve >= 99% accuracy on 10,010 test pairs (10 edge cases + 10,000 random
pairs with seed=2025) to qualify.
"""

import argparse
import importlib.util
import random
import time


def load_submission(path: str):
    spec = importlib.util.spec_from_file_location("submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "build_model"):
        raise ValueError("Submission must define build_model() -> (model, metadata)")
    if not hasattr(mod, "multiply"):
        raise ValueError("Submission must define multiply(model, a, b) -> int")

    return mod


def run_test(mod, num_tests=10000, seed=2025):
    model, metadata = mod.build_model()

    print(f"Model: {metadata.get('name', 'unnamed')}")
    print(f"Author: {metadata.get('author', 'unknown')}")
    print(f"Parameters (unique): {metadata.get('params', '?')}")
    print(f"Architecture: {metadata.get('architecture', '?')}")
    print(f"Tricks: {', '.join(metadata.get('tricks', []))}")
    print()

    edge_cases = [
        (0, 0),
        (0, 1),
        (9_999_999_999, 0),
        (9_999_999_999, 1),
        (9_999_999_999, 9_999_999_999),
        (5_000_000_000, 5_000_000_000),
        (1_111_111_111, 8_888_888_889),
        (1_234_567_890, 9_876_543_210),
        (9_876_543_210, 9_876_543_210),
        (3_141_592_653, 2_718_281_828),
    ]

    rng = random.Random(seed)
    random_cases = [
        (rng.randint(0, 9_999_999_999), rng.randint(0, 9_999_999_999))
        for _ in range(num_tests)
    ]

    all_cases = edge_cases + random_cases
    total = len(all_cases)
    passed = 0
    failures = []

    start = time.time()
    for i, (a, b) in enumerate(all_cases):
        expected = a * b
        try:
            result = mod.multiply(model, a, b)
        except Exception as e:
            failures.append((a, b, expected, f"ERROR: {e}"))
            continue

        if result == expected:
            passed += 1
        else:
            failures.append((a, b, expected, result))

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start
            print(f"  Progress: {i+1}/{total} ({passed}/{i+1} correct) [{elapsed:.1f}s]")

    elapsed = time.time() - start
    accuracy = passed / total * 100
    qualified = accuracy >= 99.0

    print()
    print(f"Results: {passed}/{total} correct ({accuracy:.2f}%)")
    print(f"Time: {elapsed:.1f}s ({total/elapsed:.0f} multiplications/sec)")
    print(f"Status: {'QUALIFIED' if qualified else 'NOT QUALIFIED'} (threshold: 99%)")

    if failures:
        print(f"\nFirst 10 failures (of {len(failures)}):")
        for a, b, expected, got in failures[:10]:
            print(f"  {a} * {b} = {expected}, got {got}")

    return {
        "passed": passed,
        "total": total,
        "accuracy": accuracy,
        "qualified": qualified,
        "time": elapsed,
        "metadata": metadata,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", help="Path to submission .py file")
    parser.add_argument("--num-tests", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    mod = load_submission(args.submission)
    run_test(mod, num_tests=args.num_tests, seed=args.seed)
