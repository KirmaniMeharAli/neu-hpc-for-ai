"""
Week 7: Verify Determinism of Test Case Generation

Runs each test generation script TWICE and verifies that the outputs
are bit-for-bit identical. This is a REQUIRED step per the assignment.
"""

import os
import json
import sys
import hashlib
import importlib.util
import shutil


def load_json(filepath):
    """Load a JSON test case file."""
    with open(filepath, "r") as f:
        return json.dumps(json.load(f), sort_keys=True)


def get_file_hash(filepath):
    """Get SHA-256 hash of a file."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_generator(script_path, test_cases_dir):
    """Run a test generation script by importing and calling main()."""
    spec = importlib.util.spec_from_file_location("generator", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def verify_determinism():
    """Run all generators twice and compare outputs."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_cases_dir = os.path.join(script_dir, "..", "test_cases")
    backup_dir = os.path.join(script_dir, "..", "test_cases_run1")

    generators = [
        "generate_mlp_tests.py",
        "generate_router_tests.py",
        "generate_experts_tests.py",
        "generate_moe_full_tests.py",
    ]

    all_passed = True

    for gen_script in generators:
        script_path = os.path.join(script_dir, gen_script)
        if not os.path.exists(script_path):
            print(f"WARNING: {gen_script} not found, skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"Verifying determinism: {gen_script}")
        print(f"{'=' * 60}")

        # --- Run 1 ---
        print(f"\n  Run 1...")
        run_generator(script_path, test_cases_dir)

        # Backup run 1 results
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.copytree(test_cases_dir, backup_dir)

        # Get hashes for run 1
        run1_hashes = {}
        for fname in os.listdir(backup_dir):
            if fname.endswith(".json"):
                run1_hashes[fname] = get_file_hash(os.path.join(backup_dir, fname))

        # --- Run 2 ---
        print(f"  Run 2...")
        run_generator(script_path, test_cases_dir)

        # Compare
        run2_hashes = {}
        for fname in os.listdir(test_cases_dir):
            if fname.endswith(".json"):
                run2_hashes[fname] = get_file_hash(os.path.join(test_cases_dir, fname))

        # Check that all run1 files match run2
        for fname in run1_hashes:
            if fname not in run2_hashes:
                print(f"  FAIL: {fname} missing from run 2!")
                all_passed = False
            elif run1_hashes[fname] != run2_hashes[fname]:
                print(f"  FAIL: {fname} differs between runs!")
                # Show detailed diff
                run1_data = load_json(os.path.join(backup_dir, fname))
                run2_data = load_json(os.path.join(test_cases_dir, fname))
                if run1_data != run2_data:
                    print(f"    Content differs (not just formatting)")
                all_passed = False
            else:
                print(f"  PASS: {fname} is deterministic (hash: {run1_hashes[fname][:16]}...)")

    # Cleanup backup
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)

    print(f"\n{'=' * 60}")
    if all_passed:
        print("ALL DETERMINISM CHECKS PASSED!")
    else:
        print("SOME CHECKS FAILED - Fix non-determinism before submitting!")
    print(f"{'=' * 60}")

    return all_passed


if __name__ == "__main__":
    success = verify_determinism()
    sys.exit(0 if success else 1)
