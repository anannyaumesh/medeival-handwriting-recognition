#!/usr/bin/env python3
"""
ROVER ensemble with strict guards against hallucination.
"""

import json
from pathlib import Path
from collections import Counter
import numpy as np
from Levenshtein import distance as edit_distance
import zipfile

OUTPUT_DIR = Path("/mnt/hdd2/rainfall_project/code/gsmap/proj/test/submissions/rover_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEBUG_LINES = 10


def load_preds(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_preds(preds, name):
    out = OUTPUT_DIR / f"{name}.json"
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
    
    zp = OUTPUT_DIR / f"{name}.zip"
    with zipfile.ZipFile(zp, 'w') as zf:
        zf.write(out, arcname='predictions.json')
    
    print(f"Saved: {out}")
    return out


# ============================================================
# ROVER CORE (character-level alignment + voting)
# ============================================================

def levenshtein_alignment(ref, hyp):
    """Align two strings character-by-character using edit distance."""
    m, n = len(ref), len(hyp)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    # Backtrace
    i, j = m, n
    aligned_ref = []
    aligned_hyp = []

    while i > 0 or j > 0:
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned_ref.append(ref[i - 1])
            aligned_hyp.append("")
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            aligned_ref.append("")
            aligned_hyp.append(hyp[j - 1])
            j -= 1
        else:
            aligned_ref.append(ref[i - 1])
            aligned_hyp.append(hyp[j - 1])
            i -= 1
            j -= 1

    return aligned_ref[::-1], aligned_hyp[::-1]


def rover_merge_two(s1, s2):
    """ROVER merge two strings with strict guards."""
    if s1 == s2:
        return s1
    
    ref, hyp = levenshtein_alignment(s1, s2)
    
    result = []
    for r, h in zip(ref, hyp):
        if r == h:
            result.append(r)
        elif r == "":
            result.append(h)
        elif h == "":
            result.append(r)
        else:
            # Disagreement: prefer the primary (first) model
            result.append(r)
    
    merged = "".join(result).strip()
    return merged


def safe_rover(strings, primary_idx=0):
    """
    Smart ROVER: only merge when safe, otherwise fall back to primary.
    
    Rules:
    1. If all agree → return that
    2. If two models and they're similar (<20% edit distance) → ROVER merge
    3. If two models and they're different → pick primary
    4. If 3+ models → majority vote on whole strings, ROVER only on close pairs
    """
    # All agree
    if len(set(strings)) == 1:
        return strings[0], "agree"
    
    primary = strings[primary_idx]
    
    if len(strings) == 2:
        s1, s2 = strings[0], strings[1]
        max_len = max(len(s1), len(s2))
        
        if max_len == 0:
            return primary, "empty"
        
        ed = edit_distance(s1, s2)
        ratio = ed / max_len
        
        # Too different → just use primary
        if ratio > 0.20:
            return primary, "too_different"
        
        # Similar enough → ROVER merge with guards
        merged = rover_merge_two(s1, s2)
        
        # Length guard: merged shouldn't be longer than longest input + 2
        if len(merged) > max(len(s1), len(s2)) + 2:
            return primary, "length_guard"
        
        # Word count guard
        wc_primary = len(primary.split())
        wc_merged = len(merged.split())
        if abs(wc_merged - wc_primary) > 1:
            return primary, "wordcount_guard"
        
        return merged, "rover_2way"
    
    else:
        # 3+ models: try majority vote first
        vote = Counter(strings)
        winner, count = vote.most_common(1)[0]
        
        if count >= 2:
            # Majority exists — use it
            return winner, "majority"
        
        # No majority — all different. Try ROVER on the two closest.
        # Find the two most similar predictions
        best_pair = (0, 1)
        best_dist = float('inf')
        
        for i in range(len(strings)):
            for j in range(i + 1, len(strings)):
                d = edit_distance(strings[i], strings[j])
                if d < best_dist:
                    best_dist = d
                    best_pair = (i, j)
        
        s1, s2 = strings[best_pair[0]], strings[best_pair[1]]
        max_len = max(len(s1), len(s2))
        
        if max_len == 0:
            return primary, "empty"
        
        ratio = best_dist / max_len
        
        if ratio > 0.25:
            # Even closest pair is too different → use primary
            return primary, "all_different"
        
        # ROVER merge the closest pair
        merged = rover_merge_two(s1, s2)
        
        # Guards
        if len(merged) > max(len(s1), len(s2)) + 2:
            return primary, "length_guard_3way"
        
        wc_primary = len(primary.split())
        wc_merged = len(merged.split())
        if abs(wc_merged - wc_primary) > 1:
            return primary, "wordcount_guard_3way"
        
        return merged, "rover_3way"


def run_rover(pred_dicts, name, debug=True):
    """Run safe ROVER across all lines."""
    keys = list(pred_dicts[0].keys())
    out = {}
    
    stats = Counter()
    changed = 0
    printed = 0
    
    for k in keys:
        strings = [p.get(k, "") for p in pred_dicts]
        merged, reason = safe_rover(strings, primary_idx=0)
        out[k] = merged
        stats[reason] += 1
        
        if merged != strings[0]:
            changed += 1
            if debug and printed < DEBUG_LINES:
                print(f"\n{'='*80}")
                print(f"LINE: {k} ({reason})")
                for i, s in enumerate(strings):
                    print(f"  MODEL {i+1}: {s[:80]}")
                print(f"  ROVER:   {merged[:80]}")
                printed += 1
    
    print(f"\nStats: {dict(stats)}")
    print(f"Changed: {changed}/{len(keys)} ({100*changed/len(keys):.1f}%)")
    
    save_preds(out, name)
    return out


# ============================================================
# EXPERIMENTS
# ============================================================

def task1():
    print("\n" + "=" * 60)
    print("TASK 1")
    print("=" * 60)

    preds = {
        "tta": load_preds("/mnt/hdd2/rainfall_project/code/gsmap/proj/test/submissions/task1_tta_beam8_spacing.json"),
        "catmus": load_preds("/mnt/hdd2/rainfall_project/code/gsmap/proj/test/submissions/catmus_task_1_predictions.json"),
        "custombpe": load_preds("/mnt/hdd2/rainfall_project/code/gsmap/proj/test/submissions/custombpe_task1.json"),
    }

    print("\nEXP1: TTA + BPE")
    run_rover([preds["tta"], preds["custombpe"]], "task1_tta_bpe")

    print("\nEXP2: TTA + CATMUS")
    run_rover([preds["tta"], preds["catmus"]], "task1_tta_catmus")

    print("\nEXP3: TTA + BPE + CATMUS")
    run_rover([preds["tta"], preds["custombpe"], preds["catmus"]], "task1_all")


def task2():
    print("\n" + "=" * 60)
    print("TASK 2")
    print("=" * 60)

    preds = {
        "tta": load_preds("/mnt/hdd2/rainfall_project/code/gsmap/proj/test/submissions/task2_predictions_tta.json"),
        "catmus": load_preds("/mnt/hdd2/rainfall_project/code/gsmap/proj/test/submissions/catmus_task_2_predictions.json"),
        "custombpe": load_preds("/mnt/hdd2/rainfall_project/code/gsmap/proj/test/submissions/custombpe_task2.json"),
    }

    print("\nEXP5: TTA + BPE")
    run_rover([preds["tta"], preds["custombpe"]], "task2_tta_bpe")

    print("\nEXP6: TTA + CATMUS")
    run_rover([preds["tta"], preds["catmus"]], "task2_tta_catmus")

    print("\nEXP8: TTA + BPE + CATMUS")
    run_rover([preds["tta"], preds["custombpe"], preds["catmus"]], "task2_all")


def task3():
    print("\n" + "=" * 60)
    print("TASK 3")
    print("=" * 60)

    preds = {
        "catmus": load_preds("/mnt/hdd2/rainfall_project/code/gsmap/proj/test/submissions/catmus_task_3_predictions.json"),
        "5model": load_preds("/mnt/hdd2/rainfall_project/code/gsmap/proj/test/submissions/task3_ensemble_5models.json"),
    }

    print("\nEXP9: CATMUS + 5MODEL")
    run_rover([preds["catmus"], preds["5model"]], "task3_catmus_5model")


if __name__ == "__main__":
    task1()
    task2()
    task3()
    print("\nDONE.")