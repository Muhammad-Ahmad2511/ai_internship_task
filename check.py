#!/usr/bin/env python3
"""Compare a results file with a reference using the brief's pass rules (section 5).

    python check.py --ref reference/*.jsonl --pred results.jsonl

--ref takes one or more files (they are combined).
Only videos that appear in the reference are scored, so --pred can cover more videos.
A reference second missing from --pred counts as "no objects, 0 people", unless
--common is given, which scores only the videos present in both files.
Pass rules: objects F1 >= 0.95, people exact >= 85%, people within 1 >= 95%.
"""

import argparse
import gzip
import json
import sys
from collections import defaultdict

NAMES = ["person", "car", "dog", "laptop", "cell phone"]


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    seconds, errors = {}, defaultdict(list)
    with opener(path, "rt", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "error" in rec:
                errors[rec["video"]].append(rec["error"])
                continue
            key = (rec["video"], rec["second"])
            if key in seconds:
                print(f"warning: duplicate line for {key} in {path} (line {n})", file=sys.stderr)
            seconds[key] = (frozenset(rec["objects"]), rec["people"])
    return seconds, errors


def score(ref, pred, keys):
    tp = fp = fn = exact = within1 = missing = 0
    per_class = {c: [0, 0, 0] for c in NAMES}   # tp, fp, fn
    for k in keys:
        r_obj, r_people = ref[k]
        if k in pred:
            p_obj, p_people = pred[k]
        else:
            p_obj, p_people = frozenset(), 0
            missing += 1
        for c in NAMES:
            a, b = c in r_obj, c in p_obj
            if a and b:
                tp += 1; per_class[c][0] += 1
            elif b:
                fp += 1; per_class[c][1] += 1
            elif a:
                fn += 1; per_class[c][2] += 1
        exact += p_people == r_people
        within1 += abs(p_people - r_people) <= 1
    n = len(keys)
    f1 = 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    return {
        "seconds": n, "missing": missing, "f1": f1,
        "exact": exact / n if n else 1.0, "within1": within1 / n if n else 1.0,
        "tp": tp, "fp": fp, "fn": fn, "per_class": per_class,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, nargs="+", help="one or more reference files")
    p.add_argument("--pred", required=True)
    p.add_argument("--per-video", action="store_true", help="also print scores for each video")
    p.add_argument("--common", action="store_true",
                   help="score only videos that appear in both the reference and --pred")
    args = p.parse_args()

    ref, ref_err = {}, defaultdict(list)
    for path in args.ref:
        r, e = load(path)
        overlap = ref.keys() & r.keys()
        if overlap:
            print(f"warning: {len(overlap)} seconds in {path} already came from another --ref file "
                  f"(the later file wins)", file=sys.stderr)
        ref.update(r)
        for v, msgs in e.items():
            ref_err[v].extend(msgs)
    pred, pred_err = load(args.pred)
    ref_videos = {v for v, _ in ref} | set(ref_err)
    if args.common:
        pred_videos = {v for v, _ in pred} | set(pred_err)
        skipped = ref_videos - pred_videos
        ref_videos &= pred_videos
        ref = {k: v for k, v in ref.items() if k[0] in ref_videos}
        if skipped:
            print(f"--common: skipping {len(skipped)} reference videos not in --pred", file=sys.stderr)
    if ref_err:
        print(f"note: reference has error lines for {len(ref_err)} videos: {dict(ref_err)}", file=sys.stderr)
    keys = sorted(ref)
    extra = sorted(k for k in pred if k[0] in ref_videos and k not in ref)

    s = score(ref, pred, keys)
    ok = s["f1"] >= 0.95 and s["exact"] >= 0.85 and s["within1"] >= 0.95
    print(f"videos: {len(ref_videos)}   seconds: {s['seconds']}   "
          f"missing in pred: {s['missing']}   extra in pred: {len(extra)}")
    print(f"objects F1     : {s['f1']:.4f}   (need >= 0.95)   tp={s['tp']} fp={s['fp']} fn={s['fn']}")
    print(f"people exact   : {s['exact']:.4f}   (need >= 0.85)")
    print(f"people within 1: {s['within1']:.4f}   (need >= 0.95)")
    print("per class (tp/fp/fn): " + ", ".join(f"{c} {v[0]}/{v[1]}/{v[2]}" for c, v in s["per_class"].items()))
    errs = {v: e for v, e in pred_err.items() if v in ref_videos}
    if errs:
        print(f"error lines in pred for reference videos: {errs}")
    if args.per_video:
        by_video = defaultdict(list)
        for k in keys:
            by_video[k[0]].append(k)
        for v, ks in sorted(by_video.items()):
            sv = score(ref, pred, ks)
            print(f"  {v}: n={sv['seconds']} F1={sv['f1']:.3f} exact={sv['exact']:.3f} within1={sv['within1']:.3f}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
