#!/usr/bin/env python3
"""Video indexing at speed.

For every second of every video stored in FineVideo-style Parquet files, record
which of five COCO objects (person, car, dog, laptop, cell phone) YOLO11l finds
and how many people are visible.

Required usage (grading):
    python run.py --input <folder of .parquet files> --output results.jsonl

Optional extras (used to process the 50 shards on free Kaggle GPUs):
    --hf-stream          download shards one at a time from Hugging Face instead of --input
    --checkpoint-dir D   keep one finished file per shard in D and skip finished shards on resume
    --max-hours H        stop cleanly before H hours (unfinished shards are redone next run)
    --max-shards N       only the first N shards (testing)
    --max-videos N       only the first N videos of each shard (testing)
    --devices 0,1        one worker process per GPU (default: all visible GPUs)
    --sample-fps N       run the model on N frames per second instead of every frame
    --half               FP16 inference
"""

import argparse
import fnmatch
import gzip
import io
import json
import math
import os
import queue
import shutil
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixed task settings (section 3 of the brief)
# ---------------------------------------------------------------------------
TARGETS = {0: "person", 2: "car", 16: "dog", 63: "laptop", 67: "cell phone"}  # COCO ids
CLASS_IDS = list(TARGETS)
NAMES = [TARGETS[c] for c in CLASS_IDS]   # output order: person, car, dog, laptop, cell phone
COL = {c: i for i, c in enumerate(CLASS_IDS)}
PERSON_ID = 0
IMGSZ = 640
CONF = 0.4
IOU = 0.7          # Ultralytics default, written out explicitly

SCRIPT_DIR = Path(__file__).resolve().parent


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def short(e):
    s = f"{type(e).__name__}: {e}"
    return s[:200]


class TimeBudgetExceeded(Exception):
    pass


class _StopShard(Exception):
    """--max-videos reached (testing only)."""


# ---------------------------------------------------------------------------
# Per-second aggregation (section 5 of the brief)
# ---------------------------------------------------------------------------
def median_floor(values):
    """Median of integers, rounded down when it falls halfway between two numbers."""
    v = sorted(values)
    n = len(v)
    if n == 0:
        return 0
    if n % 2:
        return v[n // 2]
    return (v[n // 2 - 1] + v[n // 2]) // 2


def second_record(video_id, second, agg):
    """agg = [n_frames, per-class frame counts (list of 5), per-frame people counts]."""
    if agg is None or agg[0] == 0:
        return {"video": video_id, "second": second, "objects": [], "people": 0}
    n, cls_frames, people = agg
    objects = [NAMES[i] for i in range(len(NAMES)) if 2 * cls_frames[i] >= n]  # >= half the frames
    return {"video": video_id, "second": second, "objects": objects, "people": median_floor(people)}


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------
def rotation_cw(stream, frame):
    """Degrees (0/90/180/270) the frame must be turned clockwise to be shown upright."""
    r = getattr(frame, "rotation", None)  # PyAV >= 14: counter-clockwise angle of the display matrix
    if r:
        return int(round(-r)) % 360
    try:
        tag = stream.metadata.get("rotate")  # older files / older FFmpeg keep it as a tag
        if tag:
            return int(float(tag)) % 360
    except Exception:
        pass
    return 0


def apply_rotation(img, cw):
    import numpy as np
    if cw == 90:
        img = np.rot90(img, k=-1)
    elif cw == 180:
        img = np.rot90(img, k=2)
    elif cw == 270:
        img = np.rot90(img, k=1)
    else:
        return img
    return np.ascontiguousarray(img)


# ---------------------------------------------------------------------------
# One video
# ---------------------------------------------------------------------------
def index_video(data, video_id, model, args, timers, deadline):
    """Decode one mp4 (bytes), run the model on every frame (or a sample), return output records."""
    import av

    secs = {}            # second -> [n_frames, [5 class counts], [people per frame]]
    pending_imgs, pending_secs = [], []
    error = None
    n_frames = 0         # frames decoded
    n_infer = 0          # frames sent to the model
    max_sec = -1
    rot_seen = 0
    last_slot = None

    def flush():
        if not pending_imgs:
            return
        t = time.perf_counter()
        results = model.predict(
            pending_imgs, imgsz=IMGSZ, conf=CONF, iou=IOU, classes=CLASS_IDS,
            half=args.half, device=args.device, verbose=False,
        )
        for s, r in zip(pending_secs, results):
            cls = r.boxes.cls.int().tolist() if r.boxes is not None else []
            agg = secs.get(s)
            if agg is None:
                agg = secs[s] = [0, [0] * len(NAMES), []]
            agg[0] += 1
            for c in set(cls):
                agg[1][COL[c]] += 1
            agg[2].append(cls.count(PERSON_ID))
        pending_imgs.clear()
        pending_secs.clear()
        timers["infer"] += time.perf_counter() - t

    container = None
    try:
        container = av.open(io.BytesIO(data), mode="r")
        if not container.streams.video:
            raise ValueError("no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"          # multi-threaded decoding, same frames
        tb = stream.time_base
        rate = stream.average_rate or stream.guessed_rate or 30
        t0 = None
        frames = container.decode(stream)
        while True:
            t = time.perf_counter()
            try:
                frame = next(frames)
            except StopIteration:
                timers["decode"] += time.perf_counter() - t
                break
            # timestamp, counted from the first decoded frame
            if frame.pts is not None and tb is not None:
                ts = Fraction(frame.pts) * tb
            else:
                ts = Fraction(n_frames) / Fraction(rate)
            if t0 is None:
                t0 = ts
            rel = ts - t0
            sec = max(0, math.floor(rel))
            n_frames += 1
            max_sec = max(max_sec, sec)     # every decoded frame counts for the last second
            if deadline and n_frames % 256 == 0 and time.time() > deadline:
                raise TimeBudgetExceeded()

            if args.sample_fps:
                # split time into 1/sample_fps slots; run the model on the first frame of each slot
                slot = math.floor(rel * args.sample_fps)
                if slot == last_slot:
                    timers["decode"] += time.perf_counter() - t
                    continue
                last_slot = slot

            cw = rotation_cw(stream, frame)
            img = apply_rotation(frame.to_ndarray(format="bgr24"), cw)   # BGR, upright
            timers["decode"] += time.perf_counter() - t
            rot_seen = rot_seen or cw

            pending_imgs.append(img)
            pending_secs.append(sec)
            n_infer += 1
            if len(pending_imgs) >= args.batch:
                flush()
    except TimeBudgetExceeded:
        raise
    except Exception as e:  # damaged / unusual video: keep what we have, add an error line
        error = f"could not decode video ({short(e)})" if n_frames == 0 else f"decode failed after {n_frames} frames ({short(e)})"
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass

    try:
        flush()
    except Exception as e:
        error = error or f"inference failed ({short(e)})"
        pending_imgs.clear()

    records = []
    if n_frames == 0:
        records.append({"video": video_id, "error": error or "could not decode video (no frames)"})
    elif error is None:
        records.extend(second_record(video_id, s, secs.get(s)) for s in range(max_sec + 1))
    else:
        # failed partway: keep only seconds that were complete, then one error line
        records.extend(second_record(video_id, s, secs.get(s)) for s in range(max_sec))
        records.append({"video": video_id, "error": error})
    return records, n_frames, n_infer, rot_seen


# ---------------------------------------------------------------------------
# One Parquet shard
# ---------------------------------------------------------------------------
def process_shard(path, name, model, args, out_path, deadline):
    """Write every record of one shard to out_path (atomically). Returns stats dict."""
    import pyarrow.parquet as pq

    timers = {"decode": 0.0, "infer": 0.0}
    stats = {"shard": name, "videos": 0, "errors": 0, "frames": 0, "inferred": 0, "seconds": 0, "rotated": 0}
    t_start = time.time()
    tmp = out_path.with_suffix(".jsonl.tmp")

    with open(tmp, "w", encoding="utf-8") as f:
        def write(rec):
            f.write(json.dumps(rec) + "\n")
            if "error" in rec:
                stats["errors"] += 1
            else:
                stats["seconds"] += 1

        try:
            pf = pq.ParquetFile(path)
            num_rows = pf.metadata.num_rows
        except Exception as e:
            write({"video": f"{name}:0", "error": f"could not read parquet file ({short(e)})"})
            num_rows = 0
            pf = None

        row = 0
        if pf is not None:
            try:
                for batch in pf.iter_batches(batch_size=1, columns=["mp4"]):
                    col = batch.column(0)
                    for i in range(len(col)):
                        if args.max_videos and row >= args.max_videos:
                            raise _StopShard()
                        if deadline and time.time() > deadline:
                            raise TimeBudgetExceeded()
                        video_id = f"{name}:{row}"
                        data = col[i].as_py()
                        if not isinstance(data, (bytes, bytearray)) or not data:
                            write({"video": video_id, "error": "could not decode video (empty mp4 field)"})
                        else:
                            t = time.time()
                            recs, nf, ni, rot = index_video(data, video_id, model, args, timers, deadline)
                            for r in recs:
                                write(r)
                            stats["frames"] += nf
                            stats["inferred"] += ni
                            stats["rotated"] += bool(rot)
                            dt = time.time() - t
                            log(f"  {video_id}: {nf} frames ({ni} to model), {len(recs)} lines, rot={rot}, "
                                f"{dt:.1f}s ({nf / max(dt, 1e-9):.0f} fps)")
                        stats["videos"] += 1
                        row += 1
            except _StopShard:
                pass
            except TimeBudgetExceeded:
                raise
            except Exception as e:  # file damaged partway: mark the remaining rows
                for r in range(row, num_rows):
                    if args.max_videos and r >= args.max_videos:
                        break
                    write({"video": f"{name}:{r}", "error": f"could not read row ({short(e)})"})

    os.replace(tmp, out_path)   # the shard only counts as done once this rename happens
    stats.update(decode_s=round(timers["decode"], 1), infer_s=round(timers["infer"], 1),
                 total_s=round(time.time() - t_start, 1))
    return stats


# ---------------------------------------------------------------------------
# Worker: one per GPU, pulls shards from a shared queue
# ---------------------------------------------------------------------------
def download_shard(repo, filename, dl_dir):
    from huggingface_hub import hf_hub_download   # uses HF_TOKEN from the environment
    last = None
    for attempt in range(3):
        try:
            return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", local_dir=dl_dir)
        except Exception as e:
            last = e
            log(f"download {filename} failed (attempt {attempt + 1}): {short(e)}")
            time.sleep(10 * (attempt + 1))
    raise last


def load_model(args):
    from ultralytics import YOLO
    w = Path(args.weights)
    if not w.is_absolute() and not w.exists() and (SCRIPT_DIR / w).exists():
        w = SCRIPT_DIR / w
    return YOLO(str(w))


def worker(rank, device, task_q, args, parts_dir, deadline):
    args.device = device
    model = load_model(args)
    log(f"worker {rank}: model loaded on device {device}")

    def next_task():
        try:
            return task_q.get(timeout=5)
        except queue.Empty:
            return None

    pool = None
    if args.hf_stream:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=1)
        dl_dir = str(Path(args.download_dir) / f"w{rank}")

    def start(task):
        if task is None:
            return None
        name, src = task
        fut = pool.submit(download_shard, args.hf_repo, src, dl_dir) if pool else None
        return name, src, fut

    shard_times = []
    current = start(next_task())
    while current is not None:
        name, src, fut = current
        # leave enough time for one more shard, otherwise stop cleanly
        est = sum(shard_times) / len(shard_times) if shard_times else 0
        if deadline and time.time() + est > deadline:
            log(f"worker {rank}: time budget reached, not starting {name}")
            break
        upcoming = start(next_task())   # prefetch the next shard while this one runs
        local = None
        try:
            local = fut.result() if fut else src
            log(f"worker {rank}: processing {name}")
            st = process_shard(local, name, model, args, parts_dir / f"{name}.jsonl", deadline)
            shard_times.append(st["total_s"])
            (parts_dir / f"{name}.stats.json").write_text(json.dumps(st))
            log(f"worker {rank}: done {name}: {json.dumps(st)}")
        except TimeBudgetExceeded:
            log(f"worker {rank}: time budget hit during {name}; it will be redone next run")
            tmp = parts_dir / f"{name}.jsonl.tmp"
            if tmp.exists():
                tmp.unlink()
            break
        except Exception as e:
            log(f"worker {rank}: FAILED {name}: {short(e)}")
        finally:
            if fut and local and not args.keep_downloads:
                try:
                    os.remove(local)
                except OSError:
                    pass
        current = upcoming

    if pool:
        pool.shutdown(wait=False, cancel_futures=True)
    log(f"worker {rank}: finished")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def resolve_devices(spec):
    if spec == "cpu":
        return ["cpu"]
    if spec != "auto":
        return [d.strip().replace("cuda:", "") for d in spec.split(",") if d.strip()]
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    return [str(i) for i in range(n)] or ["cpu"]


def list_tasks(args):
    if args.hf_stream:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(args.hf_repo, repo_type="dataset")
        files = sorted(f for f in files if fnmatch.fnmatch(f, args.hf_include))
        return [(Path(f).name, f) for f in files]
    files = sorted(Path(args.input).glob("*.parquet"))
    return [(f.name, str(f)) for f in files]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", help="folder of .parquet files")
    p.add_argument("--output", required=True, help="output JSON Lines file")
    p.add_argument("--weights", default="yolo11l.pt")
    p.add_argument("--devices", default="auto", help="'auto' (all GPUs), 'cpu', or e.g. '0,1'")
    p.add_argument("--batch", type=int, default=16, help="frames per model call")
    p.add_argument("--half", action="store_true", help="FP16 inference (off by default for the baseline)")
    p.add_argument("--sample-fps", type=float, default=0,
                   help="run the model on N evenly spaced frames per second (0 = every frame)")
    p.add_argument("--checkpoint-dir", help="keep per-shard results here and skip finished shards")
    p.add_argument("--max-hours", type=float, help="stop cleanly before this many hours")
    p.add_argument("--max-shards", type=int, help="only the first N shards (testing)")
    p.add_argument("--max-videos", type=int, help="only the first N videos of each shard (testing)")
    p.add_argument("--gzip", action="store_true", help="also write <output>.gz")
    p.add_argument("--hf-stream", action="store_true", help="download shards from Hugging Face one at a time")
    p.add_argument("--hf-repo", default="HuggingFaceFV/finevideo")
    p.add_argument("--hf-include", default="data/train-000[0-4]*")
    p.add_argument("--download-dir", default=os.path.join(tempfile.gettempdir(), "finevideo"))
    p.add_argument("--keep-downloads", action="store_true", help="do not delete downloaded shards")
    args = p.parse_args()
    if not args.hf_stream and not args.input:
        p.error("--input is required unless --hf-stream is used")
    if args.max_videos and args.checkpoint_dir:
        p.error("--max-videos is for tests; don't mix it with --checkpoint-dir (shards would look finished)")
    return args


def main():
    args = parse_args()
    t_start = time.time()
    deadline = t_start + args.max_hours * 3600 if args.max_hours else None

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_dir:
        parts_dir = Path(args.checkpoint_dir).resolve()
        parts_dir.mkdir(parents=True, exist_ok=True)
    else:
        parts_dir = Path(tempfile.mkdtemp(prefix=".parts_", dir=out_path.parent))

    tasks = list_tasks(args)
    if args.max_shards:
        tasks = tasks[: args.max_shards]
    todo = [t for t in tasks if not (parts_dir / f"{t[0]}.jsonl").exists()]
    devices = resolve_devices(args.devices)
    log(f"{len(tasks)} shards, {len(tasks) - len(todo)} already done, {len(todo)} to do; devices={devices}")

    if todo:
        if len(devices) == 1:
            q = queue.Queue()
            for t in todo:
                q.put(t)
            worker(0, devices[0], q, args, parts_dir, deadline)
        else:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            q = ctx.Queue()
            for t in todo:
                q.put(t)
            procs = [ctx.Process(target=worker, args=(i, d, q, args, parts_dir, deadline))
                     for i, d in enumerate(devices)]
            for pr in procs:
                pr.start()
            for pr in procs:
                pr.join()

    # merge finished shards into the single output file
    done = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for name, _ in tasks:
            part = parts_dir / f"{name}.jsonl"
            if part.exists():
                done += 1
                with open(part, "r", encoding="utf-8") as f:
                    shutil.copyfileobj(f, out)
    if args.gzip:
        with open(out_path, "rb") as src, gzip.open(str(out_path) + ".gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
    if not args.checkpoint_dir:
        shutil.rmtree(parts_dir, ignore_errors=True)

    elapsed = time.time() - t_start
    status = "COMPLETE" if done == len(tasks) else "INCOMPLETE (run again to resume)"
    log(f"{status}: {done}/{len(tasks)} shards merged into {out_path} in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
