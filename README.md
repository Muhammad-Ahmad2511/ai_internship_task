# Video Indexing at Speed

For every second of every video in a folder of FineVideo Parquet files, `run.py` records which of
**person, car, dog, laptop, cell phone** YOLO11l finds, and how many people are visible.

## 1. System packages

None. PyAV ships its own FFmpeg.

## 2. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Preparation command (run once, needs internet, from the repository folder)

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"
```

This downloads `yolo11l.pt` into the repository folder. `run.py` looks for it there, so the timed run
needs no internet.

## 4. Run

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

It uses all visible GPUs automatically (one worker process per GPU).

## How it works (baseline)

1. Lists every `*.parquet` file in `--input` and reads the `mp4` column one row (video) at a time.
2. Decodes the video with PyAV, turns each frame upright if the video has a rotation tag, and converts it to BGR.
3. Runs YOLO11l (`imgsz=640`, `conf=0.4`, Ultralytics defaults otherwise) on batches of frames, keeping the five target classes.
4. Puts each frame in second `floor(t - t_first)`. An object is present in a second if it is found in at least half of that second's frames; `people` is the median per-frame person count, rounded down.
5. Writes one line per second (0 … last second). A broken video gets an error line; a video that breaks partway keeps its finished seconds plus one error line.
6. Results are written per shard, then merged into the output file.

## Optional flags (used for the full 50-file run on Kaggle)

| Flag | Purpose |
|---|---|
| `--hf-stream` | Download shards from `HuggingFaceFV/finevideo` one at a time (needs `HF_TOKEN` in the environment), process, delete |
| `--checkpoint-dir DIR` | Keep one finished file per shard; finished shards are skipped when re-run |
| `--max-hours H` | Stop cleanly before H hours; re-run to resume |
| `--max-shards N`, `--max-videos N` | Small test runs |
| `--devices 0,1` | Choose GPUs (`auto` by default, `cpu` for CPU) |
| `--batch N`, `--half` | Frames per model call; FP16 |
| `--gzip` | Also write `<output>.gz` |

Example (Kaggle, unattended):

```bash
python run.py --hf-stream --checkpoint-dir /kaggle/working/shards --max-hours 11 \
  --download-dir /tmp/finevideo --output /kaggle/working/results.jsonl --gzip
```
