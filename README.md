# Video Indexing at Speed

For every second of every video in a folder of FineVideo Parquet files, `run.py` records which of
**person, car, dog, laptop, cell phone** YOLO11l finds, and how many people are visible.

## 1. System packages

OpenCV (pulled in by Ultralytics) needs these libraries on a minimal Ubuntu 22.04:

```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
```

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

## How it works

1. Lists every `*.parquet` file in `--input` and reads the `mp4` column one row (video) at a time.
2. A decode thread decodes each video with PyAV, turns frames upright if the video has a rotation tag,
   and keeps **5 evenly spaced frames per second** (the first frame of each 0.2 s slot), converted to BGR.
   Every frame is still decoded, so second boundaries and the last second are exact.
3. The main thread runs YOLO11l (`imgsz=640`, `conf=0.4`, IoU 0.7, letterbox, FP16) on batches of those
   frames while the decode thread prepares the next ones, keeping the five target classes.
4. Per second: an object is present if it is found in at least half of the sampled frames; `people` is the
   median per-frame person count, rounded down.
5. Writes one line per second (0 … last second). A broken video gets an error line; a video that breaks
   partway keeps its finished seconds plus one error line.
6. Results are written per shard, then merged into the output file.

Sampling rate and FP16 were chosen by comparing against full-frame FP32 results on 17 videos
(`check.py`, `reference/`): at 5 fps FP16 the objects F1 was 0.986, people exact 0.961, within 1 0.997.

## Optional flags (used for the full 50-file run on Kaggle)

| Flag | Purpose |
|---|---|
| `--hf-stream` | Download shards from `HuggingFaceFV/finevideo` one at a time (needs `HF_TOKEN` in the environment), process, delete |
| `--checkpoint-dir DIR` | Keep one finished file per shard; finished shards are skipped when re-run |
| `--max-hours H` | Stop cleanly before H hours; re-run to resume |
| `--max-shards N`, `--max-videos N` | Small test runs |
| `--devices 0,1` | Choose GPUs (`auto` by default, `cpu` for CPU) |
| `--sample-fps N` | Frames per second sent to the model (default 5; `0` = every frame) |
| `--no-half` | FP32 instead of the default FP16 |
| `--batch N` | Frames per model call (default 16) |
| `--gzip` | Also write `<output>.gz` |

Reference results (every frame, FP32, exactly like the graders' reference):

```bash
python run.py --input <folder> --output ref.jsonl --sample-fps 0 --no-half
python check.py --ref ref.jsonl --pred results.jsonl
```

Example (Kaggle, unattended):

```bash
python run.py --hf-stream --checkpoint-dir /kaggle/working/shards --max-hours 11 \
  --download-dir /tmp/finevideo --output /kaggle/working/results.jsonl --gzip
```
