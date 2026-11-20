# Hardware

The full run (all 50 Parquet files) was done on a **free Kaggle notebook** (GPU T4 x2), as an
unattended "Save & Run All" commit. Raw output of the hardware commands is in
[hardware_info.txt](hardware_info.txt); the full log is in [run_log.txt](run_log.txt).

| | |
|---|---|
| **Where** | Kaggle notebook (free tier), accelerator "GPU T4 x2" |
| **GPU** | 2 × NVIDIA Tesla T4, 15 GB (15,360 MiB) each — one worker process per GPU |
| **CPU** | Intel Xeon @ 2.00 GHz, 4 vCPUs (2 cores × 2 threads) |
| **RAM** | 31 GiB |
| **Disk** | DISK_TYPE_TODO (Kaggle VM; shards streamed to `/tmp` one at a time and deleted after use) |
| **OS** | Ubuntu 24.04.4 LTS, Python 3.13.16 |

## Full run time

**11,985 s = 3 h 19 min 45 s** from start to finish (`run.py` itself reported 199.8 min).

This includes downloading the 50 shards from Hugging Face during the run (each shard was downloaded
while the previous one was being processed). Settings were the defaults: FP16, 5 sampled frames per
second, batch 16.

Command:

```bash
python run.py --hf-stream --checkpoint-dir /kaggle/working/shards --max-hours 11 \
  --download-dir /tmp/finevideo --output /kaggle/working/results.jsonl --gzip
```

| | |
|---|---|
| Videos | 1,650 (0 errors) |
| Seconds of video | 461,757 (128.3 h) |
| Frames decoded | 12,981,615 |
| Frames sent to the model | 2,305,453 (17.8%) |
| GPU time in inference (both GPUs summed) | 23,623 s |
| GPU time waiting for decoded frames | 63 s |
