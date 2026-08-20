#!/bin/bash
# Driver for single_video_grounding_compare.py - runs each video x mode
# combination as its own fresh process (see that script's docstring for
# why: cross-video memory accumulation caused reproducible OOM when this
# was one long-running process).
cd "$(dirname "$0")/.."
source activate qwen3vl

VIDEOS=(
  "./sample_videos/231116093002_M801C06L62G031_2673_end_extract/drive_1/03/1_clip/5.mp4"
  "./sample_videos/240322135559_M801C06L62G031_2673_end_extract/drive_22/01/1_clip/5.mp4"
  "./sample_videos/240322135559_M801C06L62G031_2673_end_extract/drive_14/04/1_clip/5.mp4"
  "./sample_videos/231106095132_M801C06L62G031_2673_end_extract/drive_4/01/1_clip/5.mp4"
  "./sample_videos/240325141234_M801C06L62G031_2673_end_extract/drive_2/03/1_clip/5.mp4"
  "./sample_videos/231106095132_M801C06L62G031_2673_end_extract/drive_3/02/1_clip/5.mp4"
)

mkdir -p finetune/bvft_results

for i in "${!VIDEOS[@]}"; do
  v="${VIDEOS[$i]}"
  out="finetune/bvft_results/base_$i.json"
  if [ -f "$out" ]; then
    echo "=== [base] video $i: already done, skipping ==="
    continue
  fi
  echo "=== [base] video $i: $v ==="
  # A single video failing (even after single_video_grounding_compare.py's
  # own OOM retry) shouldn't abort the rest of the batch - each remaining
  # video still gets a fresh process/GPU state regardless.
  python finetune/single_video_grounding_compare.py --mode base --video "$v" --out "$out" \
    || echo "!!! FAILED: base video $i ($v)"
done

for i in "${!VIDEOS[@]}"; do
  v="${VIDEOS[$i]}"
  out="finetune/bvft_results/finetuned_$i.json"
  if [ -f "$out" ]; then
    echo "=== [finetuned] video $i: already done, skipping ==="
    continue
  fi
  echo "=== [finetuned] video $i: $v ==="
  python finetune/single_video_grounding_compare.py --mode finetuned --video "$v" --out "$out" \
    || echo "!!! FAILED: finetuned video $i ($v)"
done

echo "ALL_DONE"
