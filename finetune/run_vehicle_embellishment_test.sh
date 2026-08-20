#!/bin/bash
# Driver for vehicle_embellishment_test.py - one fresh process per
# (video, condition) pair, see that script's docstring for why.
cd "$(dirname "$0")/.."
source activate qwen3vl

VIDEOS=(
  "./sample_videos/231116093002_M801C06L62G031_2673_end_extract/drive_1/03/1_clip/5.mp4"
  "./sample_videos/231106095132_M801C06L62G031_2673_end_extract/drive_4/01/1_clip/5.mp4"
  "./sample_videos/240325141234_M801C06L62G031_2673_end_extract/drive_2/03/1_clip/5.mp4"
  "./sample_videos/231106095132_M801C06L62G031_2673_end_extract/drive_3/02/1_clip/5.mp4"
)

mkdir -p finetune/veh_results

for i in "${!VIDEOS[@]}"; do
  v="${VIDEOS[$i]}"
  out="finetune/veh_results/without_$i.json"
  if [ -f "$out" ]; then
    echo "=== [without] video $i: already done, skipping ==="
  else
    echo "=== [without] video $i: $v ==="
    python finetune/vehicle_embellishment_test.py --condition without --video "$v" --out "$out" \
      || echo "!!! FAILED: without video $i ($v)"
  fi
done

for i in "${!VIDEOS[@]}"; do
  v="${VIDEOS[$i]}"
  out="finetune/veh_results/with_$i.json"
  if [ -f "$out" ]; then
    echo "=== [with] video $i: already done, skipping ==="
  else
    echo "=== [with] video $i: $v ==="
    python finetune/vehicle_embellishment_test.py --condition with --video "$v" --out "$out" \
      || echo "!!! FAILED: with video $i ($v)"
  fi
done

echo "ALL_DONE"
