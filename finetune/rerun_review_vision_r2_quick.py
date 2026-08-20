"""
Quick 5-video sanity check for vision LoRA round2 (finetune_checkpoints_vision_round2/
epoch_2, the val-loss-best checkpoint of this round: 0.5385/0.4901/0.4946/0.4970/
0.4973/0.5017). Thin wrapper reusing rerun_review_vision_sample.py's model-loading/
generation logic - only the adapter path, output folder, and target list differ.

Target list: sampled from reviewed videos not in the (new, 221-record) training
set - a fast eyeball check, not the full contamination-filtered methodology used
for round 1's 33-video comparison. Output goes to sb_caption_ft_vision_r2/.

Usage:
  source activate qwen3vl && python finetune/rerun_review_vision_r2_quick.py [N]
  (N defaults to 5)
"""
import json
import os
import random
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "captioning_tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rerun_review_vision_sample as base  # noqa: E402
from caption_review_tool import find_videos, is_done  # noqa: E402

base.ADAPTER_PATH = os.path.join(REPO_ROOT, "finetune_checkpoints_vision_round2", "epoch_2")

DATASET_PATH = os.path.join(REPO_ROOT, "finetune", "finetune_dataset.jsonl")
SAMPLE_SEED = 21


def output_paths_ft(video_path):
    clip_dir = os.path.dirname(video_path)
    parent_dir = os.path.dirname(clip_dir)
    out_dir = os.path.join(parent_dir, "sb_caption_ft_vision_r2")
    return out_dir, os.path.join(out_dir, "1_caption.txt"), os.path.join(out_dir, "2_caption.txt")


base.output_paths_ft = output_paths_ft


def is_done_ft(video_path):
    _, p1, p2 = output_paths_ft(video_path)
    return os.path.isfile(p1) and os.path.isfile(p2)


base.is_done_ft = is_done_ft


def load_train_videos():
    train_videos = set()
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                train_videos.add(json.loads(line)["video"])
    return train_videos


def build_target_list(n):
    train_videos = load_train_videos()
    reviewed = [v for v in find_videos() if is_done(v)]
    available = [v for v in reviewed if v not in train_videos and not is_done_ft(v)]
    random.seed(SAMPLE_SEED)
    sample = random.sample(available, min(n, len(available)))
    print(f"    pool of never-trained, not-yet-generated videos: {len(available)} "
          f"-> sampling {len(sample)}")
    return sorted(sample)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    targets = build_target_list(n)
    print(f"Quick round2 check targets: {len(targets)} videos")

    print(f"Loading base model + vision-inclusive LoRA adapter round2 ({base.ADAPTER_PATH})...")
    model, processor = base.build_model_and_processor()

    done = 0
    for video_path in targets:
        try:
            base.generate_and_save(model, processor, video_path)
            done += 1
            print(f"  ...{done}/{len(targets)} ({video_path})")
        except Exception as e:
            print(f"  [ERROR] {video_path}: {e}")

    print(f"\nDone. Generated {done} new, out of {len(targets)} targets.")


if __name__ == "__main__":
    main()
