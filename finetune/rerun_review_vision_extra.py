"""
Thin wrapper around rerun_review_vision_sample.py: generates vision-inclusive
LoRA captions for an ADDITIONAL batch of never-trained-on videos, to grow the
clean-generalization sample beyond the first 11 (see compare_review_tool_vision_clean.py,
which picks these up automatically - no comparison-tool changes needed per batch).

Reuses that module's model loading / generation / output-path logic as-is
(same sb_caption_ft_vision output folder, same adapter) - only the target
list differs: sampled fresh from {reviewed videos} - {training-set videos}
- {videos already in sb_caption_ft_vision}, so it never re-picks a video
already covered by a prior batch and never touches a training-contaminated one.

Usage:
  source activate qwen3vl && python finetune/rerun_review_vision_extra.py [N]
  (N defaults to 25)
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

DATASET_PATH = os.path.join(REPO_ROOT, "finetune", "finetune_dataset.jsonl")
SAMPLE_SEED = 13  # different from the first batch's SAMPLE_SEED=7


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
    available = [v for v in reviewed if v not in train_videos and not base.is_done_ft(v)]
    random.seed(SAMPLE_SEED)
    sample = random.sample(available, min(n, len(available)))
    print(f"    pool of never-trained, not-yet-generated videos: {len(available)} "
          f"-> sampling {len(sample)}")
    return sorted(sample)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    targets = build_target_list(n)
    print(f"Extra batch targets: {len(targets)} videos")

    print(f"Loading base model + vision-inclusive LoRA adapter ({base.ADAPTER_PATH})...")
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
