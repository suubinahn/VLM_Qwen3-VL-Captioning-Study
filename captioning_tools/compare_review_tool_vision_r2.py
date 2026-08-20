"""
Quick video-side-by-side comparison for vision LoRA round2 (finetune_checkpoints_vision_round2/
epoch_2), reusing compare_review_tool_vision.py's UX (video playback via totem,
k=keep/s=swap/flag keys) - only the target folder differs (sb_caption_ft_vision_r2,
from rerun_review_vision_r2_quick.py's 5-video sample).

Usage (run from the repo root):
  cd /home/mobiltech/Desktop/re-label/vlm
  source activate qwen3vl
  python captioning_tools/compare_review_tool_vision_r2.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compare_review_tool_vision as base  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Separate logs from round 1's (swap_decisions_vision.txt) - round1 and round2
# samples can overlap (different sampling pools, no cross-exclusion), so sharing
# a log risks mixing decisions from two different checkpoints under one file.
base.DECISIONS_LOG = os.path.join(REPO_ROOT, "finetune", "swap_decisions_vision_r2.txt")
base.BACKUP_LOG = os.path.join(REPO_ROOT, "finetune", "swap_backup_log_vision_r2.txt")


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

if __name__ == "__main__":
    base.main()
