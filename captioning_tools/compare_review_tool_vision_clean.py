"""
Contamination-filtered variant of compare_review_tool_vision.py.

Two kinds of contamination are excluded here, both because they make
"existing" (sb_caption) not a pure baseline anymore:

1. Training-set contamination: some of sb_caption_ft_vision's videos have
   their exact video in finetune_dataset.jsonl's training set - the LoRA
   adapter may simply have memorized those targets, so existing (already
   human-corrected) vs new looking similar there is not evidence of real
   generalization, just possible memorization.

2. Prior-round-swap contamination (found 2026-08-10, after the first 11-video
   pass had already been done with only #1 filtered - 3 of those 11 turned
   out to have this problem: drive_7/01, drive_7/03 (231106095132, round 1
   swap) and drive_8/04 (231106095132, round 2 swap)): if a (video,
   caption_type) was ever "swap"-ped in compare_review_tool.py (round 1) or
   compare_review_tool_r2.py (round 2), sb_caption for that item is that
   PRIOR round's fine-tuned output, not the original base/human-corrected
   text. Comparing vision's output against it tests "vision vs round 1/2",
   not "vision vs baseline" - a different, confounded question.

Both exclusion sets are computed live every run (not hardcoded snapshots) -
so as rerun_review_vision_sample.py/rerun_review_vision_extra.py generate
more sb_caption_ft_vision entries, or as round 1/2 tools log more swaps,
this tool picks it all up automatically; no per-batch file needed here.

Reuses the SAME decision/backup/flag logs as compare_review_tool_vision.py
(not separate ones) - anything already decided there carries over and
resumes correctly here; this is a view filter, not a separate review track.
Note: re-deciding an item here does NOT retroactively fix a decision already
made in the (contaminated) full compare_review_tool_vision.py pass - if any
of drive_7/01/drive_7/03/drive_8/04's "keep" decisions matter for a writeup,
treat them as "vision vs round1/2", not "vision vs baseline".

Usage (run from the repo root):
  cd /home/mobiltech/Desktop/re-label/vlm
  source activate qwen3vl
  python captioning_tools/compare_review_tool_vision_clean.py
  python captioning_tools/compare_review_tool_vision_clean.py --flagged-only
    (only videos that had a known flagged issue during original review -
    most of the pool turned out to be flag-free "already fine" videos,
    which can only show regressions, not fixes; --flagged-only narrows to
    the videos where a real fix/non-fix is actually observable)
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compare_review_tool_vision as base  # noqa: E402
from caption_review_tool import find_videos, is_done  # noqa: E402

DATASET_PATH = os.path.join(REPO_ROOT, "finetune", "finetune_dataset.jsonl")
PRIOR_SWAP_LOGS = [
    os.path.join(REPO_ROOT, "finetune", "swap_decisions.txt"),      # round 1
    os.path.join(REPO_ROOT, "finetune", "swap_decisions_r2.txt"),   # round 2
]


def load_train_videos():
    train_videos = set()
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                train_videos.add(json.loads(line)["video"])
    return train_videos


def load_prior_swaps():
    swapped = set()
    for path in PRIOR_SWAP_LOGS:
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[3] == "swap":
                    swapped.add((parts[1], parts[2]))
    return swapped


def load_flagged_videos():
    path = os.path.join(REPO_ROOT, "flagged_videos.txt")
    flagged = set()
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    flagged.add(parts[1])
    return flagged


def build_target_sequence(flagged_only=False):
    train_videos = load_train_videos()
    prior_swaps = load_prior_swaps()
    flagged_videos = load_flagged_videos() if flagged_only else None
    sequence = []
    for video_path in find_videos():
        if video_path in train_videos:
            continue
        if flagged_only and video_path not in flagged_videos:
            continue
        if not (is_done(video_path) and base.is_done_ft(video_path)):
            continue
        for caption_type in ("1_caption", "2_caption"):
            if (video_path, caption_type) in prior_swaps:
                continue
            sequence.append((video_path, caption_type))
    return sequence


FLAGGED_ONLY = "--flagged-only" in sys.argv
base.build_target_sequence = lambda: build_target_sequence(flagged_only=FLAGGED_ONLY)

if __name__ == "__main__":
    base.main()
