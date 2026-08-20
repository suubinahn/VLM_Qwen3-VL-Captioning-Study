"""
Walks every (video, caption_type) with a "[both]" tagged flag (error
confirmed present in BOTH the existing and fine-tuned caption, from the
compare_review_tool.py review) and lets you fix the caption text directly -
these need a real correction before they can be used as training targets,
since build_finetune_dataset.py just takes whatever is CURRENTLY in
sb_caption/ and treats it as the "correct" answer.

For each item: plays the video, prints every flag note for that
(video, caption_type), then opens the actual sb_caption/{1,2}_caption.txt in
VS Code (`code --wait`) for you to correct in place. Closing the tab
(Ctrl+W) or the window marks it done and moves to the next one; closing
without saving also advances (no separate "skip" - just don't change the text).

Progress is tracked in fix_both_progress.txt so re-running resumes where you
left off.

Usage:
  source activate qwen3vl && python finetune/fix_both_flagged.py
"""
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "captioning_tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from caption_review_tool import output_paths  # noqa: E402
import build_finetune_dataset as bfd  # noqa: E402

PROGRESS_PATH = os.path.join(os.path.dirname(__file__), "fix_both_progress.txt")


def load_progress():
    done = set()
    if os.path.isfile(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    done.add((parts[0], parts[1]))
    return done


def mark_done(video_path, caption_type):
    with open(PROGRESS_PATH, "a", encoding="utf-8") as f:
        f.write(f"{video_path}\t{caption_type}\n")


def build_target_list():
    entries = bfd.parse_flags()
    both = {}
    for video_path, category, note in entries:
        if "[both]" not in note:
            continue
        caption_type = "1_caption" if "1_caption" in note else "2_caption"
        key = (video_path, caption_type)
        both.setdefault(key, []).append((category, note))

    done = load_progress()
    targets = [(k, notes) for k, notes in both.items() if k not in done]
    return sorted(targets)


def open_video_player(video_path):
    return subprocess.Popen(
        ["totem", video_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def close_video_player(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()


def caption_path(video_path, caption_type):
    _, p1, p2 = output_paths(video_path)
    return p1 if caption_type == "1_caption" else p2


def main():
    targets = build_target_list()
    print(f"{len(targets)} items left to fix.")

    for index, ((video_path, caption_type), notes) in enumerate(targets):
        print("\n" + "=" * 80)
        print(f"[{index + 1}/{len(targets)}] {video_path} ({caption_type})")
        print("-" * 80)
        print("flag notes for this item:")
        for category, note in notes:
            print(f"  - {category}: {note}")
        print("-" * 80)

        path = caption_path(video_path, caption_type)
        with open(path, "r", encoding="utf-8") as f:
            print("current text:")
            print(f.read())
        print("-" * 80)
        print("Opening in gedit - edit the text and save, then come back here.")

        proc = open_video_player(video_path)
        try:
            input("Press Enter to open the caption file in gedit (video is playing)...")
            # gedit often reuses a single running instance (new tab, not a
            # new blocking process), so we can't wait on it like `code
            # --wait` - just open it and let the user say when they're done.
            # Not tracking/terminating this process: gedit is single-instance
            # per user session, so killing it could close OTHER tabs they
            # have open unrelated to this script.
            subprocess.Popen(["gedit", path])
            input("Save the file in gedit, then press Enter here to continue...")
        finally:
            close_video_player(proc)

        mark_done(video_path, caption_type)
        print("  -> marked done")

    print("\nAll [both]-flagged items fixed.")


if __name__ == "__main__":
    main()
