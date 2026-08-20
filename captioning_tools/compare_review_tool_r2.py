"""
Round-2 variant of compare_review_tool.py: compares existing sb_caption
against sb_caption_ft_r2 (round-2 model, finetune_checkpoints_round2/epoch_3,
from finetune/rerun_review_r2_sample.py's 25-video sample) instead of round
1's sb_caption_ft. Round 1's comparison files (swap_decisions.txt,
swap_backup_log.txt, sb_caption_ft/) are untouched - this uses its own log
files so the two rounds' decisions never collide.

Same UX as compare_review_tool.py: [k]eep/[s]wap/[q]uit/Backspace, flag keys
with [existing]/[new]/[both] tagging, 'x' for info.txt corrections. See that
file's docstring for full details - only the target folder/log paths differ
here.

Usage (run from the repo root):
  cd /home/mobiltech/Desktop/re-label/vlm
  source activate qwen3vl
  python captioning_tools/compare_review_tool_r2.py
"""
import os
import subprocess
import sys
import termios
import tty
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from caption_review_tool import (  # noqa: E402
    find_videos,
    output_paths,
    is_done,
    FLAG_CATEGORIES,
    log_flag,
    INFO_KEYS,
    log_info_correction,
)

DECISIONS_LOG = os.path.join(REPO_ROOT, "finetune", "swap_decisions_r2.txt")
BACKUP_LOG = os.path.join(REPO_ROOT, "finetune", "swap_backup_log_r2.txt")
FLAGGED_LOG = os.path.join(REPO_ROOT, "flagged_videos.txt")


def load_flag_history():
    history = {}
    if not os.path.isfile(FLAGGED_LOG):
        return history
    with open(FLAGGED_LOG, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            video_path, category = parts[1], parts[2]
            note = parts[3] if len(parts) > 3 else ""
            history.setdefault(video_path, []).append((category, note))
    return history


def output_paths_ft(video_path):
    clip_dir = os.path.dirname(video_path)
    parent_dir = os.path.dirname(clip_dir)
    out_dir = os.path.join(parent_dir, "sb_caption_ft_r2")
    return out_dir, os.path.join(out_dir, "1_caption.txt"), os.path.join(out_dir, "2_caption.txt")


def is_done_ft(video_path):
    _, p1, p2 = output_paths_ft(video_path)
    return os.path.isfile(p1) and os.path.isfile(p2)


def load_decisions():
    decided = set()
    if os.path.isfile(DECISIONS_LOG):
        with open(DECISIONS_LOG, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    decided.add((parts[1], parts[2]))
    return decided


def log_decision(video_path, caption_type, decision):
    os.makedirs(os.path.dirname(DECISIONS_LOG), exist_ok=True)
    with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()}\t{video_path}\t{caption_type}\t{decision}\n")


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_original_text(video_path, caption_type, current_text):
    if not os.path.isfile(BACKUP_LOG):
        return current_text
    with open(BACKUP_LOG, "r", encoding="utf-8") as f:
        content = f.read()
    last_match = None
    for block in content.split("=" * 80 + "\n"):
        if f"video: {video_path}\n" in block and f"caption_type: {caption_type}\n" in block:
            marker = "old_text:\n"
            idx = block.find(marker)
            if idx != -1:
                last_match = block[idx + len(marker):].strip()
    return last_match if last_match is not None else current_text


def remove_matching_flag_line(video_path, category, note):
    if not os.path.isfile(FLAGGED_LOG):
        return
    with open(FLAGGED_LOG, "r", encoding="utf-8") as f:
        lines = f.readlines()
    suffix = f"\t{video_path}\t{category}\t{note}\n" if note else f"\t{video_path}\t{category}\n"
    removed = False
    kept = []
    for line in lines:
        if not removed and line.endswith(suffix):
            removed = True
            continue
        kept.append(line)
    if removed:
        with open(FLAGGED_LOG, "w", encoding="utf-8") as f:
            f.writelines(kept)


def apply_decision(video_path, caption_type, decision, original_text, new_text, old_path):
    if decision == "keep":
        with open(old_path, "w", encoding="utf-8") as f:
            f.write(original_text + "\n")
    elif decision == "swap":
        os.makedirs(os.path.dirname(BACKUP_LOG), exist_ok=True)
        with open(BACKUP_LOG, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"timestamp: {datetime.now().isoformat()}\n")
            f.write(f"video: {video_path}\n")
            f.write(f"caption_type: {caption_type}\n")
            f.write("old_text:\n" + original_text + "\n\n")
        with open(old_path, "w", encoding="utf-8") as f:
            f.write(new_text + "\n")
    log_decision(video_path, caption_type, decision)


def open_video_player(video_path):
    return subprocess.Popen(
        ["totem", video_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def close_video_player(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()


def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def show_pair(caption_type, original_text, new_text):
    print("-" * 80)
    print(f"{caption_type} - EXISTING (sb_caption):")
    print(original_text)
    print("-" * 80)
    print(f"{caption_type} - NEW (sb_caption_ft_r2, round 2):")
    print(new_text)
    print("-" * 80)
    print("[k] keep existing  [s] swap to new  [Backspace] previous  [q] quit")
    print("[t] flag road_type  [c] flag road_context  [m] flag motion  "
          "[e] flag environment")
    print("[v] flag other-vehicle behavior  [p] flag pedestrian behavior  "
          "[w] flag visibility")
    print("[i] flag sign identification  [o] flag other  "
          "[x] correct info.txt (confirmed error only)")


def ask_version_tag():
    print("  이 오류가 어디 있나요? [e] 기존(sb_caption)만  [n] 새 것(round2)만  [b] 둘 다")
    while True:
        k = getch()
        if k in ("e", "E"):
            return "existing"
        if k in ("n", "N"):
            return "new"
        if k in ("b", "B"):
            return "both"
        print("  (press e/n/b)")


def decide_one(video_path, caption_type, old_path, new_path, applied_flags):
    current_text = read_text(old_path)
    original_text = get_original_text(video_path, caption_type, current_text)
    new_text = read_text(new_path)
    show_pair(caption_type, original_text, new_text)
    item_key = (video_path, caption_type)
    while True:
        key = getch()
        if key == "k":
            apply_decision(video_path, caption_type, "keep", original_text, new_text, old_path)
            print("  -> kept existing")
            return "next"
        elif key == "s":
            apply_decision(video_path, caption_type, "swap", original_text, new_text, old_path)
            print("  -> swapped in round-2 version")
            return "next"
        elif key in ("\x7f", "\x08"):
            return "back"
        elif key == "q":
            return "quit"
        elif key.lower() in FLAG_CATEGORIES:
            category = FLAG_CATEGORIES[key.lower()]
            version_tag = ask_version_tag()
            note = input(f"  {category} - describe briefly (optional, Enter to skip): ")
            full_note = f"{caption_type} [{version_tag}] {note}" if note else f"{caption_type} [{version_tag}]"
            prior = applied_flags.setdefault(item_key, {})
            if category in prior:
                remove_matching_flag_line(video_path, category, prior[category])
            log_flag(video_path, category, full_note)
            prior[category] = full_note
            print(f"  flagged: {category} [{version_tag}]")
            continue
        elif key in ("o", "O"):
            version_tag = ask_version_tag()
            note = input("  other - describe briefly: ")
            full_note = f"{caption_type} [{version_tag}] {note}" if note else f"{caption_type} [{version_tag}]"
            prior = applied_flags.setdefault(item_key, {})
            if "other" in prior:
                remove_matching_flag_line(video_path, "other", prior["other"])
            log_flag(video_path, "other", full_note)
            prior["other"] = full_note
            print(f"  flagged: other [{version_tag}]")
            continue
        elif key in ("x", "X"):
            field = input(f"  info.txt correction - which field? {INFO_KEYS}: ").strip()
            if field not in INFO_KEYS:
                print(f"  not a valid field, skipped: {field!r}")
                continue
            value = input(f"  correct value for '{field}': ").strip()
            if not value:
                print("  empty value, skipped")
                continue
            log_info_correction(video_path, field, value)
            print(f"  info_corrections.json updated: {field} -> {value!r}")
            continue
        else:
            print("  (press k/s/q, Backspace, or a flag key)")


def build_target_sequence():
    sequence = []
    for video_path in find_videos():
        if not (is_done(video_path) and is_done_ft(video_path)):
            continue
        sequence.append((video_path, "1_caption"))
        sequence.append((video_path, "2_caption"))
    return sequence


def main():
    sequence = build_target_sequence()
    decided = load_decisions()
    flag_history = load_flag_history()

    start_pos = next(
        (i for i, item in enumerate(sequence) if item not in decided), len(sequence)
    )
    if start_pos >= len(sequence):
        print("All comparisons already done.")
        return

    print(f"{len(sequence)} total, resuming at item {start_pos + 1}.")

    pos = start_pos
    current_video = None
    proc = None
    applied_flags = {}
    try:
        while pos < len(sequence):
            video_path, caption_type = sequence[pos]

            if video_path != current_video:
                close_video_player(proc)
                current_video = video_path
                print("\n" + "=" * 80)
                print(f"[{pos + 1}/{len(sequence)}] {video_path}")
                entries = flag_history.get(video_path)
                if entries:
                    print(f"  이전 flag ({len(entries)}건):")
                    for category, note in entries:
                        print(f"    - {category}: {note}")
                else:
                    print("  이전 상태: flag 없음 (통과했던 영상)")
                print("=" * 80)
                proc = open_video_player(video_path)
            else:
                print(f"\n[{pos + 1}/{len(sequence)}] {caption_type}")

            _, old_p1, old_p2 = output_paths(video_path)
            _, new_p1, new_p2 = output_paths_ft(video_path)
            old_path = old_p1 if caption_type == "1_caption" else old_p2
            new_path = new_p1 if caption_type == "1_caption" else new_p2

            result = decide_one(video_path, caption_type, old_path, new_path, applied_flags)
            if result == "quit":
                print("\nQuitting. Progress is saved - rerun to resume.")
                return
            elif result == "back":
                if pos == 0:
                    print("Already at the first item.")
                    continue
                pos -= 1
            else:
                pos += 1
    finally:
        close_video_player(proc)

    print("\nAll comparisons done.")


if __name__ == "__main__":
    main()
