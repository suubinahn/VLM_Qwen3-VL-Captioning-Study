"""
Build a supervised fine-tuning dataset from flagged_videos.txt + the current
(corrected) sb_caption files.

For each flagged (video, caption-side) pair that represents a genuine model
error, this reads:
  - info.txt -> reference_facts, exactly like the live pipeline
  - the CURRENT sb_caption/{1,2}_caption.txt -> used as the SFT target (the
    corrected text - flagged captions have already been fixed by hand or via
    --regenerate-indices)
and builds the exact prompt that would be fed to the model at inference time,
reusing combined_caption_test.py / caption_review_tool.py's own prompt
construction (read_info, format_reference_facts, PROMPT_*_TEMPLATE) so the
training prompts can never drift out of sync with the real pipeline.

info.txt-linked categories (road_type, road_context, motion, environment):
  These four are sourced verbatim from info.txt, never the model's own visual
  judgment - so a flag on one of them means EITHER info.txt itself was wrong
  (source-data problem, not the model's fault - training on it would teach
  the model to override info.txt, the opposite of what the grounding
  architecture wants) OR the model contradicted a correct given fact (a
  genuine, fine-tuning-worthy reliability problem).

  Whether info.txt was wrong is told apart by checking info_corrections.json
  directly: a plain category flag (e.g. 'm' for motion) with no matching 'x'
  correction means info.txt was never in question - kept as training data.
  If the field WAS x-corrected, that confirms info.txt was wrong, but that
  alone doesn't mean the record should be excluded - the caption TARGET TEXT
  needs to actually reflect the correction too, since a corrected info.txt
  value paired with unrewritten, still-old target text would teach the model
  to ignore the given fact (a real bug found 2026-08-11: 8 of 11 motion
  "curved lane driving" corrections had target text still saying "moves
  straight"). See has_confirmed_info_error()/target_reflects_correction():
  x-corrected AND target text already reflects it -> kept; x-corrected AND
  target text still doesn't reflect it -> excluded as stale (needs the
  caption text fixed by hand before it can be used, same as the motion fix).
  Entries excluded this way are still written to
  finetune_dataset_source_mismatch.jsonl for the record.

Output:
  finetune_dataset.jsonl                  - training records: {video,
                                             prompt_type, prompt, target,
                                             categories, notes}
  finetune_dataset_source_mismatch.jsonl  - info.txt-linked flags where the
                                             target disagrees with info.txt
                                             (excluded as source-data issues)

No model load, no video processing - pure file/text work, runs in seconds.

Usage:
  python build_finetune_dataset.py
"""
import json
import os
import re
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "captioning_tools"))
from caption_review_tool import (  # noqa: E402
    read_info,
    format_reference_facts,
    PROMPT_DESCRIPTION_TEMPLATE,
    PROMPT_STRUCTURED_TEMPLATE,
    output_paths,
)

# read_info()/output_paths() above resolve their own paths (e.g.
# "./sample_videos") relative to the current working directory, so this
# script must still be run with REPO_ROOT as cwd - these three paths are
# made absolute instead so they're correct regardless of that requirement.
FLAGGED_LOG = os.path.join(REPO_ROOT, "flagged_videos.txt")
OUTPUT_PATH = os.path.join(THIS_DIR, "finetune_dataset.jsonl")
MISMATCH_PATH = os.path.join(THIS_DIR, "finetune_dataset_source_mismatch.jsonl")

# time_of_day has no matching flag key in FLAG_CATEGORIES (unlike road_type=t,
# road_context=c, motion=m, environment=e) - it can never appear in
# flagged_videos.txt, so it never reaches this pipeline; nothing to exclude
# or include, there's just no entry for it.
EXCLUDED_CATEGORIES = set()
INFO_LINKED_CATEGORIES = {"road_context", "motion", "environment", "road_type"}


def parse_flags():
    with open(FLAGGED_LOG, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    entries = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        video_path, category = parts[1], parts[2]
        note = parts[3] if len(parts) > 3 else ""
        entries.append((video_path, category, note))
    return entries


def sides_for(note):
    has_1 = "1_caption" in note
    has_2 = "2_caption" in note
    if has_1 and has_2:
        return {"1", "2"}
    if has_1:
        return {"1"}
    if has_2:
        return {"2"}
    return {"1", "2"}  # no caption specified -> established convention: applies to both


def build_prompt(side, video_path):
    info = read_info(video_path)
    reference_facts = format_reference_facts(info)
    template = PROMPT_DESCRIPTION_TEMPLATE if side == "1" else PROMPT_STRUCTURED_TEMPLATE
    return template.format(reference_facts=reference_facts)


INFO_CORRECTIONS_PATH = os.path.join(REPO_ROOT, "info_corrections", "info_corrections.json")
CATEGORY_TO_INFO_FIELDS = {
    "road_context": ("road_context",),
    "motion": ("motion",),
    "environment": ("surface", "weather"),
    "road_type": ("road_type",),
}

# Per confirmed-correction VALUE, words/phrases that indicate the caption
# target text already incorporates that correction. Not a truth-check -
# info_corrections.json's value is already user-confirmed ground truth (via
# the 'x' key) - just a text-consistency check to catch captions that were
# never rewritten after the correction was made. Found 2026-08-11: 8 of 11
# motion="curved lane driving" records had target captions still saying
# "moves straight" (teaching the model to ignore the given fact), and
# separately, road_type flags (49 total: single/two/three/multi-lane lane-
# count corrections, not just the earlier-verified "highway" wording issue)
# had been wholesale excluded from training on the mistaken assumption that
# ALL of them were the wording issue. A value with no entry here falls back
# to a literal substring match on the value itself.
FIELD_VALUE_KEYWORDS = {
    "motion": {
        "curved lane driving": ("curve", "curved", "curving", "bend", "winding"),
        "stop": ("stop", "stopped", "stationary", "halt", "idling", "standstill"),
        "moving straight": ("straight",),
        "turning left": ("turn left", "turns left", "turning left", "left turn"),
        "turning right": ("turn right", "turns right", "turning right", "right turn"),
    },
    "road_context": {
        "city street": ("city street", "urban street"),
        "speed bump": ("speed bump", "speed hump"),
        "underpass": ("underpass",),
        "overpass": ("overpass", "flyover"),
        "tunnel": ("tunnel",),
        "bridge": ("bridge",),
        "roundabout": ("roundabout",),
        "intersection": ("intersection", "junction"),
    },
    "surface": {
        "wet": ("wet",),
        "dry": ("dry",),
    },
    "weather": {
        "rainy": ("rain", "rainy"),
        "clear": ("clear",),
        "cloudy": ("cloudy", "overcast"),
        "sunny": ("sunny", "sunlit"),
        "snowy": ("snow", "snowy"),
        "foggy": ("fog", "foggy"),
    },
    "road_type": {
        "single-lane road": ("single-lane",),
        "two-lane road": ("two-lane",),
        "three-lane road": ("three-lane",),
        "multi-lane road": ("multi-lane",),
        "three-lane": ("three-lane",),
    },
}


def load_info_corrections():
    if not os.path.isfile(INFO_CORRECTIONS_PATH):
        return {}
    with open(INFO_CORRECTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


INFO_CORRECTIONS = load_info_corrections()


def target_reflects_correction(field, value, target_text):
    text = target_text.lower()
    keywords = FIELD_VALUE_KEYWORDS.get(field, {}).get(value.lower(), (value.lower(),))
    return any(kw in text for kw in keywords)


def has_confirmed_info_error(category, video_path, target_text):
    """True if this category's field was x-corrected (info.txt itself was
    confirmed wrong) AND the caption target text hasn't been updated to
    reflect the correction yet - a genuinely stale record, excluded so the
    model isn't trained to ignore the given fact. If the target text already
    incorporates the correction (checked via target_reflects_correction),
    prompt and target agree and this is good training signal - kept, not
    excluded, even though the field was x-corrected.

    A flag never promoted to an 'x' correction at all (plain category key,
    e.g. 'm' for motion) means info.txt was never in question - the model
    contradicted a correct given fact - always kept as training data.

    History: this function originally excluded ANY x-corrected field
    unconditionally (info.txt was wrong -> don't train on it at all), which
    conflated two different questions - was info.txt wrong, vs. does the
    target text match the correction. Once info.txt is corrected, training
    the model to comply with the corrected value is exactly what the
    grounding architecture wants; only a stale, not-yet-rewritten target text
    is actually bad training data. road_type used to be excluded outright
    (EXCLUDED_CATEGORIES) on top of that, based on an 8-flag investigation
    that found a "highway" wording issue - but 49 road_type flags exist and
    most (single/two/three/multi-lane counts) are unrelated lane-count
    corrections, not the wording issue, so that blanket exclusion was
    discarding real training signal too. Before either of these, an even
    earlier version ran this same kind of text-matching heuristic
    unconditionally (no x-correction check at all) and produced false
    positives (~27 motion entries wrongly excluded) by conflating omission
    with contradiction - the fix here avoids that failure mode by only ever
    running the text check on fields ALREADY confirmed wrong via 'x' (a
    small, human-verified set), never as a truth-detector over all flags."""
    fields = CATEGORY_TO_INFO_FIELDS.get(category, ())
    corrected_fields = INFO_CORRECTIONS.get(video_path, {})
    for field in fields:
        if field not in corrected_fields:
            continue
        if not target_reflects_correction(field, corrected_fields[field], target_text):
            return True
    return False


def read_target_text(video_path, side):
    _, p1, p2 = output_paths(video_path)
    target_path = p1 if side == "1" else p2
    if not os.path.isfile(target_path):
        return None
    with open(target_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def bucket_entries(entries):
    keep, mismatch = {}, {}
    target_cache = {}
    for video_path, category, note in entries:
        if category in EXCLUDED_CATEGORIES:
            continue
        needs_check = category in INFO_LINKED_CATEGORIES
        for side in sides_for(note):
            key = (video_path, side)
            is_stale = False
            if needs_check:
                if key not in target_cache:
                    target_cache[key] = read_target_text(video_path, side) or ""
                is_stale = has_confirmed_info_error(category, video_path, target_cache[key])
            target_bucket = mismatch if is_stale else keep

            target_bucket.setdefault(key, {"categories": set(), "notes": []})
            target_bucket[key]["categories"].add(category)
            target_bucket[key]["notes"].append(note)
    return keep, mismatch


def write_records(bucket, path):
    n_written, n_missing = 0, 0
    category_counts = {}
    with open(path, "w", encoding="utf-8") as out:
        for (video_path, side), meta in sorted(bucket.items()):
            _, p1, p2 = output_paths(video_path)
            target_path = p1 if side == "1" else p2
            if not os.path.isfile(target_path):
                n_missing += 1
                continue
            with open(target_path, "r", encoding="utf-8") as f:
                target = f.read().strip()
            try:
                prompt = build_prompt(side, video_path)
            except (FileNotFoundError, KeyError) as e:
                print(f"  WARN: skipping {video_path} ({side}) - {e!r}")
                n_missing += 1
                continue
            record = {
                "video": video_path,
                "prompt_type": "description" if side == "1" else "structured",
                "prompt": prompt,
                "target": target,
                "categories": sorted(meta["categories"]),
                "notes": meta["notes"],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1
            for c in meta["categories"]:
                category_counts[c] = category_counts.get(c, 0) + 1
    return n_written, n_missing, category_counts


def main():
    entries = parse_flags()
    keep, mismatch = bucket_entries(entries)

    n_written, n_missing, cat_counts = write_records(keep, OUTPUT_PATH)
    n_mismatch, n_mismatch_missing, mismatch_counts = write_records(mismatch, MISMATCH_PATH)

    print(f"{OUTPUT_PATH}: {n_written} training records written"
          f" ({n_missing} skipped - output file missing)")
    print(f"{MISMATCH_PATH}: {n_mismatch} records excluded as likely"
          f" info.txt source-data issues ({n_mismatch_missing} skipped)")
    print("\nIncluded records by category:")
    for c, n in sorted(cat_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {c}: {n}")
    if mismatch_counts:
        print("\nExcluded (info.txt mismatch) by category:")
        for c, n in sorted(mismatch_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
