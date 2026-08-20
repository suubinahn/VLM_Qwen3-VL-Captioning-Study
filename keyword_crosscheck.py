"""
Keyword cross-check for flagged_videos.txt.

For flag entries that mention only ONE of 1_caption/2_caption (not both, not
neither), this searches the OTHER caption of the same video for the same
distinctive keyword(s) pulled from the flag note - in case the same error
slipped into the unflagged caption too and was missed during review.

Entries with NEITHER "1_caption"/"2_caption" mentioned are skipped - per
established convention, that means both captions were already checked and
either both had the issue (info.txt-linked categories: road_type/road_context/
motion/environment, which are injected identically into both prompts, or the
reviewer explicitly confirmed both). Entries mentioning BOTH are also skipped
- already fully covered.

This is a plain-text keyword search - no video, no model, no API. It can only
catch cases where the OTHER caption repeats a similar distinctive word; a
differently-worded repeat of the same mistake won't be caught.

Usage:
  python keyword_crosscheck.py
"""
import os
import re

FLAGGED_LOG = "./flagged_videos.txt"

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "but", "in",
    "on", "at", "to", "of", "with", "for", "by", "from", "as", "it", "its",
    "this", "that", "these", "those", "not", "no", "be", "been", "has",
    "have", "had", "visible", "seen", "shown", "also", "further", "down",
    "road", "ahead", "delete", "other", "vehicle", "while", "which", "than",
    "into", "near", "side", "right", "left", "front", "behind",
    "adjacent", "lane", "lanes", "street", "city", "multi", "wet", "clear",
    "one", "two", "three", "caption",
    # Generic descriptive words that naturally co-occur across two
    # independently-generated captions of the same real scene, without that
    # overlap implying the same specific error was repeated. Excluding these
    # is what keeps the cross-check precise instead of flooding on color/
    # vehicle-type coincidences.
    "white", "black", "blue", "green", "yellow", "silver", "gray", "grey",
    "red", "orange", "purple", "brown", "dark", "light",
    "sedan", "suv", "van", "truck", "bus", "taxi", "motorcycle", "hatchback",
    "car", "cars", "sedans", "trucks", "vans", "buses",
    "sign", "signs", "indicating", "traffic", "pedestrian", "pedestrians",
    "crossing", "overhead", "directly", "stopped", "parked", "present",
    "lights", "light", "signal", "intersection", "route", "direction",
}

MIN_KEYWORD_HITS = 2


def output_paths(video_path):
    clip_dir = os.path.dirname(video_path)
    parent_dir = os.path.dirname(clip_dir)
    out_dir = os.path.join(parent_dir, "sb_caption")
    return os.path.join(out_dir, "1_caption.txt"), os.path.join(out_dir, "2_caption.txt")


def extract_keywords(note):
    quotes = re.findall(r'["“”]([^"“”]+)["“”]', note)
    keywords = set()
    for q in quotes:
        for w in re.findall(r"[A-Za-z가-힣]+", q):
            wl = w.lower()
            if len(w) >= 4 and wl not in STOPWORDS:
                keywords.add(wl)
    return keywords


def main():
    if not os.path.isfile(FLAGGED_LOG):
        print(f"No {FLAGGED_LOG} found.")
        return

    with open(FLAGGED_LOG, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    candidates = []
    skipped_no_keywords = 0

    for line in lines:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        video_path, category = parts[1], parts[2]
        note = parts[3] if len(parts) > 3 else ""

        has_1 = "1_caption" in note
        has_2 = "2_caption" in note
        if has_1 == has_2:
            continue  # both mentioned, or neither -> already covered, skip

        p1, p2 = output_paths(video_path)
        other_path = p2 if has_1 else p1
        other_label = "2_caption" if has_1 else "1_caption"

        if not os.path.isfile(other_path):
            continue

        keywords = extract_keywords(note)
        if not keywords:
            skipped_no_keywords += 1
            continue

        with open(other_path, "r", encoding="utf-8") as f2:
            other_text = f2.read().lower()

        hits = sorted(kw for kw in keywords if kw in other_text)
        if len(hits) >= MIN_KEYWORD_HITS:
            candidates.append((video_path, category, other_label, hits, note))

    print(f"Scanned {len(lines)} flag entries.")
    print(f"({skipped_no_keywords} single-caption entries had no extractable quoted keywords - skipped)")
    print(f"\n{len(candidates)} possible missed duplicates found:\n")
    print("=" * 100)
    for video_path, category, other_label, hits, note in candidates:
        print(f"video:    {video_path}")
        print(f"category: {category}")
        print(f"check:    {other_label} (not originally flagged)")
        print(f"matched keyword(s): {', '.join(hits)}")
        print(f"original note: {note}")
        print("-" * 100)


if __name__ == "__main__":
    main()
