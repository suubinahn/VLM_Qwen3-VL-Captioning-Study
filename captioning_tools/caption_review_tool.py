"""
Qwen3-VL driving video captioning - interactive review tool.

Walks through every 5.mp4 under SAMPLE_ROOT (recursively, in sorted order),
generating:
  sb_caption/1_caption.txt - free-form description
  sb_caption/2_caption.txt - structured motion/vehicles/objects/environment/weather

`sb_caption` is created as a sibling of the `1_clip` folder that holds the video
(not inside it).

Motion, road type/lane count, time of day, road surface, and weather are read
from the sibling `info.txt` of each clip and injected as given facts into both
prompts, rather than asked of the model - validated as more reliable than the
model's own visual guesses for these fields. The ego-vehicle's SPECIFIC lane
(leftmost/middle/rightmost) is never stated, only the lane count from info.txt.

Additionally, verified vehicle/pedestrian/sign-text grounding facts (YOLO
detection + EasyOCR text scan, each individually confirmed via crop+VQA
before being trusted - see captioning_tools/detection_hint.py and
ocr_hint.py for why raw unverified hints were replaced with this) are
computed once per video and merged into the same reference-facts block.
Results are cached in GROUNDING_CACHE_PATH so regenerating a video's
captions (r / --regenerate-range) after a prompt change doesn't repeat the
detection+verification work, only the caption generation itself.

Each time a video's captions are shown, its clip automatically opens in the
`totem` video player (previous video's player window is closed first) so you
can watch it side by side with the generated captions while reviewing.

Controls (single keypress, no need to hit Enter afterward for r/q):
  Enter      -> accept current result, move to next video
  Backspace  -> go back to the previous video (review / redo)
  r          -> regenerate the current video (overwrites its output)
  q          -> quit (progress is preserved - already-captioned videos are
                skipped automatically next time you run this script)

  Flagging (for building a fine-tuning error dataset later) - each key
  prompts for a short optional note (Enter to skip), logs to
  flagged_videos.txt, and stays on the current video, so you can still press
  Enter/Backspace/r afterward. Mis-flagged a video? Just edit/delete the line
  in flagged_videos.txt by hand - it's a plain text file, one flag per line.
  t          -> flag: road_type error (info.txt's road size/lane-count field
                wrong, e.g. "three-lane road" should be "two-lane road")
  c          -> flag: road_context error (info.txt's road context field
                wrong, e.g. "underpass"/"city street"/"tunnel" etc.)
  m          -> flag: motion error (info.txt's motion field wrong, e.g.
                "moving straight" should be "turning right")
  e          -> flag: environment error (info.txt's surface/weather field
                itself appears wrong vs. what's actually visible, e.g. info.txt
                says "dry" but the road is clearly wet - NOT a model error,
                the model correctly used the given fact; this is a source-data
                quality flag, not a fine-tuning candidate)
  v          -> flag: vehicle behavior fabrication (OTHER vehicles only - a
                maneuver invented/misattributed to a surrounding vehicle)
  p          -> flag: pedestrian behavior fabrication (a pedestrian's action
                invented or misattributed, e.g. "crossing the street" when
                they're actually just standing/walking on the sidewalk)
  w          -> flag: overconfident despite obscured visibility
  i          -> flag: sign identification error (a sign symbol or diagram
                misread, e.g. a junction diagram read as a parking symbol) -
                NOT vehicle brand/model or license plates, which the prompt
                now bans outright (kept only as a legacy label for old
                brand-misread entries logged before that ban)
  o          -> flag: other (prompts for a short free-text note)

  x          -> correct info.txt (NOT a flag - writes directly to
                info_corrections.json, applied on top of info.txt at read
                time; the original info.txt file is never touched). Prompts
                for a field name (motion/road_context/road_type/time_of_day/
                surface/weather) and the correct value. Only use this once a
                mismatch between info.txt and the actual video has been
                directly confirmed - not from a hunch, since this changes
                what future prompts/regenerations treat as ground truth for
                that video.

Resume behavior: on startup, the tool jumps straight to the first video that
does not yet have both output files, so stopping and restarting continues
where you left off. You can still move backward past that point to review or
redo earlier videos.

Usage:
  python caption_review_tool.py
      Normal run: skips already-done videos, resumes where you left off.

  python caption_review_tool.py --regenerate-range 1 20
      Force-regenerate videos 1 through 20 (1-indexed, inclusive) even if they
      already have output - use this after changing a prompt to redo videos
      you already processed with the old one. Once you move past video 20
      (just keep pressing Enter), the same run automatically switches back to
      normal behavior for the rest - no need to restart the script.

  python caption_review_tool.py --regenerate-indices 2 6
      Force-regenerate only videos 2 and 6 (1-indexed, not necessarily
      contiguous) - use this to redo specific flagged videos. Enter jumps
      directly between just these videos (2 -> 6), skipping everything in
      between entirely - this is NOT a sequential walk through the dataset.
      If combined with --regenerate-range, falls back to a normal sequential
      walk instead (force-regenerating whichever videos match either one).

  python caption_review_tool.py --flag-summary
      Print per-category counts from flagged_videos.txt and exit (no model
      loading, no review session) - use this to check progress toward a
      fine-tuning checkpoint.
"""
import argparse
import glob
import gc
import json
import os
import re
import subprocess
import sys
import termios
import tty
from datetime import datetime

# Must be set before the first CUDA allocation (i.e. before torch/transformers
# touch the GPU) - lets NUM_FRAMES=16 fit in the 12GB GPUs by reducing
# allocator fragmentation. Without this, 16 (and even 12) frames OOM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from combined_hint import get_combined_facts

MODEL_PATH = "./Qwen3-VL-8B-Instruct"
SAMPLE_ROOT = "./sample_videos"
NUM_FRAMES = 16
FAILED_LOG = "./failed_videos.txt"
FLAGGED_LOG = "./flagged_videos.txt"
REVIEW_PROGRESS_PATH = "./info_corrections/review_progress.json"
RELOAD_INTERVAL = 15  # reload the model after this many GENERATED (not just
# viewed) videos in one session. A long-running process accumulates GPU
# memory pressure across many generate() calls even with per-call
# gc.collect()/empty_cache() cleanup - confirmed directly: the same video
# that succeeds reliably as an isolated single-video run starts hitting
# CUDA OOM by the 2nd-3rd video when several are generated back to back in
# one process. Periodic reload is the most reliable way to reset that
# state without breaking the interactive review flow.

FLAG_CATEGORIES = {
    "t": "road_type",
    "c": "road_context",
    "m": "motion",
    "e": "environment",
    "v": "vehicle_behavior",
    "p": "pedestrian_behavior",
    "w": "visibility",
    "i": "identification",
}

# Reference facts (motion, road type/lane count, time of day, road surface,
# weather) are read from the sibling info.txt of each clip and injected as
# fixed facts into both prompts, instead of asking the model to visually
# determine them. Validated against frame-level ground truth on several
# videos and found more reliable than the model's own visual guesses for
# these specific fields (esp. lane count and turning-vs-straight, which the
# model got wrong even after many prompt iterations). The model no longer
# states which SPECIFIC lane (leftmost/middle/rightmost) the ego-vehicle is
# in - only the lane count from info.txt is used - since that specific claim
# remained unreliable across every prompt variant tried. The 'l'/'t' flags
# below still apply if info.txt itself turns out to be wrong for a video.
INFO_KEYS = ["motion", "road_context", "road_type", "time_of_day", "surface", "weather"]

# Per-video field overrides for confirmed info.txt errors, keyed by video_path.
# The original info.txt files are never modified - this file is applied on
# top of them at read time instead. Only add an entry here once a mismatch
# between info.txt and reality has been directly confirmed (e.g. against the
# actual video), not from automated heuristics alone - see
# prompt_iteration_summary.md for how each entry was verified.
INFO_CORRECTIONS_PATH = "./info_corrections/info_corrections.json"


def _load_info_corrections():
    if not os.path.isfile(INFO_CORRECTIONS_PATH):
        return {}
    with open(INFO_CORRECTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


INFO_CORRECTIONS = _load_info_corrections()


def read_info(video_path):
    """Read the 6-line info.txt sibling of the clip folder (one level above 1_clip/),
    then apply any confirmed per-video corrections from INFO_CORRECTIONS_PATH -
    info.txt itself is never touched."""
    clip_dir = os.path.dirname(video_path)   # .../XX/1_clip
    parent_dir = os.path.dirname(clip_dir)   # .../XX
    info_path = os.path.join(parent_dir, "info.txt")
    with open(info_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    info = dict(zip(INFO_KEYS, lines))
    overrides = INFO_CORRECTIONS.get(video_path)
    if overrides:
        info.update(overrides)
    return info


# Cache of verified grounding facts (YOLO detection + OCR text, each
# individually confirmed via crop+VQA - see combined_hint.get_combined_facts)
# keyed by video_path, so regenerating a video's captions after a prompt
# change doesn't repeat the detection+verification work.
GROUNDING_CACHE_PATH = "./info_corrections/grounding_cache.json"


def _load_grounding_cache():
    if not os.path.isfile(GROUNDING_CACHE_PATH):
        return {}
    with open(GROUNDING_CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


GROUNDING_CACHE = _load_grounding_cache()


def get_grounding_facts(state, video_path, ego_motion=None):
    """Returns the cached verified grounding facts_text for this video,
    computing and caching it on first use. Returns "" (and logs a failure
    rather than raising) if the detection/OCR pipeline itself errors on
    this video - a scan failure shouldn't block caption generation.
    ego_motion (info.txt's motion field) is only used on a cache miss - see
    detection_hint.TURNING_EGO_MOTIONS."""
    if video_path in GROUNDING_CACHE:
        return GROUNDING_CACHE[video_path]
    try:
        facts_text = get_combined_facts(video_path, state.model, state.processor, ego_motion=ego_motion)
    except Exception as e:
        log_failure(video_path, f"grounding scan failed: {e!r}")
        facts_text = ""
    GROUNDING_CACHE[video_path] = facts_text
    os.makedirs(os.path.dirname(GROUNDING_CACHE_PATH), exist_ok=True)
    with open(GROUNDING_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(GROUNDING_CACHE, f, ensure_ascii=False, indent=2, sort_keys=True)
    return facts_text


def format_reference_facts(info):
    # info.txt's road_type enum uses "highway" only as a road-width label
    # (widest administrative tier, 광로) - not an actual limited-access
    # expressway. Left as-is, the model parrots "highway" verbatim even in
    # city-street scenes, which misleads readers. Normalize to "road" here
    # (deterministic string fix, not a prompt instruction) so it can't recur.
    road_type = info["road_type"].replace("highway", "road")
    return (
        f"- Ego-vehicle's driving motion: {info['motion']}\n"
        f"- Road context: {info['road_context']}\n"
        f"- Road type: {road_type}\n"
        f"- Time of day: {info['time_of_day']}\n"
        f"- Road surface: {info['surface']}\n"
        f"- Weather: {info['weather']}"
    )


PROMPT_DESCRIPTION_TEMPLATE = (
    "You are an expert at analyzing driving videos for building an autonomous "
    "driving dataset. The following facts about this clip have already been "
    "determined from reference annotation - treat them as given and "
    "incorporate them naturally; do not re-derive or contradict them:\n"
    "{reference_facts}\n\n"
    "IMPORTANT: if the facts above already give a vehicle's color and/or "
    "type at a position (e.g. 'truck (ahead)', 'white sedan (right)'), use "
    "those exact words for it if you mention it - never swap in a "
    "different color or type word from your own guess. If a position has "
    "no color given, don't invent one of your own.\n\n"
    "IMPORTANT: even though the counts above (e.g. '4 people (right)') are "
    "labeled approximate, never restate them as an exact number (e.g. "
    "'four pedestrians') in the caption - vehicle/pedestrian tracking can "
    "still over- or under-count when someone is occluded or re-detected. "
    "If you mention how many, use a vague quantifier instead (e.g. "
    "'several', 'a few', 'multiple') rather than a specific digit or "
    "number word.\n\n"
    "Write one compact paragraph (about 60-100 words) describing this "
    "driving video, centered on the ego-vehicle. Lead with what the "
    "ego-vehicle is doing (using the motion and road type given above, and "
    "where/when in the video), referencing an intersection, street name, or "
    "sign only if clearly visible. Be strict about what counts as a "
    "road/directional sign: only overhead or roadside signs in the standard "
    "Korean traffic-sign format (blue or green signs listing place names, "
    "route names/numbers, or directions, typically mounted on a gantry at "
    "or near an intersection) count. Text on buildings, storefronts, "
    "business signboards, or shop/brand names is commercial signage, not a "
    "road sign, even if it looks prominent, is elevated, or is shaped like "
    "a sign - never read, transcribe, or mention it, no matter how "
    "sign-like or place-name-like it looks. If a genuine road/traffic sign "
    "shows English text (e.g. a bilingual sign like '언주로 Eonju-ro'), use "
    "that English text exactly as printed. If a genuine road/traffic sign "
    "has only Korean text with no English printed on it (e.g. a regulatory "
    "sign like '추월금지'), write the Korean text exactly as shown (Hangul) "
    "instead - do not attempt to romanize or translate Korean text "
    "yourself, since that is often inaccurate. Do not state "
    "which specific lane (e.g. "
    "'leftmost', 'middle', 'rightmost') the ego-vehicle is in - only the "
    "road type/lane count given above may be mentioned. Then briefly add "
    "only the details that matter for this specific scene - e.g. a notable "
    "nearby vehicle or pedestrian, a relevant roadside hazard, or a road "
    "marking such as a crosswalk or speed bump if clearly visible - but "
    "don't confuse the two: check the COLOR first - if the marking is "
    "plain white stripes, it's a crosswalk; if yellow paint is mixed in "
    "with the white (a yellow-and-white marking), it's a speed bump, not "
    "a crosswalk, regardless of the exact pattern shape. If "
    "mentioning another vehicle, describe only its general type, color, and "
    "position relative to the ego-vehicle (e.g. 'a black SUV in the "
    "adjacent lane') - use generic type words only (sedan, SUV, van, bus, "
    "taxi, truck, motorcycle). The position facts above (ahead/left/right) "
    "are only a rough direction bucket, not lane-level information - never "
    "claim a vehicle is in the 'same lane' as the ego-vehicle, since that "
    "is a stronger claim than what was actually confirmed; use the given "
    "position word itself (e.g. 'a truck ahead'), or 'adjacent lane' only "
    "if a separate lane is actually visible in the frames. If the facts "
    "given above already state a "
    "vehicle's color and/or type at that position (e.g. 'truck (ahead)', "
    "'white sedan (right)'), use those exact words for it - don't "
    "substitute different ones (e.g. 'van', or a different color) from "
    "your own guess, and don't add a color of your own if none was given. "
    "Never guess or state a specific brand or "
    "model name (e.g. 'Mercedes', 'Audi Q5', 'Tivoli'), since this is "
    "frequently misidentified even when the vehicle type/color is obvious. "
    "Never read or mention a license plate number. Do not describe its "
    "maneuvers (overtaking, merging, "
    "cutting in, etc.); maneuvers belong only to the ego-vehicle, unless the "
    "other vehicle causes a genuine, clearly visible hazard. One more "
    "exception: if the facts given above include a confirmed lateral-"
    "movement note for a vehicle (e.g. drifting toward a lane, or growing "
    "noticeably closer suggesting a merge), reflect that movement instead "
    "of defaulting to static language like 'parked' or 'positioned' - "
    "that note has already been verified, unlike a maneuver you'd be "
    "guessing at yourself. Avoid "
    "restating similar things multiple times or listing every object you "
    "notice. Skip anything generic or repetitive. Do not use headings, "
    "bullet points, or labels - just one tight paragraph. Only describe "
    "additional things you can clearly see in the frames beyond the facts "
    "given above; do not assume or guess at typical scene elements (e.g. "
    "pedestrians, specific vehicles) that are not actually visible."
)

PROMPT_STRUCTURED_TEMPLATE = (
    "You are an expert at analyzing driving videos for building an autonomous "
    "driving dataset. The following facts about this clip have already been "
    "determined from reference annotation - treat them as given and "
    "incorporate them naturally; do not re-derive or contradict them:\n"
    "{reference_facts}\n\n"
    "IMPORTANT: even though the counts above (e.g. '4 people (right)') are "
    "labeled approximate, never restate them as an exact number (e.g. "
    "'four pedestrians') in the caption - vehicle/pedestrian tracking can "
    "still over- or under-count when someone is occluded or re-detected. "
    "If you mention how many, use a vague quantifier instead (e.g. "
    "'several', 'a few', 'multiple') rather than a specific digit or "
    "number word.\n\n"
    "Looking at the frames in order, write one natural, flowing paragraph "
    "that covers each of the following:\n"
    "1. Driving state - the motion given above (e.g. going straight, "
    "turning, stopped), plus the road type/lane count given above. Do not "
    "state which specific lane (e.g. 'leftmost', 'middle', 'rightmost') the "
    "ego-vehicle is in.\n"
    "2. Road surface and weather conditions - use the surface and weather "
    "given above.\n"
    "3. Driving environment - use the road context/type given above, plus "
    "any road structures such as a bridge, tunnel, overpass, or underpass "
    "if clearly visible.\n"
    "4. Surrounding environment - notable nearby vehicles, pedestrians, "
    "traffic lights, signs, road markings (including crosswalks and speed "
    "bumps if clearly visible - don't confuse the two: check the COLOR "
    "first - if the marking is plain white stripes, it's a crosswalk; if "
    "yellow paint is mixed in with the white (a yellow-and-white marking), "
    "it's a speed bump, not a crosswalk, regardless of the exact pattern "
    "shape), buildings, or nature. Be strict "
    "about what counts as a road/directional sign: only overhead or "
    "roadside signs in the standard Korean traffic-sign format (blue or "
    "green signs listing place names, route names/numbers, or directions, "
    "typically mounted on a gantry at or near an intersection) count. Text "
    "on buildings, storefronts, business signboards, or shop/brand names is "
    "commercial signage, not a road sign, even if it looks prominent, is "
    "elevated, or is shaped like a sign - never read, transcribe, or "
    "mention it, no matter how sign-like or place-name-like it looks. If a "
    "genuine road/traffic sign shows English text (e.g. a bilingual sign "
    "like '언주로 Eonju-ro'), use that English text exactly as printed. If a "
    "genuine road/traffic sign has only Korean text with no English printed "
    "on it (e.g. a regulatory sign like '추월금지'), write the Korean text "
    "exactly as shown (Hangul) instead - do not attempt to romanize or "
    "translate Korean text yourself, since that is often inaccurate. For other "
    "vehicles, describe only their general type, color, and position relative "
    "to the ego-vehicle (e.g. 'a black SUV in the adjacent lane', 'a white bus "
    "ahead') - use generic type words only (sedan, SUV, van, bus, taxi, "
    "truck, motorcycle). The position facts above (ahead/left/right) are "
    "only a rough direction bucket, not lane-level information - never "
    "claim a vehicle is in the 'same lane' as the ego-vehicle, since that "
    "is a stronger claim than what was actually confirmed; use the given "
    "position word itself (e.g. 'a truck ahead'), or 'adjacent lane' only "
    "if a separate lane is actually visible in the frames. If the facts "
    "given above already state a vehicle's "
    "color and/or type at that position (e.g. 'truck (ahead)', 'white "
    "sedan (right)'), use those exact words for it - don't substitute "
    "different ones (e.g. 'van', or a different color) from your own "
    "guess, and don't add a color of your own if none was given. Never "
    "guess or state a specific brand or model name "
    "(e.g. 'Mercedes', 'Audi Q5', 'Tivoli'), since this is frequently "
    "misidentified even when the vehicle type/color is obvious. Never read "
    "or mention a license plate number. Do not describe their maneuvers "
    "(overtaking, merging, cutting "
    "in, changing lanes, etc.); maneuver descriptions are reserved for the "
    "ego-vehicle's own driving state above. There are two exceptions: (a) a "
    "genuine hazard or sudden event involving another vehicle, which "
    "belongs in point 5 below, not here; (b) if the facts given above "
    "include a confirmed lateral-movement note for a vehicle (e.g. "
    "drifting toward a lane, or growing noticeably closer suggesting a "
    "merge), reflect that movement here instead of defaulting to static "
    "language like 'parked' or 'positioned' - that note has already been "
    "verified, unlike a maneuver you'd be guessing at yourself.\n"
    "5. Notable points - any hazards, sudden situations, or unusual events "
    "clearly visible in the frames (e.g. another vehicle abruptly cutting "
    "into the ego-vehicle's lane, sudden braking, a near-collision) - state "
    "plainly if nothing notable occurred rather than inventing an event.\n"
    "Output format: one natural narrative paragraph, no labels, headings, or "
    "bullet points. Only describe additional things you can clearly see "
    "beyond the facts given above; do not guess at or invent unverifiable "
    "details - omit them instead."
)


def _natural_sort_key(path):
    # Split on digit runs so "drive_2" sorts before "drive_10" (plain string
    # sort would put "drive_10".."drive_19" before "drive_2", since '1' < '2').
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path)]


def find_videos():
    pattern = os.path.join(SAMPLE_ROOT, "**", "1_clip", "5.mp4")
    return sorted(glob.glob(pattern, recursive=True), key=_natural_sort_key)


def output_paths(video_path):
    clip_dir = os.path.dirname(video_path)   # .../XX/1_clip
    parent_dir = os.path.dirname(clip_dir)   # .../XX
    out_dir = os.path.join(parent_dir, "sb_caption")
    return out_dir, os.path.join(out_dir, "1_caption.txt"), os.path.join(out_dir, "2_caption.txt")


def is_done(video_path):
    _, p1, p2 = output_paths(video_path)
    return os.path.isfile(p1) and os.path.isfile(p2)


def load_model():
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor


class ModelState:
    """Holds the current model/processor behind ONE mutable reference that
    every caller shares, instead of passing model/processor around as
    plain local variables. This matters for reload(): if a caller's own
    `model` variable were passed into a helper as an argument, deleting
    the parameter inside that helper only drops the helper's own local
    binding - the caller's variable still references the old object the
    whole time (until the helper returns and the caller reassigns it),
    so the old and new models would briefly coexist in GPU memory at
    exactly the moment reload is trying to free space. Mutating
    state.model/state.processor in place avoids that: there's only ever
    one reference, visible to every holder of `state`, and clearing it
    here actually drops the last reference before loading a fresh one."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def ensure_loaded(self):
        """Loads the model on first actual use instead of at startup, so a
        session that only ever looks at already-done videos (no generation)
        never touches the GPU at all - lets a review-only session run
        alongside a separate --auto process without competing for memory."""
        if self.model is None:
            print("Loading model...")
            self.model, self.processor = load_model()

    def reload(self):
        print("  reloading model to clear accumulated GPU memory...")
        self.model = None
        self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
        self.model, self.processor = load_model()


MIN_NUM_FRAMES = 8  # floor for the frame-count OOM fallback in run_prompt -
# below this, temporal detail would be too sparse to trust caption
# quality, so at that point just let the error propagate up to
# generate_with_recovery's reload+retry instead.
FRAME_STEP_DOWN = 4  # how many fewer frames to try per fallback step


def _generate_once(state, video_path, prompt, num_frames):
    """One attempt at building inputs and generating, at a given frame
    count. Always cleans up its own tensors before returning/raising."""
    model, processor = state.model, state.processor
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        num_frames=num_frames,
        fps=None,
    )
    inputs = inputs.to(model.device)
    try:
        # no_repeat_ngram_size guards against a real failure mode confirmed
        # directly: greedy decoding (do_sample=False) occasionally gets
        # stuck repeating the exact same sentence dozens of times until it
        # hits max_new_tokens (found in video #263's 1_caption, 449 words
        # of "A white sedan is directly ahead..." repeated verbatim). Since
        # decoding is deterministic, simply retrying with identical input
        # would just reproduce the same loop - blocking any 4-gram from
        # repeating prevents the loop without affecting normal prose,
        # which doesn't naturally reuse a specific 4-word sequence anyway.
        generated_ids = model.generate(
            **inputs, max_new_tokens=512, do_sample=False, no_repeat_ngram_size=4
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
    finally:
        # Runs even on exception (e.g. CUDA OOM) so a failure on one video
        # doesn't leave fragmented/leaked memory that dooms the next one.
        del inputs
        if "generated_ids" in locals():
            del generated_ids
        if "generated_ids_trimmed" in locals():
            del generated_ids_trimmed
        gc.collect()
        torch.cuda.empty_cache()


def run_prompt(state, video_path, prompt):
    try:
        return _generate_once(state, video_path, prompt, NUM_FRAMES)
    except torch.OutOfMemoryError:
        pass
    # Fragmentation from the many small verification calls the grounding
    # scan makes (see get_grounding_facts above) can tip this heavier call
    # over the edge even though each of those cleans up after itself - one
    # hard cleanup + retry at the SAME frame count usually recovers if it
    # really was just fragmentation. Cleaning up only after exiting the
    # except block matters: Python keeps the failed call's traceback (and
    # everything it references) alive for as long as we're still inside
    # the except block, so gc.collect() called from in there can't
    # actually reclaim that memory yet.
    gc.collect()
    torch.cuda.empty_cache()
    try:
        return _generate_once(state, video_path, prompt, NUM_FRAMES)
    except torch.OutOfMemoryError:
        pass
    # Still OOM even fresh - this isn't fragmentation, it's this specific
    # video's own peak requirement at NUM_FRAMES genuinely exceeding what's
    # available (confirmed directly: one video failed identically across
    # 4 separate attempts, including full model reloads, but succeeded
    # immediately once frame count was reduced). Step down frame count
    # instead of failing the whole video.
    num_frames = NUM_FRAMES
    gc.collect()
    torch.cuda.empty_cache()
    while num_frames > MIN_NUM_FRAMES:
        num_frames = max(MIN_NUM_FRAMES, num_frames - FRAME_STEP_DOWN)
        print(f"  still OOM at full frame count, retrying with {num_frames} frames...")
        try:
            return _generate_once(state, video_path, prompt, num_frames)
        except torch.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
    raise torch.OutOfMemoryError(
        f"CUDA OOM even at the minimum {MIN_NUM_FRAMES} frames"
    )


def generate_and_save(state, video_path, force=False):
    out_dir, p1, p2 = output_paths(video_path)
    os.makedirs(out_dir, exist_ok=True)

    need_desc = force or not os.path.isfile(p1)
    need_structured = force or not os.path.isfile(p2)

    reference_facts = None
    if need_desc or need_structured:
        info = read_info(video_path)
        reference_facts = format_reference_facts(info)
        grounding_facts = get_grounding_facts(state, video_path, ego_motion=info["motion"])
        if grounding_facts:
            reference_facts += "\n" + grounding_facts

    if need_desc:
        print("  generating 1_caption...")
        prompt_description = PROMPT_DESCRIPTION_TEMPLATE.format(reference_facts=reference_facts)
        desc = run_prompt(state, video_path, prompt_description)
        with open(p1, "w", encoding="utf-8") as f:
            f.write(desc + "\n")
    else:
        with open(p1, "r", encoding="utf-8") as f:
            desc = f.read().strip()

    if need_structured:
        print("  generating 2_caption...")
        prompt_structured = PROMPT_STRUCTURED_TEMPLATE.format(reference_facts=reference_facts)
        structured = run_prompt(state, video_path, prompt_structured)
        with open(p2, "w", encoding="utf-8") as f:
            f.write(structured + "\n")
    else:
        with open(p2, "r", encoding="utf-8") as f:
            structured = f.read().strip()

    return desc, structured


MAX_OOM_RETRIES = 3  # a single reload+retry isn't always enough - a
# grounding-heavy video's OWN peak memory need (many small verification
# calls immediately followed by one large 16-frame generation call) can
# sit right at the 12GB ceiling, so even a freshly-reloaded model can hit
# the exact same OOM on the very next attempt (confirmed directly: retried
# once after reload, failed again with byte-for-byte identical CUDA error
# numbers - not cross-video accumulation, since this was the first video
# of the session). Retrying several times with a reload before each
# attempt gives allocator-timing variance more chances to land in the
# model's favor, which is the only lever left once accumulation itself
# has been ruled out.


def generate_with_recovery(state, video_path, force, generations_since_reload):
    """Runs generate_and_save, recovering from CUDA OOM by reloading the
    model and retrying up to MAX_OOM_RETRIES times. Also applies the
    scheduled reload on success, same as before. Mutates state.model/
    state.processor in place on reload (see ModelState) - returns
    (desc, structured, generations_since_reload)."""
    state.ensure_loaded()
    attempt = 0
    while True:
        oom_occurred = False
        oom_error = None
        try:
            desc, structured = generate_and_save(state, video_path, force=force)
        except torch.OutOfMemoryError as e:
            oom_occurred = True
            oom_error = repr(e)

        if not oom_occurred:
            break
        attempt += 1
        if attempt > MAX_OOM_RETRIES:
            raise RuntimeError(f"CUDA OOM after {MAX_OOM_RETRIES} reload+retries: {oom_error}")
        # Report the error message before reloading, then let go of it -
        # the exception's traceback keeps the failed call's tensors alive
        # for as long as we're still inside the except block, which would
        # defeat the reload's whole purpose if we tried to clean up from
        # in there.
        print(f"  OOM (attempt {attempt}/{MAX_OOM_RETRIES}), reloading model and retrying: {oom_error}")
        state.reload()

    generations_since_reload = 0 if attempt else generations_since_reload + 1
    if generations_since_reload >= RELOAD_INTERVAL:
        state.reload()
        generations_since_reload = 0
    return desc, structured, generations_since_reload


def log_failure(video_path, error):
    with open(FAILED_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()}\t{video_path}\t{error!r}\n")


def log_flag(video_path, category, note=""):
    with open(FLAGGED_LOG, "a", encoding="utf-8") as f:
        line = f"{datetime.now().isoformat()}\t{video_path}\t{category}"
        if note:
            line += f"\t{note}"
        f.write(line + "\n")


def log_info_correction(video_path, field, value):
    """Add/update a confirmed info.txt override for this video in
    INFO_CORRECTIONS_PATH - never touches the original info.txt file."""
    corrections = _load_info_corrections()
    corrections.setdefault(video_path, {})[field] = value
    with open(INFO_CORRECTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(corrections, f, ensure_ascii=False, indent=2, sort_keys=True)
    INFO_CORRECTIONS[video_path] = corrections[video_path]


def save_review_progress(index, video_path):
    """Records the last video actually shown to the user (any interactive
    mode - normal resume or --review-range), so a later session can answer
    "where did I leave off" without reconstructing it from flag/correction
    timestamps, which only capture videos that got an action, not every
    video actually viewed. Overwritten on every show(), not appended -
    only the single most recent position matters."""
    os.makedirs(os.path.dirname(REVIEW_PROGRESS_PATH), exist_ok=True)
    with open(REVIEW_PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"index": index + 1, "video_path": video_path, "timestamp": datetime.now().isoformat()},
            f, ensure_ascii=False, indent=2,
        )


def load_review_progress():
    """Returns the saved {index, video_path, timestamp} dict, or None if
    no review session has ever shown a video yet."""
    if not os.path.isfile(REVIEW_PROGRESS_PATH):
        return None
    with open(REVIEW_PROGRESS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing(video_path):
    _, p1, p2 = output_paths(video_path)
    with open(p1, "r", encoding="utf-8") as f:
        desc = f.read().strip()
    with open(p2, "r", encoding="utf-8") as f:
        structured = f.read().strip()
    return desc, structured


def open_video_player(video_path):
    return subprocess.Popen(
        ["totem", video_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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


def show(index, total, video_path, desc, structured, regenerated=False):
    save_review_progress(index, video_path)
    print("\n" + "=" * 80)
    tag = " (regenerated)" if regenerated else ""
    print(f"[{index + 1}/{total}]{tag} {video_path}")
    print("-" * 80)
    print("1_caption:")
    print(desc)
    print("-" * 80)
    print("2_caption:")
    print(structured)
    print("=" * 80)
    print("[Enter] next  [Backspace] previous  [r] regenerate this one  [q] quit")
    print("[t] flag road_type  [c] flag road_context  [m] flag motion  "
          "[e] flag environment")
    print("[v] flag other-vehicle behavior  [p] flag pedestrian behavior  "
          "[w] flag visibility")
    print("[i] flag sign identification  [o] flag other  "
          "[x] correct info.txt (confirmed error only)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regenerate-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Force-regenerate videos START..END (1-indexed, inclusive), "
             "even if already done. Videos outside this range are handled normally.",
    )
    parser.add_argument(
        "--regenerate-indices",
        nargs="+",
        type=int,
        metavar="N",
        help="Force-regenerate specific (not necessarily contiguous) "
             "1-indexed video numbers, e.g. --regenerate-indices 2 6. Can be "
             "combined with --regenerate-range.",
    )
    parser.add_argument(
        "--flag-summary",
        action="store_true",
        help="Print per-category counts from flagged_videos.txt and exit "
             "(no model loading, no review session).",
    )
    parser.add_argument(
        "--review-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Browse only already-done videos START..END (1-indexed, "
             "inclusive) - never generates, so the model is never loaded "
             "unless you press 'r' to explicitly regenerate one. Not-yet-"
             "done videos in the range are skipped (not generated). Safe "
             "to run at the same time as a separate --auto process, since "
             "it won't touch the GPU on its own.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Unattended batch mode: generate and save captions for each "
             "not-yet-done (or forced) video back-to-back, without opening "
             "a video player or waiting for a keypress. Intended for "
             "background/overnight runs - review normally afterward. Can "
             "be combined with --regenerate-range/--regenerate-indices. "
             "Stop with Ctrl+C; progress is saved per-video, so rerunning "
             "resumes where it left off.",
    )
    return parser.parse_args()


def print_flag_summary():
    if not os.path.isfile(FLAGGED_LOG):
        print(f"No {FLAGGED_LOG} yet - nothing flagged so far.")
        return

    counts = {}
    total = 0
    with open(FLAGGED_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            category = parts[2] if len(parts) > 2 else "unknown"
            counts[category] = counts.get(category, 0) + 1
            total += 1

    print(f"{FLAGGED_LOG}: {total} flagged total")
    for category, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {category}: {count}")


def main():
    args = parse_args()

    if args.flag_summary:
        print_flag_summary()
        return

    videos = find_videos()
    if not videos:
        print(f"No 5.mp4 files found under {SAMPLE_ROOT}")
        return
    total = len(videos)

    if args.review_range:
        # Browse-only: restrict the queue to already-done videos in this
        # range, with force_range/force_indices left empty so force_this is
        # always False - the main loop then always takes the "already done"
        # branch (load_existing, no generate_with_recovery call), so the
        # model never loads unless the user explicitly presses 'r'.
        start, end = args.review_range
        lo, hi = max(start - 1, 0), min(end - 1, total - 1)
        seq = [i for i in range(lo, hi + 1) if is_done(videos[i])]
        skipped = (hi - lo + 1) - len(seq)
        if skipped:
            print(f"Note: {skipped} not-yet-done video(s) in that range are skipped (not generated).")
        if not seq:
            print("No already-done videos in that range.")
            return
        start_pos = 0
        force_range = None
        force_indices = set()
    else:
        force_range = None
        if args.regenerate_range:
            start, end = args.regenerate_range
            force_range = (max(start - 1, 0), min(end - 1, total - 1))

        force_indices = set()
        if args.regenerate_indices:
            force_indices = {i - 1 for i in args.regenerate_indices if 1 <= i <= total}

        force_zone = set(force_indices)
        if force_range:
            force_zone |= set(range(force_range[0], force_range[1] + 1))

        if force_zone:
            # Visit the forced videos first (in order), then seamlessly continue
            # into the rest of the dataset - jumping straight to the next
            # not-yet-done video instead of stepping through every already-done
            # one in between.
            zone_seq = sorted(force_zone)
            resume_from = next(
                (i for i in range(total) if i not in force_zone and not is_done(videos[i])),
                None,
            )
            tail = [i for i in range(resume_from, total) if i not in force_zone] if resume_from is not None else []
            seq = zone_seq + tail
            start_pos = 0
        else:
            seq = list(range(total))
            start_pos = next((p for p, i in enumerate(seq) if not is_done(videos[i])), len(seq) - 1)

    seq_total = len(seq)

    print(f"Found {total} videos.")
    # Model isn't loaded here - state starts empty and ensure_loaded() (see
    # ModelState/generate_with_recovery) loads it on first actual generation.
    # A pure review session over already-done videos then never touches the
    # GPU, so it can run alongside a separate --auto process without either
    # one fighting the other for memory.
    state = ModelState(None, None)
    generations_since_reload = 0

    if args.auto:
        print(f"Auto mode: {seq_total} videos in the queue, unattended "
              "(no player, no review prompts). Stop with Ctrl+C; rerun to resume.")
        done_count = 0
        fail_count = 0
        for pos, index in enumerate(seq):
            video_path = videos[index]
            force_this = (
                (force_range is not None and force_range[0] <= index <= force_range[1])
                or index in force_indices
            )
            if not force_this and is_done(video_path):
                continue
            tag = "force-regenerating" if force_this else "processing"
            print(f"\n[{pos + 1}/{seq_total}] (video #{index + 1}/{total}) {tag} {video_path}")
            try:
                _desc, _structured, generations_since_reload = generate_with_recovery(
                    state, video_path, force_this, generations_since_reload
                )
                done_count += 1
            except Exception as e:
                print(f"  FAILED: {e!r}")
                log_failure(video_path, e)
                fail_count += 1
        print(f"\nAuto mode done: {done_count} generated, {fail_count} failed "
              "(see failed_videos.txt). Rerun without --auto to review.")
        return

    last_progress = load_review_progress()
    if last_progress:
        print(f"(last reviewed: #{last_progress['index']}/{total} "
              f"{last_progress['video_path']} - {last_progress['timestamp']})")

    pos = start_pos
    player_proc = None
    try:
        while True:
            index = seq[pos]
            video_path = videos[index]
            force_this = (
                (force_range is not None and force_range[0] <= index <= force_range[1])
                or index in force_indices
            )

            if force_this or not is_done(video_path):
                tag = "force-regenerating" if force_this else "processing"
                print(f"\n[{pos + 1}/{seq_total}] (video #{index + 1}/{total}) {tag} {video_path}")
                try:
                    desc, structured, generations_since_reload = generate_with_recovery(
                        state, video_path, force_this, generations_since_reload
                    )
                except Exception as e:
                    print(f"  FAILED: {e!r}")
                    log_failure(video_path, e)
                    pos += 1
                    if pos >= seq_total:
                        print("\nAll videos processed (some failures - see failed_videos.txt).")
                        return
                    continue
                regenerated = force_this
            else:
                print(f"\n[{pos + 1}/{seq_total}] (video #{index + 1}/{total}) "
                      f"already done, showing existing result {video_path}")
                desc, structured = load_existing(video_path)
                regenerated = False

            close_video_player(player_proc)
            player_proc = open_video_player(video_path)
            show(index, total, video_path, desc, structured, regenerated)

            while True:
                key = getch()
                if key in ("\r", "\n"):
                    pos += 1
                    if pos >= seq_total:
                        print("\nAll videos processed.")
                        return
                    break
                elif key == "\x7f" or key == "\x08":
                    if pos == 0:
                        print("Already at the first video.")
                        continue
                    pos -= 1
                    break
                elif key in ("r", "R"):
                    print(f"\n[{index + 1}/{total}] regenerating {video_path}")
                    try:
                        desc, structured, generations_since_reload = generate_with_recovery(
                            state, video_path, True, generations_since_reload
                        )
                    except Exception as e:
                        print(f"  FAILED: {e!r}")
                        log_failure(video_path, e)
                        continue
                    show(index, total, video_path, desc, structured, regenerated=True)
                    continue
                elif key in ("q", "Q"):
                    print("\nQuitting. Progress is saved - rerun to resume.")
                    return
                elif key.lower() in FLAG_CATEGORIES:
                    category = FLAG_CATEGORIES[key.lower()]
                    note = input(f"  {category} - describe briefly (optional, Enter to skip): ")
                    log_flag(video_path, category, note)
                    print(f"  flagged: {category}")
                    continue
                elif key in ("o", "O"):
                    note = input("  other - describe briefly: ")
                    log_flag(video_path, "other", note)
                    print("  flagged: other")
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
                    continue
    finally:
        close_video_player(player_proc)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved - rerun to resume.")
