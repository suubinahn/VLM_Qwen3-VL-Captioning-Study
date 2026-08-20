"""
Qwen3-VL driving video captioning - combined test (task 1 + task 2, one model load).

No stability safeguards yet (no try/except, no resume/skip logic) - this is just
a test to run both prompts back-to-back on a single video before scaling up.

Reference facts (motion, road type/lane count, time of day, road surface, weather)
are read from the sibling info.txt of each clip and injected as fixed facts into
both prompts, instead of asking the model to visually determine them. Validated
against frame-level ground truth on several videos and found more reliable than
the model's own visual guesses for these specific fields (esp. lane count and
turning-vs-straight, which the model got wrong even after many prompt iterations).
The model no longer states which SPECIFIC lane (leftmost/middle/rightmost) the
ego-vehicle is in - only the lane count from info.txt is used - since that specific
claim remained unreliable across every prompt variant tried. Human review via
caption_review_tool.py's flagging feature remains the safety net for the rare
cases where info.txt itself might be wrong.

Appends each prompt + output to its own experiment log, same as
1_caption_test.py / 2_caption_test.py.
"""
import gc
import json
import os
from datetime import datetime

# Must be set before the first CUDA allocation (i.e. before torch/transformers
# touch the GPU) - lets NUM_FRAMES=16 fit in the 12GB GPUs by reducing
# allocator fragmentation. Without this, 16 (and even 12) frames OOM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_PATH = "./Qwen3-VL-8B-Instruct"
VIDEO_PATH = "./sample_videos/231106095132_M801C06L62G031_2673_end_extract/drive_2/01/1_clip/5.mp4"
NUM_FRAMES = 16

LOG_1 = "./experiments/1_caption_experiments.txt"
LOG_2 = "./experiments/2_caption_experiments.txt"

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
    "nearby vehicle or pedestrian, a relevant roadside hazard. If "
    "mentioning another vehicle, describe only its general type, color, and "
    "position relative to the ego-vehicle (e.g. 'a black SUV in the "
    "adjacent lane') - use generic type words only (sedan, SUV, van, bus, "
    "taxi, truck, motorcycle); never guess or state a specific brand or "
    "model name (e.g. 'Mercedes', 'Audi Q5', 'Tivoli'), since this is "
    "frequently misidentified even when the vehicle type/color is obvious. "
    "Never read or mention a license plate number. Do not describe its "
    "maneuvers (overtaking, merging, "
    "cutting in, etc.); maneuvers belong only to the ego-vehicle, unless the "
    "other vehicle causes a genuine, clearly visible hazard. Avoid "
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
    "traffic lights, signs, road markings, buildings, or nature. Be strict "
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
    "truck, motorcycle); never guess or state a specific brand or model name "
    "(e.g. 'Mercedes', 'Audi Q5', 'Tivoli'), since this is frequently "
    "misidentified even when the vehicle type/color is obvious. Never read "
    "or mention a license plate number. Do not describe their maneuvers "
    "(overtaking, merging, cutting "
    "in, changing lanes, etc.); maneuver descriptions are reserved for the "
    "ego-vehicle's own driving state above. The only exception is a genuine "
    "hazard or sudden event involving another vehicle, which belongs in "
    "point 5 below, not here.\n"
    "5. Notable points - any hazards, sudden situations, or unusual events "
    "clearly visible in the frames (e.g. another vehicle abruptly cutting "
    "into the ego-vehicle's lane, sudden braking, a near-collision) - state "
    "plainly if nothing notable occurred rather than inventing an event.\n"
    "Output format: one natural narrative paragraph, no labels, headings, or "
    "bullet points. Only describe additional things you can clearly see "
    "beyond the facts given above; do not guess at or invent unverifiable "
    "details - omit them instead."
)


def run_prompt(model, processor, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": VIDEO_PATH},
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
        num_frames=NUM_FRAMES,
        fps=None,
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    del inputs, generated_ids, generated_ids_trimmed
    gc.collect()
    torch.cuda.empty_cache()

    return output_text


def log_result(log_path, prompt, output_text):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
        f.write(f"video: {VIDEO_PATH}\n")
        f.write(f"num_frames: {NUM_FRAMES}\n")
        f.write(f"prompt: {prompt}\n")
        f.write("-" * 80 + "\n")
        f.write(output_text + "\n\n")


model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

info = read_info(VIDEO_PATH)
print(f"[info.txt] {info}")
reference_facts = format_reference_facts(info)

prompt_description = PROMPT_DESCRIPTION_TEMPLATE.format(reference_facts=reference_facts)
prompt_structured = PROMPT_STRUCTURED_TEMPLATE.format(reference_facts=reference_facts)

print("[1/2] Running description prompt...")
description = run_prompt(model, processor, prompt_description)
log_result(LOG_1, prompt_description, description)

print("[2/2] Running structured prompt...")
structured = run_prompt(model, processor, prompt_structured)
log_result(LOG_2, prompt_structured, structured)

print("\n=== 1_caption ===")
print(description)
print("\n=== 2_caption ===")
print(structured)
