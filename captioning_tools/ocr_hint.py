"""
OCR-based prompt grounding: scans many frames across the clip with EasyOCR,
then VERIFIES each surviving candidate before it ever reaches the caption
prompt - this targets "identification" (sign-text misreads), a DIFFERENT
flag category from detection_hint.py's vehicle_behavior/pedestrian_behavior
target (see prompt_iteration_summary.md section 51 and the OCR discussion
that followed it).

Why verification, not just injecting EasyOCR's text as a "hint": earlier
iterations injected raw OCR candidates into the caption prompt with
instructions like "cross-check this against the frames, trust your own
reading if you're confident." That kept failing in new ways each time
(SISTINA storefront text mistaken for a sign; "언주로" correctly read on
its own getting overwritten by EasyOCR's "연주로" misread; a low-confidence
garbled reading of the RAEMIAN archway name slipping through). The root
cause: "trust your own reading" has no real referent when perception and
hint-reconciliation happen in the SAME generation pass - there's no
separate prior reading to compare against, so text sitting explicitly in
the prompt just wins.

The fix used here: treat EasyOCR's (frame, box) hits purely as candidate
REGIONS to zoom into, not as source-of-truth text. Each candidate region is
cropped from its source frame and shown to the VLM on its own, with a
narrow question ("is this a road sign, and if so what does it say").
Cropping doesn't add pixel information, but it changes how the vision
tower's limited token budget gets spent - a whole 16-frame video call
allocates only a handful of tokens to a small distant sign, while a single
cropped image call can spend its full budget there, which is why the model
reads these crops far more reliably than either EasyOCR's raw output or
its own glance at the full frame. Only text the model itself confirms as a
genuine road sign becomes a FACT (merged into reference_facts, same tier
as info.txt), not a hint requiring further judgment calls.

Why dense scanning to find candidate regions at all, not a few
evenly-spaced frames: road signs are small/blurry for most of a clip and
only become legible for a short window as the ego-vehicle approaches
(confirmed directly - a sign unreadable in frames sampled at
10/30/50/70/90% of the clip was caught, correctly, at frame 96/150, right
as the vehicle passed under the gantry). Scanning every few frames and
keeping only the best reading per piece of text is cheap (~14s for 50
frames on GPU) and reliably catches that window.

Tried first: PaddleOCR - hit real environment bugs on both CPU (oneDNN
"not implemented" error) and GPU (cuBLAS invalid-value error on the
Korean recognition model specifically, reproducible across multiple
frames) with this system's installed versions, not fixable via simple
config changes - abandoned in favor of EasyOCR, which worked immediately.

Usage:
  from ocr_hint import build_verified_ocr_facts, format_verified_ocr_facts
  facts = build_verified_ocr_facts(video_path, model, processor)
  fact_text = format_verified_ocr_facts(facts)  # "" if nothing confirmed
  # merge fact_text into reference_facts, same as info.txt facts
"""
import gc
import os
import re

import cv2
import easyocr
import torch
from PIL import Image

FRAME_STEP = 3  # scan every Nth frame - ~50 samples for a 150-frame/10s clip
CONF_THRESHOLD = 0.5
MIN_TEXT_LEN = 2
MAX_ITEMS_IN_HINT = 8
LONG_TEXT_LEN = 4  # phrase-length text (place names etc) - more chars to
CONF_THRESHOLD_LONG_TEXT = 0.75  # get wrong, so held to a higher bar than
# short fragments/numbers - a single garbled-but-confident phrase read
# (e.g. "래미안 대처엘리스" at 0.64 for what should read "레미안 대치팰리스")
# can otherwise slip into the hint and get trusted over the model's own
# correct reading. Genuine strong single-frame reads (e.g. "강남대로" at
# 1.00, the case that originally justified dense scanning) still clear
# this bar easily - this isn't the same as requiring multi-frame repeats,
# which would also kill that legitimate single-frame catch.

_READER = None


def _get_reader():
    global _READER
    if _READER is None:
        _READER = easyocr.Reader(["ko", "en"], gpu=True)
    return _READER


def scan_video_text(video_path, frame_step=FRAME_STEP):
    """Runs EasyOCR on every `frame_step`-th frame of the whole clip.
    Returns a list of (frame_idx, confidence, text, box), unfiltered and
    unsorted, one entry per individual text detection per sampled frame.
    box is EasyOCR's raw quadrilateral [[x,y], [x,y], [x,y], [x,y]] in the
    source frame's pixel coordinates - kept so a candidate region can later
    be cropped and shown to the VLM for verification, see
    build_verified_ocr_facts()."""
    reader = _get_reader()
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    readings = []
    for idx in range(0, total, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        for box, text, conf in reader.readtext(frame, detail=1):
            text = text.strip()
            if conf >= CONF_THRESHOLD and len(text) >= MIN_TEXT_LEN:
                readings.append((idx, float(conf), text, box))
    cap.release()
    return readings


def best_readings(readings):
    """Collapses repeated readings of the same text across nearby frames
    down to its single highest-confidence occurrence (conf, idx, box),
    drops long phrase-like text that didn't clear the higher confidence bar
    (see CONF_THRESHOLD_LONG_TEXT), then returns the top MAX_ITEMS_IN_HINT
    distinct texts by confidence. These are only CANDIDATE regions to zoom
    into - not trusted spellings. Near-duplicate EasyOCR misreads of the
    same underlying sign (e.g. "언주로" vs "연주로") are deliberately left
    for build_verified_ocr_facts() to resolve via crop+VQA, rather than
    filtered here: which exact string EasyOCR used to locate a candidate
    region doesn't matter once the VLM reads the crop itself."""
    best_by_text = {}
    for idx, conf, text, box in readings:
        if text not in best_by_text or conf > best_by_text[text][0]:
            best_by_text[text] = (conf, idx, box)

    best_by_text = {
        t: v for t, v in best_by_text.items()
        if len(t) < LONG_TEXT_LEN or v[0] >= CONF_THRESHOLD_LONG_TEXT
    }

    ranked = sorted(best_by_text.items(), key=lambda kv: -kv[1][0])
    return ranked[:MAX_ITEMS_IN_HINT]


CROP_PADDING = 0.4  # extra margin around the OCR box, as a fraction of its
# own width/height - gives the model a little surrounding context (is this
# mounted on a pole vs. painted on a storefront window) without zooming out
# so far that the text shrinks back down to illegible
CROP_MIN_SIDE = 400  # upscale small crops so the vision tower isn't handed
# a tiny image it has to downsample away anyway


def crop_region(video_path, frame_idx, box, padding=CROP_PADDING):
    """Crops the (padded) OCR box out of its source frame and returns a PIL
    Image, upscaled if small. Returns None if the frame can't be read."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    h, w = frame.shape[:2]
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * padding, bh * padding
    x1 = max(0, int(x1 - pad_x))
    x2 = min(w, int(x2 + pad_x))
    y1 = max(0, int(y1 - pad_y))
    y2 = min(h, int(y2 + pad_y))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    short_side = min(ch, cw)
    if short_side < CROP_MIN_SIDE:
        scale = CROP_MIN_SIDE / short_side
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


VERIFY_PROMPT = (
    "This is a cropped close-up region from a driving-dashcam video frame. "
    "Judge only what's in this image.\n\n"
    "1. Is this an official road/directional traffic sign - overhead or "
    "roadside, typically blue or green background, showing a place name, "
    "route name, or direction? Judge this by the TEXT ITSELF, not the "
    "background color - a blue or green background alone does not make "
    "something a road sign, construction banners/awnings and building "
    "signage are very often blue or green too. Ask: is this text something "
    "a driver would navigate BY (a district/neighborhood/landmark name, a "
    "route/road name, or a direction) - or is it naming a business, "
    "building, or construction project instead (e.g. ending in words like "
    "신축공사/공사중/건설, or a brand-style name followed by a building/lot "
    "number)? The latter is NOT a road sign even mounted overhead or on a "
    "blue/green banner. Answer NO for storefronts, advertisements, "
    "banners, building/construction signage, or license plates.\n"
    "2. If yes, transcribe the text exactly as it appears - words only, no "
    "arrow symbols or other graphical marks. If the sign uses an arrow to "
    "point toward a place name, describe that in words instead (e.g. "
    "'Hannam Bridge, pointing right') rather than including a literal "
    "arrow character.\n\n"
    "Answer in exactly this format, nothing else:\n"
    "IS_ROAD_SIGN: yes or no\n"
    "TEXT: the transcribed text, or NONE"
)

_VERIFY_RE = re.compile(
    r"IS_ROAD_SIGN:\s*(yes|no).*?TEXT:\s*(.*)", re.IGNORECASE | re.DOTALL
)


def verify_candidate(model, processor, crop_image):
    """Shows a single cropped region to the VLM and asks it to judge/read
    it independently of EasyOCR's guess. Returns the transcribed text if
    confirmed as a road sign, else None."""
    if crop_image is None:
        return None
    messages = [{"role": "user", "content": [
        {"type": "image", "image": crop_image},
        {"type": "text", "text": VERIFY_PROMPT},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    oom_occurred = False
    try:
        generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    except torch.OutOfMemoryError:
        oom_occurred = True
    if oom_occurred:
        # Cleaning up only AFTER exiting the except block matters: Python
        # keeps the failed call's traceback (and everything it
        # references) alive for as long as we're still inside the except
        # block, so gc.collect() called from in there can't actually
        # reclaim that memory yet.
        gc.collect()
        torch.cuda.empty_cache()
        generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    reply = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    del inputs, generated_ids, trimmed
    gc.collect()
    torch.cuda.empty_cache()

    m = _VERIFY_RE.search(reply)
    if not m:
        return None
    is_sign = m.group(1).strip().lower() == "yes"
    text = " ".join(m.group(2).split()).strip('"')
    # Safety net for the "no arrow symbols" instruction above - strip any
    # that slip through anyway rather than let a raw glyph like "->" reach
    # the caption prompt.
    text = re.sub(r"[←-⇿➔➠-➿]", "", text).strip()
    if not is_sign or not text or text.upper() == "NONE":
        return None
    return text


def build_verified_ocr_facts(video_path, model, processor, frame_step=FRAME_STEP):
    """Full pipeline: scan for candidate text regions, keep the strongest
    per distinct string, then verify each one independently by cropping and
    asking the VLM. Returns a list of confirmed sign-text strings (VLM's
    own transcription, not EasyOCR's) - empty if nothing confirmed."""
    readings = scan_video_text(video_path, frame_step)
    if not readings:
        return []
    candidates = best_readings(readings)
    if not candidates:
        return []

    facts = []
    seen = set()
    for _text, (_conf, idx, box) in candidates:
        crop = crop_region(video_path, idx, box)
        confirmed_text = verify_candidate(model, processor, crop)
        if confirmed_text and confirmed_text not in seen:
            seen.add(confirmed_text)
            facts.append(confirmed_text)
    return facts


def format_verified_ocr_facts(facts):
    """Formats confirmed sign text as a reference-facts bullet, matching
    format_reference_facts()'s style - this is now a given fact, not a
    hint, so it belongs in the same list as the info.txt-derived facts."""
    if not facts:
        return ""
    quoted = ", ".join(f'"{t}"' for t in facts)
    plural = "s read" if len(facts) > 1 else " reads"
    return f"- Overhead/roadside sign{plural}: {quoted}"


if __name__ == "__main__":
    import sys
    from transformers import AutoModelForImageTextToText, AutoProcessor

    video = sys.argv[1] if len(sys.argv) > 1 else "./sample_videos/231116093002_M801C06L62G031_2673_end_extract/drive_1/03/1_clip/5.mp4"
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(repo_root, "Qwen3-VL-8B-Instruct")
    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(model_path)
    facts = build_verified_ocr_facts(video, model, processor)
    print(format_verified_ocr_facts(facts))
