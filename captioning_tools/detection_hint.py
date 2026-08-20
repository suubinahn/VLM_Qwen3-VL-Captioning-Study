"""
Object-detection-based prompt grounding: runs YOLO on several sampled video
frames, then VERIFIES each surviving (class, position) group and tracked
motion claim before it reaches the caption prompt - external grounding for
vehicle_behavior/pedestrian_behavior (fabricated or missed vehicles/
pedestrians, and fabricated motion claims), the categories the model's own
vision has repeatedly struggled with even after fine-tuning (see
prompt_iteration_summary.md sections 38/46/50). Note: the "identification"
flag category is mostly sign-text misreads, not vehicle misidentification -
a detector doesn't address that, see ocr_hint.py. Section 51 - YOLO(v11m,
closed COCO class set) chosen over open-vocabulary Grounding DINO after a
side-by-side test showed it's far more reliable on this dashcam footage.

Why verification, not just injecting YOLO's counts as a "hint": mirrors
ocr_hint.py's redesign (see that module's docstring for the full
rationale) - a raw hint plus "cross-check against the frames" instruction
gives the model no real anchor to check against inside a single generation
pass, so the text sitting in the prompt tends to just get echoed (or, as
seen in testing, the mere presence of a busy-sounding vehicle hint can
prime the model to add MORE vehicles than were ever detected at all, e.g.
a "white van"/"white bus" appearing in drive_22/01's caption despite never
being in the hint's counts). Cropping each detection's box and asking the
VLM to confirm it on its own, the same way ocr_hint.py verifies text
regions, catches YOLO's own false positives (a detection that isn't really
there). It does NOT fully guarantee the model won't still add something of
its own beyond what was verified - that's a separate over-elaboration
tendency, not a hint-fidelity problem - but it removes bad detections as a
source of it, and every currently-tested case has gotten cleaner, not
worse.

Two layers, both now verified:
  1. Counts/positions - grounds "what vehicles/pedestrians are present and
     roughly where", one count per distinct TRACK (not per-frame
     detection, so the same physical car isn't counted twice). One
     representative detection per (class, position) group is cropped and
     confirmed; if confirmed, the group's full count is kept, if not the
     whole group is dropped rather than guessing which member is real.
  2. Tracked lateral movement - grounds "did this one actually shift
     lanes/cut in". Each surviving track's best-confidence detection is
     cropped and confirmed before its motion note is kept.

Tracking is done with ultralytics' built-in ByteTrack (model.track(),
tracker="bytetrack.yaml" - ships with the ultralytics package already
installed for detection, no extra service/dependency/key needed), run on
every frame of the clip. This replaced an earlier from-scratch greedy
IoU-across-sparse-samples tracker (8-30 evenly-spaced frames, matching
consecutive samples purely by box overlap) that reliably lost fast lateral
movement - a car merging into the ego's lane over 1-2 seconds would shift
too much between two sparse samples for IoU to link them, fragmenting into
several short/singleton "tracks" that never met MIN_TRACK_FRAMES, so the
merge silently vanished (confirmed directly on a real case - drive_4/02's
white SUV cutting in from the right came out as "parked"/"beside the
curb"). ByteTrack's Kalman-filter motion prediction plus its two-stage
high/low-confidence matching handles exactly this - fast motion and
partial occlusion during a cut-in - far better than raw IoU, and since
each run is a single full pass over one ~5s clip, running every frame
through YOLO is still cheap (no extra VLM/verification calls - the
crop+VQA step below still only runs once per surviving group/track).

Usage:
  from detection_hint import build_verified_detection_facts, format_detection_facts
  facts = build_verified_detection_facts(video_path, model, processor)
  fact_text = format_detection_facts(facts)  # "" if nothing confirmed
  # merge fact_text into reference_facts, same as info.txt/OCR facts
"""
import gc
import os
from collections import Counter

import cv2
import torch
from PIL import Image
from ultralytics import YOLO

DETECTION_MODEL_PATH = "yolo11m.pt"
# Core vehicle/pedestrian classes (the actual target - vehicle_behavior/
# pedestrian_behavior grounding) plus two COCO classes worth surfacing
# because captions already routinely mention them (traffic light presence/
# state, stop signs) - NOT an attempt at full scene coverage (bollards,
# traffic cones, crosswalks, general road signs aren't in COCO's 80 classes
# at all; left as a gap rather than switching detectors, per the YOLO-vs-
# Grounding-DINO decision in prompt_iteration_summary.md section 51).
RELEVANT_CLASSES = {
    "car", "bus", "truck", "motorcycle", "bicycle", "person",
    "traffic light", "stop sign",
}
# Only vehicles/pedestrians are worth tracking for motion - traffic
# lights/stop signs are static, tracking them would just measure ego's own
# motion, not theirs.
TRACKABLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle", "person"}
CONF_THRESHOLD = 0.4
TRACK_FRAME_STEP = 1  # process every Nth frame - 1 (every frame) by default,
# since a single ~5s clip is cheap for YOLO+ByteTrack and dense frames are
# what make the Kalman filter's motion prediction actually pay off
MIN_TRACK_FRAMES = 3  # a track must persist this many processed frames to be trusted
POSITION_SHIFT_FRACTION = 0.15  # min lateral movement (as a fraction of frame width) to report
AREA_GROWTH_THRESHOLD = 2.5  # min box-area growth ratio (peak/first) to flag as "approaching" -
# catches a car merging toward the ego's lane when the ego's own lane isn't
# centered in frame, so a real cut-in barely shifts x-position at all (see
# summarize_track_motion) - set well above normal same-lane-following
# growth (a car ahead in the same lane grows gradually too as the gap
# closes normally; 2.5x is closer to "cutting the distance sharply")
MIN_LATERAL_FOR_APPROACH = 0.04  # area growth alone isn't enough to call
# something a "merge" - a car directly ahead in the SAME lane grows just as
# fast when the ego brakes/stops, which isn't a lane change at all. Requiring
# at least a little lateral drift alongside the growth (smaller than
# POSITION_SHIFT_FRACTION, which alone already reports pure sideways drift)
# filters pure straight-line closing-distance out of the "merging" note.
STILL_APPROACHING_RATIO = 0.85  # the track's LAST frame area must still be
# at least this fraction of its peak area to count as "still approaching" -
# catches a second false-positive source area growth alone can't: the
# ego-vehicle passing a genuinely PARKED object (e.g. a motorcycle at the
# curb). That case also grows sharply as the ego approaches, but then
# SHRINKS again as the ego passes and leaves it behind, whereas an object
# actually merging/traveling alongside keeps growing (or holds near its
# peak) right up to the end of the track. Confirmed directly on a real
# case: a parked motorcycle's last/peak area ratio was 0.70 (clearly past
# its peak and shrinking), vs. 0.99 for a car that was still genuinely
# closing in at the end of the clip. Cheaper and more reliable than an
# earlier VLM-based moving-vs-stationary check (which struggled to judge
# motion from just two disconnected crops, especially once the ego had
# traveled far enough that the background changed completely) - this uses
# the track's own size-over-time data that's already computed.
MAX_MOTION_CANDIDATES = 4  # hard cap on how many motion notes get
# crop+verified per video - a busy multi-lane scene can otherwise produce
# a dozen+ candidates, and each one is an extra VLM call on top of the
# count-group verifications already happening. Briefly lowered to 3 while
# chasing a real, reproducible OOM on one video - turned out to be caused
# by that video's 16-frame caption call itself, not by verification-call
# volume (confirmed directly: reducing to 12 frames fixed it instantly at
# the ORIGINAL cap of 4/8), so restored here - the actual fix is
# caption_review_tool.py's frame-count OOM fallback instead.
MAX_COUNT_GROUPS = 8  # same reasoning, applied to the (class, position)
# count groups - a busy intersection can have close to a dozen distinct
# groups (multiple classes x 3 positions), each needing its own
# verification call; keeps the largest/most-salient groups only.
CROP_PADDING = 0.3  # margin around a detection box, as a fraction of its own size
CROP_MIN_SIDE = 400  # upscale small crops so the vision tower isn't handed a tiny image

PLURAL = {
    "car": "cars", "bus": "buses", "truck": "trucks",
    "motorcycle": "motorcycles", "bicycle": "bicycles", "person": "people",
    "traffic light": "traffic lights", "stop sign": "stop signs",
}

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO(DETECTION_MODEL_PATH)
    return _MODEL


def _position_label(x_center, frame_width):
    frac = x_center / frame_width
    if frac < 0.4:
        return "left"
    if frac > 0.6:
        return "right"
    return "ahead"


def _run_tracking(video_path, frame_step=TRACK_FRAME_STEP):
    """Runs YOLO+ByteTrack (ultralytics' built-in model.track(), see this
    module's docstring for why this replaced a from-scratch greedy-IoU
    tracker) across the clip, keeping every `frame_step`-th frame's
    detections grouped by persistent track ID. Returns a dict track_id ->
    {"class": str, "detections": [(frame_idx, box, frame_w, confidence),
    ...]} - each entry is one physical object's full trajectory, already
    deduplicated across frames (unlike raw per-frame detections)."""
    model = _get_model()
    results = model.track(
        source=video_path, tracker="bytetrack.yaml", conf=CONF_THRESHOLD,
        verbose=False, stream=True,
    )
    tracks = {}
    for frame_idx, r in enumerate(results):
        if frame_idx % frame_step != 0:
            continue
        if r.boxes.id is None:
            continue
        w = r.orig_shape[1]
        for box, track_id, cls_idx, conf in zip(
            r.boxes.xyxy, r.boxes.id, r.boxes.cls, r.boxes.conf
        ):
            cls_name = model.names[int(cls_idx)]
            if cls_name not in RELEVANT_CLASSES:
                continue
            entry = tracks.setdefault(int(track_id), {"class": cls_name, "detections": []})
            entry["detections"].append((frame_idx, box.tolist(), w, float(conf)))
    return tracks


def summarize_tracks(tracks):
    """Per (class, position): one count per distinct TRACK (each physical
    object counted exactly once, using its most recent known position),
    plus the single highest-confidence detection across that group's
    tracks (used to crop+verify the group before it's trusted - see
    build_verified_detection_facts()). Returns (counts, best_example)
    where best_example maps (class, position) -> (confidence, frame_idx,
    box)."""
    counts = Counter()
    best_example = {}
    for track in tracks.values():
        dets = track["detections"]
        if not dets:
            continue
        _, last_box, last_w, _ = dets[-1]
        position = _position_label((last_box[0] + last_box[2]) / 2, last_w)
        key = (track["class"], position)
        counts[key] += 1
        frame_idx, box, _w, conf = max(dets, key=lambda d: d[3])
        if key not in best_example or conf > best_example[key][0]:
            best_example[key] = (conf, frame_idx, box)
    return counts, best_example


def summarize_track_motion(tracks):
    """For tracks persisting >= MIN_TRACK_FRAMES processed frames, flags
    two different kinds of notable motion, checked independently since
    they catch different situations:

    1. Lateral drift - the object's horizontal position shifted enough
       between the track's first and last frame to suggest a real lane
       change/cut-in, not gated on whether the coarse left/ahead/right
       label crossed a bucket boundary (a drift from x=0.16 to x=0.32 is
       real even though both fall in "left"). Judged on NET start-to-end
       shift, not the maximum deviation reached mid-track - an earlier
       version used max-deviation on the theory that a car settling into
       its new lane shouldn't have its shift "washed out" by the track
       continuing to run, but that was never actually observed, and
       max-deviation caused a real, confirmed bug instead: a car that
       drifted right and then drifted most of the way back left (a
       there-and-back wobble, ending close to where it started) got
       reported as "drifting rightward" purely because that was the
       peak excursion point, even though the second half of the clip -
       and the ending position - was actually moving back left. Net
       shift reports what a viewer watching the whole clip would say:
       where did it actually end up relative to where it started.
    2. Approach/growth - the object's box area grew sharply AND it drifted
       laterally by at least a little (MIN_LATERAL_FOR_APPROACH - smaller
       than POSITION_SHIFT_FRACTION, which alone already reports pure
       sideways drift). Area growth by itself isn't enough: a car directly
       ahead in the SAME lane grows just as fast whenever the ego brakes
       or the gap closes normally, which isn't a lane change at all -
       requiring some sideways component too is what distinguishes an
       actual merge from ordinary same-lane following distance changing.
       This still catches a case pure lateral-only tracking would miss:
       on a road where the ego's own lane sits toward one side of the
       frame (not centered), a car merging INTO that lane from an
       adjacent one may shift x only a little even while genuinely
       cutting in - the bulk of what's visible is "getting closer/
       bigger" (confirmed directly: a real cut-in tracked at a
       near-constant x~0.8-0.9 the whole time - a lateral shift too small
       to flag on its own - but its box area grew 6x by the end).

    Results are ranked by how far each candidate clears its own
    threshold and capped to MAX_MOTION_CANDIDATES, since each one costs
    an extra crop+VQA verification call downstream - a busy multi-lane
    scene can otherwise produce a dozen+ candidates and measurably raise
    CUDA OOM risk. Returns a list of (note_text, track) pairs so callers
    can crop+verify the track before trusting the note."""
    candidates = []
    for track in tracks.values():
        if track["class"] not in TRACKABLE_CLASSES:
            continue
        dets = track["detections"]
        if len(dets) < MIN_TRACK_FRAMES:
            continue
        _, first_box, w, _ = dets[0]
        first_frac = (first_box[0] + first_box[2]) / 2 / w
        last_box = dets[-1][1]
        last_frac = (last_box[0] + last_box[2]) / 2 / w
        shift = last_frac - first_frac
        start_pos = _position_label(first_frac * w, w)

        first_area = max(1.0, (first_box[2] - first_box[0]) * (first_box[3] - first_box[1]))
        areas = [(d[1][2] - d[1][0]) * (d[1][3] - d[1][1]) for d in dets]
        max_area = max(areas)
        growth = max_area / first_area

        # A large apparent position shift can be pure parallax from the ego
        # catching up to / driving past a slow or stationary vehicle in an
        # adjacent lane, not that vehicle actually steering - confirmed
        # directly: a car queued in stopped traffic in the next lane showed
        # shift=+0.34 (well past POSITION_SHIFT_FRACTION, so it would have
        # been reported as "drifting rightward") and 160x area growth as
        # the ego drove past it, then its box area dropped to 37% of that
        # peak by the last frame - the same "already been overtaken"
        # signature STILL_APPROACHING_RATIO was built to catch for the
        # merge check below, just reached via the drift branch instead.
        # Checking growth/shrink before either branch catches it in both.
        if growth >= AREA_GROWTH_THRESHOLD and areas[-1] / max_area < STILL_APPROACHING_RATIO:
            continue

        if abs(shift) >= POSITION_SHIFT_FRACTION:
            direction = "rightward" if shift > 0 else "leftward"
            note = f"a {track['class']} (starting {start_pos}) drifting {direction} across the clip"
            strength = abs(shift) / POSITION_SHIFT_FRACTION
            candidates.append((strength, note, track))
            continue

        if abs(shift) < MIN_LATERAL_FOR_APPROACH:
            continue
        if growth >= AREA_GROWTH_THRESHOLD:
            note = (
                f"a {track['class']} ({start_pos}) growing noticeably closer across "
                "the clip, suggesting it may be merging toward the ego-vehicle's lane"
            )
            strength = growth / AREA_GROWTH_THRESHOLD
            candidates.append((strength, note, track))

    # If multiple independently-tracked vehicles all drift the SAME
    # direction, that's a much more likely signature of the ego itself
    # moving laterally (e.g. a lane change - info.txt's motion field has
    # no category for that, so it's still recorded as "moving straight",
    # unlike an intersection turn) than of several unrelated real vehicles
    # all just happening to change lanes the same way at once - confirmed
    # directly: a "moving straight" clip showed 4 different cars, starting
    # from 4 different initial positions (right, ahead, ahead, left), ALL
    # "drifting leftward" - only explained by the ego shifting lanes
    # underneath them. Drop "drifting" notes (not "growing closer" ones -
    # a lateral ego shift doesn't explain a genuine closing-distance
    # approach) once 2+ agree on the same direction.
    drift_directions = [
        "rightward" if "drifting rightward" in note else "leftward"
        for _s, note, _t in candidates if "drifting" in note
    ]
    if drift_directions.count("leftward") >= 2 or drift_directions.count("rightward") >= 2:
        candidates = [c for c in candidates if "drifting" not in c[1]]

    candidates.sort(key=lambda c: -c[0])
    return [(note, track) for _strength, note, track in candidates[:MAX_MOTION_CANDIDATES]]


def crop_box_region(video_path, frame_idx, box, padding=CROP_PADDING):
    """Crops the (padded) detection box out of its source frame and
    returns a PIL Image, upscaled if small. Returns None if the frame
    can't be read."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
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


def _verify_prompt(cls_name):
    return (
        "This is a cropped close-up region from a driving-dashcam video "
        f"frame, centered on something an automated detector flagged as a "
        f"possible {cls_name}. Judge only what's actually in this image. "
        f"Is there really a {cls_name} visible here? Answer with exactly "
        "one word: yes or no."
    )


def verify_object_presence(model, processor, crop_image, cls_name):
    """Shows a single cropped detection region to the VLM and asks it to
    confirm the class independently of YOLO's guess. Returns True/False;
    False (not verified) on any unparseable/missing response."""
    if crop_image is None:
        return False
    messages = [{"role": "user", "content": [
        {"type": "image", "image": crop_image},
        {"type": "text", "text": _verify_prompt(cls_name)},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    oom_occurred = False
    try:
        generated_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
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
        generated_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    reply = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip().lower()
    del inputs, generated_ids, trimmed
    gc.collect()
    torch.cuda.empty_cache()
    return reply.startswith("yes")


def _facing_prompt(cls_name):
    return (
        "This is a cropped close-up region from a driving-dashcam video "
        f"frame, showing a {cls_name} that a tracker flagged as possibly "
        "closing in toward the ego-vehicle's lane. Judge only what's "
        f"visible in this image. Which side of the {cls_name} is facing "
        "the camera: its FRONT (headlights, grille, front bumper/badge "
        "visible - meaning it's traveling TOWARD the camera, i.e. "
        "oncoming traffic in the opposite direction) or its REAR/SIDE "
        "(taillights, trunk, or a side profile visible - meaning it's "
        "traveling the same direction as the camera, or crossing at an "
        "angle)? Answer with exactly one word: front or rear."
    )


def verify_not_oncoming(model, processor, crop_image, cls_name):
    """Shows a single cropped detection region and asks whether the
    vehicle's front or rear/side faces the camera. A vehicle's FRONT
    facing the camera means it's simply oncoming traffic in the opposite
    lane, naturally growing closer/bigger as it approaches head-on - not
    a merge into the ego's lane at all, even though it passes every other
    "growing closer" check (confirmed directly: a car crossing the double
    yellow line into the opposite lane, front-on, was flagged as
    "merging toward the ego-vehicle's lane" by the growth signal, when it
    was just normal oncoming traffic). Returns True (not oncoming, safe to
    trust as a real approach/merge) only if the reply is "rear"; False
    (including "front" or any unparseable/missing response - the safer
    default) otherwise."""
    if crop_image is None:
        return False
    messages = [{"role": "user", "content": [
        {"type": "image", "image": crop_image},
        {"type": "text", "text": _facing_prompt(cls_name)},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    oom_occurred = False
    try:
        generated_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
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
        generated_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    reply = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip().lower()
    del inputs, generated_ids, trimmed
    gc.collect()
    torch.cuda.empty_cache()
    return reply.startswith("rear")


SUBTYPES = ["sedan", "SUV", "van", "hatchback", "taxi", "pickup"]
COLORS = ["white", "black", "silver", "gray", "red", "blue", "green", "yellow", "brown", "orange"]
_SUBTYPE_PROMPT = (
    "This is a cropped close-up region from a driving-dashcam video frame, "
    "showing a vehicle a detector flagged as a car. Judge only what's in "
    "this image. Answer with exactly two words separated by a space, color "
    "then type:\n"
    "<color> <type>\n"
    "Color must be exactly one word from: " + ", ".join(COLORS) + ". Type "
    "must be exactly one word from: " + ", ".join(SUBTYPES) + ". If a word "
    "is genuinely unclear, use \"unclear\" for that word only (e.g. "
    "\"unclear sedan\" or \"white unclear\")."
)


def classify_vehicle_details(model, processor, crop_image):
    """Shows a single cropped, already-confirmed 'car' detection to the
    VLM and asks it to name both the color and the specific body type
    (sedan/SUV/van/etc.) in one call - YOLO's classes don't distinguish
    either, so this is the only point in the pipeline where they get
    checked. Color used to be left entirely to the caption-writing model's
    own free guess (never verified), which is exactly the kind of
    unverified claim this project's whole grounding pipeline exists to
    avoid - confirmed directly on a video where the caption called a
    plainly white car "silver". Folding color into this existing call
    costs nothing extra (one call already happening, one more word in the
    same answer) instead of adding a whole separate verification call.
    Only meaningful for a SINGLE verified instance, not a count>1 group -
    see build_verified_detection_facts() for why. Returns (color, subtype),
    either element None if unclear/unparseable for that word."""
    if crop_image is None:
        return None, None
    messages = [{"role": "user", "content": [
        {"type": "image", "image": crop_image},
        {"type": "text", "text": _SUBTYPE_PROMPT},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    oom_occurred = False
    try:
        generated_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
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
        generated_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    reply = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip().lower()
    del inputs, generated_ids, trimmed
    gc.collect()
    torch.cuda.empty_cache()

    color = next((c for c in COLORS if c in reply), None)
    subtype = next((s for s in SUBTYPES if s.lower() in reply), None)  # preserves "SUV"'s casing
    return color, subtype


TURNING_EGO_MOTIONS = {"turning left", "turning right", "curved lane driving"}
# info.txt motion values (see caption_review_tool.INFO_KEYS) under which the
# ego-vehicle's own camera is rotating. Confirmed directly on drive_6/03: a
# car driving straight through a roundabout appeared to drift rightward in
# frame purely because the ego was turning left underneath it, not because
# the other car actually moved that way - box-position drift only measures
# real relative motion when the ego itself isn't rotating, so lateral-motion
# facts are skipped entirely for these ego states rather than risk reporting
# an artifact of the ego's own turn as another vehicle's behavior.


def build_verified_detection_facts(video_path, model, processor, frame_step=TRACK_FRAME_STEP, ego_motion=None):
    """Full pipeline: track on the clip (see _run_tracking - ByteTrack via
    ultralytics), group into (class, position) counts and lateral-motion
    tracks, then verify each group/track by cropping its best example and
    asking the VLM to confirm it independently. Returns a list of fact
    strings - empty if nothing survives verification."""
    tracks = _run_tracking(video_path, frame_step)
    if not tracks:
        return []

    counts, best_example = summarize_tracks(tracks)
    facts = []
    if counts:
        confirmed_parts = []
        top_keys = sorted(counts, key=lambda k: -counts[k])[:MAX_COUNT_GROUPS]
        for key in sorted(top_keys):
            cls_name, position = key
            count = counts[key]
            _conf, frame_idx, box = best_example[key]
            crop = crop_box_region(video_path, frame_idx, box)
            if verify_object_presence(model, processor, crop, cls_name):
                noun = cls_name if count == 1 else PLURAL[cls_name]
                # Color and subtype (sedan/SUV/van/etc.) only for a single
                # confirmed car - YOLO's "car" class doesn't distinguish
                # either, and for a count>1 group the one cropped example
                # can't stand in for all of them (likely a mix of colors/
                # types).
                if cls_name == "car" and count == 1:
                    color, subtype = classify_vehicle_details(model, processor, crop)
                    if subtype:
                        noun = subtype
                    if color:
                        noun = f"{color} {noun}"
                confirmed_parts.append(f"{count} {noun} ({position})")
        if confirmed_parts:
            facts.append("Vehicle/pedestrian counts and positions (approximate): " + "; ".join(confirmed_parts))

    motion_candidates = [] if ego_motion in TURNING_EGO_MOTIONS else summarize_track_motion(tracks)
    confirmed_motion = []
    for note, track in motion_candidates:
        frame_idx, box, _w, _conf = max(track["detections"], key=lambda d: d[3])
        crop = crop_box_region(video_path, frame_idx, box)
        if not verify_object_presence(model, processor, crop, track["class"]):
            continue
        # Any motion note - "drifting" as well as "may be merging" - needs
        # this check: a vehicle in the OPPOSITE lane passing by head-on
        # naturally shows a big apparent position shift too (ordinary
        # oncoming traffic, not the vehicle changing lanes), so reject it
        # if the crop shows the vehicle's front facing the camera. This
        # used to only gate "may be merging" notes on the assumption that
        # "drifting" notes wouldn't be confused with oncoming traffic, but
        # there's no real reason a large position shift from an oncoming
        # vehicle couldn't land in that branch too - better to check both.
        if not verify_not_oncoming(model, processor, crop, track["class"]):
            continue
        confirmed_motion.append(note)
    if confirmed_motion:
        facts.append("Lateral movement observed across the clip: " + "; ".join(confirmed_motion))

    return facts


def format_detection_facts(facts):
    """Formats confirmed detection facts as reference-facts bullets,
    matching format_reference_facts()'s style - these are now given facts,
    not hints, so they belong in the same list as the info.txt-derived
    facts and ocr_hint.py's verified sign text."""
    if not facts:
        return ""
    return "\n".join(f"- {f}" for f in facts)


if __name__ == "__main__":
    import sys
    from transformers import AutoModelForImageTextToText, AutoProcessor

    video = sys.argv[1] if len(sys.argv) > 1 else "./sample_videos/231106095132_M801C06L62G031_2673_end_extract/drive_1/02/1_clip/5.mp4"
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(repo_root, "Qwen3-VL-8B-Instruct")
    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(model_path)
    facts = build_verified_detection_facts(video, model, processor)
    print(format_detection_facts(facts))
