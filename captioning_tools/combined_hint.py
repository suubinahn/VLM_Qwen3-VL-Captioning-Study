"""
Combines detection_hint.py's verified vehicle/pedestrian/motion facts
(crop+VQA-confirmed, targeting vehicle_behavior/pedestrian_behavior) and
ocr_hint.py's verified sign-text facts (crop+VQA-confirmed, targeting
identification) into one merged facts block - both are now fully verified
before this module ever sees them (see each module's docstring for why
raw, unverified hints kept causing regressions and were replaced with a
crop+VQA verification step), so this module only concatenates.

Usage:
  from combined_hint import get_combined_facts
  facts_text = get_combined_facts(video_path, model, processor)
  combined_facts = reference_facts + ("\\n" + facts_text if facts_text else "")
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection_hint import build_verified_detection_facts, format_detection_facts  # noqa: E402
from ocr_hint import build_verified_ocr_facts, format_verified_ocr_facts  # noqa: E402


def get_combined_facts(video_path, model, processor, ego_motion=None):
    """Returns a single formatted facts_text string (reference-facts-style
    bullets) combining verified detection facts and verified OCR facts -
    "" if neither found anything. Merge directly into reference_facts.
    ego_motion (info.txt's motion field) is passed through to skip lateral-
    motion facts while the ego itself is turning - see
    detection_hint.TURNING_EGO_MOTIONS."""
    detection_facts = build_verified_detection_facts(video_path, model, processor, ego_motion=ego_motion)
    ocr_facts = build_verified_ocr_facts(video_path, model, processor)

    parts = []
    detection_text = format_detection_facts(detection_facts)
    if detection_text:
        parts.append(detection_text)
    ocr_text = format_verified_ocr_facts(ocr_facts)
    if ocr_text:
        parts.append(ocr_text)

    return "\n".join(parts)


if __name__ == "__main__":
    from transformers import AutoModelForImageTextToText, AutoProcessor

    video = sys.argv[1] if len(sys.argv) > 1 else "./sample_videos/231116093002_M801C06L62G031_2673_end_extract/drive_1/03/1_clip/5.mp4"
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(repo_root, "Qwen3-VL-8B-Instruct")
    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(model_path)
    print(get_combined_facts(video, model, processor))
