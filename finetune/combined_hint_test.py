"""
Does the VERIFICATION-based combined grounding (detection_hint.py's soft
YOLO+tracking hint + ocr_hint.py's crop+VQA-verified sign-text facts,
composed via captioning_tools/combined_hint.py's get_combined_grounding())
fix the recurring regressions the earlier raw-OCR-hint approach kept
hitting (SISTINA storefront text mistaken for a sign in drive_3/02;
"언주로" correctly read on its own getting overwritten by a misread in
drive_1/03; a garbled RAEMIAN archway reading in drive_14/04)?

Base model only (no LoRA), same rationale as the other *_hint_test.py
scripts. Mixed video set: half previously flagged for vehicle_behavior,
half for identification, to check both categories at once.

Usage:
  source activate qwen3vl && python finetune/combined_hint_test.py
"""
import gc
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "captioning_tools"))
os.chdir(REPO_ROOT)

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from caption_review_tool import (  # noqa: E402
    read_info,
    format_reference_facts,
    PROMPT_STRUCTURED_TEMPLATE,
)
from combined_hint import get_combined_facts  # noqa: E402

MODEL_PATH = os.path.join(REPO_ROOT, "Qwen3-VL-8B-Instruct")
NUM_FRAMES = 16

TEST_VIDEOS = [
    "./sample_videos/231116093002_M801C06L62G031_2673_end_extract/drive_1/03/1_clip/5.mp4",
    "./sample_videos/240322135559_M801C06L62G031_2673_end_extract/drive_22/01/1_clip/5.mp4",
    "./sample_videos/240322135559_M801C06L62G031_2673_end_extract/drive_14/04/1_clip/5.mp4",
    "./sample_videos/231106095132_M801C06L62G031_2673_end_extract/drive_4/01/1_clip/5.mp4",
    "./sample_videos/240325141234_M801C06L62G031_2673_end_extract/drive_2/03/1_clip/5.mp4",
    "./sample_videos/231106095132_M801C06L62G031_2673_end_extract/drive_3/02/1_clip/5.mp4",
]


def run(model, processor, video_path, prompt):
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": prompt},
        ]}
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt", num_frames=NUM_FRAMES, fps=None,
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    text = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    del inputs, generated_ids, trimmed
    gc.collect()
    torch.cuda.empty_cache()
    return text


def main():
    print("Loading base model (no LoRA)...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    for video_path in TEST_VIDEOS:
        print("\n" + "=" * 90)
        print(f"video: {video_path}")

        info = read_info(video_path)
        reference_facts = format_reference_facts(info)
        base_prompt = PROMPT_STRUCTURED_TEMPLATE.format(reference_facts=reference_facts)

        print("\n--- WITHOUT any hint ---")
        print(run(model, processor, video_path, base_prompt))

        print("\nRunning verified YOLO detection + OCR crop verification...")
        facts_text = get_combined_facts(video_path, model, processor)
        print(f"verified facts: {facts_text}")
        combined_facts = reference_facts
        if facts_text:
            combined_facts += "\n" + facts_text
        hint_prompt = PROMPT_STRUCTURED_TEMPLATE.format(reference_facts=combined_facts)

        print("\n--- WITH combined grounding ---")
        print(run(model, processor, video_path, hint_prompt))


if __name__ == "__main__":
    main()
