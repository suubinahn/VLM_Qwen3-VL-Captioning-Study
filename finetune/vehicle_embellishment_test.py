"""
Single-video, single-condition script (WITHOUT or WITH verified grounding),
invoked repeatedly by run_vehicle_embellishment_test.sh - one fresh process
per (video, condition) pair, same isolation rationale as
single_video_grounding_compare.py (looping over multiple videos in one
process hit reproducible CUDA OOM on these 12GB GPUs once grounding's
extra verification calls are in the mix).

Purpose: checks a specific hypothesis raised in review - that adding YOLO
vehicle-count hints (which only ever give a bare class + position, e.g.
"6 cars (ahead)" - never color or specific type) pushes the model to
enumerate MORE vehicles, with MORE specific (but ungrounded - YOLO never
verifies color/type) color/type guesses, than it would unprompted. Color/
type is NOT something detection_hint.py verifies (only class presence via
crop+VQA) - so any color/type mentioned is equally "the model's own guess"
in both conditions; the question is only whether grounding changes how
MUCH of that guessing happens.

Usage:
  source activate qwen3vl && python finetune/vehicle_embellishment_test.py \\
      --condition without --video <path> --out <result.json>
  source activate qwen3vl && python finetune/vehicle_embellishment_test.py \\
      --condition with --video <path> --out <result.json>
"""
import argparse
import gc
import json
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
    try:
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    except torch.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    text = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    del inputs, generated_ids, trimmed
    gc.collect()
    torch.cuda.empty_cache()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["without", "with"], required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    video_path = args.video
    info = read_info(video_path)
    reference_facts = format_reference_facts(info)

    print("Loading base model (no LoRA)...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    facts_text = ""
    if args.condition == "with":
        facts_text = get_combined_facts(video_path, model, processor)
        print(f"verified facts: {facts_text}")
        reference_facts = reference_facts + ("\n" + facts_text if facts_text else "")

    prompt = PROMPT_STRUCTURED_TEMPLATE.format(reference_facts=reference_facts)
    caption = run(model, processor, video_path, prompt)
    print(f"caption: {caption}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "video": video_path,
            "condition": args.condition,
            "facts_text": facts_text,
            "caption": caption,
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
