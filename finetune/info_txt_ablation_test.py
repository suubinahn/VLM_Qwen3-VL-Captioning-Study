"""
One-video test: does the model do noticeably worse WITHOUT info.txt facts
injected, compared to with them? Revisits an early-project finding (model's
own visual judgment for motion/lane-count/etc. was unreliable even after
extensive prompting) with a direct, fresh side-by-side test.

Generates the SAME video's structured caption twice:
  1. WITH info.txt facts (current pipeline, PROMPT_STRUCTURED_TEMPLATE as-is)
  2. WITHOUT info.txt facts (same prompt, but asks the model to determine
     motion/road type/road context/surface/weather itself from the video,
     instead of being told)

Base model only (no LoRA), NUM_FRAMES=16 (inference setting).

Usage:
  source activate qwen3vl && python finetune/info_txt_ablation_test.py
"""
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "captioning_tools"))

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from caption_review_tool import (  # noqa: E402
    read_info,
    format_reference_facts,
    PROMPT_STRUCTURED_TEMPLATE,
)

MODEL_PATH = os.path.join(REPO_ROOT, "Qwen3-VL-8B-Instruct")
VIDEO_PATH = os.path.join(
    REPO_ROOT, "sample_videos",
    "231106095132_M801C06L62G031_2673_end_extract", "drive_1", "02", "1_clip", "5.mp4",
)
NUM_FRAMES = 16

# Same structured prompt, but with the "given facts" block removed and
# replaced with an instruction to determine those same fields visually -
# everything else (what to describe, sign rules, banned content) kept
# identical so this isolates just the "given facts vs guess" variable.
NO_FACTS_PREFIX = (
    "You are an expert at analyzing driving videos for building an autonomous "
    "driving dataset. Determine the following from the video frames yourself: "
    "the ego-vehicle's driving motion (e.g. moving straight, turning left/right, "
    "stopped), the road context (e.g. city street, bridge, tunnel, underpass), "
    "the road type/lane count, the time of day, the road surface condition, "
    "and the weather.\n\n"
)


def build_no_facts_prompt():
    # Reuse the structured template's body (point 1 onward) but drop its
    # "{reference_facts}" block and swap the opening sentence for one that
    # doesn't claim the facts are already given.
    body = PROMPT_STRUCTURED_TEMPLATE.split("\n\n", 1)[1]
    return NO_FACTS_PREFIX + body


def run(model, processor, prompt):
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": VIDEO_PATH},
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
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def main():
    info = read_info(VIDEO_PATH)
    print(f"Ground truth (info.txt, corrections applied): {info}\n")

    print("Loading base model (no LoRA)...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    with_facts_prompt = PROMPT_STRUCTURED_TEMPLATE.format(reference_facts=format_reference_facts(info))
    without_facts_prompt = build_no_facts_prompt()

    print("\n=== WITH info.txt facts ===")
    print(run(model, processor, with_facts_prompt))

    print("\n=== WITHOUT info.txt facts (model determines itself) ===")
    print(run(model, processor, without_facts_prompt))


if __name__ == "__main__":
    main()
