"""
Simple test: does adding object_captions.txt content as an UNVERIFIED hint
(not authoritative like info.txt) help or hurt the caption, compared to the
current info.txt-only prompt? object_captions.txt is known to contain some
clearly wrong entries (e.g. "a set of train tracks" on an urban bridge scene)
so it's framed as "possibly relevant, verify before using" rather than given
fact - this tests whether that hint helps the model notice more real detail
without also picking up the noise.

Same video as info_txt_ablation_test.py, same base model, same info.txt
facts - only difference is the added object_captions hint block.

Usage:
  source activate qwen3vl && python finetune/object_captions_hint_test.py
"""
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "captioning_tools"))

from transformers import AutoModelForImageTextToText, AutoProcessor

from caption_review_tool import (  # noqa: E402
    read_info,
    format_reference_facts,
    PROMPT_STRUCTURED_TEMPLATE,
)

MODEL_PATH = os.path.join(REPO_ROOT, "Qwen3-VL-8B-Instruct")
VIDEO_DIR = os.path.join(
    REPO_ROOT, "sample_videos",
    "231106095132_M801C06L62G031_2673_end_extract", "drive_1", "02",
)
VIDEO_PATH = os.path.join(VIDEO_DIR, "1_clip", "5.mp4")
NUM_FRAMES = 16


def build_hint_prompt(info):
    base_prompt = PROMPT_STRUCTURED_TEMPLATE.format(reference_facts=format_reference_facts(info))
    with open(os.path.join(VIDEO_DIR, "object_captions.txt"), encoding="utf-8") as f:
        phrases = [line.strip() for line in f if line.strip()]
    hint_block = (
        "\n\nAn automated object detector also produced these short notes about "
        "this clip - they are UNVERIFIED and sometimes wrong or irrelevant, so "
        "only use ones that match what you actually see in the frames; ignore "
        "the rest silently:\n- " + "\n- ".join(phrases)
    )
    return base_prompt + hint_block


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
    print("Loading base model (no LoRA)...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    hint_prompt = build_hint_prompt(info)
    print("\n=== WITH info.txt + object_captions.txt hint ===")
    print(run(model, processor, hint_prompt))


if __name__ == "__main__":
    main()
