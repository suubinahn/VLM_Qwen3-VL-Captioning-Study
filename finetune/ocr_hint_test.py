"""
Does adding an OCR text-scan hint (see captioning_tools/ocr_hint.py) help
the BASE model on videos previously flagged for identification errors
(sign-text misreads - see prompt_iteration_summary.md section 51 and the
OCR discussion following it, which found "identification" is mostly this,
not vehicle misidentification)?

Base model only (no LoRA) - isolates the hint's own effect, same rationale
as info_txt_ablation_test.py/object_captions_hint_test.py/yolo_hint_test.py.

For each test video, generates the structured (2_caption) caption twice:
  1. WITHOUT the OCR hint (current pipeline, PROMPT_STRUCTURED_TEMPLATE as-is)
  2. WITH the OCR hint appended to the reference-facts block (same
     injection point as detection_hint.py - right after the given facts,
     before the numbered instructions)

Usage:
  source activate qwen3vl && python finetune/ocr_hint_test.py
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
from ocr_hint import format_ocr_hint  # noqa: E402

MODEL_PATH = os.path.join(REPO_ROOT, "Qwen3-VL-8B-Instruct")
NUM_FRAMES = 16

TEST_VIDEOS = [
    "./sample_videos/240322135559_M801C06L62G031_2673_end_extract/drive_15/04/1_clip/5.mp4",
    "./sample_videos/240205094744_M801C06L62G031_2673_end_extract/drive_2/02/1_clip/5.mp4",
    "./sample_videos/231106095132_M801C06L62G031_2673_end_extract/drive_4/01/1_clip/5.mp4",
    "./sample_videos/231106095132_M801C06L62G031_2673_end_extract/drive_3/02/1_clip/5.mp4",
    "./sample_videos/240205094744_M801C06L62G031_2673_end_extract/drive_5/01/1_clip/5.mp4",
    "./sample_videos/240325141234_M801C06L62G031_2673_end_extract/drive_2/03/1_clip/5.mp4",
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

        print("\n--- WITHOUT OCR hint ---")
        print(run(model, processor, video_path, base_prompt))

        print("\nRunning OCR scan...")
        hint = format_ocr_hint(video_path)
        print(f"hint: {hint}")
        combined_facts = reference_facts + ("\n\n" + hint if hint else "")
        hint_prompt = PROMPT_STRUCTURED_TEMPLATE.format(reference_facts=combined_facts)

        print("\n--- WITH OCR hint ---")
        print(run(model, processor, video_path, hint_prompt))


if __name__ == "__main__":
    main()
