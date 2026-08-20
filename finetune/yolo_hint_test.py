"""
Does adding a YOLO object-detection hint (see captioning_tools/detection_hint.py)
help the BASE model on videos previously flagged for vehicle_behavior errors
(fabricated/misidentified vehicles - see prompt_iteration_summary.md sections
38/46/50, where this category stayed weak across every fine-tuning round)?

Base model only (no LoRA) - isolates the hint's own effect, same rationale as
info_txt_ablation_test.py/object_captions_hint_test.py testing base-only.

For each test video, generates the structured (2_caption) caption twice:
  1. WITHOUT the detection hint (current pipeline, PROMPT_STRUCTURED_TEMPLATE as-is)
  2. WITH the detection hint appended (YOLO counts/positions from a few frames)

Usage:
  source activate qwen3vl && python finetune/yolo_hint_test.py
"""
import gc
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "captioning_tools"))
os.chdir(REPO_ROOT)

from transformers import AutoModelForImageTextToText, AutoProcessor

from caption_review_tool import (  # noqa: E402
    read_info,
    format_reference_facts,
    PROMPT_STRUCTURED_TEMPLATE,
)
from detection_hint import format_detection_hint  # noqa: E402

MODEL_PATH = os.path.join(REPO_ROOT, "Qwen3-VL-8B-Instruct")
NUM_FRAMES = 16

TEST_VIDEOS = [
    "./sample_videos/231106095132_M801C06L62G031_2673_end_extract/drive_7/04/1_clip/5.mp4",
    "./sample_videos/231116093002_M801C06L62G031_2673_end_extract/drive_1/02/1_clip/5.mp4",
    "./sample_videos/231116093002_M801C06L62G031_2673_end_extract/drive_1/03/1_clip/5.mp4",
    "./sample_videos/240322135559_M801C06L62G031_2673_end_extract/drive_9/01/1_clip/5.mp4",
    "./sample_videos/240322135559_M801C06L62G031_2673_end_extract/drive_18/03/1_clip/5.mp4",
    "./sample_videos/240322135559_M801C06L62G031_2673_end_extract/drive_22/01/1_clip/5.mp4",
    "./sample_videos/231116093002_M801C06L62G031_2673_end_extract/drive_4/04/1_clip/5.mp4",
    "./sample_videos/240322135559_M801C06L62G031_2673_end_extract/drive_12/03/1_clip/5.mp4",
    "./sample_videos/240322135559_M801C06L62G031_2673_end_extract/drive_14/04/1_clip/5.mp4",
    "./sample_videos/240205170303_M801C06L62G031_2673_end_extract/drive_2/02/1_clip/5.mp4",
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

        print("\n--- WITHOUT detection hint ---")
        print(run(model, processor, video_path, base_prompt))

        print("\nRunning YOLO detection...")
        hint = format_detection_hint(video_path)
        print(f"hint: {hint}")
        # Inserted right after the given-facts block (before the numbered
        # instructions), same position info.txt facts occupy - not appended
        # after the whole template - so it reads as part of the same
        # "context given up front" structure instead of a bolted-on
        # afterthought, keeping output style consistent with the rest of
        # the pipeline's captions.
        combined_facts = reference_facts + ("\n\n" + hint if hint else "")
        hint_prompt = PROMPT_STRUCTURED_TEMPLATE.format(reference_facts=combined_facts)

        print("\n--- WITH detection hint ---")
        print(run(model, processor, video_path, hint_prompt))


if __name__ == "__main__":
    main()
