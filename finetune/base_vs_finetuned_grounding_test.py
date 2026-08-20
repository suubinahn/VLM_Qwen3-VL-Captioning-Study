"""
Now that verified grounding (captioning_tools/combined_hint.py - YOLO
detection + OCR text, each individually confirmed via crop+VQA before
being trusted) is validated and wired into the real pipeline
(caption_review_tool.py), this checks a question that was never tested
before: given the SAME verified facts, does the fine-tuned model produce a
better caption than the base model, or does fine-tuning add nothing (or
even hurt) once real grounding is already doing the heavy lifting?

Every earlier fine-tuning round (language-only round1/round2, vision
round1/round2) was tested WITHOUT any detection/OCR grounding, and every
grounding test so far has used the BASE model only (see combined_hint_test.py
etc.) - the two were never combined, so this closes that gap.

Adapter: finetune_checkpoints_vision_round2/epoch_2, the val-loss-best
checkpoint from vision round 2 (0.5385/0.4901/0.4946/0.4970/0.4973/0.5017 -
epoch 2 lowest). Vision round 2 (not round 1, not either language-only
round) chosen as the most-corrected/most-recent checkpoint available.

Fairness: grounding facts are computed ONCE via the base model (crop+VQA
verification is about reading the video correctly, not about caption
style, so there's no reason to duplicate it per-model) and the identical
facts_text is then injected into both the base and fine-tuned generation
passes - isolates the comparison to "which model writes the better
caption from the same given facts", not "which model also grounds better".

Usage:
  source activate qwen3vl && python finetune/base_vs_finetuned_grounding_test.py
"""
import gc
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "captioning_tools"))
os.chdir(REPO_ROOT)

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

from caption_review_tool import (  # noqa: E402
    read_info,
    format_reference_facts,
    PROMPT_STRUCTURED_TEMPLATE,
)
from combined_hint import get_combined_facts  # noqa: E402

MODEL_PATH = os.path.join(REPO_ROOT, "Qwen3-VL-8B-Instruct")
ADAPTER_PATH = os.path.join(REPO_ROOT, "finetune_checkpoints_vision_round2", "epoch_2")
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
    try:
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    except torch.OutOfMemoryError:
        # Fragmentation from many small verification calls earlier in this
        # long-running process can tip a later, heavier caption-generation
        # call over the edge even though each call cleans up after itself -
        # one hard cleanup + retry is usually enough to recover.
        gc.collect()
        torch.cuda.empty_cache()
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    text = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    del inputs, generated_ids, trimmed
    gc.collect()
    torch.cuda.empty_cache()
    return text


def build_finetuned_model_and_processor():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    NUM_LAYERS = 36
    device_map = {
        "model.visual": 0,
        "model.language_model.embed_tokens": 0,
        "model.language_model.norm": 1,
        "model.language_model.rotary_emb": 1,
        "lm_head": 1,
    }
    for i in range(NUM_LAYERS):
        device_map[f"model.language_model.layers.{i}"] = 0 if i < NUM_LAYERS // 2 else 1

    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map=device_map,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor


def main():
    print("=== Phase 1: base model + verified grounding ===")
    print("Loading base model (no LoRA)...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    facts_by_video = {}
    base_results = {}
    for video_path in TEST_VIDEOS:
        print(f"\n{video_path}")
        info = read_info(video_path)
        reference_facts = format_reference_facts(info)

        facts_text = get_combined_facts(video_path, model, processor)
        facts_by_video[video_path] = (reference_facts, facts_text)
        print(f"  verified facts: {facts_text}")

        combined_facts = reference_facts + ("\n" + facts_text if facts_text else "")
        prompt = PROMPT_STRUCTURED_TEMPLATE.format(reference_facts=combined_facts)
        base_results[video_path] = run(model, processor, video_path, prompt)
        print(f"  base+grounding: {base_results[video_path]}")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    print("\n\n=== Phase 2: fine-tuned model (vision round2 epoch2) + SAME verified grounding ===")
    print("Loading fine-tuned model...")
    ft_model, ft_processor = build_finetuned_model_and_processor()

    ft_results = {}
    for video_path in TEST_VIDEOS:
        reference_facts, facts_text = facts_by_video[video_path]
        combined_facts = reference_facts + ("\n" + facts_text if facts_text else "")
        prompt = PROMPT_STRUCTURED_TEMPLATE.format(reference_facts=combined_facts)
        ft_results[video_path] = run(ft_model, ft_processor, video_path, prompt)
        print(f"\n{video_path}")
        print(f"  finetuned+grounding: {ft_results[video_path]}")

    print("\n\n" + "=" * 90)
    print("SIDE BY SIDE")
    print("=" * 90)
    for video_path in TEST_VIDEOS:
        print("\n" + "-" * 90)
        print(f"video: {video_path}")
        _reference_facts, facts_text = facts_by_video[video_path]
        print(f"verified facts: {facts_text}")
        print(f"\n[BASE + grounding]\n{base_results[video_path]}")
        print(f"\n[FINE-TUNED + grounding]\n{ft_results[video_path]}")


if __name__ == "__main__":
    main()
