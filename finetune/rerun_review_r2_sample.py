"""
Cheap early check for round 2 (finetune_checkpoints_round2/epoch_3, the
val-loss-best checkpoint - see prompt_iteration_summary.md section 43):
generates captions with the round-2 model on a small random SAMPLE of the
same 160-video pool used for round 1's full comparison (130 flagged + 30
unflagged), instead of redoing all 160 immediately. The idea is to get an
early signal - if the sample shows a clearly higher fix rate than round 1's
78-flags/1-fixed baseline, do the full 160 next; if not, skip straight to
the next strategy (OCR model, vision LoRA, etc.) without spending the full
manual review effort again.

Output goes to a NEW sibling folder "sb_caption_ft_r2" (round 1's
"sb_caption_ft" is left untouched, still useful as the round-1 reference).

Usage:
  source activate qwen3vl && python finetune/rerun_review_r2_sample.py
"""
import gc
import os
import random
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "captioning_tools"))

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

from caption_review_tool import (  # noqa: E402
    read_info,
    format_reference_facts,
    PROMPT_DESCRIPTION_TEMPLATE,
    PROMPT_STRUCTURED_TEMPLATE,
    find_videos,
    is_done,
)

MODEL_PATH = os.path.join(REPO_ROOT, "Qwen3-VL-8B-Instruct")
ADAPTER_PATH = os.path.join(REPO_ROOT, "finetune_checkpoints_round2", "epoch_3")
FLAGGED_LOG = os.path.join(REPO_ROOT, "flagged_videos.txt")
NUM_FRAMES = 16
SAMPLE_UNFLAGGED = 30
SAMPLE_SIZE = 25  # how many of the 160-video pool to sample for this early check
SEED = 42
SAMPLE_SEED = 7  # different from round 1's SEED so this sample isn't just "the first N"


def load_flagged_videos():
    with open(FLAGGED_LOG, "r", encoding="utf-8") as f:
        paths = {line.split("\t")[1] for line in f if line.strip() and "\t" in line}
    return sorted(paths)


def output_paths_ft(video_path):
    clip_dir = os.path.dirname(video_path)
    parent_dir = os.path.dirname(clip_dir)
    out_dir = os.path.join(parent_dir, "sb_caption_ft_r2")
    return out_dir, os.path.join(out_dir, "1_caption.txt"), os.path.join(out_dir, "2_caption.txt")


def is_done_ft(video_path):
    _, p1, p2 = output_paths_ft(video_path)
    return os.path.isfile(p1) and os.path.isfile(p2)


def build_target_list():
    flagged = load_flagged_videos()
    all_reviewed = [v for v in find_videos() if is_done(v)]
    flagged_set = set(flagged)
    unflagged = [v for v in all_reviewed if v not in flagged_set]

    random.seed(SEED)
    random.shuffle(unflagged)
    sampled_unflagged = sorted(unflagged[:SAMPLE_UNFLAGGED])

    pool = flagged + sampled_unflagged  # the same 160-video pool round 1 used
    random.seed(SAMPLE_SEED)
    sample = random.sample(pool, min(SAMPLE_SIZE, len(pool)))
    return sorted(sample)


def build_model_and_processor():
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


def run_prompt(model, processor, video_path, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        num_frames=NUM_FRAMES,
        fps=None,
    )
    device = model.get_input_embeddings().weight.device
    inputs = inputs.to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    del inputs, generated_ids, generated_ids_trimmed
    gc.collect()
    torch.cuda.empty_cache()
    return output_text


def generate_and_save(model, processor, video_path):
    out_dir, p1, p2 = output_paths_ft(video_path)
    os.makedirs(out_dir, exist_ok=True)

    info = read_info(video_path)
    reference_facts = format_reference_facts(info)

    desc_prompt = PROMPT_DESCRIPTION_TEMPLATE.format(reference_facts=reference_facts)
    desc = run_prompt(model, processor, video_path, desc_prompt)
    with open(p1, "w", encoding="utf-8") as f:
        f.write(desc + "\n")

    struct_prompt = PROMPT_STRUCTURED_TEMPLATE.format(reference_facts=reference_facts)
    structured = run_prompt(model, processor, video_path, struct_prompt)
    with open(p2, "w", encoding="utf-8") as f:
        f.write(structured + "\n")


def main():
    targets = build_target_list()
    print(f"Sample targets: {len(targets)} videos (from the same 160-video round-1 pool)")

    print(f"Loading base model + round-2 LoRA adapter ({ADAPTER_PATH})...")
    model, processor = build_model_and_processor()

    done = 0
    skipped = 0
    for video_path in targets:
        if is_done_ft(video_path):
            skipped += 1
            continue
        try:
            generate_and_save(model, processor, video_path)
            done += 1
            print(f"  ...{done + skipped}/{len(targets)} ({video_path})")
        except Exception as e:
            print(f"  [ERROR] {video_path}: {e}")

    print(f"\nDone. Generated {done} new, skipped {skipped} already-done, "
          f"out of {len(targets)} targets.")


if __name__ == "__main__":
    main()
