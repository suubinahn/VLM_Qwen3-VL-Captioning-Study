"""
Real-world check for the vision-inclusive LoRA run (finetune_checkpoints_vision/
epoch_4, the val-loss-best checkpoint - see finetune_train_vision.py's run:
0.6491/0.6088/0.6089/0.5962/0.6101/0.6117, epoch 4 lowest before it started
rising again): generates captions with this checkpoint on the SAME 25-video
sample round 2 used (same SAMPLE_SEED=7, same 160-video pool), so base
(sb_caption) / round 2 (sb_caption_ft_r2) / this run (sb_caption_ft_vision)
are all directly comparable on identical videos - not just three separate
uncoordinated samples.

Mirrors rerun_review_r2_sample.py's structure; only the adapter path and
output folder differ. Base model load is still 4-bit + standard inference
device_map (NOT the training-time 14/22 split - that was only needed to fit
backward-pass memory; inference has no optimizer/gradient-checkpointing
memory pressure, so the same split round 1/round 2 inference used is fine
here too). NUM_FRAMES=16, matching the inference pipeline (decoupled from
training's NUM_FRAMES=4, same as round 1/2's decoupling from their
NUM_FRAMES=6 training setting).

Output goes to a NEW sibling folder "sb_caption_ft_vision" - round 1's
sb_caption_ft and round 2's sb_caption_ft_r2 are left untouched.

Usage:
  source activate qwen3vl && python finetune/rerun_review_vision_sample.py
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
ADAPTER_PATH = os.path.join(REPO_ROOT, "finetune_checkpoints_vision", "epoch_4")
FLAGGED_LOG = os.path.join(REPO_ROOT, "flagged_videos.txt")
NUM_FRAMES = 16
SAMPLE_UNFLAGGED = 30
SAMPLE_SIZE = 25
SEED = 42
SAMPLE_SEED = 7  # same as rerun_review_r2_sample.py - keeps the sample identical across rounds


def load_flagged_videos():
    with open(FLAGGED_LOG, "r", encoding="utf-8") as f:
        paths = {line.split("\t")[1] for line in f if line.strip() and "\t" in line}
    return sorted(paths)


def output_paths_ft(video_path):
    clip_dir = os.path.dirname(video_path)
    parent_dir = os.path.dirname(clip_dir)
    out_dir = os.path.join(parent_dir, "sb_caption_ft_vision")
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

    pool = flagged + sampled_unflagged
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
    print(f"Sample targets: {len(targets)} videos (same pool/sample as round 2)")

    print(f"Loading base model + vision-inclusive LoRA adapter ({ADAPTER_PATH})...")
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
