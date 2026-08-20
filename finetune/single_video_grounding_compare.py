"""
Processes ONE video, with ONE model (base or fine-tuned), in its own
process - invoked repeatedly (once per video x model combination) by a
driver script/shell loop, never looping over multiple videos within a
single Python process.

Why: base_vs_finetuned_grounding_test.py, which looped over all 6 test
videos within one process, hit reproducible CUDA OOM on the 3rd video
(drive_14/04, the one with the most verification calls) THREE times in a
row, on both GPUs, even with a hard cleanup+retry on OOM. Cross-call
memory accumulation across many videos in one long-running process is a
real, reproducible risk on these 12GB GPUs once grounding's extra
verification calls are added - not something a per-call retry reliably
recovers from. Isolating each video into its own process is the guaranteed
fix - each process starts with a completely clean GPU (confirmed via
nvidia-smi between runs).

Grounding facts are computed once (mode=base always computes+caches them,
since crop+VQA verification is about reading the video correctly, not
caption style) and cached to FACTS_CACHE_PATH so mode=finetuned reuses the
identical facts rather than recomputing - keeps the comparison fair AND
avoids redundant work.

Usage:
  source activate qwen3vl && python finetune/single_video_grounding_compare.py \\
      --mode base --video <path> --out <result.json>
  source activate qwen3vl && python finetune/single_video_grounding_compare.py \\
      --mode finetuned --video <path> --out <result.json>
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
FACTS_CACHE_PATH = os.path.join(REPO_ROOT, "finetune", "base_vs_finetuned_facts_cache.json")


def load_facts_cache():
    if not os.path.isfile(FACTS_CACHE_PATH):
        return {}
    with open(FACTS_CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_facts_cache(cache):
    with open(FACTS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)


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


def build_base_model():
    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor


def build_finetuned_model():
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
        MODEL_PATH, quantization_config=bnb_config, device_map=device_map,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "finetuned"], required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    video_path = args.video
    info = read_info(video_path)
    reference_facts = format_reference_facts(info)

    print(f"Loading {args.mode} model...")
    if args.mode == "base":
        model, processor = build_base_model()
    else:
        model, processor = build_finetuned_model()

    facts_cache = load_facts_cache()
    if video_path in facts_cache:
        facts_text = facts_cache[video_path]
        print("(reusing cached grounding facts)")
    else:
        facts_text = get_combined_facts(video_path, model, processor)
        facts_cache[video_path] = facts_text
        save_facts_cache(facts_cache)
    print(f"verified facts: {facts_text}")

    combined_facts = reference_facts + ("\n" + facts_text if facts_text else "")
    prompt = PROMPT_STRUCTURED_TEMPLATE.format(reference_facts=combined_facts)
    caption = run(model, processor, video_path, prompt)
    print(f"caption: {caption}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "video": video_path,
            "mode": args.mode,
            "facts_text": facts_text,
            "caption": caption,
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
