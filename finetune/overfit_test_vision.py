"""
"Step 2" diagnostic: same overfit test as overfit_test.py (vehicle_behavior,
8 examples, 30 epochs), but this time the vision tower is ALSO included in
the LoRA targets (not just the language model). Language targets are kept
too - this adds vision capacity on top, it doesn't replace the language LoRA.

Rationale: round 1 and round 2 (language-only LoRA, more data/epochs in
round 2) both showed weak real-world improvement despite the language-only
overfit test succeeding perfectly on its 8 training examples - round 2 also
showed a val-loss overfitting signal (rising after epoch 3). Combined with
vehicle_behavior/visibility errors being reclassified as hallucination-type
(not language-policy-type - see prompt_iteration_summary.md), this tests
whether the vision encoder itself needs adjusting, not just the language
decoder.

  - Succeeds cleanly (similar to language-only's perfect memorization,
    without OOM) -> vision-inclusive training is technically feasible;
    worth trying for real (more data, proper train/val split).
  - OOMs or fails to memorize even 8 examples -> vision-inclusive LoRA isn't
    viable at this scale/hardware, or the vision tower genuinely can't
    represent this distinction - reconsider strategy entirely (e.g. lean on
    info.txt + human review rather than further fine-tuning investment).

Usage:
  source activate qwen3vl && python finetune/overfit_test_vision.py
"""
import gc
import json
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finetune_train as ft

# The language-only overfit test fit fine at 6 frames, but adding vision to
# the LoRA targets pushed every single step to OOM (confirmed: GPU0 hit
# 11.99/12.29GB before training even began, and all 8 examples OOM'd every
# epoch, all 30 epochs, avg_loss stayed exactly 0.0 the whole run - not one
# successful step). build_inputs()/compute_loss() in finetune_train.py read
# NUM_FRAMES from that module's own global, not a parameter, so overriding
# it here (before any model/data work) affects both training and the final
# generation check consistently.
ft.NUM_FRAMES = 4

import torch
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

DATASET_PATH = os.path.join(os.path.dirname(__file__), "finetune_dataset.jsonl")
N_EXAMPLES = 8
NUM_EPOCHS = 30
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 1e-4
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "overfit_test_vision_checkpoint")

LANGUAGE_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
VISION_SUFFIXES = ("qkv", "proj", "linear_fc1", "linear_fc2")


def find_lora_targets_with_vision(model):
    """Same as finetune_train.find_lora_targets(), but does NOT skip the
    vision tower - language suffixes are matched everywhere, vision suffixes
    (qkv/proj/linear_fc1/linear_fc2, the vision tower's actual Linear layer
    names, confirmed by inspecting model.named_modules()) only within
    "visual"-path modules, so generic names like "proj" don't accidentally
    match unrelated language-side modules."""
    targets = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) and "Linear4bit" not in type(module).__name__:
            continue
        is_vision = "visual" in name or "vision" in name
        leaf = name.split(".")[-1]
        if is_vision:
            if leaf in VISION_SUFFIXES:
                targets.add(leaf)
        else:
            if name.endswith(LANGUAGE_SUFFIXES):
                targets.add(leaf)
    return sorted(targets)


def load_vehicle_behavior_subset(path, n):
    records = [json.loads(line) for line in open(path, "r", encoding="utf-8")]
    vb = [r for r in records if "vehicle_behavior" in r.get("categories", [])]
    return vb[:n]


def main():
    print("[1/3] Loading model in 4-bit + attaching LoRA (language + vision)...")
    bnb_config = ft.BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    NUM_LAYERS = 36
    device_map = {
        "model.visual": 0,
        "model.language_model.embed_tokens": 0,
        "model.language_model.norm": 1,
        "model.language_model.rotary_emb": 1,
        "lm_head": 1,
    }
    # GPU 0 also carries the vision tower + embed_tokens, and now the vision
    # tower needs full backward (it's LoRA-adapted too, unlike the
    # language-only run) - give it fewer decoder layers than GPU 1 to
    # compensate, instead of the even 18/18 split used for language-only.
    GPU0_LAYERS = 14
    for i in range(NUM_LAYERS):
        device_map[f"model.language_model.layers.{i}"] = 0 if i < GPU0_LAYERS else 1

    model = ft.AutoModelForImageTextToText.from_pretrained(
        ft.MODEL_PATH, quantization_config=bnb_config, device_map=device_map,
        attn_implementation="flash_attention_2",
    )
    processor = ft.AutoProcessor.from_pretrained(ft.MODEL_PATH)
    model = ft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()

    target_modules = find_lora_targets_with_vision(model)
    print(f"    LoRA target modules (language + vision): {target_modules}")
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    device = model.get_input_embeddings().weight.device
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    print("[2/3] Loading tiny vehicle_behavior subset...")
    records = load_vehicle_behavior_subset(DATASET_PATH, N_EXAMPLES)
    print(f"    {len(records)} examples selected for overfitting")
    for r in records:
        print(f"    - {r['video']} ({r['prompt_type']})")

    total_steps = max(1, (len(records) * NUM_EPOCHS) // GRAD_ACCUM_STEPS)
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(total_steps * 0.05)), num_training_steps=total_steps
    )

    print(f"[3/3] Overfitting: {NUM_EPOCHS} epochs on {len(records)} examples, "
          f"total optimizer steps={total_steps}")
    model.train()
    micro_step, step = 0, 0
    optimizer.zero_grad()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for record in records:
            try:
                loss = ft.compute_loss(model, processor, record, device)
                (loss / GRAD_ACCUM_STEPS).backward()
            except torch.OutOfMemoryError:
                print(f"    [skip OOM] {record['video']}")
                optimizer.zero_grad()
                gc.collect()
                torch.cuda.empty_cache()
                continue
            epoch_loss += loss.item()
            micro_step += 1
            del loss
            gc.collect()
            torch.cuda.empty_cache()
            if micro_step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    epoch {epoch + 1}/{NUM_EPOCHS} avg_loss={epoch_loss / len(records):.4f}")
            for i in range(torch.cuda.device_count()):
                print(f"      GPU {i}: peak {torch.cuda.max_memory_allocated(i) / 1024**3:.2f} GB")

    model.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved overfit adapter to {OUTPUT_DIR}")

    print("\n=== Generating on the SAME training videos to check compliance ===")
    model.eval()
    for record in records:
        messages = [
            {"role": "user", "content": [
                {"type": "video", "video": record["video"]},
                {"type": "text", "text": record["prompt"]},
            ]}
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", num_frames=ft.NUM_FRAMES, fps=None,
        )
        inputs = inputs.to(device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        print("\n" + "=" * 80)
        print(f"video: {record['video']}")
        print(f"TARGET (what training aimed for):\n{record['target']}")
        print(f"GENERATED (after overfitting):\n{output_text}")
        del inputs, generated_ids, trimmed
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
