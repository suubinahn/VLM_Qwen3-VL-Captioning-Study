"""
Real QLoRA training run with the vision tower included in the LoRA targets
(language + vision, not language-only), on the full finetune_dataset.jsonl
with a proper train/val split - the follow-up to overfit_test_vision.py's
8-example diagnostic, which confirmed this configuration trains cleanly (no
OOM, clean convergence) after rebalancing the GPU layer split to 14/22. That
test only proved the run is technically feasible; it says nothing about real
generalization, which is what this script checks.

Round 2 (2026-08-11, OUTPUT_DIR=finetune_checkpoints_vision_round2): the
first vision-inclusive run (finetune_checkpoints_vision/, 165 records)
trained with road_type entirely excluded (0 examples - a blanket-exclusion
bug in build_finetune_dataset.py) and motion/road_context/environment
partially excluded due to stale caption text contradicting x-corrected
info.txt facts. Both bugs were found and fixed this session (see
build_finetune_dataset.py's has_confirmed_info_error() history) - the
dataset grew from 165 to 221 records, road_type alone going from 0 to 92.
This round trains fresh (new OUTPUT_DIR, not resumed from round 1's
epoch_6) so round 1's epoch_4 checkpoint - already partway through
real-world comparison via compare_review_tool_vision_clean.py - stays
untouched as a reference point.

Mirrors finetune_train.py's (round 2, language-only) training loop
(train/val split, per-epoch checkpointing, resume-from-latest-epoch, cosine
schedule with warmup) so the two runs are comparable on data/epochs/LR -
the only intentional differences are vision-inclusive LoRA targets,
NUM_FRAMES=4 (vs 6; vision-inclusive backward pass needs more headroom per
overfit_test_vision.py's investigation), and a 14/22 GPU0/GPU1 layer split
(vs 18/18; GPU0 also carries the now-LoRA-adapted vision tower).

Usage:
  source activate qwen3vl && python finetune/finetune_train_vision.py
"""
import gc
import os
import random
import sys
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finetune_train as ft

ft.NUM_FRAMES = 4

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune_dataset.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "finetune_checkpoints_vision_round2")
OUTPUT_DIR = os.path.normpath(OUTPUT_DIR)
NUM_EPOCHS = 6
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.03
VAL_FRACTION = 0.1
SEED = 42
LOG_EVERY = 10

LANGUAGE_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
VISION_SUFFIXES = ("qkv", "proj", "linear_fc1", "linear_fc2")


def find_lora_targets_with_vision(model):
    """Same rationale as overfit_test_vision.py's version: language suffixes
    matched everywhere, vision suffixes only within "visual"-path modules so
    generic names like "proj" don't leak into unrelated language modules."""
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


def build_model_and_processor(resume_from=None):
    bnb_config = ft.BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    NUM_LAYERS = 36
    GPU0_LAYERS = 14  # verified OOM-free split for vision-inclusive LoRA (overfit_test_vision.py)
    device_map = {
        "model.visual": 0,
        "model.language_model.embed_tokens": 0,
        "model.language_model.norm": 1,
        "model.language_model.rotary_emb": 1,
        "lm_head": 1,
    }
    for i in range(NUM_LAYERS):
        device_map[f"model.language_model.layers.{i}"] = 0 if i < GPU0_LAYERS else 1

    model = AutoModelForImageTextToText.from_pretrained(
        ft.MODEL_PATH, quantization_config=bnb_config, device_map=device_map,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(ft.MODEL_PATH)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()

    if resume_from:
        print(f"    resuming LoRA weights from {resume_from}")
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
    else:
        target_modules = find_lora_targets_with_vision(model)
        print(f"    LoRA target modules (language + vision): {target_modules}")
        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    resume_dir, resume_epoch = ft.find_latest_checkpoint(OUTPUT_DIR)
    if resume_dir:
        print(f"[1/3] Found checkpoint at epoch {resume_epoch} - resuming from there "
              f"(loading model in 4-bit + attaching saved LoRA)...")
    else:
        print("[1/3] Loading model in 4-bit + attaching LoRA (language + vision)...")
    model, processor = build_model_and_processor(resume_from=resume_dir)
    device = model.get_input_embeddings().weight.device
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    print("[2/3] Loading dataset and splitting train/val...")
    records = ft.load_dataset(DATASET_PATH)
    random.shuffle(records)
    n_val = max(1, int(len(records) * VAL_FRACTION))
    val_records = records[:n_val]
    train_records = records[n_val:]
    print(f"    train: {len(train_records)}  val: {len(val_records)}")

    total_steps = max(1, (len(train_records) * NUM_EPOCHS) // GRAD_ACCUM_STEPS)
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * WARMUP_RATIO)),
        num_training_steps=total_steps,
    )

    print(f"[3/3] Training: {NUM_EPOCHS} epochs, grad_accum={GRAD_ACCUM_STEPS}, "
          f"effective batch={GRAD_ACCUM_STEPS}, total optimizer steps={total_steps}")

    log_path = os.path.join(OUTPUT_DIR, "train_log.txt")
    steps_per_epoch = max(1, len(train_records) // GRAD_ACCUM_STEPS)
    step = resume_epoch * steps_per_epoch
    for _ in range(step):
        scheduler.step()
    if resume_epoch:
        print(f"    fast-forwarded scheduler by {step} steps (resuming after epoch {resume_epoch})")
    micro_step = 0
    running_loss = 0.0
    optimizer.zero_grad()
    model.train()

    for epoch in range(resume_epoch, NUM_EPOCHS):
        random.shuffle(train_records)
        for record in train_records:
            try:
                loss = ft.compute_loss(model, processor, record, device)
            except torch.OutOfMemoryError:
                print(f"    [train skip OOM] {record['video']} ({record['prompt_type']})")
                optimizer.zero_grad()
                gc.collect()
                torch.cuda.empty_cache()
                continue

            (loss / GRAD_ACCUM_STEPS).backward()
            running_loss += loss.item()
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

                if step % LOG_EVERY == 0:
                    avg_loss = running_loss / (LOG_EVERY * GRAD_ACCUM_STEPS)
                    msg = (f"epoch {epoch + 1}/{NUM_EPOCHS} step {step}/{total_steps} "
                           f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
                    print(msg)
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"{datetime.now().isoformat()} {msg}\n")
                    running_loss = 0.0

        val_loss = ft.run_validation(model, processor, val_records, device)
        msg = f"epoch {epoch + 1}/{NUM_EPOCHS} done - val_loss={val_loss:.4f}"
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} {msg}\n")

        ckpt_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch + 1}")
        model.save_pretrained(ckpt_dir)
        print(f"    saved LoRA adapter to {ckpt_dir}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
