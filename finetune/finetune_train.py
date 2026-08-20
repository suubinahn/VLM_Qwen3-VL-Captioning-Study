"""
QLoRA training loop for Qwen3-VL-8B-Instruct captioning fine-tune.

Builds on the configuration verified in finetune_feasibility_test.py:
4-bit quant + LoRA (language-model attention/MLP only, vision tower frozen)
+ flash-attention-2 + gradient checkpointing + manual 18/18 GPU layer split
+ logits_to_keep (avoids computing lm_head over the whole prompt, which
OOM'd on its own). Training uses NUM_FRAMES=6 - inference/captioning stays
at 16 frames in combined_caption_test.py/caption_review_tool.py, unchanged;
see finetune_feasibility_test.py's NUM_FRAMES comment for why they differ.

Batch size is 1 sample (a single video's prompt+target won't fit more than
one at a time in 12GB with gradient checkpointing); GRAD_ACCUM_STEPS
simulates a larger effective batch by accumulating gradients before each
optimizer step.

Saves a LoRA adapter checkpoint (via peft's save_pretrained, NOT the full
base model) after every epoch to OUTPUT_DIR/epoch_N.

Usage:
  source activate qwen3vl && python finetune_train.py
"""
import gc
import json
import os
import random
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

MODEL_PATH = "./Qwen3-VL-8B-Instruct"
# DATASET_PATH must be absolute - the dataset lives in finetune/, not the
# repo root, but this script (and MODEL_PATH/OUTPUT_DIR above/below) still
# needs to be run with the repo root as cwd for those to resolve correctly.
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune_dataset.jsonl")
OUTPUT_DIR = "./finetune_checkpoints_round2"
NUM_FRAMES = 6
NUM_EPOCHS = 6
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.03
VAL_FRACTION = 0.1
SEED = 42
LOG_EVERY = 10

LORA_TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def load_dataset(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_lora_targets(model):
    """Attention/MLP linear layers in the language model decoder only -
    skips the vision tower/merger so the adapter learns text-generation
    behavior, not visual feature extraction."""
    targets = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) and "Linear4bit" not in type(module).__name__:
            continue
        if "visual" in name or "vision" in name:
            continue
        if name.endswith(LORA_TARGET_SUFFIXES):
            targets.add(name.split(".")[-1])
    return sorted(targets)


def find_latest_checkpoint(output_dir):
    """Highest-numbered OUTPUT_DIR/epoch_N found, or None. Used to resume a
    run that got killed mid-training (e.g. VS Code closed) instead of
    restarting from epoch 0 - there's no saved optimizer/scheduler state, so
    the resumed run gets a fresh optimizer and a scheduler fast-forwarded to
    the right point (see main()), not a bit-exact continuation."""
    if not os.path.isdir(output_dir):
        return None, 0
    best_epoch, best_dir = 0, None
    for name in os.listdir(output_dir):
        if name.startswith("epoch_") and os.path.isdir(os.path.join(output_dir, name)):
            try:
                n = int(name.split("_")[1])
            except (IndexError, ValueError):
                continue
            if n > best_epoch:
                best_epoch, best_dir = n, os.path.join(output_dir, name)
    return best_dir, best_epoch


def build_model_and_processor(resume_from=None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # device_map="auto" put 35 of 36 decoder layers on GPU 1 alone (confirmed
    # in finetune_feasibility_test.py) - balance layers by hand instead.
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

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map=device_map,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()

    if resume_from:
        print(f"    resuming LoRA weights from {resume_from}")
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
    else:
        target_modules = find_lora_targets(model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def build_inputs(processor, record, device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": record["video"]},
                {"type": "text", "text": record["prompt"]},
            ],
        }
    ]
    prompt_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        num_frames=NUM_FRAMES,
        fps=None,
    )
    target_ids = processor.tokenizer(
        record["target"] + processor.tokenizer.eos_token,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids

    input_ids = torch.cat([prompt_inputs.input_ids, target_ids], dim=1)
    attention_mask = torch.cat([prompt_inputs.attention_mask, torch.ones_like(target_ids)], dim=1)

    model_inputs = dict(prompt_inputs)
    model_inputs.pop("input_ids")
    model_inputs.pop("attention_mask")
    # mm_token_type_ids is computed by the processor for the prompt-only
    # sequence - pad with zeros (text) to match the appended target tokens.
    if "mm_token_type_ids" in model_inputs:
        text_pad = torch.zeros_like(target_ids)
        model_inputs["mm_token_type_ids"] = torch.cat(
            [model_inputs["mm_token_type_ids"], text_pad], dim=1
        )

    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    target_ids = target_ids.to(device)

    return input_ids, attention_mask, target_ids, model_inputs


def compute_loss(model, processor, record, device):
    input_ids, attention_mask, target_ids, model_inputs = build_inputs(processor, record, device)
    # Restrict lm_head to the target region (+1 token to predict the first
    # target token) instead of the full prompt - computing logits over the
    # whole ~10k-token prompt OOM'd on its own (see finetune_feasibility_test.py).
    logits_to_keep = target_ids.shape[1] + 1
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=logits_to_keep,
        **model_inputs,
    )
    shift_logits = outputs.logits[:, :-1, :].float()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)), target_ids.reshape(-1)
    )
    return loss


def run_validation(model, processor, val_records, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for record in val_records:
            try:
                loss = compute_loss(model, processor, record, device)
                total += loss.item()
                count += 1
                del loss
            except torch.OutOfMemoryError:
                print(f"    [val skip OOM] {record['video']} ({record['prompt_type']})")
            gc.collect()
            torch.cuda.empty_cache()
    model.train()
    return total / max(1, count)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    resume_dir, resume_epoch = find_latest_checkpoint(OUTPUT_DIR)
    if resume_dir:
        print(f"[1/3] Found checkpoint at epoch {resume_epoch} - resuming from there "
              f"(loading model in 4-bit + attaching saved LoRA)...")
    else:
        print("[1/3] Loading model in 4-bit + attaching LoRA...")
    model, processor = build_model_and_processor(resume_from=resume_dir)
    device = model.get_input_embeddings().weight.device
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    print("[2/3] Loading dataset and splitting train/val...")
    records = load_dataset(DATASET_PATH)
    random.shuffle(records)
    n_val = max(1, int(len(records) * VAL_FRACTION))
    val_records = records[:n_val]
    train_records = records[n_val:]
    print(f"    train: {len(train_records)}  val: {len(val_records)}")

    total_steps = max(1, (len(train_records) * NUM_EPOCHS) // GRAD_ACCUM_STEPS)
    # No saved optimizer/scheduler state across a resume (only the LoRA
    # weights are checkpointed) - a fresh AdamW is standard practice here,
    # but the scheduler is fast-forwarded below so the LR picks up roughly
    # where it left off instead of restarting the warmup/cosine curve.
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
                loss = compute_loss(model, processor, record, device)
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

        val_loss = run_validation(model, processor, val_records, device)
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
