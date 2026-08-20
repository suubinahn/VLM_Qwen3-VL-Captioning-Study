"""
Diagnostic overfit test: can QLoRA training even learn to fix
vehicle_behavior errors (fabricated/misdescribed vehicles) on a tiny,
deliberately easy subset, given enough epochs?

Rationale: round-1 fine-tuning (166 examples, 3 epochs) showed ~0% fix rate
for vehicle_behavior in the comparison review, despite this category looking
"language-behavior fixable" in theory. Before concluding round 1 just needed
more data/epochs, this checks whether the mechanism can learn AT ALL - if it
can't even memorize a small, clean subset after heavy overfitting, that's
evidence of a deeper problem (e.g. the errors are actually vision-grounding
hallucinations, not a suppressible language habit), not a data-volume issue.

Trains for many epochs on just ~8 vehicle_behavior examples, then generates
on those SAME videos to check whether the fabricated/misdescribed vehicle
language is gone.

Usage:
  source activate qwen3vl && python finetune/overfit_test.py
"""
import gc
import json
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finetune_train as ft

import torch
from transformers import get_cosine_schedule_with_warmup

DATASET_PATH = os.path.join(os.path.dirname(__file__), "finetune_dataset.jsonl")
N_EXAMPLES = 8
NUM_EPOCHS = 30
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 1e-4
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "overfit_test_checkpoint")


def load_vehicle_behavior_subset(path, n):
    records = [json.loads(line) for line in open(path, "r", encoding="utf-8")]
    vb = [r for r in records if "vehicle_behavior" in r.get("categories", [])]
    return vb[:n]


def main():
    print("[1/3] Loading model + LoRA...")
    model, processor = ft.build_model_and_processor()
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
            except torch.OutOfMemoryError:
                print(f"    [skip OOM] {record['video']}")
                optimizer.zero_grad()
                gc.collect()
                torch.cuda.empty_cache()
                continue
            (loss / GRAD_ACCUM_STEPS).backward()
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
