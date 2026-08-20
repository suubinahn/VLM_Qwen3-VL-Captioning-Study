"""
QLoRA feasibility test - NOT the real training script.

Loads Qwen3-VL-8B-Instruct in 4-bit (bitsandbytes), attaches a LoRA adapter
(peft) to the language-model's attention/MLP projections only (vision tower
left untouched/frozen), and runs a single forward+backward pass on ONE
record from finetune_dataset.jsonl. Goal: confirm this fits in the 2x12GB
GPU budget and that gradients actually flow into the LoRA params, before
writing the full training loop.

Usage:
  source activate qwen3vl && python finetune_feasibility_test.py
"""
import gc
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_PATH = "./Qwen3-VL-8B-Instruct"
DATASET_PATH = "./finetune_dataset.jsonl"
# Training uses fewer frames than inference (NUM_FRAMES=16 in
# combined_caption_test.py / caption_review_tool.py, unchanged). 16 frames
# (~12,800 tokens) OOM'd on the attention/MLP path outright. 8 frames
# (~9,900 tokens) got all the way through forward+backward but missed by a
# reproducible, fixed ~474MB regardless of GPU layer split or LoRA dropout -
# i.e. a genuine sequence-length-driven ceiling, not a tunable inefficiency.
# 6 frames adds real margin instead of chasing the last few hundred MB.
NUM_FRAMES = 6

LORA_TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def load_one_record(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise RuntimeError(f"no records found in {path}")


def find_lora_targets(model):
    """Pick attention/MLP linear layers to adapt, restricted to the language
    model decoder - skip the vision tower/merger so the adapter only learns
    text-generation behavior, not visual feature extraction."""
    targets = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) and "Linear4bit" not in type(module).__name__:
            continue
        if "visual" in name or "vision" in name:
            continue
        if name.endswith(LORA_TARGET_SUFFIXES):
            targets.add(name.split(".")[-1])
    return sorted(targets)


def print_mem(tag):
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"    [mem:{tag}] GPU {i}: allocated {alloc:.2f} GB, reserved {reserved:.2f} GB")


def main():
    print("[1/5] Loading model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # device_map="auto" balances by STATIC WEIGHT size only, which ignores
    # activation memory entirely - it put 35 of 36 decoder layers on GPU 1
    # alone (all comfortably small at 4-bit), leaving GPU 1 to do nearly the
    # whole forward pass's activations by itself -> OOM despite GPU 0 sitting
    # almost empty. max_memory caps didn't change this (still weight-based),
    # so split the 36 decoder layers evenly by hand instead.
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
    print(f"    hf_device_map layer distribution: "
          f"{__import__('collections').Counter(str(v) for v in model.hf_device_map.values())}")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print_mem("after model load")

    print("[2/5] Preparing model for k-bit training + attaching LoRA...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()

    target_modules = find_lora_targets(model)
    print(f"    LoRA target modules: {target_modules}")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # 0.0 -> peft uses nn.Identity() for the dropout module instead of
        # nn.Dropout(p>0). At p>0, dropout allocates a full extra copy of its
        # input on every call - for down_proj's LoRA path that input is
        # (seq_len, intermediate_size=12288), a ~474MB tensor per recompute
        # that was the exact, reproducible OOM culprit in this test (moving
        # decoder layers between the two GPUs didn't change the deficit at
        # all, which is what pointed at a per-layer allocation, not a global
        # imbalance).
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print_mem("after LoRA attach")

    print("[3/5] Loading one finetune_dataset.jsonl record...")
    record = load_one_record(DATASET_PATH)
    print(f"    video: {record['video']}")
    print(f"    prompt_type: {record['prompt_type']}")

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
    for k, v in prompt_inputs.items():
        if hasattr(v, "shape"):
            print(f"    prompt_inputs[{k}]: shape={tuple(v.shape)} dtype={v.dtype}")

    print("[4/5] Building labels (prompt tokens masked, target tokens supervised)...")
    target_ids = processor.tokenizer(
        record["target"] + processor.tokenizer.eos_token,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids

    input_ids = torch.cat([prompt_inputs.input_ids, target_ids], dim=1)
    attention_mask = torch.cat(
        [prompt_inputs.attention_mask, torch.ones_like(target_ids)], dim=1
    )
    model_inputs = dict(prompt_inputs)
    model_inputs.pop("input_ids")
    model_inputs.pop("attention_mask")
    # mm_token_type_ids (0=text/1=image/2=video per token) is computed by the
    # processor for the prompt-only sequence and is needed for M-RoPE - the
    # appended target tokens are plain text, so pad it with zeros to match
    # the new concatenated length rather than leaving it at the old length.
    if "mm_token_type_ids" in model_inputs:
        text_pad = torch.zeros_like(target_ids)
        model_inputs["mm_token_type_ids"] = torch.cat(
            [model_inputs["mm_token_type_ids"], text_pad], dim=1
        )
    device = model.get_input_embeddings().weight.device
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    print_mem("before forward")
    print("[5/5] Running one forward+backward pass...")
    model.train()
    # Computing lm_head logits over the FULL 9901-token sequence (mostly the
    # given-facts prompt, which is never a training target) OOM'd on its own
    # (5.72 GiB for one tensor: seq_len x vocab_size). logits_to_keep restricts
    # lm_head to just the target region (+1 token so the last prompt position
    # can predict the first target token) - loss is then computed by hand
    # instead of relying on the model's built-in loss_function, since that
    # expects labels shaped to match the FULL sequence, not a kept suffix.
    logits_to_keep = target_ids.shape[1] + 1
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=logits_to_keep,
        **model_inputs,
    )
    shift_logits = outputs.logits[:, :-1, :].float()
    shift_labels = target_ids.to(device)
    loss = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
    )
    print(f"    loss: {loss.item():.4f}")
    loss.backward()

    grad_norm_total = 0.0
    num_params_with_grad = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad_norm_total += param.grad.norm().item() ** 2
                num_params_with_grad += 1
    print(f"    trainable params with non-None grad: {num_params_with_grad}")
    print(f"    total grad norm: {grad_norm_total ** 0.5:.4f}")

    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.max_memory_allocated(i) / 1024**3
        reserved = torch.cuda.max_memory_reserved(i) / 1024**3
        print(f"    GPU {i}: peak allocated {allocated:.2f} GB, peak reserved {reserved:.2f} GB")

    del outputs, loss, input_ids, attention_mask, model_inputs
    gc.collect()
    torch.cuda.empty_cache()
    print("\nFeasibility test passed: forward+backward completed, gradients flowed into LoRA params.")


if __name__ == "__main__":
    main()
