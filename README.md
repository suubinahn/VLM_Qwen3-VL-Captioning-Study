# Dashcam VLM Captioning Pipeline

Generates two English captions (`1_caption.txt`, a free-form paragraph; `2_caption.txt`, a structured driving-scene paragraph) per dashcam clip, using Qwen3-VL-8B-Instruct grounded with YOLO+ByteTrack vehicle/pedestrian tracking and EasyOCR sign reading — both verified via crop+VQA before being trusted as fact, rather than injected as unverified hints.

**Not included in this repo** (see `.gitignore`): the video dataset, model weights, fine-tuning checkpoints, and any logs/JSON that quote real caption text or dataset file paths. This repo is the pipeline code only — see "What you need to supply" below.

## Repo layout

- `captioning_tools/` — the production pipeline.
  - `caption_review_tool.py` — main entry point (generation + interactive review).
  - `detection_hint.py` — YOLO detection + ByteTrack tracking + crop/VQA verification (counts, positions, lateral-motion/merge detection).
  - `ocr_hint.py` — EasyOCR text localization + crop/VQA verification (road/directional signs only).
  - `combined_hint.py` — merges the two into one grounding-facts block.
  - `compare_review_tool*.py` — side-by-side base-vs-finetuned caption review tools (from the fine-tuning evaluation rounds; kept for reference).
- `captioning_tests/` — early standalone prompt-experiment scripts (superseded by `caption_review_tool.py`; kept for history).
- `finetune/` — QLoRA fine-tuning scripts and evaluation tooling. **Fine-tuning was tried across several rounds and ultimately abandoned** — a base model given the same verified grounding facts performed at least as well. Kept for reference; not part of the current production path.
- `keyword_crosscheck.py` — one-off script that cross-checks flagged errors between the two caption files for a video.

## Setup

1. Python 3.10, CUDA 12.1, an NVIDIA GPU (developed on 2x 12GB GPUs — see the memory-management notes in `caption_review_tool.py`/`detection_hint.py` if running on less).
2. `pip install -r requirements.txt`
3. `flash-attn` can be finicky to install depending on your OS/glibc version — if the prebuilt wheel fails to import, build from source (`FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-build-isolation`), which needs a matching CUDA toolkit (`nvcc`, `cuda_runtime.h`) available — installable via conda (`cuda-nvcc`, `cuda-cudart-dev`, `cuda-cccl`, `cuda-nvrtc-dev` for CUDA 12.1) if your system CUDA doesn't match.
4. Download the model into `./Qwen3-VL-8B-Instruct/` — this is [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) on Hugging Face, unmodified.
5. YOLO weights (`yolo11m.pt`) auto-download on first run via `ultralytics` — no manual step needed.

## What you need to supply

The pipeline expects a `sample_videos/` directory (path set by `SAMPLE_ROOT` in `caption_review_tool.py`) containing one folder per recording session, each with `drive_N/0X/` clip folders. Each clip folder needs:

- `1_clip/5.mp4` — the video clip.
- `info.txt` — 6 lines: motion, road_context, road_type, time_of_day, surface, weather (used as given facts, not re-derived by the model — see `format_reference_facts()`).

Output is written to a sibling `sb_caption/` folder per clip (`1_caption.txt`, `2_caption.txt`).

## Running

```bash
# Unattended batch generation (no player, no review prompts)
python captioning_tools/caption_review_tool.py --auto

# Interactive review of already-generated captions only (no GPU use)
python captioning_tools/caption_review_tool.py --review-range 1 100

# Force-regenerate specific videos
python captioning_tools/caption_review_tool.py --regenerate-indices 12 47 103
python captioning_tools/caption_review_tool.py --regenerate-range 1 50
```

Run with no flags for the full interactive review loop (generates as it goes, opens a video player per clip, supports flagging categories and `info.txt` correction overrides — see the in-tool key hints).
