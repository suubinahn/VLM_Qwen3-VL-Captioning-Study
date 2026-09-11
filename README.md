# Dashcam VLM Captioning Pipeline

Qwen3-VL-8B-Instruct를 기반으로, YOLO+ByteTrack 기반 차량/보행자 추적과 EasyOCR 기반 표지판 인식으로 근거(grounding)를 보강하여, 블랙박스(주행) 영상 클립 하나당 2종의 영어 캡션(`1_caption.txt`: 자유 서술형 문단, `2_caption.txt`: 구조화된 주행 장면 문단)을 생성합니다. YOLO/OCR로 얻은 정보는 검증 없이 그대로 힌트로 주입하는 것이 아니라, crop 이미지 + VQA(시각 질의응답)를 거쳐 사실 여부를 확인한 뒤에만 근거로 사용합니다.

**이 저장소에 포함되지 않은 것** (`.gitignore` 참고): 영상 데이터셋, 모델 가중치, 파인튜닝 체크포인트, 그리고 실제 캡션 문장이나 데이터셋 파일 경로가 담긴 로그/JSON 파일. 이 저장소에는 파이프라인 코드만 포함되어 있습니다 — 아래 "직접 준비해야 하는 것" 항목을 참고하세요.

## 저장소 구조

- `captioning_tools/` — 실제 운영(프로덕션) 파이프라인
  - `caption_review_tool.py` — 메인 실행 파일 (캡션 생성 + 인터랙티브 검수)
  - `detection_hint.py` — YOLO 객체 탐지 + ByteTrack 추적 + crop/VQA 검증 (개수, 위치, 차선 변경/합류 등 측면 이동 감지)
  - `ocr_hint.py` — EasyOCR 기반 텍스트 위치 검출 + crop/VQA 검증 (도로·방향 표지판만 대상)
  - `combined_hint.py` — 위 두 결과를 하나의 근거(grounding) 정보 블록으로 병합
  - `compare_review_tool*.py` — 파인튜닝 평가 라운드에서 사용했던, 베이스 모델과 파인튜닝 모델의 캡션을 나란히 비교하는 검수 도구 (참고용으로 보존)
- `captioning_tests/` — 초기 단계의 개별 프롬프트 실험 스크립트 (현재는 `caption_review_tool.py`로 대체됨, 기록 보존 목적)
- `finetune/` — QLoRA 파인튜닝 스크립트 및 평가 도구. **여러 라운드에 걸쳐 파인튜닝을 시도했으나 최종적으로는 채택하지 않았습니다** — 동일한 검증된 근거 정보를 제공했을 때 베이스 모델이 최소한 동등하거나 더 나은 성능을 보였기 때문입니다. 참고용으로 남겨두었으며 현재 운영 파이프라인에는 포함되지 않습니다.
- `keyword_crosscheck.py` — 한 영상에 대해 두 캡션 파일 간에 표시된(flag) 오류를 상호 대조하는 1회성 스크립트

## 환경 설정

1. Python 3.10, CUDA 12.1, NVIDIA GPU (12GB GPU 2장 환경에서 개발됨 — 그보다 낮은 사양이라면 `caption_review_tool.py` / `detection_hint.py` 내 메모리 관리 관련 주석을 참고하세요)
2. `pip install -r requirements.txt`
3. `flash-attn`은 OS·glibc 버전에 따라 설치가 까다로울 수 있습니다. 미리 빌드된 wheel의 import가 실패하면 소스에서 직접 빌드하세요(`FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-build-isolation`). 이 경우 일치하는 CUDA 툴킷(`nvcc`, `cuda_runtime.h`)이 필요하며, 시스템 CUDA 버전이 맞지 않는다면 conda로 설치할 수 있습니다(CUDA 12.1 기준 `cuda-nvcc`, `cuda-cudart-dev`, `cuda-cccl`, `cuda-nvrtc-dev`).
4. `./Qwen3-VL-8B-Instruct/` 경로에 모델을 다운로드하세요 — Hugging Face의 [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) 모델을 수정 없이 그대로 사용합니다.
5. YOLO 가중치(`yolo11m.pt`)는 `ultralytics` 패키지를 통해 최초 실행 시 자동으로 다운로드되므로 별도 작업이 필요 없습니다.

## 직접 준비해야 하는 것

이 파이프라인은 `sample_videos/` 디렉터리(경로는 `caption_review_tool.py`의 `SAMPLE_ROOT`에서 설정)를 필요로 하며, 이 디렉터리 안에는 촬영 세션별 폴더가 있고 그 안에 각각 `drive_N/0X/` 형태의 클립 폴더가 들어 있어야 합니다. 각 클립 폴더에는 다음이 필요합니다:

- `1_clip/5.mp4` — 영상 클립 파일
- `info.txt` — 6줄 구성: motion(주행 동작), road_context(도로 맥락), road_type(도로 유형), time_of_day(시간대), surface(노면), weather(날씨). 이 값들은 모델이 다시 추론하는 대상이 아니라 **주어진 사실(given facts)**로 그대로 사용됩니다 (`format_reference_facts()` 참고).

결과물은 각 클립과 같은 위치(형제 경로)의 `sb_caption/` 폴더에 저장됩니다 (`1_caption.txt`, `2_caption.txt`).

## 실행 방법

```bash
# 무인 배치 생성 (영상 플레이어·검수 프롬프트 없이 전체 자동 실행)
python captioning_tools/caption_review_tool.py --auto

# 이미 생성된 캡션만 인터랙티브로 검수 (GPU 미사용)
python captioning_tools/caption_review_tool.py --review-range 1 100

# 특정 영상만 강제로 재생성
python captioning_tools/caption_review_tool.py --regenerate-indices 12 47 103
python captioning_tools/caption_review_tool.py --regenerate-range 1 50
```

옵션 없이 실행하면 전체 인터랙티브 검수 루프가 동작합니다 (생성과 동시에 클립별로 영상 플레이어가 열리며, 오류 카테고리 플래깅과 `info.txt` 수정·덮어쓰기 기능을 지원합니다 — 자세한 키 조작법은 도구 내 안내를 참고하세요).
