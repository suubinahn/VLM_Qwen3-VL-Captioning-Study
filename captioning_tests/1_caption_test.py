"""
Qwen3-VL driving video captioning - task 1 (free-form description), PROMPT EXPERIMENT mode.

Each run appends the prompt + generated output to experiments/1_caption_experiments.txt
(with a timestamp) instead of overwriting anything, so multiple prompt attempts
can be compared side by side.

Edit PROMPT_DESCRIPTION below between runs to try different wording.
"""
import os
from datetime import datetime
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_PATH = "./Qwen3-VL-8B-Instruct"
VIDEO_PATH = "./sample_videos/231106095132_M801C06L62G031_2673_end_extract/drive_1/02/1_clip/5.mp4"
NUM_FRAMES = 8
EXPERIMENT_LOG = "./experiments/1_caption_experiments.txt"

PROMPT_DESCRIPTION = (
    "Write one compact paragraph (about 60-100 words) describing this driving "
    "video, centered on the ego-vehicle. Lead with what the ego-vehicle is doing "
    "(lane, maneuvers such as going straight, turning, or changing lanes, and "
    "where/when in the video), referencing an intersection, street name, or sign "
    "only if clearly visible. Then briefly add only the details that matter for "
    "this specific scene - e.g. a notable nearby vehicle or pedestrian, a relevant "
    "roadside hazard, or the general setting and weather - without restating "
    "similar things multiple times or listing every object you notice. Skip "
    "anything generic or repetitive. Do not use headings, bullet points, or "
    "labels - just one tight paragraph. Only describe things you can clearly "
    "see in the frames; do not assume or guess at typical scene elements "
    "(e.g. pedestrians, specific vehicles) that are not actually visible."
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": VIDEO_PATH},
            {"type": "text", "text": PROMPT_DESCRIPTION},
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
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0].strip()

os.makedirs(os.path.dirname(EXPERIMENT_LOG), exist_ok=True)
with open(EXPERIMENT_LOG, "a", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write(f"timestamp: {datetime.now().isoformat()}\n")
    f.write(f"video: {VIDEO_PATH}\n")
    f.write(f"num_frames: {NUM_FRAMES}\n")
    f.write(f"prompt: {PROMPT_DESCRIPTION}\n")
    f.write("-" * 80 + "\n")
    f.write(output_text + "\n\n")

print(f"Appended to {EXPERIMENT_LOG}")
print("\n=== output ===")
print(output_text)
