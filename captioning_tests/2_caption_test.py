"""
Qwen3-VL driving video captioning - task 2 (structured key:value fields), PROMPT EXPERIMENT mode.

Each run appends the prompt + generated output to experiments/2_caption_experiments.txt
(with a timestamp) instead of overwriting anything, so multiple prompt attempts
can be compared side by side.

Edit PROMPT_STRUCTURED below between runs to try different wording.
"""
import os
from datetime import datetime
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_PATH = "./Qwen3-VL-8B-Instruct"
VIDEO_PATH = "./sample_videos/231106095132_M801C06L62G031_2673_end_extract/drive_1/02/1_clip/5.mp4"
NUM_FRAMES = 8
EXPERIMENT_LOG = "./experiments/2_caption_experiments.txt"

PROMPT_STRUCTURED = (
    "Analyze this driving video and write a single paragraph that covers, in "
    "flowing prose, each of the following: the ego-vehicle's driving motion "
    "(e.g. moving straight, turning left, turning right, changing lanes, "
    "stopped); any notable nearby vehicles and their behavior; other relevant "
    "surrounding objects such as pedestrians, traffic signs, traffic lights, or "
    "road markings; the driving environment (e.g. intersection, highway, urban "
    "street, residential road, and any road structures such as a bridge, tunnel, "
    "overpass, or underpass if present); and the weather and road surface conditions. "
    "Cover every element but do not use labels, headings, or bullet points - "
    "just one cohesive paragraph. Only describe things you can clearly see in "
    "the frames; do not assume or guess at typical scene elements (e.g. "
    "pedestrians, specific vehicles) that are not actually visible. If a "
    "category has nothing notable, state that briefly rather than inventing detail."
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
            {"type": "text", "text": PROMPT_STRUCTURED},
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
    f.write(f"prompt: {PROMPT_STRUCTURED}\n")
    f.write("-" * 80 + "\n")
    f.write(output_text + "\n\n")

print(f"Appended to {EXPERIMENT_LOG}")
print("\n=== output ===")
print(output_text)
