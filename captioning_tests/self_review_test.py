"""
One-off test: can Qwen3-VL catch its own known errors when asked to review
a caption against the video, instead of generating from scratch?

Uses 3 videos with already-confirmed ground-truth errors (from flagged_videos.txt)
to see if a review-style prompt surfaces them.
"""
import gc
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_PATH = "./Qwen3-VL-8B-Instruct"
NUM_FRAMES = 16

CASES = [
    {
        "name": "drive_4/01 (known error: Nissan Juke should be Tivoli)",
        "video": "./sample_videos/231106095132_M801C06L62G031_2673_end_extract/drive_4/01/1_clip/5.mp4",
        "caption": (
            "The ego-vehicle is stopped at an intersection on a multi-lane highway under "
            "cloudy skies, with a wet road surface. Ahead, a white Nissan Juke is "
            "stationary, and to the right, a white sedan is positioned in the adjacent "
            "lane. A green overhead road sign indicates directions to '사당역' and '부산', "
            "with a 'P' symbol for parking. Trees line the roadside, and buildings are "
            "visible in the background."
        ),
    },
    {
        "name": "drive_3/04 (known error: parking symbol should be junction diagram)",
        "video": "./sample_videos/231106095132_M801C06L62G031_2673_end_extract/drive_3/04/1_clip/5.mp4",
        "caption": (
            "The ego-vehicle is stopped in traffic on a multi-lane highway under cloudy "
            "skies, with a wet road surface. Ahead, a white SUV is stationary, and to "
            "its right, a dark blue Audi Q5 is also stopped. A green overhead road sign "
            "indicates directions to '사당역' and '부산', with a parking symbol. A silver "
            "sedan passes on the right, and a white car enters the frame from the right "
            "edge."
        ),
    },
    {
        "name": "drive_5/03 (known error: green traffic light missed)",
        "video": "./sample_videos/231106095132_M801C06L62G031_2673_end_extract/drive_5/03/1_clip/5.mp4",
        "caption": (
            "The ego-vehicle maintains a steady straight trajectory along the multi-lane "
            "highway, navigating through a city street slick with rainwater that "
            "reflects the bright, diffused daylight. Raindrops streak across the "
            "windshield, blurring the view of the wet asphalt and the towering "
            "glass-and-steel buildings lining both sides of the road. The urban "
            "environment is dense, with vehicles including a blue bus marked with route "
            "number 341 ahead, a white SUV to the left, and a black SUV in the adjacent "
            "lane, all moving in the same direction. Roadside barriers and orange "
            "traffic cones are visible along the edges, and the street is flanked by "
            "trees with autumnal foliage and Korean flags fluttering near the sidewalks. "
            "No road signs or traffic lights are clearly visible in the frames, and no "
            "sudden maneuvers or hazards are evident as the vehicles proceed in a calm, "
            "orderly flow."
        ),
    },
]

REVIEW_PROMPT_TEMPLATE = (
    "You are reviewing a caption written for this driving video, checking it for "
    "factual errors. Here is the caption:\n\n\"{caption}\"\n\n"
    "Carefully compare every claim in the caption against what is actually visible "
    "in the video frames. List each factual error, misidentification, or "
    "hallucination you find, and state what should have been said instead. If a "
    "vehicle brand/model is named, check it carefully against any visible badges or "
    "distinctive shape. If a sign or symbol is described, check its actual shape "
    "against what's claimed. If the caption claims something is NOT visible, "
    "double check the frames for it. If you find no errors, say 'No errors found.'"
)


def run_review(model, processor, video_path, caption):
    prompt = REVIEW_PROMPT_TEMPLATE.format(caption=caption)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt},
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

    del inputs, generated_ids, generated_ids_trimmed
    gc.collect()
    torch.cuda.empty_cache()

    return output_text


if __name__ == "__main__":
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    for case in CASES:
        print("\n" + "=" * 80)
        print(case["name"])
        print("=" * 80)
        result = run_review(model, processor, case["video"], case["caption"])
        print(result)
