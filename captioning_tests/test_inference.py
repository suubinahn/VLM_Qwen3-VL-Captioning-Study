from transformers import AutoModelForImageTextToText, AutoProcessor

model_path = "./Qwen3-VL-8B-Instruct"
image_path = "./Qwen3-VL/cookbooks/assets/spatial_understanding/lots_of_people.jpeg"

print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

messages = [
    {"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": "이 이미지를 한국어로 자세히 설명해줘."},
    ]}
]
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
).to(model.device)

print("Generating...")
output = model.generate(**inputs, max_new_tokens=512)
result = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n=== Caption ===")
print(result)
