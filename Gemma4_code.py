# pip install transformers accelerate bitsandbytes  # or flash-attn for speed
from transformers import AutoModelForMultimodalLM, AutoProcessor
import torch
from PIL import Image
import requests

# Load model (use E2B for edge, 31B-IT for max quality; quantized versions available)
model_id = "google/gemma-4-E4B-it"   # or "google/gemma-4-31B-it" / "google/gemma-4-26B-A4B-it"
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",          # or "cuda" / "cpu"
    attn_implementation="flash_attention_2"  # for speed
)
processor = AutoProcessor.from_pretrained(model_id)

# Example image + prompt (replace with your URL or local path)
image_url = "https://picsum.photos/id/1015/800/600"  # sample UI/diagram image
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# Define a tool (function calling)
tools = [
    {
        "type": "function",
        "function": {
            "name": "generate_html_from_image",
            "description": "Generate clean HTML/CSS from a UI screenshot or diagram",
            "parameters": {
                "type": "object",
                "properties": {
                    "html_code": {"type": "string", "description": "Full HTML code"}
                },
                "required": ["html_code"]
            }
        }
    }
]

# Messages with multimodal input + thinking mode + tool
messages = [
    {
        "role": "system",
        "content": "You are a helpful visual agent. Use thinking mode for step-by-step reasoning. Always use tools when appropriate."
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},   # or image path/bytes
            {"type": "text", "text": "Analyze this image in detail. Then use the generate_html_from_image tool to output production-ready HTML/CSS that recreates the UI exactly."}
        ]
    }
]

# Process inputs
inputs = processor.apply_chat_template(
    messages,
    tools=tools,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=True   # activates <|think|> mode
).to(model.device)

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.95,
    do_sample=True
)

response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
