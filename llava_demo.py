import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16
)

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

from PIL import Image
import requests

image_url = "https://basketballhq.com/wp-content/uploads/2014/06/Playing-College-Basketball.jpg"
user_question = "What are the people doing?"

conversation = [
    {   "role": "user",
        "content": [
            {"type": "image", "url": image_url},
            {"type": "text", "text": user_question},
        ],
    },
]

prompt = processor.apply_chat_template(conversation,
    add_generation_prompt=True
)

raw_image = Image.open(requests.get(image_url, stream=True).raw)

inputs = processor(
    text=prompt,
    images=raw_image,
    return_tensors="pt"
).to(model.device)

print("Generating answer...")
generate_ids = model.generate(
    **inputs,
    max_new_tokens=50
)

output = processor.batch_decode(
    generate_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]
print(output)

"""## AudioLDM Demo"""

from diffusers import AudioLDMPipeline
import torch
import scipy

repo_id = "cvssp/audioldm-s-full-v2"
pipe = AudioLDMPipeline.from_pretrained(
    repo_id, torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"

audio = pipe(
    prompt,
    num_inference_steps=10,
    audio_length_in_s=5.0
).audios[0]

scipy.io.wavfile.write(
    "techno.wav",
    rate=16000,
    data=audio
)