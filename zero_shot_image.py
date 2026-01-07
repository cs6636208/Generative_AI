from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

# Load the model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of two cats", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
output = model(**inputs)
logits_per_image = output.logits_per_image
probs = logits_per_image.softmax(dim=1)
print(probs)

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

######## CLIP's visual embedding ###########
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
img_inputs = processor(images=image, return_tensors="pt", padding=True)
image_features = model.get_image_features(**img_inputs)
print(f"Image shape : {image_features.shape}")

##########  CLIP's text embedding ###########
query_text = "a photo of cat"
text_inputs = processor(text=[query_text], return_tensors="pt", padding=True)
text_features = model.get_text_features(**text_inputs)
print(f"Text Shape: {text_features.shape}")