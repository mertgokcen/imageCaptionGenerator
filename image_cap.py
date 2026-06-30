import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def load_image_in_rgb(img_path):
    return Image.open(img_path).convert('RGB')

image_paths = [f"./images/img{i}.jpg" for i in range(1,4)]
images = [load_image_in_rgb(img_path) for img_path in image_paths]



text = "the image of"
all_inputs = [
    processor(images=image, text=text, return_tensors="pt")
    for image in images
]


all_outputs = [
    model.generate(**inputs, max_length=50)
    for inputs in all_inputs
]

captions = [
    processor.decode(outputs[0], skip_special_tokens=True)
    for outputs in all_outputs
]

for idx, caption in enumerate(captions, start=1):
    print(f"img{idx}.jpg:", caption)