from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load once globally
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(image):
    if image is None:
        return "No image provided."
    inputs = processor(image.convert("RGB"), return_tensors="pt")
    output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)
