import numpy as np
import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import requests
import gradio as gr
import os

# Load pretrained VGG16 model
VGG16 = models.vgg16(weights="IMAGENET1K_V1")
use_cuda = torch.cuda.is_available()
if use_cuda:
    VGG16 = VGG16.cuda()

# Global cache for labels
LABELS_CACHE = None

def prefetch_labels():
    global LABELS_CACHE
    LABELS_MAP_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        LABELS_CACHE = requests.get(LABELS_MAP_URL, timeout=5).json()
    except requests.exceptions.RequestException as e:
        LABELS_CACHE = None
        print(f"Error fetching labels: {e}")

# Fetch labels on startup
prefetch_labels()

def load_convert_image_to_tensor(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'), 'RGB')
    elif isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    in_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor()
    ]) 
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image

def get_human_readable_label_for_class_id(class_id, labels_cache=None):
    if labels_cache is None or class_id >= len(labels_cache):
        return f"Unknown class ID: {class_id}"
    return labels_cache[class_id]

def classify_image(image, confidence_threshold=0.0):
    global LABELS_CACHE
    if LABELS_CACHE is None:
        return "Error: Labels not loaded"

    try:
        image_tensor = load_convert_image_to_tensor(image)
        if use_cuda:
            image_tensor = image_tensor.cuda()

        output = VGG16(image_tensor)
        softmax_output = torch.nn.functional.softmax(output, dim=1)
        top_probs, top_classes = torch.topk(softmax_output, 3)
        top_probs = top_probs.cpu().detach().numpy() if use_cuda else top_probs.detach().numpy()
        top_classes = top_classes.cpu().detach().numpy() if use_cuda else top_classes.detach().numpy()

        result = {}
        for prob, cls_id in zip(top_probs[0], top_classes[0]):
            if prob >= confidence_threshold:
                label = get_human_readable_label_for_class_id(int(cls_id), LABELS_CACHE)
                result[label] = prob
        return result if result else "No predictions above the confidence threshold."
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
image_input = gr.Image()
confidence_slider = gr.Slider(minimum=0, maximum=1, default=0.0, label="Confidence Threshold (Optional)")
label_output = gr.Label(num_top_classes=3)

interface = gr.Interface(fn=classify_image, inputs=[image_input, confidence_slider], outputs=label_output)

# Launch Gradio with shareable link
interface.launch(share=True)
