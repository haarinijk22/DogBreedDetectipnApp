import numpy as np
import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import requests
import gradio as gr

VGG16 = models.vgg16(pretrained=True)
use_cuda = torch.cuda.is_available()
if use_cuda:
    VGG16 = VGG16.cuda()

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

def get_human_readable_label_for_class_id(class_id):
    LABELS_MAP_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    labels = requests.get(LABELS_MAP_URL).json()
    return labels[class_id]

def classify_image(image):
    try:
        image_tensor = load_convert_image_to_tensor(image)
        if use_cuda:
            image_tensor = image_tensor.cuda()

        output = VGG16(image_tensor)
        _, preds_tensor = torch.max(output, 1)
        pred = np.squeeze(preds_tensor.cpu().numpy()) if use_cuda else np.squeeze(preds_tensor.numpy())
        class_description = get_human_readable_label_for_class_id(int(pred))
        return class_description
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
image_input = gr.Image()
label_output = gr.Label(num_top_classes=1)
interface = gr.Interface(fn=classify_image, inputs=image_input, outputs=label_output)
interface.launch()
