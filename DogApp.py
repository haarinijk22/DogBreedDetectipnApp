import numpy as np
import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import requests
import gradio as gr

# Load pretrained VGG16 model
VGG16 = models.vgg16(pretrained=True)
use_cuda = torch.cuda.is_available()
if use_cuda:
    VGG16 = VGG16.cuda()

def load_convert_image_to_tensor(image):
    # Check if the input is a NumPy array (Gradio may pass it as NumPy)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'), 'RGB')
    elif isinstance(image, str):
        image = Image.open(image).convert('RGB')  # If it's a path, load the image
    
    in_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),  # Correct input size for VGG16
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
        # If there's any error, return the error message for debugging
        return f"Error: {str(e)}"

# Adding the Disclaimer
disclaimer_text = "Disclaimer: Predictions are based on VGG16 model accuracy and may not be 100% accurate."

# Gradio Interface
def create_interface():
    # Gradio Interface
    image_input = gr.Image(label="Upload an Image")
    label_output = gr.Label(num_top_classes=1)
    disclaimer_output = gr.Markdown(disclaimer_text)

    # Interface Layout
    interface = gr.Blocks()  # Using Blocks for custom layout

    with interface:
        gr.Markdown("# Dog Breed Identification")
        gr.Markdown("This tool uses a VGG16 model to predict the dog breed based on the uploaded image.")
        with gr.Row():
            with gr.Column():
                image_input.render()  # Input on the left
            with gr.Column():
                label_output.render()  # Output on the right
        disclaimer_output.render()  # Disclaimer at the bottom

    return interface

# Launch the interface
create_interface().launch()
