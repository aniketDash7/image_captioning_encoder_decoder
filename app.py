import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from model import CNNtoRNN
from get_loader import get_loader

# Load the model checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("my_checkpoint.pth.tar", map_location=device)

# Recreate the model
embed_size = 256
hidden_size = 256
num_layers = 1

# Load the dataset to get the vocabulary
_, dataset = get_loader(
    root_folder="archive/Images",
    annotation_file="archive/captions.txt",
    transform = transforms.Compose(
        [
            transforms.Resize((356,356)),
            transforms.RandomCrop((299,299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

        ]
    ),
    num_workers=1,
)

vocab_size = len(dataset.vocab)
model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Image transformation
transform = transforms.Compose(
    [
        transforms.Resize((356,356)),
        transforms.RandomCrop((299,299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

    ]
)

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    caption = model.caption_image(image, dataset.vocab)
    return " ".join(caption)

# Streamlit UI
st.title("Image Captioning with CNN-RNN")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    
    st.write("")
    st.write("Generating caption...")

    # Save the uploaded image
    image_path = f"temp_image.{uploaded_file.name.split('.')[-1]}"
    image.save(image_path)

    # Generate the caption
    caption = generate_caption(image_path)
    st.write("Caption: ", caption)
