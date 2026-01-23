import streamlit as st
import torch
import pickle
from PIL import Image
import torchvision.transforms as transforms

from model import TransformerEncoderViT
from model import TransformerDecoder
from vocabulary_class import Vocabulary
from inference import generate_caption_beam

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Image Caption Generator", layout="centered")

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    encoder = TransformerEncoderViT(256).to(device)
    decoder = TransformerDecoder(embed_size=256, vocab_size=len(vocab)).to(device)

    encoder.load_state_dict(
        torch.load("models/encoder.pth", map_location=device)
    )
    decoder.load_state_dict(
        torch.load("models/decoder.pth", map_location=device)
    )

    encoder.eval()
    decoder.eval()
    return encoder, decoder

@st.cache_resource
def load_vocab():
    return torch.load("models/vocab.pkl", weights_only=False)

vocab = load_vocab()
encoder, decoder = load_models()

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

# -----------------------------
# UI
# -----------------------------
st.title("🖼️ Image Caption Generator")
st.write("ViT Encoder + Transformer Decoder")

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            img_tensor = transform(image)
            caption = generate_caption_beam(
                img_tensor,
                encoder,
                decoder,
                vocab,
                device=device
            )

        st.success("Caption generated!")
        st.markdown(f"### 📝 **{caption}**")
