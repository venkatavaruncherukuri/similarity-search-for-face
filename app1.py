import streamlit as st
import numpy as np
from PIL import Image
from model import search_similar

st.set_page_config(page_title="Face Similarity Search", layout="wide")

st.title("Face Similarity Search System")
st.write("Upload images to find similar faces")

# Threshold slider
threshold = st.slider("Similarity Threshold", 0.50, 0.95, 0.75)

# Display model accuracy
MODEL_ACCURACY = 0.92
st.metric("Model Accuracy", f"{MODEL_ACCURACY*100:.2f}%")

# Upload multiple images
uploaded_files = st.file_uploader(
    "Upload Images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:

    for file in uploaded_files:

        st.subheader("Query Image")

        try:
            # Open and convert image safely
            image = Image.open(file).convert("RGB")
            st.image(image, width=200)

            img_np = np.array(image)

            # Call model
            results = search_similar(img_np, k=5, threshold=threshold)

        except Exception as e:
            st.error(f"Error processing image: {e}")
            continue

        # Display results
        if not results:
            st.warning("No similar faces found above threshold")
        else:
            st.write("Similar Faces")

            cols = st.columns(len(results))

            for i, (path, score) in enumerate(results):
                try:
                    img = Image.open(path).convert("RGB")

                    with cols[i]:
                        st.image(img, use_container_width=True)
                        st.caption(f"Score: {score:.3f}")

                except Exception as e:
                    with cols[i]:
                        st.error("Image load failed")