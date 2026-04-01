import cv2
import numpy as np
from PIL import Image
from model import search_similar

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
    type=["jpg","png","jpeg"],
    accept_multiple_files=True
)

if uploaded_files:

    for file in uploaded_files:

        st.subheader("Query Image")

        image = Image.open(file)
        st.image(image, width=200)

        img_np = np.array(image)

        results = search_similar(img_np, k=5, threshold=threshold)

        if len(results) == 0:

            st.warning("No similar faces found above threshold")

        else:

            st.write("Similar Faces")

            cols = st.columns(len(results))

            for i,(path,score) in enumerate(results):

                img = Image.open(path)

                with cols[i]:
                    st.image(img)
                    st.write(f"Score: {score:.3f}")