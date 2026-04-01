import cv2
import numpy as np
import faiss
import pickle
import os
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity


# Load the FaceNet model (used to convert face images into embeddings)
embedder = FaceNet()


# Load the FAISS index file (used for fast similarity search)
index = faiss.read_index("faiss_index.bin")


# Load the saved embeddings of all dataset images
embeddings = np.load("embeddings.npy")


# Load the image paths corresponding to each embedding
with open("image_paths.pkl", "rb") as f:
    image_paths = pickle.load(f)


# Folder where the dataset images are stored locally
DATASET_FOLDER = "lfw_funneled"


# Function to convert Kaggle paths to local paths
# Example: /kaggle/working/lfw_funneled/person/img.jpg
# becomes: lfw_funneled/person/img.jpg
def fix_path(path):
    filename = os.path.basename(path)  # extract image filename
    person = os.path.basename(os.path.dirname(path))  # extract person folder
    return os.path.join(DATASET_FOLDER, person, filename)


# Function to extract embedding from a query image
def extract_embedding(img):

    # Convert BGR to RGB (FaceNet expects RGB images)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize image to FaceNet input size
    img = cv2.resize(img, (160, 160))

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Generate embedding vector
    embedding = embedder.embeddings(img)

    # Return embedding
    return embedding[0]


# Function to search similar faces using FAISS
def search_similar(img, k=5, threshold=0.75):

    # Extract embedding for the query image
    query_embedding = extract_embedding(img)

    # Normalize the embedding vector
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Reshape for FAISS input
    query_embedding = query_embedding.reshape(1, -1).astype("float32")

    # Search the FAISS index for top k similar embeddings
    distances, indices = index.search(query_embedding, k)

    results = []

    # Loop through the returned indices
    for idx in indices[0]:

        # Get the embedding of the matched image
        candidate_embedding = embeddings[idx].reshape(1, -1)

        # Calculate cosine similarity between query and dataset image
        similarity = cosine_similarity(query_embedding, candidate_embedding)[0][0]

        # Apply similarity threshold
        if similarity >= threshold:

            # Fix path from Kaggle format to local dataset format
            img_path = fix_path(image_paths[idx])

            # Store result as (image_path, similarity_score)
            results.append((img_path, similarity))

    return results