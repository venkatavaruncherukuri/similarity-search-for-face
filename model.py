import cv2
import numpy as np
import faiss
import os

# Example dataset folder
DATASET_PATH = "dataset"   # make sure this folder exists

image_paths = []
features = []

def extract_features(img):
    img = cv2.resize(img, (100, 100))
    return img.flatten().astype('float32')

# Load dataset once
for file in os.listdir(DATASET_PATH):
    path = os.path.join(DATASET_PATH, file)
    img = cv2.imread(path)
    if img is not None:
        image_paths.append(path)
        features.append(extract_features(img))

features = np.array(features)

# Build FAISS index
index = faiss.IndexFlatL2(features.shape[1])
index.add(features)

def search_similar(query_img, k=5, threshold=0.5):
    q = extract_features(query_img).reshape(1, -1)
    distances, indices = index.search(q, k)

    results = []
    for i, d in zip(indices[0], distances[0]):
        score = 1 / (1 + d)
        if score >= threshold:
            results.append((image_paths[i], score))

    return results