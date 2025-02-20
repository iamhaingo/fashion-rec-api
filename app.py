# app.py
import streamlit as st
import numpy as np
from PIL import Image
import random
from sklearn.metrics.pairwise import cosine_similarity


# Load embeddings and image paths
@st.cache_data
def load_embeddings():
    data = np.load("clothing_embeddings.npz", allow_pickle=True)
    return data["embeddings"], data["image_paths"]


embeddings, image_paths = load_embeddings()

# Streamlit app
st.title("fashion recommender system")
# st.markdown("Find similar fashion items using FashionCLIP embeddings!")

# Button to randomize query
if st.button("🎲 Random Query"):
    random_idx = random.randint(0, len(image_paths) - 1)
    query_image_path = image_paths[random_idx]
    query_embedding = embeddings[random_idx].reshape(1, -1)

    # Compute similarities
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    similar_indices = np.argsort(similarities)[::-1][1:6]  # Exclude self

    # Display query image
    st.subheader("Query Image")
    st.image(Image.open(query_image_path), width=200)

    # Display similar images in columns
    st.subheader("Top 5 Similar Items")
    cols = st.columns(5)
    for i, idx in enumerate(similar_indices):
        with cols[i]:
            st.image(
                Image.open(image_paths[idx]),
                caption=f"Similarity: {similarities[idx]:.2f}",
                use_container_width=True,
            )
