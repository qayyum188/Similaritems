import os
import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import pickle
import gdown  # For downloading files from Google Drive

# Google Drive file link for the pickle file containing model and features
pickle_file_url = "https://drive.google.com/uc?id=1i2IfrKdXUAs6zi-O62UlugYiXl2pjSDq"  # Replace with your file ID

# Path to save the pickle file locally
pickle_file_path = "features_and_model.pkl"

# Function to download the pickle file from Google Drive
def download_file(url, output_path):
    if not os.path.exists(output_path):  # Avoid downloading repeatedly
        st.write(f"Downloading {output_path}...")
        gdown.download(url, output_path, quiet=False)

# Download the pickle file containing the model and features
download_file(pickle_file_url, pickle_file_path)

# Function to extract features using ResNet50
def extract_features(img_path, model):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image {img_path}")
            return None
        
        img = cv2.resize(img, (224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        features = model.predict(img)
        return features.flatten()
    except Exception as e:
        print(f"Error extracting features from {img_path}: {e}")
        return None

# Load the model and features from pickle file
def load_from_pickle():
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, "rb") as f:
            data = pickle.load(f)
        model = data["model"]
        image_paths = data["image_paths"]
        feature_list = data["feature_list"]
        print("Model and features loaded from pickle file!")
        return model, image_paths, feature_list
    else:
        print("Pickle file not found. Please ensure the pickle file is accessible.")
        return None, None, None

# Function to calculate the Euclidean distance between the feature of the test image and all images
def find_similar_images(image_paths, feature_list, test_img_path, model, num_similar_images=4):
    test_img_features = extract_features(test_img_path, model)
    if test_img_features is None:
        print(f"Error extracting features for test image: {test_img_path}")
        return []
    
    test_img_features = np.array(test_img_features, dtype=np.float64).reshape(1, -1)
    distances = euclidean_distances(test_img_features, feature_list)
    closest_indices = np.argsort(distances[0])[:num_similar_images]

    return closest_indices

# Streamlit interface
st.title("THE VOGUE STORE")
st.markdown("<h3 style='text-align: left;'>Elegance is the only beauty that never fades</h3>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: left;'>Created by Abdul Qayyum</h5>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image to find similar ones:", type=["jpg", "jpeg", "png"])

model, image_paths, feature_list = load_from_pickle()

if model is None or image_paths is None or feature_list is None:
    st.write("Model and features are not loaded. Please ensure the pickle file is accessible.")
else:
    if uploaded_file is not None:
        with open("temp_uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        closest_indices = find_similar_images(image_paths, feature_list, "temp_uploaded_image.jpg", model, num_similar_images=4)

        if len(closest_indices) == 0:
            st.write("No similar images found. Please check the dataset or try again.")

        st.image("temp_uploaded_image.jpg", caption="Uploaded Image", use_container_width=True)
        st.write("Similar Images:")
        fig, ax = plt.subplots(1, 4, figsize=(15, 10))

        for i, idx in enumerate(closest_indices):
            img = cv2.imread(image_paths[idx])
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax[i].imshow(img)
            ax[i].axis('off')
            ax[i].set_title(f"Similar {i + 1}")

        st.pyplot(fig)
