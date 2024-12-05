import os
import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.models import load_model
import h5py
import gdown
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# Function to download the .h5 file from Google Drive
def download_h5_file():
    h5_url = "https://drive.google.com/uc?id=1MI9frsLSXKlYeQSp-zbLuxaSfawLzwZJ"  # Your Google Drive link
    output = "model.h5"
    if not os.path.exists(output):  # Only download if not already downloaded
        print("Downloading model from Google Drive...")
        gdown.download(h5_url, output, quiet=False)
    else:
        print("Model file already downloaded.")

# Function to load the model and features from the .h5 file
def load_model_and_features():
    download_h5_file()  # Ensure the model file is downloaded
    
    h5_path = "model.h5"
    if os.path.exists(h5_path):
        print(f"Loading model and features from {h5_path}")
        
        # Load the model
        model = load_model(h5_path)
        
        # Load the features (assuming they are stored as 'features' in the h5 file)
        with h5py.File(h5_path, 'r') as f:
            feature_list = np.array(f['features'])  # Features should be stored under the key 'features'
            image_paths = np.array(f['image_paths'])  # Image paths under the key 'image_paths'
        
        return model, image_paths, feature_list
    else:
        print(f"Model file {h5_path} not found.")
        return None, None, None

# Function to extract features from the test image
def extract_features(img_path, model):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image {img_path}")
            return None
        
        img = cv2.resize(img, (224, 224))  # Resize to model's input size
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        
        # If using a pre-trained model like ResNet50 or similar, apply preprocessing if needed
        features = model.predict(img)
        return features.flatten()
    except Exception as e:
        print(f"Error extracting features from {img_path}: {e}")
        return None

# Function to calculate the Euclidean distance between the test image and all images
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

# File uploader for the test image
uploaded_file = st.file_uploader("Upload an image to find similar ones:", type=["jpg", "jpeg", "png"])

# Load the model and features
model, image_paths, feature_list = load_model_and_features()

if model is None:
    st.write("Model is not loaded. Please train the model first.")
else:
    if uploaded_file is not None:
        # Save the uploaded image temporarily
        with open("temp_uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Find similar images
        closest_indices = find_similar_images(image_paths, feature_list, "temp_uploaded_image.jpg", model, num_similar_images=4)

        if len(closest_indices) == 0:
            st.write("No similar images found. Please check the dataset or try again.")

        # Display the uploaded image
        st.image("temp_uploaded_image.jpg", caption="Uploaded Image", use_container_width=True)

        # Display the similar images
        st.write("Similar Images: ")
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
