import os
import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.models import load_model
import gdown
import h5py
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# Function to load the model and features directly from Google Drive
def load_model_from_drive():
    # Google Drive link for the model file
    h5_url = "https://drive.google.com/file/d/1MI9frsLSXKlYeQSp-zbLuxaSfawLzwZJ/view?usp=drive_link"  # Replace with your file ID
    output = "model.h5"

    # Download the model file from Google Drive if it's not already downloaded
    if not os.path.exists(output):  # Only download if not already downloaded
        print(f"Downloading model from Google Drive...")
        gdown.download(h5_url, output, quiet=False)
    else:
        print("Model file already downloaded.")

    # Load the model and features from the .h5 file
    with h5py.File(output, "r") as f:
        model = load_model(f['model'])  # Load the model
        feature_list = np.array(f['features'])  # Load features
        image_paths = list(f['image_paths'])  # Load image paths
    
    return model, image_paths, feature_list

# Function to extract features using ResNet50 (or any other model you're using)
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

# Function to find similar images based on Euclidean distance
def find_similar_images(image_paths, feature_list, test_img_path, model, num_similar_images=4):
    # Extract features for the test image
    test_img_features = extract_features(test_img_path, model)
    if test_img_features is None:
        print(f"Error extracting features for test image: {test_img_path}")
        return []
    
    # Ensure test image features are in the correct dtype for comparison
    test_img_features = np.array(test_img_features, dtype=np.float64).reshape(1, -1)

    # Compute Euclidean distances between the test image and all images in the dataset
    distances = euclidean_distances(test_img_features, feature_list)
    
    # Get the top 'num_similar_images' closest images
    closest_indices = np.argsort(distances[0])[:num_similar_images]

    return closest_indices

# Streamlit interface
st.title("THE VOGUE STORE")

# Add the new motto and "Created by Abdul Qayyum" text left-aligned under the title
st.markdown("<h3 style='text-align: left;'>Elegance is the only beauty that never fades</h3>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: left;'>Created by Abdul Qayyum</h5>", unsafe_allow_html=True)

# File uploader for the test image
uploaded_file = st.file_uploader("Upload an image to find similar ones:", type=["jpg", "jpeg", "png"])

# Load model and features from Google Drive
model, image_paths, feature_list = load_model_from_drive()

# Ensure that model is loaded
if model is None:
    st.write("Model is not loaded. Please train the model first.")
else:
    # When an image is uploaded
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
