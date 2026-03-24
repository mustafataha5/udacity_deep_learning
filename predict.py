import os

# 1. SILENCE WARNINGS: Must be done before importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

# Global configuration
IMAGE_SIZE = 224

def process_image(image: np.ndarray) -> np.ndarray:
    """Prepares a raw image for the model: resize and normalize."""
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image /= 255.0
    return image.numpy()

def predict(image_path: str, model: tf.keras.Model, top_k: int):
    """Predicts top K classes. Returns probabilities and class labels."""
    try:
        with Image.open(image_path) as img:
            image = np.asarray(img.convert("RGB"))
    except Exception as e:
        print(f"Error: Could not open image file. {e}")
        sys.exit(1)

    processed_image = process_image(image)
    # Add batch dimension
    processed_image = np.expand_dims(processed_image, axis=0)

    # Run prediction
    predictions = model.predict(processed_image, verbose=0)[0]

    # Extract top K
    top_k_indices = np.argsort(predictions)[-top_k:][::-1]
    top_k_probs = predictions[top_k_indices]
    top_k_classes = [str(idx) for idx in top_k_indices]

    return top_k_probs.tolist(), top_k_classes

def load_category_names(json_path: str) -> dict:
    """Loads class-to-name mapping from JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def format_output(probs, classes, category_names=None):
    """Maps indices to names and prints a formatted table."""
    print(f"\n{'FLOWER SPECIES':<25} | {'PROBABILITY':<10}")
    print("-" * 40)
    
    for prob, cls in zip(probs, classes):
        # Handle 0-based vs 1-based indexing logic
        name = cls
        if category_names:
            name = category_names.get(cls) or category_names.get(str(int(cls) + 1)) or f"Class {cls}"
        
        print(f"{name:<25} | {prob:.4%}")
    print()

def main():
    parser = argparse.ArgumentParser(description="Professional Flower Image Classifier")
    parser.add_argument("image_path", help="Path to input image (e.g., image.jpg)")
    parser.add_argument("model_path", help="Path to trained .h5 model")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results")
    parser.add_argument("--category_names", help="JSON file for name mapping")

    args = parser.parse_args()

    # Verify paths exist
    if not Path(args.model_path).is_file():
        print(f"Error: Model file '{args.model_path}' not found.")
        sys.exit(1)

    # Load Model safely with Custom Objects
    try:
        # custom_object_scope handles the Keras 2/3 Hub layer compatibility
        with tf.keras.utils.custom_object_scope({'KerasLayer': hub.KerasLayer}):
            model = tf.keras.models.load_model(args.model_path)
    except Exception as e:
        print(f"Error: Failed to load model. Ensure it is a valid .h5 file. \n{e}")
        sys.exit(1)

    # Run Prediction
    probs, classes = predict(args.image_path, model, args.top_k)

    # Load Names
    category_names = None
    if args.category_names:
        category_names = load_category_names(args.category_names)

    # Display Results
    format_output(probs, classes, category_names)

if __name__ == "__main__":
    main()