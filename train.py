"""Training script for image captioning model."""

import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from dataset import load_dataset
from feature_extractor import extract_features
from model import build_model
from generator import CaptionGenerator

# -------------------------------
# YOUR WINDOWS PATHS
# -------------------------------
CAPTION_PATH = r"C:\Users\abd2a\Downloads\image-captioning\data\captions.txt"
IMAGE_FOLDER = r"C:\Users\abd2a\Downloads\image-captioning\data\images"
OUTPUT_PATH = r"C:\Users\abd2a\Downloads\image-captioning\models"
# -------------------------------


def train_model(
    caption_path=CAPTION_PATH,
    image_folder=IMAGE_FOLDER,
    output_path=OUTPUT_PATH,
    epochs=50,
    batch_size=64,
):
    """Train the image captioning model."""

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    df = load_dataset(caption_path)
    print(f"Loaded {len(df)} captions")

    # Extract features
    print("Extracting image features...")
    features = extract_features(
        image_folder,
        df,
        output_path=os.path.join(output_path, "features.pkl"),
    )
    print(f"Extracted features for {len(features)} images")

    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df["caption"].values)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")

    # Max caption length
    max_length = max(len(c.split()) for c in df["caption"].values)
    print(f"Max caption length: {max_length}")

    # Data generator
    print("Creating data generator...")
    generator = CaptionGenerator(df, tokenizer, features, max_length, vocab_size, batch_size)

    # Build model
    print("Building model...")
    model = build_model(vocab_size, max_length)
    model.summary()

    # Train
    print("Training model...")
    history = model.fit(generator, epochs=epochs, verbose=1)

    # Save model
    model_path = os.path.join(output_path, "model.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save tokenizer
    tokenizer_path = os.path.join(output_path, "tokenizer.pkl")
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {tokenizer_path}")

    return model, tokenizer, history


if __name__ == "__main__":
    train_model()
