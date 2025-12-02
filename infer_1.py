import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from feature_extractor import create_feature_extractor


# =========================
#   YOUR PATHS HERE
# =========================
MODEL_PATH = r"C:\Users\abd2a\Downloads\image-captioning\models\model.h5"
TOKENIZER_PATH = r"C:\Users\abd2a\Downloads\image-captioning\models\tokenizer.pkl"
IMG_SIZE = 224
MAX_LENGTH = 37  # Must match training max_length
# =========================


def generate_caption(image_path):
    """Generate and display caption for a single image."""

    # Load model
    caption_model = load_model(MODEL_PATH)

    # Load tokenizer
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    # Recreate the CNN extractor used during training
    feature_extractor = create_feature_extractor()

    # Preprocess image
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_arr = img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    # Extract features
    image_features = feature_extractor.predict(img_arr, verbose=0)

    # Generate caption
    in_text = "startseq"
    for _ in range(MAX_LENGTH):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)

        # Use dictionary format for model prediction (matches how model was trained)
        yhat = caption_model.predict({
            'image_input': image_features,
            'text_input': sequence
        }, verbose=0)
        yhat_index = np.argmax(yhat)

        word = tokenizer.index_word.get(yhat_index)
        if word is None:
            break

        if word == "endseq":
            break

        in_text += " " + word

    final_caption = in_text.replace("startseq", "").strip()

    # Display image + caption
    plt.figure(figsize=(8, 8))
    plt.imshow(load_img(image_path))
    plt.axis("off")
    plt.title(final_caption, fontsize=16, color="red")
    plt.show()

    return final_caption


# ===================
# Example usage:
# ===================

# Change this to your image
image_path = r"C:\Users\abd2a\Downloads\image-captioning\data\images\97406261_5eea044056.jpg"

generate_caption(image_path)