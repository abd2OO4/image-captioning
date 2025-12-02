import numpy as np
import os
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tqdm import tqdm

def create_feature_extractor():
    """Create a DenseNet201 feature extractor model.
    
    Returns:
        Keras Model that outputs features (1920 dimensions)
    """
    model = DenseNet201(weights="imagenet")
    # Remove the last layer to get feature vector
    extractor = Model(model.input, model.layers[-2].output)
    return extractor

def extract_features(image_folder, df, output_path=None):
    """Extract features from images using DenseNet201.
    
    Args:
        image_folder: Path to folder containing images
        df: DataFrame with 'image' column containing image filenames
        output_path: Optional path to save features as pickle file
        
    Returns:
        Dictionary mapping image names to feature vectors (1920 dims)
        
    Raises:
        FileNotFoundError: If image_folder doesn't exist
        ValueError: If 'image' column not in dataframe
    """
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")
    
    if 'image' not in df.columns:
        raise ValueError("DataFrame must have 'image' column")
    
    extractor = create_feature_extractor()
    features = {}
    
    missing_images = []
    for img_name in tqdm(df['image'].unique(), desc="Extracting features"):
        path = os.path.join(image_folder, img_name)
        
        # Check if image exists
        if not os.path.exists(path):
            missing_images.append(img_name)
            continue
            
        try:
            img = load_img(path, target_size=(224, 224))
            img = img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)
            feature = extractor.predict(img, verbose=0)
            features[img_name] = feature
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            
    if missing_images:
        print(f"Warning: {len(missing_images)} images not found")
    
    if output_path:
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(features, f)
        print(f"Features saved to {output_path}")
    
    return features