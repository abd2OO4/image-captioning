import pandas as pd
import os
import sys

# Handle imports for when this is run directly or as a module
try:
    from .preprocessing import clean_caption
except ImportError:
    from preprocessing import clean_caption

def load_dataset(caption_path):
    """Load dataset from captions file and clean captions.
    
    Handles the Flickr8k format where each line is:
    image_name#caption_id<TAB>caption_text
    
    Args:
        caption_path: Path to captions file (tab-separated, no header)
        
    Returns:
        DataFrame with 'image' and 'caption' columns
    """
    if not os.path.exists(caption_path):
        raise FileNotFoundError(f"Caption file not found: {caption_path}")
    
    # Read the file format: image_id<TAB>caption
    data = []
    with open(caption_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split by tab
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"Warning: Skipping malformed line: {line}")
                continue
            
            image_id = parts[0].split('#')[0]  # Extract just the image name
            caption = parts[1]
            
            data.append({
                'image': image_id,
                'caption': caption
            })
    
    if not data:
        raise ValueError(f"No captions found in {caption_path}")
    
    df = pd.DataFrame(data)
    
    # Clean captions
    df['caption'] = df['caption'].apply(clean_caption)
    
    print(f"Loaded {len(df)} captions for {df['image'].nunique()} unique images")
    return df
