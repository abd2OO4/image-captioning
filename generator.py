import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

class CaptionGenerator(Sequence):
    """Data generator for image captioning model.
    
    Generates batches of (image_features, caption_sequences) -> target_words
    """
    
    def __init__(self, df, tokenizer, features, max_length, vocab_size, batch_size=64):
        """Initialize the data generator.
        
        Args:
            df: DataFrame with 'image' and 'caption' columns
            tokenizer: Keras tokenizer fitted on captions
            features: Dictionary mapping image names to feature vectors
            max_length: Maximum caption length
            vocab_size: Size of vocabulary
            batch_size: Batch size for training
        """
        super().__init__()
        self.df = df.copy()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.features = features
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.n = len(self.df)
        self.missing_count = 0
        
        # Get feature dimension from first available feature
        if self.features:
            first_feature = next(iter(self.features.values()))
            # Flatten if needed
            if len(first_feature.shape) > 1:
                self.feature_dim = first_feature.flatten().shape[0]
            else:
                self.feature_dim = first_feature.shape[0]
        else:
            self.feature_dim = 1920

    def __len__(self):
        return max(1, self.n // self.batch_size)

    def __getitem__(self, idx):
        batch = self.df.iloc[idx * self.batch_size : (idx + 1) * self.batch_size]
        X1, X2, y = [], [], []

        for _, row in batch.iterrows():
            # Validate required columns
            if 'image' not in row or 'caption' not in row:
                continue
                
            img_name = row['image']
            
            # Check if features exist for this image
            if img_name not in self.features:
                self.missing_count += 1
                continue
                
            try:
                feature = self.features[img_name]
                
                # Flatten feature if it's multi-dimensional
                if len(feature.shape) > 1:
                    if feature.shape[0] == 1:
                        feature = feature.flatten()  # (1, 1920) -> (1920,)
                    else:
                        feature = feature[0] if len(feature.shape) > 2 else feature.flatten()
                
                # Ensure feature is 1D
                if len(feature.shape) != 1:
                    feature = feature.flatten()
                
                # Tokenize caption
                seq = self.tokenizer.texts_to_sequences([row['caption']])[0]
                
                if len(seq) < 2:
                    continue

                # Generate training pairs
                for i in range(1, len(seq)):
                    in_seq = pad_sequences([seq[:i]], maxlen=self.max_length)[0]
                    out_seq = to_categorical([seq[i]], num_classes=self.vocab_size)[0]

                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue

        if len(X1) == 0:
            # Return empty batch of correct shapes
            X1 = np.zeros((1, self.feature_dim), dtype=np.float32)
            X2 = np.zeros((1, self.max_length), dtype=np.int32)
            y = np.zeros((1, self.vocab_size), dtype=np.float32)
        else:
            X1 = np.array(X1, dtype=np.float32)
            X2 = np.array(X2, dtype=np.int32)
            y = np.array(y, dtype=np.float32)
            
        # Return as dictionary for multiple inputs
        return {'image_input': X1, 'text_input': X2}, y
