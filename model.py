from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, concatenate, Reshape, Dropout, add, RepeatVector, TimeDistributed

def build_model(vocab_size, max_length, image_feature_dim=1920, embedding_dim=256, lstm_units=256):
    """Build image captioning model.
    
    Architecture:
    - Image features (1920D) -> Dense(256) -> RepeatVector(max_length) -> shape (max_length, 256)
    - Text (max_length) -> Embedding(256) -> shape (max_length, 256)
    - Concatenate -> LSTM -> Dense output
    
    Args:
        vocab_size: Size of vocabulary
        max_length: Maximum caption length
        image_feature_dim: Dimension of image features (default 1920 from DenseNet201)
        embedding_dim: Dimension of word embeddings (default 256)
        lstm_units: Number of LSTM units (default 256)
        
    Returns:
        Compiled Keras model
    """
    # Image input branch
    image_input = Input(shape=(image_feature_dim,), name='image_input')
    x1 = Dense(embedding_dim, activation="relu")(image_input)
    # Repeat image features to match sequence length
    x1 = RepeatVector(max_length)(x1)  # Shape: (max_length, embedding_dim)

    # Text input branch
    text_input = Input(shape=(max_length,), name='text_input')
    x2 = Embedding(vocab_size, embedding_dim, name='embedding')(text_input)  # Shape: (max_length, embedding_dim)

    # Merge branches - now they have matching shapes
    merged = concatenate([x1, x2], axis=-1)  # Shape: (max_length, 2*embedding_dim)
    
    # LSTM to process the sequence
    lstm_out = LSTM(lstm_units, return_sequences=False, name='lstm')(merged)
    x = Dropout(0.5)(lstm_out)
    
    x = Dense(128, activation="relu", name='dense_1')(x)
    output = Dense(vocab_size, activation="softmax", name='output')(x)

    # Use dictionary input names to match generator output
    model = Model(inputs={'image_input': image_input, 'text_input': text_input}, outputs=output)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model
