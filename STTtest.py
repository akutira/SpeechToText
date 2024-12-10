import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D, LSTM, BatchNormalization, Bidirectional
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def prepare_dataset(wavs_dir, metadata_path, target_length=16000):
    """
    Prepare dataset using .wav files and metadata.
    """
    metadata = pd.read_csv(metadata_path, sep='|', header=None, 
                           names=['id', 'transcription', 'normalized_transcription'])
    grouped = metadata  # Skip filtering for debugging
    all_wave = []
    all_transcriptions = []

    success_count = 0
    fail_count = 0

    for _, row in grouped.iterrows():
        wav_path = os.path.join(wavs_dir, row['id'] + '.wav')
        if not os.path.exists(wav_path):
            print(f"Missing file: {wav_path}")
            fail_count += 1
            continue
        try:
            samples_audio, _ = librosa.load(wav_path, sr=1600)
            if len(samples_audio) > target_length:
                samples_audio = samples_audio[:target_length]
            else:
                samples_audio = np.pad(samples_audio, (0, target_length - len(samples_audio)))
            all_wave.append(samples_audio)
            all_transcriptions.append(row['transcription'].lower())
            success_count += 1
        except Exception as e:
            print(f"Error processing file {wav_path}: {e}")
            fail_count += 1
    
    print(f"Successfully loaded files: {success_count}")
    print(f"Failed files: {fail_count}")

    if len(all_wave) < 2:
        raise ValueError("Not enough samples to create a meaningful dataset. Check file paths or formats.")
    
    return np.array(all_wave), all_transcriptions



def augment_audio(audio, sr):
    # Add noise
    noise = np.random.normal(0, 0.005, len(audio))
    audio = audio + noise

    # Pitch shifting
    audio = librosa.effects.pitch_shift(audio, sr, n_steps=np.random.randint(-5, 5))

    # Speed adjustment
    speed_factor = np.random.uniform(0.9, 1.1)
    audio = librosa.effects.time_stretch(audio, speed_factor)

    return audio

def create_advanced_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Convolutional layers with more filters
    x = Conv1D(64, 13, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.4)(x)

    x = Conv1D(128, 11, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.4)(x)

    # Use Bidirectional LSTMs
    x = LSTM(128, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(128))(x)

    # Dense layers with L2 regularization
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)

    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model


# def create_speech_model(input_shape, num_classes):
#     """
#     Create a 1D Convolutional Neural Network for speech recognition
    
#     Args:
#     input_shape (tuple): Shape of input data
#     num_classes (int): Number of output classes
    
#     Returns:
#     tf.keras.Model: Compiled speech recognition model
#     """
#     inputs = Input(shape=input_shape)

#     # Convolutional layers
#     conv = Conv1D(8, 13, padding='valid', activation='relu', strides=1)(inputs)
#     conv = MaxPooling1D(3)(conv)
#     conv = Dropout(0.3)(conv)

#     conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
#     conv = MaxPooling1D(3)(conv)
#     conv = Dropout(0.3)(conv)

#     conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
#     conv = MaxPooling1D(3)(conv)
#     conv = Dropout(0.3)(conv)

#     conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
#     conv = MaxPooling1D(3)(conv)
#     conv = Dropout(0.3)(conv)

#     # Flatten and dense layers
#     conv = Flatten()(conv)
#     conv = Dense(256, activation='relu')(conv)
#     conv = Dropout(0.3)(conv)
#     conv = Dense(128, activation='relu')(conv)
#     conv = Dropout(0.3)(conv)

#     outputs = Dense(num_classes, activation='softmax')(conv)

#     model = Model(inputs, outputs)
    
#     # Compile the model
#     model.compile(
#         loss='categorical_crossentropy',
#         optimizer='adam',
#         metrics=['accuracy']
#     )
    
#     return model

def prepare_audio(audio_path, target_length=16000):
    """
    Preprocess the audio file to match the required input size for the model
    
    Args:
    audio_path (str): Path to the audio file
    
    Returns:
    np.array: Preprocessed audio array
    """
    try:
        # Load the audio file
        audio, _ = librosa.load(audio_path, sr=1600)
        
        # Ensure the audio has a fixed length
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)))
        
        return audio
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def predict_audio(model, label_encoder, audio_path):
    """
    Predict the transcription for a given audio file
    
    Args:
    model (tf.keras.Model): The trained model
    label_encoder (LabelEncoder): The label encoder used for encoding transcriptions
    audio_path (str): Path to the audio file
    
    Returns:
    str: Predicted transcription
    """
    # Preprocess the audio
    audio = prepare_audio(audio_path)
    
    if audio is None:
        return None

    # Reshape audio to match the input shape (batch_size, 8000, 1)
    audio = audio.reshape(1, 16000, 1)
    
    # Predict transcription
    prob = model.predict(audio)
    index = np.argmax(prob[0])  # Get index of the highest probability
    transcription = label_encoder.inverse_transform([index])[0]  # Decode the index back to text
    
    return transcription

def main():
    # Set paths
    wavs_dir = 'wavs'  # Replace with your actual path
    metadata_path = 'metadata.csv'  # Replace with your actual path

    # augmented_audio = augment_audio(audio, sr)
    # Prepare dataset
    try:
        all_wave, all_transcriptions = prepare_dataset(wavs_dir, metadata_path)
        print(all_wave.shape)
        print(len(all_transcriptions))
    except ValueError as e:
        print(f"Dataset preparation error: {e}")
        return None

    # Print dataset information
    unique_transcriptions = set(all_transcriptions)
    print(f"Total samples: {len(all_wave)}")
    print(f"Unique transcriptions: {len(unique_transcriptions)}")
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(all_transcriptions)
    
    # One-hot encode labels
    y_cat = to_categorical(y)
    
    # Reshape audio data
    X = all_wave.reshape(-1, 16000, 1)

    # Check class distribution before splitting
    class_counts = np.bincount(y)
    print("Class distribution:", class_counts)
    
    # Attempt to split data
    try:
        x_tr, x_val, y_tr, y_val = train_test_split(
            X, y_cat, 
            test_size=0.2, 
            stratify=y, 
            random_state=42
        )
    except ValueError as e:
        print(f"Train-test split error: {e}")
        print("Trying without stratification...")
        x_tr, x_val, y_tr, y_val = train_test_split(
            X, y_cat, 
            test_size=0.2, 
            random_state=42
        )

    # Create model
    model = create_advanced_model(
        input_shape=(16000, 1), 
        num_classes=y_cat.shape[1]
    )

    # Callbacks
    es = EarlyStopping(
    monitor='val_loss', 
    mode='min', 
    verbose=1, 
    patience=15, 
    min_delta=0.00001
)

    mc = ModelCheckpoint(
        'best_model.hdf5.keras', 
        monitor='val_accuracy', 
        verbose=1, 
        save_best_only=True, 
        mode='max'
    )

    # Train model
    history = model.fit(
        x_tr, y_tr,
        epochs=100, 
        callbacks=[es, mc], 
        batch_size=32, 
        validation_data=(x_val, y_val)
    )

    # Path to the test audio file
    test_audio_path = 'testwavsfold/LJ033-0060.wav'  # Replace with your actual file path

    # Predict transcription for the test audio
    predicted_transcription = predict_audio(model, le, test_audio_path)
    
    if predicted_transcription:
        print(f"Predicted Transcription for Recordtester.wav: {predicted_transcription}")
    else:
        print("Error in predicting transcription.")

if __name__ == '__main__':
    main()
