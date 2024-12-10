import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import StratifiedShuffleSplit

class SpeechRecognitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x, lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_input)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        logits = self.fc(out)
        
        return logits

class AudioDataset(Dataset):
    def __init__(self, dataset_path, max_samples=None):
        self.metadata_path = os.path.join(dataset_path, 'metadata.csv')
        self.wavs_path = os.path.join(dataset_path, 'wavs')
        
        # Load metadata
        self.metadata = pd.read_csv(self.metadata_path, 
                                    sep='|', 
                                    header=None, 
                                    names=['id', 'transcription', 'normalized_transcription'])
        
        # Limit samples if specified
        if max_samples:
            self.metadata = self.metadata.head(max_samples)
        
        # Validate wav files
        self.metadata = self.metadata[
            self.metadata['id'].apply(
                lambda x: os.path.exists(os.path.join(self.wavs_path, f"{x}.wav"))
            )
        ]
        
        # Prepare vocabulary
        self.char_to_idx = self.build_vocab()
    
    def build_vocab(self):
        all_chars = set(''.join(self.metadata['transcription']).lower())
        
        char_to_idx = {
            '<pad>': 0,
            '<sos>': 1,
            '<eos>': 2
        }
        
        for char in sorted(all_chars):
            if char not in char_to_idx:
                char_to_idx[char] = len(char_to_idx)
        
        return char_to_idx
    
    def extract_features(self, audio_file):
        y, sr = librosa.load(audio_file, sr=22050)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return torch.FloatTensor(mfccs.T)
    
    def encode_transcription(self, transcription):
        encoded = [self.char_to_idx['<sos>']]
        encoded.extend([self.char_to_idx.get(char.lower(), 0) for char in transcription])
        encoded.append(self.char_to_idx['<eos>'])
        
        return torch.LongTensor(encoded)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        wav_file = os.path.join(self.wavs_path, f"{row['id']}.wav")
        
        try:
            features = self.extract_features(wav_file)
            transcription = self.encode_transcription(row['transcription'])
            
            return features, transcription
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            return torch.zeros(13, 10), torch.LongTensor([1])
    
    def __len__(self):
        return len(self.metadata)

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    
    features, transcriptions = zip(*batch)
    
    features_padded = pad_sequence(features, batch_first=True)
    features_lengths = torch.LongTensor([len(f) for f in features])
    
    transcriptions_padded = pad_sequence(transcriptions, batch_first=True, padding_value=0)
    
    return features_padded, transcriptions_padded, features_lengths

def train_model(dataset_path, model_save_path=None):
    # Set default model save path if not provided
    if model_save_path is None:
        model_save_path = os.path.join(os.getcwd(), 'speech_recognition_model.pt')
    
    # Create dataset
    full_dataset = AudioDataset(dataset_path)
    
    print(f"Total dataset size: {len(full_dataset)}")
    
    # Use StratifiedShuffleSplit for better sampling
    labels = full_dataset.metadata['transcription'].str.len()
    train_indices, val_indices = next(iter(
        StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
            np.zeros(len(labels)), labels
        )
    ))
    
    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    # Create data loaders
    train_loader = DataLoader(full_dataset, batch_size=32, sampler=train_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(full_dataset, batch_size=32, sampler=val_sampler, collate_fn=collate_fn)
    
    # Initialize model
    model = SpeechRecognitionModel(
        input_dim=13,  # MFCC features
        hidden_dim=128, 
        output_dim=len(full_dataset.char_to_idx)
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    for epoch in range(10):
        model.train()
        train_loss = 0
        
        for batch_idx, (features, transcriptions, lengths) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(features, lengths)
            batch_size, max_seq_len, vocab_size = logits.shape
            
            # Reshape
            logits_reshaped = logits.view(-1, vocab_size)
            transcriptions_reshaped = transcriptions.view(-1)
            
            # Mask
            mask = (transcriptions_reshaped != 0)
            masked_logits = logits_reshaped[mask]
            masked_transcriptions = transcriptions_reshaped[mask]
            
            # Loss and backprop
            loss = criterion(masked_logits, masked_transcriptions)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {train_loss/len(train_loader)}")
    
    # Save model using PyTorch's save method
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': full_dataset.char_to_idx
    }, model_save_path)
    
    print(f"Model saved to {model_save_path}")
    
    return model, full_dataset.char_to_idx

def main():
    # Specify your dataset path
    dataset_path = 'wavs'
    
    try:
        # Train model
        train_model(dataset_path)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()