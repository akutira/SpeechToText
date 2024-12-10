# SpeechToText
This repository contains the implementation of a Speech-to-Text model, training on audio samples stored in .wav format and corresponding transcriptions from a metadata file. The project also includes evaluation on test datasets to ensure model accuracy and robustness.

Project Overview

The goal of this project is to preprocess and train a neural network model on audio data to predict transcriptions accurately. The dataset includes .wav files paired with their textual descriptions provided in CSV metadata files. The model is tested on a separate set of .wav files and their corresponding transcriptions.

Features

Data Preprocessing:
Handles variable-length .wav files.
Pads or trims audio to a uniform length for model consistency.
Model Training:
Trains a neural network on audio-transcription pairs.
Leverages state-of-the-art preprocessing techniques.
Testing and Evaluation:
Evaluates model performance on unseen test data.
Reports key metrics like accuracy and transcription error rate.
Dataset Structure

Training Data
Audio Files: Located in the wavs/ directory.
Metadata: A metadata.csv file contains the mapping:
id: Unique identifier for each audio file (matches filename).
transcription: Text corresponding to the audio.
Test Data
Audio Files: Located in the testwavsfold/ directory.
Metadata: A testmetadataf.csv file with the same structure as the training metadata.
