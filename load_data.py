import librosa
import numpy as np
import os

DATASET_PATH = 'data'

def get_spectrogram(file_path, n_fft, hop_length):

    signal, sample_rate = librosa.load(file_path)
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    spectrogram = librosa.amplitude_to_db(np.abs(stft))

    return spectrogram



def get_mfcc(file_path, n_fft, hop_length, n_mfcc):
    
    signal, sample_rate = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, hop_length=hop_length, n_fft=n_fft, n_mfcc=n_mfcc)

    return mfcc
