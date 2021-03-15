import numpy as np
import matplotlib.pyplot as plt
import librosa,os
import librosa.display
import datetime
import speech_recognition as sr
r = sr.Recognizer()
mic = sr.Microphone()
import datetime
import pyaudio
import wave
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import Compression as cp
import Recorder as rec
from scipy.spatial import distance
def waveplot1(y_file,fs):
    fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True)
    librosa.display.waveplot(y_file, sr=fs, ax=ax)
    ax.set(title='Waveplot1')
    ax.label_outer()
    plt.show()
def waveplot2(y_file,fs):
    y_harm, y_perc = librosa.effects.hpss(y_file)
    fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True)
    librosa.display.waveplot(y_harm, sr=fs, color='r', alpha=0.25, ax=ax)
    librosa.display.waveplot(y_perc, sr=fs, color='b', alpha=0.5, ax=ax)
    ax.set(title='Waveplot2')
    plt.show()
def bspectrogram(y_file,fs):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(y_file, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title='Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()
def spectrogram(y_file,fs):
    D = librosa.stft(y_file)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax,sr=fs)
    ax.set(title='Now with labeled axes!')
    fig.colorbar(img, ax=ax, format="%+2.f dB")