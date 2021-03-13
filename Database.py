import numpy as np
import matplotlib.pyplot as plt
import librosa,os
import librosa.display
import datetime
import pyaudio
import wave
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import Recorder as rec
from scipy.spatial import distance
import CompareAlgorithms as ca
def updatedatabase():
    audios=os.listdir("AudioLib")
    audios.sort()
    audiotexts=os.listdir("AudioTexts")
    for audio in audios:
        filename=audio
        songstar=datetime.datetime.now()
        audio = audio.replace("-WAV", "")
        audio = audio.split(" ")
        audio.pop(0)
        audio = " ".join(audio)
        if audio.replace(".wav",".txt") not in audiotexts and "DS_" not in audio and audio != "":
            data=ca.hashsupercompression("AudioLib/"+filename,True)
            np.savetxt("AudioTexts/"+audio.replace(".wav",".txt"),data, fmt='%d')
            print(audio,"done in",datetime.datetime.now()-songstar)
def speechupdatedatabase():
    audios=os.listdir("SpeechLib")
    audios.sort()
    audiotexts=os.listdir("SpeechTexts")
    for audio in audios:
        filename=audio
        songstar=datetime.datetime.now()
        if audio.replace(".wav",".txt") not in audiotexts and "DS_" not in audio and audio != "":
            data=ca.supercompression("SpeechLib/"+filename,False,False)
            np.savetxt("SpeechTexts/"+audio.replace(".wav",".txt"),data, fmt='%d')
            print(audio,"done in",datetime.datetime.now()-songstar)
