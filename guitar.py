import wave
import numpy as np
import librosa
import time
import pyaudio

import sys
import matplotlib.pyplot as plt
import Compression as cp
import Plots as pl
import datetime
def actualhertz(hzlist):
    hzlist.sort()
    actuallist = hzlist[int(len(hzlist)/20):-int(len(hzlist)/20)]
    return sum(actuallist) / len(actuallist)
def findnote(data,chunk,fs):
    freqlist = []
    swidth=1
    window = np.blackman(chunk)
    while len(data) == chunk * swidth:
        # write data out to the audio stream
        # stream.write(data)
        # unpack the data and times by the hamming window
        indata = np.array(wave.struct.unpack("%dh" % (len(data) / swidth), \
                                             data)) * window
        # Take the fft and square each value
        fftData = abs(np.fft.rfft(indata)) ** 2
        # find the maximum
        which = fftData[1:].argmax() + 1
        # use quadratic interpolation around the max
        if which != len(fftData) - 1:
            y0, y1, y2 = np.log(fftData[which - 1:which + 2:])
            x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
            # find the frequency and output it
            thefreq = (which + x1) * fs / chunk
            freqlist.append(thefreq)
        else:
            thefreq = which * fs / chunk
            freqlist.append(thefreq)
    hz = actualhertz(freqlist)
    note = librosa.hz_to_note(hz)
    print(note)
def findnotestream(stream):
    chunk = 512
    data = stream.read(chunk)
    fs=48000
    freqlist = []
    swidth = 2
    window = np.blackman(chunk)
    p = pyaudio.PyAudio()
    stream2 = p.open(format=pyaudio.paInt16, channels=1, rate=fs, output=True,frames_per_buffer=chunk)
    while len(data) == chunk * swidth:
        # write data out to the audio stream
        start=datetime.datetime.now()
        stream2.write(data)
        # unpack the data and times by the hamming window
        indata = np.array(wave.struct.unpack("%dh" % (len(data) / swidth), data)) * window
        # Take the fft and square each value
        fftData = abs(np.fft.rfft(indata)) ** 2
        # find the maximum
        which = fftData[1:].argmax() + 1
        # use quadratic interpolation around the max
        if which != len(fftData) - 1:
            y0, y1, y2 = np.log(fftData[which - 1:which + 2:])
            x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
            # find the frequency and output it
            thefreq = (which + x1) * fs / chunk
            freqlist.append(thefreq)
        else:
            thefreq = which * fs / chunk
            freqlist.append(thefreq)
        try:
            note = librosa.hz_to_note(thefreq)
            print(note,"delay",str((datetime.datetime.now()-start).microseconds/100000))
        except:
            pass
        data = stream.read(chunk,exception_on_overflow=False)


    stream.close()
    p.terminate()
    return [note,hz]

def findnote2(filename):
    chunk = 2048
    # open up a wave
    wf = wave.open(filename, 'rb')
    swidth = wf.getsampwidth()
    RATE = wf.getframerate()
    # use a Blackman window
    window = np.blackman(chunk)
    # open stream
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=RATE,
                        output=True)

    # read some data
    data = wf.readframes(chunk)

    # play stream and find the frequency of each chunk
    freqlist=[]
    while len(data) == chunk * swidth:
        # write data out to the audio stream
        #stream.write(data)
        # unpack the data and times by the hamming window
        indata = np.array(wave.struct.unpack("%dh" % (len(data) / swidth), \
                                             data)) * window
        # Take the fft and square each value
        fftData = abs(np.fft.rfft(indata)) ** 2
        # find the maximum
        which = fftData[1:].argmax() + 1
        # use quadratic interpolation around the max
        if which != len(fftData) - 1:
            y0, y1, y2 = np.log(fftData[which - 1:which + 2:])
            x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
            # find the frequency and output it
            thefreq = (which + x1) * RATE / chunk
            freqlist.append(thefreq)
        else:
            thefreq = which * RATE / chunk
            freqlist.append(thefreq)
        # read some more data
        data = wf.readframes(chunk)
    hz=actualhertz(freqlist)
    note=librosa.hz_to_note(hz)
    if data:
        stream.write(data)
    stream.close()
    p.terminate()
    return [note,hz]
def soundplot(stream):
    t1=time.time()
    fs = 44100

    CHUNK = 2048
    data = np.frombuffer(stream.read(CHUNK,exception_on_overflow = False),dtype=np.int16)
    findnote(data,CHUNK,fs)
def callback(in_data, frame_count, time_info, flag):
        return in_data, pyaudio.paContinue
def liveproccess():
    p = pyaudio.PyAudio()
    CHANNELS = 1
    RATE = 48000
    stream = p.open(format=pyaudio.paFloat32,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,
                    input=True,
                    stream_callback=callback)

    stream.start_stream()
    while stream.is_active():
        time.sleep(20)
        stream.stop_stream()
        print("Stream is stopped")

    stream.close()

    p.terminate()
def liveplot():
    RATE = 48000
    CHUNK = int(RATE / 20)  # RATE / number of updates per second
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=RATE,
                    output=True,
                    input=True,
                    stream_callback=callback)
    for i in range(sys.maxsize ** 10):

        soundplot(stream)
    stream.stop_stream()
    stream.close()
    p.terminate()

def livefindnote():
    RATE = 48000
    CHUNK = int(RATE / 20)  # RATE / number of updates per second
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    for i in range(sys.maxsize ** 10):
        findnotestream(stream)
    stream.stop_stream()
    stream.close()
    p.terminate()
#print(findnotestream("GuitarNotes/e4.wav"))
livefindnote()