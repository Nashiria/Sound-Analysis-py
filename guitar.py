import Recorder as rc
import Plots as pl
import Compression as cp
import librosa
import datetime
def actualhertz(hzlist):
    hzlist.sort()
    actuallist = hzlist[int(len(hzlist)/20):-int(len(hzlist)/20)]
    return sum(actuallist) / len(actuallist)
def findnote(filename):
    import pyaudio
    import wave
    import numpy as np
    import librosa
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
start=datetime.datetime.now()
print(findnote("GuitarNotes/e4.wav"))
print("Found in",str(datetime.datetime.now()-start))