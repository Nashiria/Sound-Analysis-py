import datetime
import os
import wave

import librosa
import librosa.display
import numpy as np
import pyaudio
from pydub import AudioSegment
from pydub.silence import split_on_silence


def speechcompare(y1, y2):
    y2index = 0
    maxacc = {"accuracy": 0, "start": 0}
    y1 = np.transpose(y1)
    for line in y2:
        if len(y2) - y2index + 1 > len(y1):
            a = np.array(y2)[y2index:y2index + len(y1)].flatten()
            B = np.array(y1).flatten()
            # trues=(a-B)
            # lineaccuracy=np.count_nonzero(trues==0)/len(trues)
            lineaccuracy = np.sum(a == B) / len(a)
            if maxacc["accuracy"] < 100 * lineaccuracy:
                maxacc = {"start": y2index, "accuracy": (100 * lineaccuracy)}
        else:
            break
        y2index += 1
    percetangeofstart = maxacc["start"] / len(y2)
    start = len(y2) * 0.256 * percetangeofstart
    time = datetime.timedelta(seconds=start)
    return {"accuracy": maxacc["accuracy"], "start": str(time)}


def speechfindmostaccuraterec():
    songlist = os.listdir("SpeechTexts")
    songlist.sort()
    songs = []
    index = 0
    t = datetime.datetime.now()
    filename = speechrecord()
    words = []
    for audio_chunk in load_chunks(filename):
        audio_chunk.export("temp.wav", format="wav")

        for song in songlist:
            perc = (100 * index / len(songlist))
            passedtime = datetime.datetime.now() - t
            try:
                perctime = (100 * (passedtime.total_seconds() / perc)) - passedtime.total_seconds()
                # if perctime<lasttime:
                # print("%"+str(int(perc)),"remaining time",str(int(perctime)),"seconds")

            except:
                pass
            if ".DS_" not in song:
                songdata = np.loadtxt("SpeechTexts/" + song)
                samplecount = len(songdata)
                compedtest = np.transpose(speechsupercompression())
                output = speechcompare(compedtest, songdata)
                entry = {"name": song.replace(".txt", ""), "accuracy": output["accuracy"], "start": output["start"]}
                songs.append(entry)
                # print(entry)
            index += 1
        retmax = {"accuracy": 0}
        for match in songs:
            try:
                songacc = float(match["accuracy"])
                maxac = float(retmax["accuracy"])
                if songacc > maxac:
                    retmax = match
            except:
                pass
        words.append(retmax)
    return words


def load_chunks(filename):
    long_audio = AudioSegment.from_mp3(filename)
    audio_chunks = split_on_silence(
        long_audio, min_silence_len=250,
        silence_thresh=-50
    )
    return audio_chunks


def speechrecord():
    library = os.listdir("Speeches")
    x = datetime.datetime.now()
    timestamp = "-" + str(x.day) + "-" + str(x.month) + "-" + str(x.year)
    recordid = 0
    for data in library:
        if (timestamp in data):
            recordid += 1
    timestamp += "-" + str(recordid)
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 48000  # Record at 44100 samples per second
    seconds = 5
    filename = "Speeches/Speech" + timestamp + ".wav"
    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    print('Recording')
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)
    frames = []  # Initialize array to store frames
    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()
    print('Finished recording')
    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    return filename


def speechsupercompression():
    y_file, fs = librosa.load("temp.wav", mono=True)
    D = librosa.stft(y_file)
    S_db = np.transpose(np.array(librosa.amplitude_to_db(np.abs(D), ref=np.max))).tolist()
    edited = []
    average = 0
    count = 0
    for line in S_db:
        temp = []
        for data in line:
            data += 80
            if data > 0:
                average += (data)
                count += 1
    average = average / count
    for line in S_db:
        temp = []
        for data in line:
            category = round((data + 80) / 10)
            temp.append(category)
        edited.append(temp)
    return edited


def runspeechtest():
    guess = speechfindmostaccuraterec()
    sent = []
    for word in guess:
        sent.append(word["name"])
    print(" ".join(sent))
