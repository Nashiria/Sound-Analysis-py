import librosa
import librosa.display
import numpy as np
import speech_recognition as sr

r = sr.Recognizer()
mic = sr.Microphone()
import datetime
import os
import Database as db
import Compression as cp
import Recorder as rec
import CompareAlgorithms as ca

start = datetime.datetime.now()


def findmostaccurate(test, songlist, resample):
    tests = []
    for _ in range(len(songlist)):
        r = ca.comparealgorithhm(test, songlist[_], resample)
        r["song"] = songlist[_]
        tests.append(r)
    maxacc = {"accuracy": 0}
    for test in tests:
        if test["accuracy"] > maxacc["accuracy"]:
            maxacc = test
    return maxacc


def findmostaccuratetext(test):
    decompressedsongs = []
    songlist = os.listdir("AudioTexts")
    compedtest = np.loadtxt("AudioTexts/" + test + ".txt")
    songs = []
    for song in songlist:
        if ".DS_" not in song and test not in song:
            sstart = datetime.datetime.now()
            songdata = np.loadtxt("AudioTexts/" + song)
            output = ca.textcomparealgorithhmv2(compedtest, songdata)
            songs.append({"name": song.replace(".txt", ""), "accuracy": output["accuracy"], "start": output["start"]})
            # print("Compared in", datetime.datetime.now() - sstart, song)
    retmax = {"accuracy": 0}
    for match in songs:
        try:
            songacc = float(match["accuracy"])
            maxac = float(retmax["accuracy"])
            if songacc > maxac:
                retmax = match
        except:
            pass
    return retmax


def findmostaccuraterec(test):
    songlist = os.listdir("AudioTexts")
    songlist.sort()
    compedtest = np.transpose(cp.hashsupercompression(test, True))
    songs = []
    index = 0
    t = datetime.datetime.now()
    lasttime = 100000
    for song in songlist:
        perc = (100 * index / len(songlist))
        passedtime = datetime.datetime.now() - t
        try:
            perctime = (100 * (passedtime.total_seconds() / perc)) - passedtime.total_seconds()
            if perctime < lasttime:
                print("%" + str(int(perc)), "remaining time", str(int(perctime)), "seconds")
            lasttime = perctime
        except:
            pass
        if ".DS_" not in song and test not in song:
            songdata = np.loadtxt("AudioTexts/" + song)
            output = ca.textcomparealgorithhmv2(compedtest, songdata)
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
    retmax["found in"]=str(datetime.datetime.now()-t)
    return retmax


def runtests():
    testcountingstars = "counting_stars_1-45_1-50"
    testlikeastone = "like_a_stone_1-25_1-30-WAV"
    testbeliever = "believer_2-00_2-05"
    songlist = ["believer.wav", "counting_stars.wav", "like_a_stone.wav"]
    start1 = datetime.datetime.now()
    print(testcountingstars, findmostaccuratetext(testcountingstars, songlist), "found in",
          datetime.datetime.now() - start1)
    start2 = datetime.datetime.now()
    print(testlikeastone, findmostaccuratetext(testlikeastone, songlist), "found in", datetime.datetime.now() - start2)
    start3 = datetime.datetime.now()
    print(testbeliever, findmostaccuratetext(testbeliever, songlist), "found in", datetime.datetime.now() - start3)


def lookdata(filename, resample):
    y_file, fs = librosa.load(filename)
    if resample:
        targetsample = 2000
        y_file = librosa.resample(y_file, target_sr=targetsample, orig_sr=fs)
    D = librosa.stft(y_file)
    S_db = np.transpose(np.array(librosa.amplitude_to_db(np.abs(D), ref=np.max))).tolist()

    song = {"name": filename.replace(".wav", ""), "data": D, "shape": np.shape(np.array(D))}
    print(song["name"], fs, song["shape"])


def runrecordingtest():

    recording = rec.Record()
    rtime=datetime.datetime.now()
    result=findmostaccuraterec(recording)
    #runtime = datetime.datetime.now() - rtime
    #print("Proccess completed in", str(runtime.seconds + (round(100 * runtime.microseconds / 1000000) / 100)), "seconds")
    print('Currently playing "'+result["name"]+" current")


#db.newdatabase()
