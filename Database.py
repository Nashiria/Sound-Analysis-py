import datetime
import os

import numpy as np

import Compression as cp


def updatedatabase():
    audios = os.listdir("AudioLib")
    audios.sort()
    audiotexts = os.listdir("AudioTexts")
    for audio in audios:
        filename = audio
        songstar = datetime.datetime.now()
        audio = audio.replace("-WAV", "")
        audio = audio.split(" ")
        audio.pop(0)
        audio = " ".join(audio)
        if audio.replace(".wav", ".txt") not in audiotexts and "DS_" not in audio and audio != "":
            data = cp.hashsupercompression("AudioLib/" + filename, True)
            np.savetxt("AudioTexts/" + audio.replace(".wav", ".txt"), data, fmt='%d')
            print(audio, "done in", datetime.datetime.now() - songstar)


def speechupdatedatabase():
    audios = os.listdir("SpeechLib")
    audios.sort()
    audiotexts = os.listdir("SpeechTexts")
    for audio in audios:
        filename = audio
        songstar = datetime.datetime.now()
        if audio.replace(".wav", ".txt") not in audiotexts and "DS_" not in audio and audio != "":
            data = cp.supercompression("SpeechLib/" + filename, False, False)
            np.savetxt("SpeechTexts/" + audio.replace(".wav", ".txt"), data, fmt='%d')
            print(audio, "done in", datetime.datetime.now() - songstar)

def deletedatabase():
    audiotexts = os.listdir("AudioTexts")
    for file in audiotexts:
        os.remove("AudioTexts/"+file)

def createdatabasefile():
    decompressedsongs = []
    songlist = os.listdir("AudioTexts")
    songs = []
    t = datetime.datetime.now()
    lasttime = 100000
    index = 0
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
        if ".DS_" not in song:
            songdata = np.loadtxt("AudioTexts/" + song, dtype=int)
            songs.append([song.replace(".txt","").replace(" ","_"),songdata.tolist()])
        index+=1
    np.savetxt("data",songs, fmt='%s')
def loaddatabase():
    songlist = os.listdir("AudioTexts")
    print("Loading database.")
    songs = []
    t = datetime.datetime.now()
    lasttime = 100000
    index = 0
    for song in songlist:
        #perc = (100 * index / len(songlist))
        #passedtime = datetime.datetime.now() - t
        ####try:
        ###    perctime = (100 * (passedtime.total_seconds() / perc)) - passedtime.total_seconds()
        ##    if perctime < lasttime:
        #        print("%" + str(int(perc)), "remaining time", str(int(perctime)), "seconds")
         #   lasttime = perctime
        #except:
        #    pass
        if ".DS_" not in song:
            songdata = np.loadtxt("AudioTexts/" + song, dtype=int)
            songs.append([song.replace(".txt", "").replace(" ", "_"), songdata.tolist()])
        index += 1
    print("Database loaded in",datetime.datetime.now()-t)
    return songs
def newdatabase():
    deletedatabase()
    updatedatabase()