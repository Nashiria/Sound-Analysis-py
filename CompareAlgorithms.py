import numpy as np
import speech_recognition as sr

r = sr.Recognizer()
mic = sr.Microphone()
import datetime
import Compression as cp


def comparealgorithhm(filename1, filename2, resample):
    import os
    unsupported = ["aiff", "mp3", "m4a"]
    name1 = filename1.split(".")[0]
    name2 = filename2.split(".")[0]
    if filename1.split(".")[1] in unsupported:
        os.system("ffmpeg -i " + filename1 + " " + name1 + ".wav")
    if filename2.split(".")[1] in unsupported:
        os.system("ffmpeg -i " + filename2 + " " + name2 + ".wav")
    filename1 = name1 + ".wav"
    filename2 = name2 + ".wav"
    y1, fsof1, length1 = cp.editedspectogramdata(filename1, resample)
    y2, fsof2, length2 = cp.editedspectogramdata(filename2, resample)
    a = 0
    linecount = 0
    acclist = []
    maxacc = {"accuracy": 0}
    index = 0
    y2index = 0
    lastperc = 0
    print(len(y1), len(y2))
    for line in y2:
        # if round(100*y2index/(len(y2)-len(y1)))>lastperc:
        #    lastperc=round(100*y2index/(len(y2)-len(y1)))
        #    print("%"+str(lastperc))
        if len(y2) - y2.index(line) > len(y1):
            lineaccuracy = 0
            linecount += 1
            placementacc = 0
            indexofstart = y2.index(line)
            index = indexofstart
            for line in y1:
                c1 = line
                c2 = y2[indexofstart]
                correct = 0
                wrong = 0
                for index in range(len(c1)):
                    if c1[index] == c2[index]:
                        correct += 1
                    else:
                        wrong += 1
                lineaccuracy += correct / (correct + wrong)
                indexofstart += 1
            lineaccuracy = lineaccuracy / len(y1)
            if maxacc["accuracy"] < 100 * lineaccuracy:
                maxacc = {"start": y2index, "accuracy": 100 * lineaccuracy}
            acclist.append({"start": y2index, "accuracy": 100 * lineaccuracy})
        else:
            break
        y2index += 1
    start = length2 * maxacc["start"] / (len(y2))
    time = datetime.timedelta(seconds=start)

    return {"accuracy": maxacc["accuracy"], "start": str(time)}


def textcomparealgorithm(y1, y2):
    linecount = 0
    maxacc = {"accuracy": 0}
    for y2index in range(len(y2)):
        if len(y2) - y2index > len(y1):
            lineaccuracy = 0
            linecount += 1
            indexofstart = y2index
            for line in y1:
                c1 = line
                c2 = y2[indexofstart]
                correct = 0
                for index in range(len(c1)):
                    if c1[index] == c2[index]:
                        correct += 1
                    elif c1[index] == c2[index] + 1 or c1[index] == c2[index] - 1:
                        correct += 0.5
                lineaccuracy += correct / (len(line))
                indexofstart += 1
            lineaccuracy = lineaccuracy / len(y1)
            if maxacc["accuracy"] < 100 * lineaccuracy:
                maxacc = {"start": y2index, "accuracy": 100 * lineaccuracy}
        else:
            break
    percetangeofstart = maxacc["start"] / len(y2)
    start = len(y2) * 0.128 * percetangeofstart * (4000 / 2000)
    time = datetime.timedelta(seconds=start)
    return {"accuracy": maxacc["accuracy"], "start": str(time)}


def textcomparealgorithhmv2(y1, y2):
    y2index = 0
    maxacc = {"accuracy": 0, "start": 0}
    y1 = np.transpose(y1)
    for line in y2:
        if len(y2) - y2index + 1 > len(y1):
            A = np.array(y2)[y2index:y2index + len(y1)].flatten()
            B = np.array((y1)).flatten()
            lineaccuracy = np.sum(A==B) / len(A)
            if maxacc["accuracy"] < 100 * lineaccuracy:
                maxacc = {"start": y2index, "accuracy": (100 * lineaccuracy)}
        else:
            break
        y2index += 1
    percetangeofstart = maxacc["start"] / len(y2)
    start = len(y2) * 0.256 * percetangeofstart/16
    time = datetime.timedelta(seconds=start)
    return {"accuracy": maxacc["accuracy"], "start": str(time)}
