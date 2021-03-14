import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import speech_recognition as sr

r = sr.Recognizer()
mic = sr.Microphone()


def readwav(filename):
    y_file, fs = librosa.load(filename)
    targetsample = 2000
    y_file = librosa.resample(y_file, target_sr=targetsample, orig_sr=fs)
    D = librosa.stft(y_file)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S_db


def editedspectrogram(y_file, filename):
    y_file, fs = y_file
    D = librosa.stft(y_file)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    edited = []
    for line in S_db:
        temp = []
        for data in line:
            if data > -40:
                a = 1
            else:
                a = 0
            temp.append(a)
        edited.append(temp)
    edited = np.array(edited)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(edited, cmap="gray", x_axis='time', y_axis='linear', ax=ax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.savefig('Plots/Spectrogram_Edited/' + filename.replace(".wav", "") + "_spectogram_edited.png",
                bbox_inches='tight', pad_inches=0)
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.set(title='Edited Spectrogram')
    plt.show()


def compression(y_file):
    D = librosa.stft(np.array(y_file))
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    edited = []
    for line in S_db:
        temp = []
        for data in line:
            if data > -40:
                a = 1
            else:
                a = 0
            temp.append(a)
        edited.append(temp)
    compressed = []
    columnlength = 0
    rowlength = len(edited)
    size = 0
    returnlist = [rowlength]
    for line in edited:
        if returnlist == [rowlength]:
            columnlength = len(line)
            returnlist.append(columnlength)
        if 1 in line:
            linelist = [edited.index(line)]
            ii = np.where(np.array(line) == 1)[0].tolist()
            again = len(ii) > 1
            nums = sorted(set(ii))
            gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
            edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
            temp = list(zip(edges, edges))
            for data in temp:
                num1 = int(data[0])
                num2 = int(data[1])
                linelist.append([num1, 1 + num2 - num1])
            if linelist not in returnlist:
                returnlist.append(linelist)
    return returnlist


def decompression(compressed_y_file):
    decompressed = []
    column = compressed_y_file[1]
    row = compressed_y_file[0]
    compressed_y_file.remove(column)
    compressed_y_file.remove(row)
    for entry in range(row):
        temp = []
        if len(compressed_y_file) > 0:
            if compressed_y_file[0][0] == entry:
                temp += ([0] * column)
                compressed_y_file[0].remove(compressed_y_file[0][0])
                for oneloc in compressed_y_file[0]:
                    for _ in range(int(oneloc[1])):
                        temp[oneloc[0] + _ - 1] = 1
                compressed_y_file.remove(compressed_y_file[0])
            else:
                temp += ([0] * column)
        else:
            temp += ([0] * column)
        decompressed.append(temp)
    return decompressed


def decompressiontext(compressed_y_file):
    decompressed = []
    column = compressed_y_file[1]
    row = len(compressed_y_file)
    compressed_y_file.remove(column)
    compressed_y_file.remove(row)
    for entry in range(row):
        temp = []
        if len(compressed_y_file) > 0:
            if compressed_y_file[0][0] == entry:
                temp += ([0] * column)
                compressed_y_file[0].remove(compressed_y_file[0][0])
                index = 0
                for _ in range(int(len(compressed_y_file[0]) / 2)):
                    oneloc = [compressed_y_file[0][0], compressed_y_file[0][1]]
                    compressed_y_file[0].remove(oneloc[0])
                    compressed_y_file[0].remove(oneloc[1])
                    for _ in range(int(oneloc[1])):
                        temp[oneloc[0] + _ - 1] = 1
                compressed_y_file.remove(compressed_y_file[0])
            else:
                temp += ([0] * column)
        else:
            temp += ([0] * column)
        decompressed.append(temp)
    return decompressed


def fourier(y_file):
    D = np.abs(librosa.stft(y_file))
    fig, ax = plt.subplots()
    db = librosa.display.waveplot(D)
    ax.set(title='Fourier Transform', )
    plt.show()


def supercompression(filename, mono, resample):
    y_file, fs = librosa.load(filename, mono=mono)
    if resample:
        targetsample = 2000
        y_file = librosa.resample(y_file, target_sr=targetsample, orig_sr=fs)
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
            data += 80
            if data == 0:
                category = 0
            elif data > 0 and data < average * 0.5:
                category = 1
            elif data > average * 0.5 and data < average:
                category = 2
            elif data > average and data < average * 1.5:
                category = 3
            elif data > average * 1.5:
                category = 4
            temp.append(category)
        edited.append(temp)
    return edited


def hashsupercompression(filename, mono):
    y_file, fs = librosa.load(filename, mono=mono,sr=4000)
    D = librosa.stft(y_file,n_fft=256)
    S_db = np.transpose(np.array(librosa.amplitude_to_db(np.abs(D), ref=np.max))).tolist()
    edited = []
    average = 0
    count = 0
    for line in S_db:
        for data in line:
            data += 80
            if data > 0:
                average += (data)
                count += 1
    average = average / count
    targets=5
    sample=targets
    for line in S_db:
        if sample==targets:
            temp = []
            index=0
            for data in line:
                category = round(((data + 80) * (80 / (2 * average))) / 10)
                temp.append(category)
                index += 1
            edited.append(temp)
            sample=0
        sample+=1
    return edited


def supercompressioncategorized(filename):
    y_file, fs = librosa.load(filename)
    targetsample = 2000
    y_file = librosa.resample(y_file, target_sr=targetsample, orig_sr=fs)
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
            data += 80
            if data < average * 0.5:
                category = 0
            elif data > average * 0.5 and data < average:
                category = 1
            elif data > average and data < average * 1.5:
                category = 2
            elif data > average * 1.5:
                category = 3
            temp.append(category)
        edited.append(temp)
    return edited


def compressedspectrogram(y_file, filename):
    edited = np.array(decompression(y_file))
    fig, ax = plt.subplots()
    img = librosa.display.specshow(edited, cmap="gray", x_axis='time', y_axis='linear', ax=ax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.savefig('Plots/Spectrogram_Edited/' + filename.replace(".wav", "") + "_spectogram_compressed.png",
                bbox_inches='tight', pad_inches=0)
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.set(title='Compressed Spectrogram')
    plt.show()


def supercompressedspectogram(y_file, filename):
    edited = np.array(y_file)
    edited = np.transpose(edited)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(edited, cmap="gray", x_axis='time', y_axis='linear', ax=ax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.savefig('Plots/Spectrogram_Edited/' + filename.replace(".wav", "") + "_spectogram_supercompressed.png",
                bbox_inches='tight', pad_inches=0)
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.set(title='Supercompressed Spectrogram')
    plt.show()


def writecompresseddata(data, filename):
    with open("Plots/Text_Data/" + filename.replace(".wav", "") + '_compressed.txt', 'w') as outfile:
        for line in compression(data, filename):
            line = str(line)
            line = line.replace("[", "").replace("]", "").replace(", ", " ")
            outfile.write(str(line + "\n"))


def writesupercompresseddata(file, path):
    with open(path, 'w') as outfile:
        for line in file:
            line = str(line)
            line = line.replace("[", "").replace("]", "").replace(", ", " ")
            outfile.write(str(line + "\n"))


def writedecompresseddata(data, filename):
    with open("Plots/Text_Data/" + filename.replace(".wav", "") + '_decompressed.txt', 'w') as outfile:
        for line in decompression(data):
            line = str(line)
            line = line.replace("[", "").replace("]", "").replace(", ", " ")
            if line != "":
                outfile.write(str(line + "\n"))


def writerawdata(data, filename):
    D = librosa.stft(data)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    edited = []
    for line in S_db:
        temp = []
        for data in line:
            if data > -40:
                a = 1
            else:
                a = 0
            temp.append(a)
        edited.append(temp)
    data = edited
    with open("Plots/Text_Data/" + filename.replace(".wav", "") + '_raw.txt', 'w') as outfile:
        for line in data:
            line = str(line)
            line = line.replace("[", "").replace("]", "").replace(", ", " ")
            if line != "":
                outfile.write(str(line + "\n"))


def editedspectogramdata(filename, resample):
    y, fs = librosa.load(filename)
    length = int(100 * len(y) / fs) / 100
    if resample:
        targetsample = 4000
        y = librosa.resample(y, target_sr=targetsample, orig_sr=fs)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    edited = []
    for line in S_db:
        temp = []
        for data in line:
            if data > -40:
                a = 1
            else:
                a = 0
            temp.append(a)
        edited.append(temp)
    edited = np.transpose(np.array(edited)).tolist()
    return (edited, fs, length)


def readdecompressed(filename):
    filepath = "Plots/Text_Data/" + filename
    f = open(filepath, "r")
    ret = []
    lineindex = 0
    for line in f:
        if " " in line:
            line = (str(line).replace("\n", "")).split(" ")
            for _ in range(len(line)):
                line[_] = int(line[_])
            ret.append(line)
        else:
            ret.append(int(line))
    return decompressiontext(ret)


def readraw(filename):
    f = open(filename, "r")
    ret = []
    lineindex = 0
    for line in f:
        if " " in line:
            line = (str(line).replace("\n", "")).split(" ")
            for _ in range(len(line)):
                line[_] = int(line[_])
            ret.append(line)
        else:
            ret.append(int(line))
    n = 0
    returnlist = []
    total = 1025
    desired = 1025
    for line in ret:
        newline = []
        for _ in range(desired):
            indexs = range(int(total / desired))
            av = 0
            for index in indexs:
                av += line[_ * (index + 1)]
            av = av / 5
            av = round(av)
            newline.append(av)
        returnlist.append(newline)
    return returnlist
