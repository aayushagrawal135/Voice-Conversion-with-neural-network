from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import pandas as pd

#librosa
def readWav(filename):
    rate, signal = wav.read(filename)
    return rate, signal

def totalTime(rate, signal):
    return (len(signal)/rate)*1000;

# mfcc(signal, samplerate, winlen, winstep, numcep, nfilt, nfft)

def mfcc_mat(rate, sampledWav):
    mfcc_matrix = []
    for i in range(len(sampledWav)):
        mfcc_feat = mfcc(sampledWav[i], rate)
        mfcc_matrix.append(mfcc_feat)
    mfcc_matrix = np.asarray(mfcc_matrix)
    return mfcc_matrix


def frameDivision(timeWindow, stride, rate, signal):
    sampleWindow = (len(signal)*timeWindow)/totalTime(rate, signal)
    stride = int(sampleWindow*stride)
    sampledWav = []
    count = 0
    while count+sampleWindow<len(signal):
        temp = signal[count:int(count+sampleWindow)]
        sampledWav.append(temp)
        count = count+stride

    sampledWav = np.asarray(sampledWav)
    return sampledWav

rate, signal = readWav("BDL_natural_a0058.wav")
#sampledWav = frameDivision(25, 0.3, rate, signal)
#nMfccCoeff = 13
#mfcc_matrix = mfcc_mat(rate, sampledWav)

#frames, samplePerFrame = np.shape(sampledWav)

#mfcc_matrix = mfcc_matrix.reshape(frames, nMfccCoeff)
feat = mfcc(signal)
delta_feat = delta(feat, 2)
print(np.shape(delta_feat))
print(np.shape(feat))
