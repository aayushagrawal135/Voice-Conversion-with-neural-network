'''
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

sample_rate, samples = wavfile.read('A.wav')
frequencies, times, spectogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, spectogram)
plt.imshow(spectogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
'''
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np

(rate,sig) = wavfile.read("BDL_natural_a0058.wav")

'''
mfcc_feat = mfcc(sig,rate)
d_mfcc_feat = delta(mfcc_feat, 2)
fbank_feat = logfbank(sig,rate)

#print(fbank_feat[1:3,:])
'''
