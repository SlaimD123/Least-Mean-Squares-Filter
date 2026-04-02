# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 14:43:05 2025

@author: XSY
"""


import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
import matplotlib.pyplot as plt

sample_rate, audio_data = wavfile.read(r"C:\Users\XSY\Documents\2025\ls_mono.wav")
x = np.array(audio_data)

sample_rate, audio_data = wavfile.read(r"C:\Users\XSY\Documents\2025\mic_mono.wav")
d = np.array(audio_data)

l = len(d)
filter_l = 128
a_temp = np.zeros(filter_l - 1, dtype = np.float)
X = np.concatenate((a_temp, x))
mu = 1e-2
x_temp = np.zeros(filter_l, dtype = np.float)
h = np.zeros(filter_l, dtype = np.float)
e = np.zeros(l)
for i in range(l):
    x_temp = np.flip(X[i:i+filter_l])
    d_estimate = np.dot(x_temp,np.conj(h))
    e[i] = d[i] - d_estimate
    h = h + (mu * x_temp * np.conj(e[i]))/(0.001 + np.dot(x_temp, np.conj(x_temp)))

plt.plot(d)
plt.plot(e)
plt.show()