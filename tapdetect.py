#! /usr/bin/env python

import numpy as np
from scipy.fft import rfft, irfft
from scipy.io import wavfile
from scipy.signal import stft, correlate, cheby2, sosfilt
from scipy.optimize import least_squares, fmin
from math import cos, sin, pi, sqrt
from itertools import product
from pandas import read_csv
from wx import Rect
import matplotlib.pyplot as plt
from acoustics.cepstrum import real_cepstrum
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# https://raw.githubusercontent.com/scivision/piradar/master/piradar/delayseq.py
from delayseq import delayseq

def ir(x):
    return int(round(x))

def abs_corr(a, b):
    return np.abs(correlate(a, b))

def hpass(Y1, Y2):
    n = len(Y1)
    return np.array([ 1 if i>2500.0/fs*n else 0 for i in range(0, n) ])

def hpass2000(Y1, Y2):
    n = len(Y1)
    return np.array([ 1 if i>2000.0/fs*n else 0 for i in range(0, n) ])

def hpass3500(Y1, Y2):
    n = len(Y1)
    return np.array([ 1 if i>3500.0/fs*n else 0 for i in range(0, n) ])

def hpass3500(Y1, Y2):
    n = len(Y1)
    return np.array([ 1 if i>4500.0/fs*n else 0 for i in range(0, n) ])

def sig_coherence(Y1, Y2):
    nominator = Y1 * np.conj(Y2)
    denominator = np.conj(Y1) * Y2
    denominator[denominator == 0] = 1e-16
    return nominator / denominator


def eckart_hpass(Y1, Y2):
    n = len(Y1)
    G_y1y2_a = np.abs(Y1 * np.conj(Y2))
    denominator = (Y1 * np.conj(Y1) - G_y1y2_a) * (Y2 * np.conj(Y2) - G_y1y2_a)
    denominator[denominator == 0] = 1e-16
    Ψ = G_y1y2_a / denominator
    return np.array([ Ψ[k] if k>2500.0/fs*n else 0 for k in range(0, n) ])

def roth_hpass(Y1, Y2):
    n = len(Y1)
    denominator = Y1 * np.conj(Y1)
    denominator[denominator == 0] = 1e-16
    result = 1. / denominator
    result[:int(2500./fs*n)] = 0
    return result

def phat_hpass(Y1, Y2):
    n = len(Y1)
    phat = Y1 * np.conj(Y2)
    phat[phat==0] = 1e-16
    phat = 1 / phat
    phat[:int(2500./fs*n)] = 0
    return phat

def gcc(Ψ):
    def gcc_fn(y1, y2):
        Y1 = rfft(np.pad(y1, (0, len(y1))))
        Y2 = rfft(np.pad(y2, (len(y2), 0)))
        G = Y1 * np.conj(Y2) * Ψ(Y1, Y2)
        return irfft(G)[1:]
    return gcc_fn

corr_fn = gcc(hpass)
corr_fn = gcc(roth_hpass)
corr_fn = gcc(phat_hpass)

name_prefix = "ttaps_test_"
target = "flounder"
recorder = "pi"
recording_id = "9-10-4"
sound_samples_dir = "/home/zarandy/Documents/sound-samples"

filename = sound_samples_dir+"/"+name_prefix+""+target+"-"+recorder+"-"+recording_id+".wav"
filename_diff = sound_samples_dir+"/"+name_prefix+"-"+target+"-"+recorder+"-"+recording_id+"_diff.wav"
filename_lbl = sound_samples_dir+"/ttaps_ontable-"+target+"-"+recording_id+"-delayed.txt"
#filename = "/home/zarandy/Documents/sound-samples/ttaps-inhand-pi-flounder-5.wav"
#filename = "/home/zarandy/Documents/sound-samples/ttaps-inhand-pi-flounder-6.wav"
#filename = "/home/zarandy/Documents/sound-samples/ttaps_test_flounder-pi-7-10-22.wav"
#filename = "/home/zarandy/Documents/sound-samples/ttaps_test_HWLYA-pi-7-11-13.wav"
#filename = "/home/zarandy/Documents/sound-samples/ttaps_test_CO2N_sprout-pi-7-12-50.wav"
fs, x = wavfile.read(filename)

f, t, Zxx = stft(x, fs=fs, window='hann', nperseg=128, noverlap=96, axis=0)

fc, c, l = Zxx.shape
Zxxa = np.absolute(Zxx)

radius_cm = 4.63
speed_of_sound_cm_per_sec = 34029.0
r06 = range(0, 6)
mic_coordinates = np.array([ [cos((2.-k)*pi/3), sin((2.-k)*pi/3), 0] for k in r06]) * radius_cm

# Wavfile noise cancel
xnc = x[:, :3] - x[:, 3:]
wavfile.write(filename_diff, fs, xnc)

t_tresh = 20
t_tresh_s = 5
tap_cands = []
for i in range(0, l - 16):
    t_found = np.zeros(6);
    for ci in r06:
        tapvol = np.average(Zxxa[8:30, ci, (i+4):(i+7)])
        baseindx = np.concatenate((np.arange(4) + i, np.arange(4) + i + 7))
        basevol = np.average(Zxxa[8:30, ci, baseindx])
        if basevol < tapvol * 0.85:
            for fi in range(8, 30):
                if (np.average(Zxxa[fi, ci, i:(i+4)]) < 0.7 * np.average(Zxxa[fi, ci, (i+4):(i+7)]) ) and (np.average(Zxxa[fi, ci, (i+7):(i+11)]) < 0.7 * np.average(Zxxa[fi, ci, (i+4):(i+7)])):
                    t_found[ci] += 1
    if (t_found[3] >= t_tresh_s or t_found[4] >= t_tresh_s) and np.sum(t_found) > t_tresh:
        tap_cands += [(i+4, np.sum(t_found))]
        print("tap ", t[i+4], '(', i+4, ') :', np.sum(t_found), '; ', t_found)

# 
test_tap_idx = 13848
test_tap_start = test_tap_idx * 32
test_tap_filt_start = test_tap_idx * 32 - 512
test_tap_end = test_tap_idx * 32 + 64       #

sos = cheby2(5, 40, 2500, 'high', fs=fs, output='sos')

def compute_TDoA(filt_start, start, end, ref_ci, max_diff=30):
    s = start - filt_start
    n = end - start
    d = min(max_diff, n)
    correlations = [ corr_fn(x[start:end, i], x[start:end, ref_ci])[(n-d):(n+d)] for i in r06 ]
    return np.argmax(correlations, axis=1) - d + 1

def compute_position(arrivals):
    diffs = (arrivals - np.average(arrivals)) * speed_of_sound_cm_per_sec / fs
    x0 = np.array([0, -8, .1])
    def residuals(u):
        n = np.linalg.norm(u - mic_coordinates, axis=1)
        return n - np.average(n) - diffs
    def jacobian(u):
        jd = np.array([ (u - s_i) / np.linalg.norm(u - s_i) for s_i in mic_coordinates ])
        return jd - np.average(jd, axis=0)
    p = least_squares(residuals, x0, jac=jacobian)
    return p

def score_pw(crosscorr, u, mid):
    n = np.linalg.norm(u - mic_coordinates, axis=1)
    r = 0.0
    for i, j in product(r06, r06):
        d = int(n[i] - n[j])
        r += crosscorr[i, j, mid+d]
    return r


def compute_position_pw(crosscorr, u0, mid):
    p = fmin(lambda u: -score_pw(crosscorr, u, mid), u0, disp=False)
    return p


#test_taps_idcs = [ 4064, 8281, 9457, 10978, 13848, 16707, 19248, 20510, 22013, 26062, 27485 ]
#test_taps_idcs = [ 
#        5630,
#        6854,
#        8477,
#        10930,
#        12171,
#        13580,
#        16477,
#        # 12.9?
#        19212,
#        # 15.911
#        23306, #?
#        24636,
#        27526,
#        28726,
#        30206,
#        33074,
#        34404, # 24.965
#        35688,
#        ]
test_taps_idcs = [
        4339,
        5546,
        6966, 
        9814,
        10978,
        12304,
        15235,
        16461,
        #17360, # not on screen, but looks like a legit tap
        17840,
        20748,
        22039,
        23382,
        26604,
        27489,
        28962,
        31724,
        32945,
        34376,
        37126,
        38516,
        39923,
        42680,
        44028,
        45448,
        48180,
        49668,
        50975,
        53801, 
        55231,
        56448, #56455, # ?
        59271,
        60692,
        62016,
        64717,
        66110,
        67605
        ]




for idx in test_taps_idcs:
    arrivals = compute_TDoA(idx * 32 - 512, idx * 32, idx * 32 + 64, 4, 12)
    p = compute_position(arrivals)
    print(idx, t[idx], p.x)

def score(start, u, length=30):
    dists = np.linalg.norm(u - mic_coordinates, axis=1) * fs / speed_of_sound_cm_per_sec
    delays = dists - np.average(dists)
    end = start + length
    #translated = np.array([ x[(start+ir(dists[i])):(end+ir(dists[i])), i] for i in r06 ])
    translated = np.array([ delayseq(x[(start+int(dists[i])):(end+int(dists[i])), i], float(-dists[i]) % 1, 1) for i in r06 ])
    total = np.sum(translated, axis = 0)
    autocorr = corr_fn(total, total)
    return autocorr[int(len(autocorr) / 2)]


def compute_position_subsample(start, u0, length):
    r = fmin(lambda u: -score(start, u, length), u0, disp=False)
    return r
    

def compute_TDoA_multiple(filt_start, start, end, ref_ci, max_diff=30, verbose=False):
    n = end - start
    correlations = np.array([ corr_fn(x[start:end, i], x[start:end, ref_ci]) for i in r06 ])
    crosscorr2d = np.array([[corr_fn(x[start:end, i], x[start:end, j]) for j in r06] for i in r06 ])
    def get_peaks(i):
        d = int(sqrt(2-2*cos((i-ref_ci)*pi/3)) * radius_cm/speed_of_sound_cm_per_sec*fs + 1)
        c = correlations[i, (n-1-d):(n+d)]
        p = np.max(c)
        result = []
        for j in range(0, len(c)):
            if c[j] >= 0.8 * p and (j == 0 or c[j] >= c[j-1]) and (j == len(c)-1 or c[j] >= c[j+1]):
                result += [j-d]
        return result
    corr_local_peaks = [ get_peaks(i) for i in r06 ]
    candidates = []
    if verbose:
        print(corr_local_peaks)
    for idx in product(*corr_local_peaks):
        u = compute_position(np.array(idx)).x
        #print(u)
        if (u ** 2).sum() > radius_cm ** 2 / 4 and (u ** 2).sum() < 1000000:
            u = compute_position_pw(crosscorr2d, u, n-1)
        #    u = compute_position_subsample(sos, start, u, n)
            candidates.append(u)
    if verbose:
        print(np.array(candidates))
    if len(candidates) == 0:
        return np.array([0., 0., 0.])
    return candidates[np.argmax([ score(start, u, n) for u in candidates ])]
    return candidates[np.argmax([ score_pw(crosscorr2d, u, n-1) for u in candidates ])]

for idx in test_taps_idcs:
    print()
    u = compute_TDoA_multiple(idx * 32, idx * 32, idx * 32 + 128, 4, 12, verbose=True)
    print(idx, t[idx], u)

#

fig, axs = plt.subplots(6, 1, sharex=True)
for i in r06:
    axs[i].plot(x[(test_taps_idcs[2]*32):(test_taps_idcs[2]*32+64), i])

plt.show()

fig, axs = plt.subplots(6, 1, sharex=True)
s = 0
n = 128
tn = 2
start = test_taps_idcs[tn] * 32 
end = start + n
test_taps_idcs[tn] * 32
test_taps_idcs[tn] * 32 / fs
max_diff = 30
ref_ci = 4
for i in r06:
    xseg = x[(test_taps_idcs[tn]*32):(test_taps_idcs[tn]*32+n), i]
    corr = corr_fn(x[start:end, i], x[start:end, ref_ci])[int(n/2):int(3*n/2)]
    axs[i].plot(corr / np.max(np.absolute(corr)))
    axs[i].plot(xseg / np.max(np.absolute(xseg)))

plt.show()

fig, axs = plt.subplots(2, 1, sharex=True)
fig.patch.set_facecolor('black')
sos = cheby2(5, 40, 2500, 'high', fs=fs, output='sos')
s = 512
n = 128
tn = 3
test_taps_idcs[tn] * 32 
test_taps_idcs[tn] * 32 / fs
filtered = sosfilt(sos, x[(test_taps_idcs[tn]*32-s):(test_taps_idcs[tn]*32+n), :], axis=0)
max_diff = 30
ref_ci = 4
xseg = x[(test_taps_idcs[tn]*32 - s):(test_taps_idcs[tn]*32+n), ref_ci]
axs[0].specgram(xseg, Fs=fs, NFFT=128, window=np.hanning(128), noverlap=120)
axs[1].specgram(filtered[:, i], Fs=fs, NFFT=128, window=np.hanning(128), noverlap=127)
plt.show()


#700
#1000
#1400
##400
#filename_lbl = filename[:-4]+'-delayed.txt'

txtfile = read_csv(filename_lbl, sep='\t|[\tTap (]|, |[)]', engine='python', header=None)
plt.scatter(txtfile[8], txtfile[7])
plt.show()
rect_classes = [ Rect(0,0,700,400), Rect(700,0,300,400), Rect(1000,0,400,400), Rect(1400,0,600,400), Rect(0,400,700,1000), Rect(700,400,300,1000), Rect(1000,400,400,1000), Rect(1400,400,600,1000) ]

def coordinates_to_class(x, y, c):
    for i in range(0, len(c)):
        if c[i].Contains(int(x), int(y)):
            return i
    return -1

def tap_idc_to_class(idc): 
    tid = -1
    for j in range(0, len(txtfile[0])):
        if abs(txtfile[0][j] - idc * 32 / fs) < 0.1:
            tid = j
            break
    if tid != -1:
        return coordinates_to_class(txtfile[8][tid], txtfile[7][tid], rect_classes)
    return -1;

test_taps_classes = np.array([ tap_idc_to_class(i) for i in test_taps_idcs ])

def find_tap_idcs():
    label_timestamps = txtfile[0]
    candidates = []
    for (a, b) in tap_cands:
        if t[a] < pidied:
            candidates += [(a, b)]
        else:
            break
    output = [];
    i = 0
    fps = 0
    for ts in label_timestamps:
        c = -1
        vol = 0
        while i < len(candidates) and t[candidates[i][0]] < ts - 0.05:
            i += 1
            fps += 1
        while i < len(candidates) and t[candidates[i][0]] <= ts + 0.05:
            if candidates[i][1] > vol:
                c = i
                vol = candidates[i][1]
            i += 1
        output += [(c, vol)]
    return output, fps/len(candidates)

find_tap_idcs()


def extract_cepstrum_features(idc):
    return real_cepstrum(x[(idc*32):(idc*32+256), 4])

test_taps_cepstrum_features = np.array([ extract_cepstrum_features(i) for i in test_taps_idcs ])

train_idcs = np.array([ 1,  2, 3, 4,  5,  7,  8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35])
test_idcs = np.array([0, 6, 12, 18, 24, 27, 30, 33])

lda = LinearDiscriminantAnalysis()
lda.fit(test_taps_cepstrum_features[train_idcs], test_taps_classes[train_idcs])
lda.predict(test_taps_cepstrum_features[test_idcs])
test_taps_classes[test_idcs]



