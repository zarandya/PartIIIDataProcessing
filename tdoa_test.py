#! /usr/bin/env python

# load as in classify.py
from utils import *
import matplotlib.pyplot as plt
from matplotlib.colors import cnames
from data_augmentation import augment, augment_inorder
import numpy as np
from os import listdir
from os.path import join, isdir, isfile
from scipy.io import wavfile
from acoustics.cepstrum import real_cepstrum
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from scipy.signal import correlate, stft
from math import isinf, isnan
from librosa.feature import mfcc
from tensorflow.keras import datasets, layers, models, losses

#experimentName = "tblword"
#experimentName = "lscpword"
experimentName = "QWERTYu"
#experimentName = "Sik2"
#max_cands = 25000

base_dir = '/home/zarandy/Documents/sound-samples/sync/'
#base_dir = '/mnt/sdc1/almos/sound-samples/sync'
jsons = []
for f in listdir(base_dir):
    d = join(base_dir, f)
    if (isdir(d)):
        json_file = join(d, "syncd.json")
        if (isfile(json_file)):
            jsons += [open_json(json_file)]

audio_in, tap_cands_in_jsons, num_found, num_correctly_syncd = open_audio_files(jsons, experimentName, verbose=False)


tap_anz_ = []
tap_anz = []
tap_ch = []
keys = {}
jj = None
for json, fsx in zip(jsons, audio_in):
    if (json.experimentName == experimentName) and not hasattr(json, "taps_syncd_orig"):
        jj = json
        fs, x = fsx
        taps_syncd = json.taps_syncd
        actual_taps = [ int(i.time_samples) for i in taps_syncd ]
        for t in json.taps_syncd:
            print('=====')
            ts = int(t.time_samples)
            fgcc = gcc(hpassf(3000), fs)
            #D = compute_TDoA_longer_filter(x, ts, ts+64, 4, fgcc, fs, ts-512, ts+512, 2500, max_diff=12)
            score = 0
            start_ts = 0
            print(ts, ' wf ', len(x))
            for tsd in range(ts-128, min(ts+128, len(x)-64), 16):
                dcd = compute_corr_pw(x, tsd, tsd+64, fgcc, max_diff=12)
                #print(tsd, dcd[4, 4, 11])
                if dcd[4, 4, 11] > score:
                    score = dcd[4, 4, 11]
                    start_ts = tsd
            #D = compute_TDoA(x, start_ts, start_ts + 64, 4, fgcc, max_diff=12)
            #D = compute_TDoA_cancel_noise(x, start_ts, start_ts+64, 4, fgcc, start_ts-512, start_ts-512+64, max_diff=12)
            dc = compute_corr_pw(x, start_ts, start_ts+64, fgcc, max_diff=12)
            D = np.array([
                np.argmax(dc[4, 0, 11:23]),
                np.argmax(dc[4, 1, 11:23]),
                np.argmax(dc[4, 2, 11:21]),
                np.argmax(dc[4, 3, 5:18]) - 6,
                0,
                np.argmax(dc[4, 5, 11:18])
                ])
            dd_ = compute_TDoA_pw(x, start_ts, start_ts+64, fgcc, max_diff=12)
            dd = np.array([
                [
                    0,
                    np.argmax(dc[0, 1, 8:15]) - 3,
                    np.argmax(dc[0, 2, 5:14]) - 6,
                    np.argmax(dc[0, 3, 0:12]) - 11,
                    np.argmax(dc[0, 4, 0:12]) - 11,
                    np.argmax(dc[0, 5, 5:12]) - 6
                    ],
                [
                    np.argmax(dc[1, 0, 8:15]) - 3,
                    0,
                    np.argmax(dc[1, 2, 5:12]) - 6,
                    np.argmax(dc[1, 3, 0:12]) - 11,
                    np.argmax(dc[1, 4, 0:12]) - 11,
                    np.argmax(dc[1, 5, 5:14]) - 6,
                    ],
                [
                    np.argmax(dc[2, 0, 9:19]) - 2,
                    np.argmax(dc[2, 1, 11:18]),
                    0,
                    np.argmax(dc[2, 3, 5:12]) - 6,
                    np.argmax(dc[2, 4, 2:12]) - 9,
                    np.argmax(dc[2, 5, 3:20]) - 8
                    ],
                [
                    np.argmax(dc[3, 0, 11:23]),
                    np.argmax(dc[3, 1, 11:23]),
                    np.argmax(dc[3, 2, 11:18]),
                    0,
                    np.argmax(dc[3, 4, 5:18]) - 6,
                    np.argmax(dc[3, 5, 11:21])
                    ],
                [
                    np.argmax(dc[4, 0, 11:23]),
                    np.argmax(dc[4, 1, 11:23]),
                    np.argmax(dc[4, 2, 11:21]),
                    np.argmax(dc[4, 3, 5:18]) - 6,
                    0,
                    np.argmax(dc[4, 5, 11:18])
                    ],
                [
                    np.argmax(dc[5, 0, 11:18]),
                    np.argmax(dc[5, 1, 9:19]) - 2,
                    np.argmax(dc[5, 2, 3:20]) - 8,
                    np.argmax(dc[5, 3, 2:12]) - 9,
                    np.argmax(dc[5, 4, 5:12]) - 6,
                    0
                    ]
                ])
            try:
                #p = compute_position_sa_sim(D, 4)
                φ_ = compute_DOA_azimuth(dd_)
                φ = compute_DOA_azimuth(dd)
                #p = compute_position_lse(D, fs).x
                #p = compute_position_pw(dc, np.array([0, -8, .1]))
                print(json.wav)
                print(t)
                print(D)
                #print(p)
                print(φ)
                #fig, axs = plt.subplots(6, 1, sharex=True)
                #for i in r06:
                #    axs[i].plot(dc[4, i, :])
                #plt.show()
                if True:#np.isfinite(p).all() and np.linalg.norm(p) < 40:
                    if keys.keys().isdisjoint(t.ch):
                        keys[t.ch] = []
                    keys[t.ch] += [φ]
                    tap_anz += [φ]
                    tap_anz_ += [φ_]
                    tap_ch += [t.ch]
            except:
                raise

tap_anz = np.array(tap_anz)
tap_anz_ = np.array(tap_anz_)
tap_ch = np.array(tap_ch)

letters = ['e', 't', 'a', 'r', 'i', 'o', 'n', 's', 'h', 'd']
letters = np.array(['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'])
#letters = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l']
#letters = ['z', 'x', 'c', 'v', 'b', 'n', 'm']
colours = np.array(['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown', 'grey', 'pink', 'black'])

for ch, cl in zip(letters, colours):#zip(keys.keys(), cnames.keys()):
    if not keys.keys().isdisjoint(ch):
        print(ch, len(keys[ch]))
        xy = np.array(keys[ch])
        x = xy[:, 0, 0]
        y = xy[:, 1, 0]
        mx = np.average(x)
        my = np.average(y)
        w, v = np.linalg.eig(np.array([x,y]) @ (np.array([x, y]).T))
        i = np.argmax(w)
        #plt.scatter(x, y, c=cl, label=ch)
        #plt.plot([mx - 10 * v[0, i], mx + 10 * v[0, i]], [my - 10 * v[1, i], my + 10 * v[1, i]], '-', c=cl)
        plt.scatter(np.arctan(y/ x), np.linalg.norm(np.array([x, y]), axis=0), c=cl, label=ch)
        #b, m = np.polynomial.polynomial.polyfit(y, x, 1)
        #plt.plot([b - 10 * m, b + 10 * m], [-10, 10], '-', c=cl)
        #b, m = np.polynomial.polynomial.polyfit(x, y, 1)
        #plt.plot([-10, 10], [b - 10 * m, b + 10 * m], '-', c=cl)
        #plt.scatter(xy[:, 0], xy[:, 1], c=cl, label=ch)

for ch, cl in zip(letters, colours):#zip(keys.keys(), cnames.keys()):
    if not keys.keys().isdisjoint(ch):
        print(ch, len(keys[ch]))
        φ = np.array(keys[ch]).flatten()
        y = np.random.uniform(size=φ.shape)
        plt.scatter(φ, y, c=cl, label=ch)

fig, ax = plt.subplots(1, 2, sharey=True)
for ch, cl in zip(letters, colours):#zip(keys.keys(), cnames.keys()):
    ax[0].scatter(tap_anz[tap_ch == ch, 0] % pi, np.random.uniform(size=np.shape(tap_anz[tap_ch == ch, 0])), c=cl, label=ch)
    ax[1].scatter(tap_anz_[tap_ch == ch, 0] % pi, np.random.uniform(size=np.shape(tap_anz_[tap_ch == ch, 0])), c=cl, label=ch)

dy = -17
wx = 18
fig, ax = plt.subplots(len(letters), sharex=True)
for i, (ch, cl) in enumerate(zip(letters, colours)):
    ax[i].hist(tap_anz[tap_ch == ch, 0] % (2*pi), bins=np.arange(pi, 2*pi, step=pi/36), color='black', label=ch)
    key = getattr(jj.keys, ch)
    correct_anz = (np.arctan2(dy, wx/2* ((key.left+key.right) / jj.screenWidth - 1))) % (2*pi)
    print(dy, wx/2* ((key.left+key.right) / jj.screenWidth - 1), correct_anz)
    ax[i].plot([correct_anz, correct_anz], [0, 20], c='blue')
    #mean = np.average(tap_anz[tap_ch == ch, 0] % (2*pi))
    #ax[i].plot([mean, mean], [0, 20], c='green')
    ax[i].legend(loc='upper left')

#fig, ax = plt.subplots(len(letters), sharex=True)
for i, (ch, cl) in enumerate(zip(letters, colours)):
    plt.hist(tap_anz[tap_ch == ch, 0] % pi, bins=np.arange(pi, step=pi/72), color=cl, label=ch, alpha=0.7)

plt.legend()
plt.show()

u = int(len(tap_ch) * .9)
train_idcs = np.arange(u)
test_idcs = np.arange(u, len(tap_ch))
lda = LinearDiscriminantAnalysis()
lda.fit(tap_anz[train_idcs], tap_ch[train_idcs])
pred = lda.predict(tap_anz[test_idcs])

