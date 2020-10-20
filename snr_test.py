#! /usr/bin/env python


from utils import *
import matplotlib
#matplotlib.use('pgf')
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
import sys

print('there is probably not enough RAM to load all data, this file must run in multiple passes')
experimentName = re.compile('tdis.5')
experimentName = re.compile('(atnPin|onehand|cvr|lscpword|bj)')
experimentName = "atPin"
experimentName = re.compile('(tdis.5|atnPin|atPin|onehand|cvr|lscpword)')
#max_cands = 25000

jsons = get_jsons()

audio_in, tap_cands_in_jsons, num_found, num_correctly_syncd = open_audio_files(jsons, experimentName, fallback_mode='use_syncd_to_self', verbose=True)

def get_snrs(jsons, audio_in, experimentName, fallback_mode='ignore', also_get_self=False, l=512):
    snrs = []
    snrs_self = []
    for json, fsx in zip(jsons, audio_in):
        if json.experimentName == experimentName and (fallback_mode != 'ignore' or not hasattr(json, "taps_syncd_orig")):
            fs, x = fsx
            taps_syncd = json.taps_syncd
            actual_taps = [ int(i.time_samples) for i in taps_syncd ]
            print(json.wav)
            for t in actual_taps:
                if t < len(x) - 4096:
                    snr = (compute_snr_per_frequency(x, t, length=l, noise_start=t-512))
                    snrs += [snr]
            if also_get_self:
                fs, x = wavfile.read(json.self_recording_wav)
                taps_syncd = json.taps_syncd_to_self_recording
                actual_taps = [ int(i.time_samples) for i in taps_syncd ]
                print(json.wav)
                for t in actual_taps:
                    if t < len(x) - 4096:
                        snr = (compute_snr_per_frequency(x, t, length=l, noise_start=t-512))
                        snrs_self += [snr]
    if also_get_self:
        return snrs, snrs_self
    return snrs
    

l=512
snrs = [[],[],[],[],[]]
snrs_atPin = []
snrs_self_atPin = []
snrs_atnPin = []
snrs_self_atnPin = []
snrs_onehand = []
snrs_self_onehand = []
snrs_cvr = []
snrs_self_cvr = []
snrs_lscpword = []
snrs_self_lscpword = []
snrs_bj = []
snrs_self_bj = []
for json, fsx in zip(jsons, audio_in):
    for i in range(5):
        if (json.experimentName == ('tdis%s5' % (i+1))):#
            fs, x = fsx
            taps_syncd = json.taps_syncd
            actual_taps = [ int(i.time_samples) for i in taps_syncd ]
            print(json.wav)
            for t in actual_taps:
                if t < len(x) - 4096:
                    snr = (compute_snr_per_frequency(x, t, length=l, noise_start=t-512))
                    snrs[i] += [snr]
    if json.experimentName == 'atPin' and not hasattr(json, "taps_syncd_orig"):
        fs, x = fsx
        taps_syncd = json.taps_syncd
        actual_taps = [ int(i.time_samples) for i in taps_syncd ]
        print(json.wav)
        for t in actual_taps:
            if t < len(x) - 4096:
                snr = (compute_snr_per_frequency(x, t, length=l, noise_start=t-512))
                snrs_atPin += [snr]
        fs, x = wavfile.read(json.self_recording_wav)
        taps_syncd = json.taps_syncd
        actual_taps = [ int(i.time_samples) for i in taps_syncd ]
        print(json.wav)
        for t in actual_taps:
            if t < len(x) - 4096:
                snr = (compute_snr_per_frequency(x, t, length=l, noise_start=t-512))
                snrs_self_atPin += [snr]
    if json.experimentName == 'atnPin' and not hasattr(json, "taps_syncd_orig"):
        fs, x = fsx
        taps_syncd = json.taps_syncd
        actual_taps = [ int(i.time_samples) for i in taps_syncd ]
        print(json.wav)
        for t in actual_taps:
            if t < len(x) - 4096:
                snr = (compute_snr_per_frequency(x, t, length=l, noise_start=t-512))
                snrs_atnPin += [snr]
        fs, x = wavfile.read(json.self_recording_wav)
        taps_syncd = json.taps_syncd
        actual_taps = [ int(i.time_samples) for i in taps_syncd ]
        print(json.wav)
        for t in actual_taps:
            if t < len(x) - 4096:
                snr = (compute_snr_per_frequency(x, t, length=l, noise_start=t-512))
                snrs_self_atnPin += [snr]
    if json.experimentName == 'onehand':
        fs, x = fsx
        taps_syncd = json.taps_syncd
        actual_taps = [ int(i.time_samples) for i in taps_syncd ]
        print(json.wav)
        for t in actual_taps:
            if t < len(x) - 4096:
                snr = (compute_snr_per_frequency(x, t, length=l, noise_start=t-512))
                snrs_onehand += [snr]
    if json.experimentName == 'cvr':
        fs, x = fsx
        taps_syncd = json.taps_syncd
        actual_taps = [ int(i.time_samples) for i in taps_syncd ]
        print(json.wav)
        for t in actual_taps:
            if t < len(x) - 4096:
                snr = (compute_snr_per_frequency(x, t, length=l, noise_start=t-512))
                snrs_cvr += [snr]
    if json.experimentName == 'lscpword' and not hasattr(json, "taps_syncd_orig"):
        fs, x = fsx
        taps_syncd = json.taps_syncd
        actual_taps = [ int(i.time_samples) for i in taps_syncd ]
        print(json.wav)
        for t in actual_taps:
            if t < len(x) - 4096:
                snr = (compute_snr_per_frequency(x, t, length=l, noise_start=t-512))
                snrs_lscpword += [snr]
    if json.experimentName == 'bj' and not hasattr(json, "taps_syncd_orig"):
        fs, x = fsx
        taps_syncd = json.taps_syncd
        actual_taps = [ int(i.time_samples) for i in taps_syncd ]
        print(json.wav)
        for t in actual_taps:
            if t < len(x) - 4096:
                snr = (compute_snr_per_frequency(x, t, length=l, noise_start=t-512))
                snrs_bj += [snr]

asnrs = [ np.average(10*np.log10(s), axis=0) for s in snrs ]
aasnrs = [ np.average(s, axis=1) for s in asnrs ]
aaasnrs = np.array([ np.average(s[11:25], axis=0) for s in aasnrs ]).flatten()
asnrs_atPin = np.average(10*np.log10(snrs_atPin), axis=0)
aasnrs_atPin = np.average(asnrs_atPin, axis=1)
aaasnrs_atPin = np.average(aasnrs_atPin[11:25], axis=0).flatten()
asnrs_self_atPin = np.average(10*np.log10(snrs_self_atPin), axis=0)
aasnrs_self_atPin = np.average(asnrs_self_atPin, axis=1)
aaasnrs_self_atPin = np.average(aasnrs_self_atPin[11:25], axis=0).flatten()
asnrs_atnPin = np.average(10*np.log10(snrs_atnPin), axis=0)
aasnrs_atnPin = np.average(asnrs_atnPin, axis=1)
aaasnrs_atnPin = np.average(aasnrs_atnPin[11:25], axis=0).flatten()
asnrs_self_atnPin = np.average(10*np.log10(snrs_self_atnPin), axis=0)
aasnrs_self_atnPin = np.average(asnrs_self_atnPin, axis=1)
aaasnrs_self_atnPin = np.average(aasnrs_self_atnPin[11:25], axis=0).flatten()
asnrs_onehand = np.average(10*np.log10(snrs_onehand), axis=0)
aasnrs_onehand = np.average(asnrs_onehand, axis=1)
asnrs_cvr = np.average(10*np.log10(snrs_cvr), axis=0)
aasnrs_cvr = np.average(asnrs_cvr, axis=1)
asnrs_lscpword = np.average(10*np.log10(snrs_lscpword), axis=0)
aasnrs_lscpword = np.average(asnrs_lscpword, axis=1)
asnrs_bj = np.average(10*np.log10(snrs_bj), axis=0)
aasnrs_bj = np.average(asnrs_bj, axis=1)
#snrs = np.average(snrs, axis=0)

#l=512
#snrs_self = [[],[],[],[],[],[],[]]
#for json, fsx in zip(jsons, audio_in):
#    for i in range(1, 6):
#        if (json.experimentName == ('tdis%s5' % i)) and not hasattr(json, "taps_syncd_orig"):
#            fs, x = wavfile.read(json.self_recording_wav)
#            taps_syncd = json.taps_syncd_to_self_recording
#            actual_taps = [ int(i.time_samples) for i in taps_syncd ]
#            print(json.self_recording_wav)
#            for t in actual_taps:
#                if t < len(x) - 4096:
#                    snr = (compute_snr_per_frequency(x, t, length=l, noise_start=t-512))
#                    snrs_self[i] += [snr]
#
#snrs_self = np.average(snrs_self, axis=0)

#colours = ['blue', 'green', 'orange', 'red', 'purple', 'grey']



#max_f_idx = int(l/8)
#colours = ['red', 'green', 'cyan', 'blue', 'deeppink']
#for j in range(5):
#    plt.plot(np.arange(48001, step=48000/l*2)[1:max_f_idx], aasnrs[j][1:max_f_idx, 0], c=colours[j])
plt.clf()
matplotlib.use('pgf')
max_f_idx = int(l/8)
colours_2d = [
        ['indianred', 'brown', 'firebrick', 'maroon', 'red', 'darkred'],
        ['olivedrab', 'forestgreen', 'limegreen', 'darkgreen', 'green', 'darkolivegreen'],
        ['turquoise', 'lightseagreen', 'teal', 'cyan', 'deepskyblue', 'dodgerblue'],
        ['midnightblue', 'navy', 'darkblue', 'mediumblue', 'blue', 'royalblue'],
        ['indigo', 'darkviolet', 'purple', 'fuchsia', 'deeppink', 'crimson']
        ]
for j in range(5):
    aa = asnrs[j]
    cc = colours_2d[j]
    for i in range(6):
        plt.plot(np.arange(48001, step=48000/l*2)[1:max_f_idx], aa[1:max_f_idx, i, 0], c=cc[i],
                label=("%d cm" % (15+10*j) if i==3 else None))

plt.grid()
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Signal-to-Noise Ratio (dB)')

plt.savefig('AlmosPartIII/figs/snr-tdist.5.pgf')
    
plt.clf()
matplotlib.use('pgf')

max_f_idx = int(l/8)
colours = ['brown', 'firebrick', 'maroon', 'crimson', 'red', 'darkred']
labels=[None, None, None, None, 'External microphone', None]
for i in range(6):
    plt.plot(np.arange(48001, step=48000/l*2)[1:max_f_idx], asnrs_atPin[1:max_f_idx, i, 0], c=colours[i], label=labels[i])

max_f_idx = int(l/8)
colours = ['green', 'lime']
labels = ['Internal microphone', None]
for i in range(2):
    plt.plot(np.arange(48001, step=48000/l*2)[1:max_f_idx], asnrs_self_atPin[1:max_f_idx, i, 0], c=colours[i], label=labels[i])

plt.grid()
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Signal-to-Noise Ratio (dB)')

plt.savefig('AlmosPartIII/figs/snr-atPin.pgf')

plt.clf()
matplotlib.use('pgf')

max_f_idx = int(l/8)
colours = ['brown', 'firebrick', 'maroon', 'crimson', 'red', 'darkred']
labels=[None, None, None, None, 'External microphone', None]
for i in range(6):
    plt.plot(np.arange(48001, step=48000/l*2)[1:max_f_idx], asnrs_atnPin[1:max_f_idx, i, 0], c=colours[i], label=labels[i])

max_f_idx = int(l/8)
colours = ['green', 'lime']
labels = ['Internal microphone', None]
for i in range(2):
    plt.plot(np.arange(48001, step=48000/l*2)[1:max_f_idx], asnrs_self_atnPin[1:max_f_idx, i, 0], c=colours[i], label=labels[i])


plt.grid()
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Signal-to-Noise Ratio (dB)')

plt.savefig('AlmosPartIII/figs/snr-atnPin.pgf')

plt.clf()
matplotlib.use('pgf')

max_f_idx = int(l/8)
plt.plot(np.arange(48001, step=48000/l*2)[1:max_f_idx], aasnrs_bj[1:max_f_idx,  0], label='On a solid surface,\ntapping with one finger')
plt.plot(np.arange(48001, step=48000/l*2)[1:max_f_idx], aasnrs_atnPin[1:max_f_idx,  0], label='In one hand, tapping \nwith finger of other')
plt.plot(np.arange(48001, step=48000/l*2)[1:max_f_idx], aasnrs_lscpword[1:max_f_idx,  0], label='Held in two hands, landscape,\ntapping with two thumbs')
plt.plot(np.arange(48001, step=48000/l*2)[1:max_f_idx], aasnrs_cvr[1:max_f_idx, 0], label='Cradled two hands, portrait,\ntappung with two thumbs')
plt.plot(np.arange(48001, step=48000/l*2)[1:max_f_idx], aasnrs_onehand[1:max_f_idx, 0], label='In one hand, tapping \nwith thumb of same hand')

plt.grid()
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Signal-to-Noise Ratio (dB)')

plt.savefig('AlmosPartIII/figs/snr-holding.pgf')

sys.exit(0)

snrs = get_snrs(jsons, audio_in, experimentName, fallback_mode='use_syncd_to_self')

asnrs = np.average(10*np.log10(snrs), axis=0)

max_f_idx = int(l/8)
colours = ['brown', 'firebrick', 'maroon', 'crimson', 'red', 'darkred']
labels=[None, None, None, None, 'External microphone', None]
for i in range(6):
    plt.plot(np.arange(48001, step=48000/l*2)[1:max_f_idx], asnrs[1:max_f_idx, i, 0], c=colours[i], label=labels[i])

matplotlib.use('Qt5Agg')
plt.show()


