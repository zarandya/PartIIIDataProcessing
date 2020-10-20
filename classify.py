#! /usr/bin/env python

from utils import *
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

experimentName = "atPin"

jsons = get_jsons()

audio_in, tap_cands_in_jsons, num_found, num_correctly_syncd = open_audio_files(jsons, experimentName, verbose=True)

waveforms = []
mfcc_features = []
ceps_features = []
fourier_features = []
mfcc_6mic_coarse_features = []
mfcc_6mic_all_features = []
fourier_6mic_all_features = []
mfcc_6mic_1024nf78fs_features = []

tap_cand_json_name = []
tap_cand_waveforms = []
tap_cand_mfcc_features = []
tap_cand_ceps_features = []
tap_cand_fourier_features = []
tap_cand_mfcc_6mic_coarse_features = []
tap_cand_mfcc_6mic_all_features = []
tap_cand_fourier_6mic_all_features = []
tap_cand_mfcc_6mic_1024nf78fs_features = []
tap_cand_labels = []
tap_cand_ch = []
for jid, (json, tap_cands, fsx) in enumerate(zip(jsons, tap_cands_in_jsons, audio_in)):
#    if (json.experimentName == experimentName):
    if (json.experimentName == experimentName) and not hasattr(json, "taps_syncd_orig"):
        fs, x = fsx
        taps_syncd = json.taps_syncd
        actual_taps = [ int(i.time_samples) for i in taps_syncd ]
        i = 0
        run = 0
        tap_cands = np.array(tap_cands).astype(int)
        while i < len(tap_cands):
            if run == 0:
                selected = -1
                while i+run+1 < len(tap_cands) and tap_cands[i+run][0] + 1 == tap_cands[i+run+1][0]:
                    if tap_cands[i+run][0] * 32 in actual_taps:
                        selected = run
                    run += 1
                if tap_cands[i+run][0] * 32 in actual_taps:
                    selected = run
                run += 1
                if selected == -1:
                    probs = np.array([ c[0] for c in tap_cands[i:(i+run)] ])
                    probs = probs / np.sum(probs)
                    selected = np.random.choice(run, p=probs)
            if selected == 0:
                idx = tap_cands[i][0] * 32
                seg = x[(idx-64):(idx+448), :]
                if (len(seg) == 512) and (idx < len(x) - 9600 or idx <= actual_taps[-1]):
                    #mfcc_features = mfcc(seg[:, 4].astype(float), sr=fs, hop_length=32, n_fft=128, n_mfcc=20)[np.newaxis].swapaxes(1, 2)
                    #ceps_features = real_cepstrum(seg[:, 4])
                    #fourier_features = stft(seg[:, 4], fs=fs, nperseg=128, noverlap=96)[2][np.newaxis, 0:16].swapaxes(1, 2)
                    mfcc_6mic_all_features = np.array([mfcc(seg[:, j].astype(float), sr=fs, hop_length=32, n_fft=128).swapaxes(0, 1) for j in range(0, 6)])
                    #mfcc_6mic_coarse_features = np.array([mfcc(seg[:, j].astype(float), sr=fs).swapaxes(0, 1) for j in range(0, 6)])
                    #fourier_6mic_all_features = np.moveaxis(stft(seg, fs=fs, nperseg=128, noverlap=96, axis=0)[2][0:16], 0, 2)
                    mfcc_6mic_1024nf78fs_features = np.array([mfcc(seg[:, j].astype(float), sr=int(fs * 7 / 8), hop_length=2048, n_fft=1024) for j in range(0, 6)])
                    labels = 0
                    ch = 'FALSE_POSITIVE'
                    for tap in taps_syncd:
                        if idx == int(tap.time_samples):
                            labels = 1
                            if hasattr(tap, "ch"):
                                ch = tap.ch
                            break
                    if np.isfinite(mfcc_features).all() and np.isfinite(ceps_features).all() and np.isfinite(fourier_features).all() and  np.isfinite(mfcc_6mic_all_features).all() and np.isfinite(mfcc_6mic_coarse_features).all() and np.isfinite(fourier_6mic_all_features).all() and np.isfinite(mfcc_6mic_1024nf78fs_features).all():
                        tap_cand_json_name += [jid]
                        tap_cand_waveforms += [seg]
                        tap_cand_mfcc_features += [mfcc_features]
                        tap_cand_ceps_features += [ceps_features]
                        tap_cand_fourier_features += [fourier_features]
                        tap_cand_mfcc_6mic_all_features += [mfcc_6mic_all_features]
                        tap_cand_mfcc_6mic_coarse_features += [mfcc_6mic_coarse_features]
                        tap_cand_fourier_6mic_all_features += [fourier_6mic_all_features]
                        tap_cand_mfcc_6mic_1024nf78fs_features += [mfcc_6mic_1024nf78fs_features]
                        tap_cand_labels += [labels]
                        tap_cand_ch += [ch]
            run -= 1
            selected -= 1
            i += 1
        print(len(tap_cand_labels))
        #if len(tap_cand_labels) > max_cands:
        #    break

ch_class_names_arr = sorted(np.unique(tap_cand_ch))
ch_class_names = np.array(ch_class_names_arr)
for i in range(len(tap_cand_ch)):
    tap_cand_ch[i] = ch_class_names_arr.index(tap_cand_ch[i])

tap_cand_ch = U.np.array([ ch_class_names_arr.index(ch) for ch in tap_cand_ch ])
###

tap_cand_waveforms = np.array(tap_cand_waveforms)
tap_cand_labels = np.array(tap_cand_labels)
tap_cand_ch = np.array(tap_cand_ch)
tap_waveforms = tap_cand_waveforms[tap_cand_labels==1]
tap_ch = tap_cand_ch[tap_cand_labels==1]
aug_data, aug_ch = augment(tap_waveforms, tap_ch, tap_cand_waveforms[tap_cand_labels==0], np.sum(tap_cand_labels)*10)
aug2_data, aug2_ch = augment_inorder(tap_waveforms, tap_ch, tap_cand_waveforms[tap_cand_labels==0], 10)

tap_fourier_features = []
tap_fourier_6mic_all_features = []
for d in tap_waveforms:
    tap_fourier_features += [stft(d[:, 4], fs=fs, nperseg=128, noverlap=96)[2][np.newaxis, 0:16].swapaxes(1, 2)]
    tap_fourier_6mic_all_features += [np.moveaxis(stft(d, fs=fs, nperseg=128, noverlap=96, axis=0)[2][0:16], 0, 2)]

aug_fourier_features = []
aug_fourier_6mic_all_features = []
for d in aug_data:
    aug_fourier_features += [stft(d[:, 4], fs=fs, nperseg=128, noverlap=96)[2][np.newaxis, 0:16].swapaxes(1, 2)]
    aug_fourier_6mic_all_features += [np.moveaxis(stft(d, fs=fs, nperseg=128, noverlap=96, axis=0)[2][0:16], 0, 2)]

aug2_fourier_features = []
aug2_fourier_6mic_all_features = []
for d in aug2_data:
    aug2_fourier_features += [stft(d[:, 4], fs=fs, nperseg=128, noverlap=96)[2][np.newaxis, 0:16].swapaxes(1, 2)]
    aug2_fourier_6mic_all_features += [np.moveaxis(stft(d, fs=fs, nperseg=128, noverlap=96, axis=0)[2][0:16], 0, 2)]

tap_cand_fourier_features = []
tap_cand_fourier_6mic_all_features = []
for i, d in enumerate(tap_cand_waveforms):
    if (i % 100) == 0:
        print(i, '/', len(tap_cand_waveforms))
    tap_cand_fourier_features += [stft(d[:, 4], fs=fs, nperseg=128, noverlap=96)[2][np.newaxis, 0:16].swapaxes(1, 2)]
    tap_cand_fourier_6mic_all_features += [np.moveaxis(stft(d, fs=fs, nperseg=128, noverlap=96, axis=0)[2][0:16], 0, 2)]

tap_cand_mfcc_features = []
tap_cand_mfcc_6mic_all_features = []
for i, d in enumerate(tap_cand_waveforms):
    if (i % 100) == 0:
        print(i, '/', len(tap_cand_waveforms))
    tap_cand_mfcc_features += [mfcc(d[:, 4].astype(float), sr=fs, hop_length=32, n_fft=128, n_mfcc=20)[np.newaxis].swapaxes(1, 2)]
    tap_cand_mfcc_6mic_all_features += [np.array([mfcc(d[:, j].astype(float), sr=fs, hop_length=32, n_fft=128).swapaxes(0, 1) for j in range(0, 6)])]

pred, act = traintest_ch_cnn(aug_fourier_features, np.ones(aug_ch.shape), aug_ch, ch_class_names, verbose=True)



traintest(tap_cand_fourier_features, tap_cand_labels, verbose=True)
traintest(tap_cand_ceps_features, tap_cand_labels)
traintest(tap_cand_mfcc_features, tap_cand_labels)
pred_fourier_diff, act = traintest(tap_cand_fourier_diff_features, tap_cand_labels)
pred_ceps_diff, act = traintest(tap_cand_ceps_diff_features, tap_cand_labels)
pred_mfcc_diff, act = traintest(tap_cand_mfcc_diff_features, tap_cand_labels)
traintest(tap_cand_mfcc_4mic_features, tap_cand_labels)
traintest(tap_cand_mfcc_6mic_all_features, tap_cand_labels)

crossval(tap_cand_mfcc_6mic_all_features, tap_cand_labels)

pred, act = traintest_ch(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, ch_class_names, verbose=True)

pred, act = traintest_ch_cnn(tap_cand_fourier_6mic_all_features, tap_cand_labels, tap_cand_ch, ch_class_names, verbose=True)




##
tap_cand_fourier_6mic_all_features = np.array(tap_cand_fourier_6mic_all_features)
l, nd,nt, nf = np.shape(tap_cand_fourier_6mic_all_features)
selection_features = np.swapaxes(tap_cand_fourier_6mic_all_features, 1, 3)
ch_features = np.reshape(tap_cand_fourier_6mic_all_features, (l, nd*nt, nf, 1))

gu = find_pin_cnn(selection_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features = ch_features, ch_model_gen = gen_model_44, max_guess=20000, num_digits=5, verbose=True)

Q,R = find_pin_lda(tap_cand_mfcc_6mic_all_features, tap_cand_labels, tap_cand_ch, jsons, tap_cand_json_name, ch_class_names, ch_features=tap_cand_mfcc_6mic_1024nf78fs_features, max_guess=20000, num_digits=5, verbose=True, num_threads=1, return_probs=True)
def find_pin_lda(features, labels, ch_labels, jsons, json_id, class_names, ch_features=None, max_guess=200, tp_prob_power=1, num_digits=None, num_threads=4, verbose=False, penalise_non_chosen=False, return_probs=False, ngram_model=None):



lda_gu = find_pin_lda(tap_cand_mfcc_6mic_all_features[:1500], tap_cand_labels[:1500], tap_cand_ch[:1500], jsons, tap_cand_json_name[:1500], ch_class_names, max_guess=20000, num_digits=5, verbose=True,  return_probs=True)
