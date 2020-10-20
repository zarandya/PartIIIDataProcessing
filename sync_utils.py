#! /usr/bin/env python

import numpy as np
import re
from scipy.signal import stft
from scipy.io import wavfile
from os.path import join, isdir, isfile
from types import SimpleNamespace

def find_tap_cands(Zxxa, t, l, t_tresh, vol_max=50., verbose=False):
    """
    Finds tap candidates in a norm spectrogram
    """
    tap_cands = []
    freqs = np.array(range(3, 15))
    freq_weight = np.array([2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1])
    for i in range(0, l - 16):
        t_found = 0
        t_verb = []
        tapvol = np.average(Zxxa[freqs][:, (i+4):(i+7)])
        baseindx = np.concatenate((np.arange(4) + i, np.arange(4) + i + 7))
        basevol = np.average(Zxxa[freqs][:, baseindx])
        if basevol < tapvol * 0.85:
            for fi, fw in zip(freqs, freq_weight):
                vol_before = np.average(Zxxa[fi, i:(i+4)])
                vol_here = min(vol_max, np.average(Zxxa[fi, (i+4):(i+7)]))
                vol_after = np.average(Zxxa[fi, (i+7):(i+11)])
                if (fi < 6):
                    vol_after = 0
                if (vol_before < 0.7 * vol_here) and (vol_after < 0.7 * vol_here):
                    t_found += fw * (vol_here - vol_before / 2 - vol_after / 2)
                    t_verb += [fi]
        if t_found >= t_tresh:
            if verbose: 
                print(len(tap_cands), "tap ", t[i+4], '(', i+4, ') : ', t_found, t_verb)
            tap_cands += [(i+4, np.sum(t_found))]
    return tap_cands


def find_sync_sequence(arr1, samplerate):
    """
    Finds the sync sequence in a recording
    """
    arr = arr1
    FREQ1 = 18000
    FREQ2 = 18500
    max_len = 96000
    if len(arr) > max_len:
        arr = arr[:max_len]
    f, t, Zxx = stft(arr, fs=samplerate, nperseg=256, noverlap=240)
    Zxxa = np.abs(Zxx)
    f1 = int(round(FREQ1 / f[1]))
    f2 = int(round(FREQ2 / f[1]))
    dt = int(round(4096. / 48000 / t[1]))
    index = 0;
    best = 0;
    for i in range(-dt, 0):
        vol = np.sum(Zxxa[f2, :(i+dt)]) - np.sum(Zxxa[f1, :(i+dt)])
        if (vol > best):
            best = vol
            index = i
    for i in range(0, dt):
        vol = np.sum(Zxxa[f1, :i]) - np.sum(Zxxa[f2, :i]) + np.sum(Zxxa[f2, i:(i+dt)]) - np.sum(Zxxa[f1, i:(i+dt)])
        if (vol > best):
            best = vol
            index = i
    for i in range(dt, len(arr) - dt - 1):
        vol = np.sum(Zxxa[f1, (i-dt):i]) - np.sum(Zxxa[f2, (i-dt):i]) + np.sum(Zxxa[f2, i:(i+dt)]) - np.sum(Zxxa[f1, i:(i+dt)])
        if (vol > best):
            best = vol
            index = i
    return int(t[index] * samplerate)

def sync_recording_to_labels(tap_cands, taps, time_scale, scrmax=180):
    """
    Automatic labelling function. 
    Selects the set of tap candidates that most closely resemble a sequence of labels. 
    Aims to maximise likelihood of candidate being a tap and minimise variance in label delays. 
    """
    n = len(tap_cands)
    m = len(taps)
    sm = np.zeros((n, m))
    sqs = np.zeros((n, m))
    scr = np.zeros((n, m))
    nf = np.zeros((n, m))
    prev = np.zeros((n, m), dtype=int)
    previd = np.zeros((n, m), dtype=int)
    C=1000.0
    B=30
    gbest=0
    endpoint = (-1, -1)
    for i in range(0, n):
        d = tap_cands[i][0] - taps[0].timestamp / time_scale
        sm[i, 0] = d
        sqs[i, 0] = d * d
        scr[i, 0] = min(B + tap_cands[i][1], scrmax)
        prev[i, 0] = -1
        previd[i, 0] = -1
        nf[i, 0] = 1
        for j in range(1, m):
            d = tap_cands[i][0] - taps[j].timestamp / time_scale
            prev[i, j] = -1
            previd[i, j] = -1
            sm[i, j] = d
            sqs[i, j] = d * d
            score = min(B + tap_cands[i][1], scrmax)
            scr[i, j] = score
            nf[i, j] = 1
            best = score / C
            for k in range(max(0, j - 15), j):
                dt = (taps[j].timestamp - taps[k].timestamp) / time_scale
                rs = 0
                re = i
                while rs < re:
                    if tap_cands[rs+re>>1][0] < tap_cands[i][0] - dt:
                        rs = (rs+re >> 1) + 1
                    else:
                        re = rs+re >> 1
                for l in range(max(0, rs - 15), min(i, rs + 15)):
                    lsm = sm[l, k] + d
                    lsqs = sqs[l, k] + d * d
                    lscr = scr[l, k] + score
                    lnf = nf[l, k] + 1 
                    s = lscr / (C + lsqs / lnf - lsm * lsm / lnf / lnf)
                    if s < 0:
                        import pdb; pdb.set_trace()
                    #print(i, j, l, k, ':', s)
                    if s > best:
                        best = s
                        sm[i, j] = lsm
                        sqs[i, j] = lsqs
                        scr[i, j] = lscr
                        prev[i, j] = l
                        previd[i, j] = k
                        nf[i, j] = lnf
                        if best >= gbest:
                            gbest = best
                            endpoint = (i, j)
            #print(i, j, best, prev[i, j], previd[i, j])
    result = -np.ones(m, dtype=int)
    i, j = endpoint
    while i != -1 and j != -1:
        result[j] = i
        i, j = (prev[i, j], previd[i, j])
    return result, gbest

def open_audio_files(jsons, experimentName, fallback_mode='ignore', verbose=False):
    tap_cands_in_jsons = []
    audio_in = []
    num_found = 0
    num_correctly_syncd = 0
    for i, json in enumerate(jsons):
        tap_cands = None
        fsx = None
        if json.experimentName == experimentName or (type(experimentName) is re.Pattern and experimentName.fullmatch(json.experimentName) is not None):
            num_found += 1
            found = 0
            avg_diff = 0
            for tap in json.taps_syncd:
                for s in json.taps_syncd_to_self_recording:
                    if abs(s.time_samples - tap.time_samples) < 1024:
                        found += 1
                        avg_diff += s.time_samples - tap.time_samples
                        break
            misaligned = found * 2 <= len(json.taps_syncd) + 1
            if misaligned:
                print("Labels misaligned:", json.recordingId)
                setattr(json, "taps_syncd_orig", json.taps_syncd) # used to mark misalignments for compatibility
                if fallback_mode == 'ignore':
                    print("ignoring input")
                    skip = True
                if fallback_mode == 'use_syncd_to_self':
                    print("Falling back to syncd to self label")
                    json.taps_syncd = json.taps_syncd_to_self_recording
                if fallback_mode == 'use_syncd':
                    print("using labels anyway")
            if not misaligned or fallback_mode != 'ignore':
                num_correctly_syncd += 1
                fs, x = wavfile.read(json.wav)
                fsx = [fs, x]
                fn = json.dir + '/tap_cands.npy'
                if found < len(json.taps_syncd) and not misaligned:
                    avg_diff /= found
                    for s in json.taps_syncd_to_self_recording:
                        found_this_one = False
                        for tap in json.taps_syncd:
                            if abs(s.time_samples - tap.time_samples) < 1024:
                                found_this_one = True
                                break
                        if not found_this_one:
                            new_tap = SimpleNamespace(**s.__dict__)
                            new_tap.time_samples -= avg_diff
                            new_tap.time_seconds -= avg_diff / fs
                            json.taps_syncd += [new_tap]
                if isfile(fn):
                    with open(fn, 'rb') as f:
                        tap_cands = np.load(f, allow_pickle=False)
                else:
                    f, t, Zxx = stft(x[:, 4], fs, nperseg=128, noverlap=96)
                    fc, l = Zxx.shape
                    Zxxa = np.abs(Zxx)
                    tap_cands = find_tap_cands(Zxxa, t, l, get_tap_tresh_for_device(json.deviceName))
                    with open(fn, 'wb') as f:
                        np.save(f, np.array(tap_cands), allow_pickle=False)
        tap_cands_in_jsons += [tap_cands]
        audio_in += [fsx]
        if verbose:
            print("found", num_found, '(',i+1,'/',len(jsons),') correctly sync\'d:', num_correctly_syncd)
    return audio_in, tap_cands_in_jsons, num_found, num_correctly_syncd

def get_tap_tresh_for_device(name):
    if name == "flounder":
        return 30
    if name == "CO2N_sprout":
        return 30
    return 30

def compute_snr_per_frequency(x, start, length=128, noise_start=None):
    if noise_start is None:
        noise_start = start - 512
    f, t, signal = stft(x[start:(start+length), :], axis=0, nperseg=length, noverlap=-length)
    f, t, noise = stft(x[noise_start:(noise_start+length), :], axis=0, nperseg=length, noverlap=-length)
    return np.abs(signal**2) / np.abs(noise**2)
